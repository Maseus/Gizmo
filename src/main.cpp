#include "chat_tui.hpp"
#include "chat_template.hpp"
#include "cli_parser.hpp"
#include "inference_engine.hpp"
#include "model_discovery.hpp"
#include "model_manager.hpp"
#include "proc_status.hpp"
#include "server/server.hpp"
#include "tokenizer.hpp"
#include "tui.hpp"
#include "util/string.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <functional>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;


// One benchmark cell: a (N, path) measurement.
struct bench_row_t {
    int      n_tokens;
    bool     sharded;
    double   wall_s;
    size_t   rss_bytes;
    size_t   hwm_bytes;
    // For the sharded path: VmRSS/VmHWM captured right after the
    // sharded prefill completed, BEFORE the un-sharded prefill
    // re-faulted the model. This is the actual "sharded steady
    // state" and what users care about for "would qwen3.8:27b
    // fit in 13 GB". For the un-sharded path, these are 0
    // (sharding is disabled, so the engine's getter returns 0).
    size_t   sharded_prefill_rss;
    size_t   sharded_prefill_hwm;
};

// Build a deterministic prompt string that tokenizes to roughly
// `target_tokens` tokens. Used by bench/sweep to control prefill size.
std::string build_prompt_for_token_count(
    const llama_vocab* vocab,
    int32_t target_tokens
) {
    const std::string prompt_template =
        "The capital of France is Paris. "
        "The capital of Germany is Berlin. "
        "The capital of Spain is Madrid. "
        "The capital of Italy is Rome. "
        "The capital of Japan is Tokyo. ";
    std::string repeated;
    while (repeated.size() < 4096) {
        repeated += prompt_template;
    }

    int32_t n = 0;
    std::vector<llama_token> tokens = gizmo::tokenize_text(
        vocab, repeated, /*add_special=*/true, /*parse_special=*/true, &n);
    if (n <= 0) {
        return "";
    }
    const int use_n = std::min(target_tokens, n);

    // Detokenize the first use_n tokens back to a string so it can
    // be passed to the text-based generate() API.
    return gizmo::tokens_to_string(
        vocab,
        std::vector<llama_token>(tokens.begin(), tokens.begin() + use_n),
        /*special=*/true);
}

// Run a single (N, sharded?) measurement against the engine.
// Returns the wall time of generate() and the post-run memory
// stats. The engine is reset_for_next_run() before the call so
// repeated calls don't accumulate KV cache.
//
// When prefill_only is true, the sharded-path run uses
// engine.prefill_only_tokens() (no decode, no un-sharded
// prefill, no model-page re-fault). The un-sharded-path run
// uses the normal llama_decode path (it's the baseline). This
// gives a clean wall-time comparison of "sharded prefill" vs
// "un-sharded prefill" without the double-prefill cost masking
// the sharded time.
bench_row_t run_one_bench(
    gizmo::InferenceEngine & engine,
    const std::vector<llama_token> & tokens,
    int n_tokens,
    bool sharded,
    int max_decode_tokens,
    bool prefill_only
) {
    const std::string prompt = "(benchmark)";  // unused; we feed tokens directly
    (void) prompt;

    engine.reset_for_next_run();

    gizmo::InferenceConfig cfg;
    cfg.max_tokens  = max_decode_tokens;
    cfg.temperature = 0.0f;  // deterministic for benchmarking

    const auto t0 = std::chrono::steady_clock::now();
    // We need a prompt string; tokenize it via the engine's vocab.
    // The simplest approach: feed the tokens back as a string of
    // placeholder characters. llama_tokenize the placeholder, but
    // that produces different tokens. Instead, decode the tokens
    // to a string via the model's vocab.
    // Simpler still: just call engine.generate with an empty
    // prompt and let the engine build the input. But the engine
    // itself tokenizes the prompt, and we want to control N
    // exactly. So: use the vocab to detokenize the tokens back
    // to a string, then pass that to generate. Slight rounding
    // error possible but good enough for memory benchmarks.
    // (Phase 7+ can add a direct tokens-passthrough API.)
    std::vector<llama_token> use_tokens(tokens.begin(), tokens.begin() + n_tokens);
    const std::string prompt_str = gizmo::tokens_to_string(
        llama_model_get_vocab(engine.raw_model()), use_tokens, /*special=*/true);
    if (sharded && prefill_only) {
        // Sharded-prefill-only path: skip the un-sharded prefill
        // and the decode loop. Measures only the sharded prefill
        // wall time, which is what users care about for "would
        // sharded prefill alone be fast enough for qwen3.8:27b?"
        if (n_tokens <= (int)tokens.size()) {
            (void) engine.prefill_only_tokens(tokens.data(), n_tokens);
        } else {
            (void) engine.prefill_only(prompt_str);
        }
    } else {
        (void) engine.generate(prompt_str, cfg);
    }
    const auto t1 = std::chrono::steady_clock::now();

    const double wall_s =
        std::chrono::duration<double>(t1 - t0).count();
    const size_t rss = gizmo::read_vm_rss_bytes();
    const size_t hwm = gizmo::read_vm_hwm_bytes();
    // For the sharded path, capture the steady-state VmRSS/HWM
    // from right after the sharded prefill. These are the real
    // "sharded path" memory numbers when the model pages have not
    // yet been forced resident by an un-sharded run.
    const size_t sharded_rss = engine.last_sharded_prefill_rss();
    const size_t sharded_hwm = engine.last_sharded_prefill_hwm();
    return { n_tokens, sharded, wall_s, rss, hwm, sharded_rss, sharded_hwm };
}

void print_bench_table(
    const std::vector<bench_row_t> & rows,
    const std::string & model_path,
    int n_layer,
    int n_embd,
    bool prefill_only
) {
    std::cout << "\nGizmo bench: " << model_path << "\n";
    std::cout << "  layers=" << n_layer << "  embd=" << n_embd << "\n";
    std::cout << "  decode tokens per run: 1\n";
    std::cout << "\n";
    // For sharded columns, show the *sharded prefill* VmHWM/VmRSS
    // (captured right after the per-block engine finished). This is
    // the real "sharded path" memory cost when pages have not been
    // forced resident by an earlier un-sharded run.
    std::cout << " tokens | un-shard (s) | un-shard HWM (MB) | un-shard RSS (MB)"
              << " | shard (s) | shard prefill HWM (MB) | shard prefill RSS (MB)"
              << " | saved (MB)\n";
    std::cout << "-------+--------------+-------------------+------------------"
              << " +-----------+----------------------+--------------------"
              << "+------------\n";

    // Rows are stored in two contiguous blocks: first all
    // un-sharded runs, then all sharded runs. Pair them by N.
    const size_t npairs = rows.size() / 2;
    for (size_t i = 0; i < npairs; ++i) {
        const bench_row_t & u = rows[i];              // un-sharded
        const bench_row_t & s = rows[npairs + i];      // sharded
        // Savings is the un-sharded VmHWM (full model resident)
        // minus the sharded prefill VmRSS (one block resident
        // after eviction). HWM is monotonic and not a fair
        // comparison for the sharded path since the un-sharded
        // prefill for KV cache warming re-faults the model.
        // Compute as signed to avoid unsigned underflow.
        const long saved_mb = (long)(
            ((long long)u.hwm_bytes - (long long)s.sharded_prefill_rss)
            / (1024 * 1024));
        std::cout << std::setw(6) << u.n_tokens << " | "
                  << std::fixed << std::setprecision(2)
                  << std::setw(13) << u.wall_s << " | "
                  << std::setw(17) << (u.hwm_bytes / (1024 * 1024)) << " | "
                  << std::setw(16) << (u.rss_bytes / (1024 * 1024)) << " | "
                  << std::setw(10) << s.wall_s << " | "
                  << std::setw(20) << (s.sharded_prefill_hwm / (1024 * 1024)) << " | "
                  << std::setw(18) << (s.sharded_prefill_rss / (1024 * 1024)) << " | "
                  << std::setw(10) << saved_mb << "\n";
    }
    std::cout << "\nNote: sharded prefill HWM/RSS are captured at the end of the"
              << "\nsharded per-block engine, before the un-sharded prefill re-faults"
              << "\nthe model for KV-cache warming. The HWM is high because the"
              << "\nmodel load + first few blocks' pages are still in the page"
              << "\ncache; the RSS is what the sharded path holds in RAM after"
              << "\neach block is evicted.\n";
    if (!prefill_only) {
        std::cout << "Sharded wall time in this table includes a 2x prefill"
                  << "\n(sharded + un-sharded); with --prefill-only it is one"
                  << "\nprefill. A future phase removes the duplicate.\n";
    } else {
        std::cout << "--prefill-only was set: sharded wall time is one prefill.\n";
    }

    // Extrapolation: if this is Qwen3-8B (8B params, 36 layers),
    // project what a qwen3.8:27b run would look like on the same
    // machine.
    //
    // The sharded prefill RSS has two parts:
    //   - scheduler overhead (~313 MB, constant, doesn't scale)
    //   - per-block weights + intermediate tensors
    //     (per-block scales with model_size / n_layer)
    //
    // qwen3.8:27b has 64 layers and 3.4x the parameters of 8B,
    // so its per-block is 3.4x * 36/64 = 1.9x the 8B per-block.
    // Intermediate tensors depend on n_embd (4096 for both 8B and
    // 27B) and n_tokens, so they don't scale up. The scheduler
    // overhead is constant.
    //
    // So a reasonable estimate is:
    //   27B sharded prefill RSS ~=  scheduler_overhead
    //                              + (8B RSS - 8B scheduler) * 1.9
    //
    // For un-sharded, the resident model is the whole GGUF which
    // scales with the model size (3.4x).
    const bool is_qwen3_8b =
        (n_layer == 36 && model_path.find("Qwen3-8B") != std::string::npos);
    if (is_qwen3_8b && !rows.empty()) {
        // Use the largest-N sharded prefill RSS (most representative
        // for a real-world prompt). For un-sharded, the HWM is the
        // model's mmap footprint which is the same regardless of N.
        const bench_row_t & shard_last = rows[rows.size() - 1];
        const bench_row_t & unshard_last = rows[rows.size() / 2 - 1];
        const long long k_param = 27;     // 27B params
        const long long k_8b   = 8;       // 8B params
        const long long n_layer_8b  = 36; // Qwen3-8B layer count
        const long long n_layer_27b = 64; // qwen3.8:27b layer count
        const long long scheduler_overhead_mb = 313;  // CPU buffer pool
        const long long shard_8b_mb =
            (long long)(shard_last.sharded_prefill_rss / (1024 * 1024));
        const long long unshard_8b_mb =
            (long long)(unshard_last.hwm_bytes / (1024 * 1024));
        // 27B sharded prefill RSS: scheduler + scaled per-block
        const long long per_block_8b_mb = shard_8b_mb - scheduler_overhead_mb;
        const long long per_block_27b_mb =
            per_block_8b_mb * k_param * n_layer_8b / (k_8b * n_layer_27b);
        const long long shard_27b_mb = scheduler_overhead_mb + per_block_27b_mb;
        // 27B un-sharded: model scales linearly with param count
        const long long unshard_27b_mb = unshard_8b_mb * k_param / k_8b;
        std::cout << "\nProjection for qwen3.8:27b on this machine:\n";
        std::cout << "  un-sharded 27B VmHWM: ~" << unshard_27b_mb
                  << " MB  (does NOT fit 13 GB if > 13000 MB)\n";
        std::cout << "  sharded 27B prefill VmRSS: ~" << shard_27b_mb
                  << " MB  (fits 13 GB if < 13000 MB)\n";
        std::cout << "  Estimate assumes 27B has same Q4_K_M quantization,\n"
                  << "  same n_embd=4096, 64 layers (vs 36 for 8B), and\n"
                  << "  same sharded-engine scheduler overhead (~313 MB).\n";
    }
}

}  // namespace

static int do_validate(
    const std::string& model_path,
    int32_t resident_layers,
    bool no_evict,
    int32_t row_size,
    int32_t threads,
    bool verbose,
    bool progress,
    bool json_output,
    bool argmax_only,
    float max_diff_threshold,
    float mean_diff_threshold,
    int32_t context_size
) {
    // Built-in prompt suite. Mix of lengths and content to exercise
    // embedding lookup, causal attention, and (on qwen3.5) hybrid
    // full-attention + recurrent routing.
    const std::vector<std::string> prompts = {
        "Hi",
        "Hello world",
        "What is the capital of France?",
        "The capital of France is Paris. The capital of Germany is Berlin. "
        "The capital of Spain is Madrid. The capital of Italy is Rome. "
        "The capital of Japan is Tokyo.",
    };

    struct validate_case_t {
        std::string prompt;
        int n_tokens = 0;
        int baseline_argmax = -1;
        int sharded_argmax = -1;
        float baseline_best = 0.0f;
        float sharded_best = 0.0f;
        float max_abs_diff = 0.0f;
        double mean_abs_diff = 0.0;
        bool pass = false;
    };
    auto run_suite = [&](
        gizmo::InferenceEngine& engine,
        bool is_baseline,
        std::vector<validate_case_t>& out_results
    ) -> bool {
        out_results.clear();
        out_results.reserve(prompts.size());
        for (const auto& prompt : prompts) {
            // Each prompt must start from an empty KV/recurrent cache so the
            // logits match a fresh llama_decode run. Without this reset the
            // previous prompt's K/V or SSM state leaks into the next case.
            engine.reset_for_next_run();
            std::vector<float> logits;
            if (engine.validate_prefill(prompt, logits) != 0) {
                if (verbose) {
                    std::cerr << "[validate] failed to run prompt: " << prompt << "\n";
                }
                return false;
            }
            if (logits.empty()) {
                return false;
            }
            validate_case_t r;
            r.prompt = prompt;
            r.n_tokens = 0;  // filled later if needed
            int best_id = 0;
            float best_val = logits[0];
            for (size_t i = 1; i < logits.size(); ++i) {
                if (logits[i] > best_val) { best_val = logits[i]; best_id = (int)i; }
            }
            // We need per-case token counts for the report. Tokenize here
            // using the engine vocab; the value is the same for baseline
            // and sharded so either branch can fill it.
            {
                int32_t n = 0;
                (void)gizmo::tokenize_text(
                    llama_model_get_vocab(engine.raw_model()), prompt,
                    /*add_special=*/true, /*parse_special=*/true, &n);
                r.n_tokens = n > 0 ? n : 0;
            }
            if (is_baseline) {
                r.baseline_argmax = best_id;
                r.baseline_best = best_val;
            } else {
                r.sharded_argmax = best_id;
                r.sharded_best = best_val;
            }
            out_results.push_back(r);
        }
        return true;
    };

    // Baseline: load without sharding so llama_decode prefaults the
    // full model and gives the un-sharded reference logits.
    gizmo::InferenceEngine baseline_engine;
    baseline_engine.set_verbose(verbose);
    baseline_engine.set_progress(progress);
    baseline_engine.set_threads(threads);
    baseline_engine.set_context_size(context_size);
    if (!baseline_engine.initialize(model_path, /*layer_shard_lazy=*/false)) {
        std::cerr << "[validate] failed to load model (baseline): " << model_path << "\n";
        return 1;
    }

    std::vector<validate_case_t> baseline_results;
    if (!run_suite(baseline_engine, /*is_baseline=*/true, baseline_results)) {
        std::cerr << "[validate] baseline suite failed\n";
        return 1;
    }

    // Sharded: reload with lazy mmap and enable the per-block engine.
    gizmo::InferenceEngine sharded_engine;
    sharded_engine.set_verbose(verbose);
    sharded_engine.set_progress(progress);
    sharded_engine.set_threads(threads);
    sharded_engine.set_context_size(context_size);
    if (!sharded_engine.initialize(model_path, /*layer_shard_lazy=*/true)) {
        std::cerr << "[validate] failed to load model (sharded): " << model_path << "\n";
        return 1;
    }
    const int32_t val_resident = resident_layers > 0 ? resident_layers : 1;
    const int32_t val_row_size = row_size > 0 ? row_size : 1;
    sharded_engine.enable_sharded_engine(
        val_resident, /*evict_weights=*/!no_evict, val_row_size);

    std::vector<validate_case_t> sharded_results;
    if (!run_suite(sharded_engine, /*is_baseline=*/false, sharded_results)) {
        std::cerr << "[validate] sharded suite failed\n";
        return 1;
    }

    if (baseline_results.size() != sharded_results.size()) {
        std::cerr << "[validate] mismatched result counts\n";
        return 1;
    }

    // Compare and fill pass/fail metrics.
    int failures = 0;
    for (size_t i = 0; i < baseline_results.size(); ++i) {
        const auto& b = baseline_results[i];
        auto& s = sharded_results[i];
        s.baseline_argmax = b.baseline_argmax;
        s.baseline_best = b.baseline_best;

        // Need logits to compute diffs. Re-run the sharded prefill for
        // this prompt and keep the logits. This is the simplest path
        // without adding a temp member to validate_case_t.
        sharded_engine.reset_for_next_run();
        std::vector<float> shard_logits;
        if (sharded_engine.validate_prefill(s.prompt, shard_logits) != 0) {
            s.pass = false;
            ++failures;
            continue;
        }

        // Re-run baseline to get its logits for diff. Reset the cache
        // before every prompt so the baseline matches a fresh context.
        baseline_engine.reset_for_next_run();
        std::vector<float> base_logits;
        if (baseline_engine.validate_prefill(s.prompt, base_logits) != 0) {
            s.pass = false;
            ++failures;
            continue;
        }

        if (shard_logits.size() != base_logits.size() || shard_logits.empty()) {
            s.pass = false;
            ++failures;
            continue;
        }

        double sum_abs = 0.0;
        float max_abs = 0.0f;
        for (size_t j = 0; j < base_logits.size(); ++j) {
            const float d = shard_logits[j] - base_logits[j];
            const float ad = d < 0 ? -d : d;
            sum_abs += ad;
            if (ad > max_abs) max_abs = ad;
        }
        s.max_abs_diff = max_abs;
        s.mean_abs_diff = sum_abs / (double)base_logits.size();
        const bool argmax_match = (s.sharded_argmax == s.baseline_argmax);
        const bool logit_match = argmax_only ||
                                 (s.max_abs_diff < max_diff_threshold &&
                                  s.mean_abs_diff < mean_diff_threshold);
        s.pass = argmax_match && logit_match;
        if (!s.pass) ++failures;
    }

    // Output.
    if (json_output) {
        std::cout << "{\n";
        std::cout << "  \"model\": \"" << model_path << "\",\n";
        std::cout << "  \"pass\": " << (failures == 0 ? "true" : "false") << ",\n";
        std::cout << "  \"resident_layers\": " << val_resident << ",\n";
        std::cout << "  \"row_size\": " << val_row_size << ",\n";
        std::cout << "  \"cases\": [\n";
        for (size_t i = 0; i < sharded_results.size(); ++i) {
            const auto& r = sharded_results[i];
            std::cout << "    {\n";
            std::cout << "      \"prompt\": \"" << r.prompt << "\",\n";
            std::cout << "      \"n_tokens\": " << r.n_tokens << ",\n";
            std::cout << "      \"baseline_argmax\": " << r.baseline_argmax << ",\n";
            std::cout << "      \"sharded_argmax\": " << r.sharded_argmax << ",\n";
            std::cout << "      \"max_abs_diff\": " << r.max_abs_diff << ",\n";
            std::cout << "      \"mean_abs_diff\": " << r.mean_abs_diff << ",\n";
            std::cout << "      \"pass\": " << (r.pass ? "true" : "false") << "\n";
            std::cout << "    }" << (i + 1 < sharded_results.size() ? "," : "") << "\n";
        }
        std::cout << "  ]\n";
        std::cout << "}\n";
    } else {
        std::cout << "\nGizmo Validate: " << model_path << "\n";
        std::cout << "  resident_layers=" << val_resident
                  << " row_size=" << val_row_size << "\n\n";
        std::cout << "prompt                                 | tokens | argmax_match | max_abs_diff | mean_abs_diff | status\n";
        std::cout << "---------------------------------------+--------+--------------+--------------+---------------+-------\n";
        for (const auto& r : sharded_results) {
            std::string display = r.prompt;
            if (display.size() > 38) display = display.substr(0, 35) + "...";
            std::cout << std::left << std::setw(39) << display << "| ";
            std::cout << std::right << std::setw(6) << r.n_tokens << " | ";
            std::cout << std::setw(12) << (r.sharded_argmax == r.baseline_argmax ? "yes" : "NO")
                      << " | ";
            std::cout << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.max_abs_diff << " | ";
            std::cout << std::scientific << std::setprecision(2)
                      << std::setw(13) << r.mean_abs_diff << " | ";
            std::cout << (r.pass ? "PASS" : "FAIL") << "\n";
        }
        std::cout << "\n" << (failures == 0 ? "All cases passed." : std::to_string(failures) + " case(s) failed.")
                  << "\n";
    }

    return failures > 0 ? std::min(failures, 127) : 0;
}

// Minimal interactive chat. Maintains a simple text context, prints the
// assistant response as it streams, and shows tok/s plus memory at the end
// of each turn. Commands: /quit, /reset, /clear.
static int do_chat(
    const std::string& model_path,
    int32_t resident_layers,
    bool no_shard,
    bool no_evict,
    int32_t row_size,
    int32_t threads,
    int32_t max_tokens,
    int32_t context_size,
    bool measure_ram,
    int32_t measure_interval_ms,
    bool verbose,
    bool progress
) {
    gizmo::InferenceEngine engine;
    engine.set_verbose(verbose);
    engine.set_progress(progress);
    engine.set_threads(threads);
    engine.set_context_size(context_size);
    if (!engine.initialize(model_path, /*layer_shard_lazy=*/!no_shard)) {
        std::cerr << "Failed to initialize inference engine\n";
        return 1;
    }

    if (!no_shard) {
        engine.enable_sharded_engine(
            resident_layers > 0 ? resident_layers : 8,
            /*evict_weights=*/!no_evict,
            row_size > 0 ? row_size : 1);
        if (!engine.is_sharded()) {
            std::cout << "Note: sharded engine was not enabled; falling back to llama_decode.\n";
        }
    }

    const struct llama_vocab* vocab = llama_model_get_vocab(engine.raw_model());

    std::cout << "\nGizmo Chat: " << model_path << "\n";
    std::cout << "Type a message and press Enter. Commands: /quit, /reset, /clear\n";
    std::cout << "---------------------------------------------------------------\n";

    // Chat history used with the model's built-in chat template.
    std::vector<std::pair<std::string, std::string>> messages;
    messages.push_back({"system", "You are a helpful assistant."});

    // Background memory measurement thread.
    std::atomic<bool> stop_measure{false};
    std::thread measure_thread;
    if (measure_ram) {
        measure_thread = std::thread([&]() {
            while (!stop_measure.load()) {
                const size_t rss = gizmo::read_vm_rss_bytes();
                const size_t hwm = gizmo::read_vm_hwm_bytes();
                std::cerr << "[measure] VmRSS=" << (rss / (1024 * 1024))
                          << " MB  VmHWM=" << (hwm / (1024 * 1024)) << " MB\n";
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(measure_interval_ms));
            }
        });
    }

    gizmo::InferenceConfig cfg;
    cfg.max_tokens = max_tokens > 0 ? max_tokens : 128;
    cfg.temperature = 0.8f;

    std::string input;
    while (true) {
        std::cout << "\nYou: ";
        std::cout.flush();
        if (!std::getline(std::cin, input)) {
            break;
        }

        if (input == "/quit" || input == "/exit") {
            break;
        }
        if (input == "/reset" || input == "/clear") {
            messages.clear();
            messages.push_back({"system", "You are a helpful assistant."});
            engine.reset_for_next_run();
            std::cout << "[chat] context cleared\n";
            continue;
        }
        if (input.empty()) {
            continue;
        }

        // Add the new user message and format the full conversation with the
        // model's chat template, ending with the assistant prefix.
        messages.push_back({"user", input});
        std::string prompt = gizmo::apply_chat_template(engine.raw_model(), messages, /*add_ass=*/true);
        if (prompt.empty()) {
            // Fall back to a simple format if no template is available.
            prompt.clear();
            for (const auto& m : messages) {
                prompt += m.first + ": " + m.second + "\n\n";
            }
            prompt += "assistant: ";
        }

        std::cout << "Assistant: ";
        std::cout.flush();

        engine.reset_for_next_run();
        const auto t0 = std::chrono::steady_clock::now();
        std::string response = engine.generate(prompt, cfg, /*quiet=*/true);
        const auto t1 = std::chrono::steady_clock::now();

        // Store the assistant response in the history for multi-turn chat.
        messages.push_back({"assistant", response});

        std::cout << response;

        // Count generated tokens for speed reporting.
        int32_t n_response_tokens = 0;
        (void)gizmo::tokenize_text(
            vocab, response,
            /*add_special=*/false, /*parse_special=*/false, &n_response_tokens);
        if (n_response_tokens < 0) n_response_tokens = 0;

        const double wall_s = std::chrono::duration<double>(t1 - t0).count();
        const double tok_s = wall_s > 0.0 ? n_response_tokens / wall_s : 0.0;
        const size_t rss = gizmo::read_vm_rss_bytes();
        const size_t hwm = gizmo::read_vm_hwm_bytes();

        std::cout << "\n  [" << n_response_tokens << " tokens in "
                  << std::fixed << std::setprecision(2) << wall_s << " s => "
                  << std::setprecision(1) << tok_s << " tok/s; RSS "
                  << (rss / (1024 * 1024)) << " MB; HWM "
                  << (hwm / (1024 * 1024)) << " MB]\n";

    }

    stop_measure.store(true);
    if (measure_thread.joinable()) measure_thread.join();

    std::cout << "\nChat ended.\n";
    return 0;
}

// One sweep cell: a (t, r, K, path) measurement over a full generate() run.
struct sweep_row_t {
    std::string config;    // "unshard" or "shard" or "fallback"
    int32_t     t = -1;    // CPU threads; -1 if not varied
    int32_t     r = -1;    // resident_layers; -1 for un-sharded baseline
    int32_t     K = -1;    // row_size; -1 for un-sharded baseline
    double      wall_s = 0.0;
    size_t      rss_bytes = 0;
    size_t      hwm_bytes = 0;
    size_t      prefill_rss = 0;  // 0 for un-sharded / fallback
    size_t      prefill_hwm = 0;  // 0 for un-sharded / fallback
    bool        failed = false;
    bool        fallback = false;
};

static double tok_per_s(int32_t prefill_tokens, int32_t decode_tokens, double wall_s) {
    if (wall_s <= 0.0) return 0.0;
    return static_cast<double>(prefill_tokens + decode_tokens) / wall_s;
}

static int do_sweep(
    const std::string& model_path,
    const std::vector<int32_t>& resident_layers_list,
    const std::vector<int32_t>& row_size_list,
    const std::vector<int32_t>& threads_list,
    bool no_evict,
    int32_t max_tokens,
    int32_t prefill_tokens,
    bool json_output
) {
    // Load the model once with lazy mmap for the sharded sweep configs.
    gizmo::InferenceEngine sharded_engine;
    if (!sharded_engine.initialize(model_path, /*layer_shard_lazy=*/true)) {
        std::cerr << "Failed to load model for sweep (sharded): " << model_path << "\n";
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(sharded_engine.raw_model());

    const std::string prompt = build_prompt_for_token_count(vocab, prefill_tokens);
    if (prompt.empty()) {
        std::cerr << "Failed to build sweep prompt\n";
        return 1;
    }

    // Actual token count of the generated prompt.
    int32_t actual_prefill = 0;
    (void)gizmo::tokenize_text(vocab, prompt,
                               /*add_special=*/true, /*parse_special=*/true, &actual_prefill);
    if (actual_prefill < 0) actual_prefill = 0;

    // Deterministic generation config.
    gizmo::InferenceConfig cfg;
    cfg.max_tokens = max_tokens > 0 ? max_tokens : 16;
    cfg.temperature = 0.0f;

    std::vector<sweep_row_t> rows;
    rows.reserve(threads_list.size() *
                  (1 + resident_layers_list.size() * row_size_list.size()));

    // Load one baseline engine with full prefault. Threads are varied at
    // runtime via set_threads(), so the model is loaded only once.
    gizmo::InferenceEngine baseline_engine;
    if (!baseline_engine.initialize(model_path, /*layer_shard_lazy=*/false)) {
        std::cerr << "Failed to load model for sweep (baseline): " << model_path << "\n";
        return 1;
    }

    // Iterate over thread counts in the outer loop so each thread value gets
    // its own un-sharded baseline and a full (r, K) matrix.
    for (int32_t t : threads_list) {
        const int32_t use_t = t > 0 ? t : 1;

        // 1) Un-sharded baseline for this thread count.
        baseline_engine.set_threads(use_t);
        baseline_engine.reset_for_next_run();

        std::cout << "[sweep] threads=" << use_t << " running un-sharded baseline...\n";
        const auto tb0 = std::chrono::steady_clock::now();
        (void) baseline_engine.generate(prompt, cfg, /*quiet=*/true);
        const auto tb1 = std::chrono::steady_clock::now();

        sweep_row_t base_row;
        base_row.config = "unshard";
        base_row.t = use_t;
        base_row.wall_s = std::chrono::duration<double>(tb1 - tb0).count();
        base_row.rss_bytes = gizmo::read_vm_rss_bytes();
        base_row.hwm_bytes = gizmo::read_vm_hwm_bytes();
        rows.push_back(base_row);

        // 2) Sharded configurations: for each (r, K), set threads, enable the
        //    scheduler, reset KV cache, and run a quiet generate().
        for (int32_t r : resident_layers_list) {
            for (int32_t K : row_size_list) {
                const int32_t use_r = r > 0 ? r : 1;
                const int32_t use_K = K > 0 ? K : 1;

                sharded_engine.set_threads(use_t);
                sharded_engine.enable_sharded_engine(
                    use_r, /*evict_weights=*/!no_evict, use_K);
                if (!sharded_engine.is_sharded()) {
                    // The architecture isn't supported by the sharded engine; this
                    // configuration runs the same un-sharded path as the baseline.
                    // Measure it once and report it as a fallback, not a failure.
                    std::cout << "[sweep] threads=" << use_t
                              << " r=" << use_r << " K=" << use_K
                              << " (sharded unsupported; running fallback)...\n";
                    sharded_engine.reset_for_next_run();

                    const auto t0 = std::chrono::steady_clock::now();
                    (void) sharded_engine.generate(prompt, cfg, /*quiet=*/true);
                    const auto t1 = std::chrono::steady_clock::now();

                    sweep_row_t row;
                    row.config = "fallback";
                    row.t = use_t;
                    row.r = use_r;
                    row.K = use_K;
                    row.fallback = true;
                    row.wall_s = std::chrono::duration<double>(t1 - t0).count();
                    row.rss_bytes = gizmo::read_vm_rss_bytes();
                    row.hwm_bytes = gizmo::read_vm_hwm_bytes();
                    rows.push_back(row);
                    continue;
                }

                sharded_engine.reset_for_next_run();

                std::cout << "[sweep] threads=" << use_t
                          << " r=" << use_r << " K=" << use_K << " ...\n";
                const auto t0 = std::chrono::steady_clock::now();
                (void) sharded_engine.generate(prompt, cfg, /*quiet=*/true);
                const auto t1 = std::chrono::steady_clock::now();

                sweep_row_t row;
                row.config = "shard";
                row.t = use_t;
                row.r = use_r;
                row.K = use_K;
                row.wall_s = std::chrono::duration<double>(t1 - t0).count();
                row.rss_bytes = gizmo::read_vm_rss_bytes();
                row.hwm_bytes = gizmo::read_vm_hwm_bytes();
                row.prefill_rss = sharded_engine.last_sharded_prefill_rss();
                row.prefill_hwm = sharded_engine.last_sharded_prefill_hwm();
                rows.push_back(row);
            }
        }
    }

    // Output.
    if (json_output) {
        std::cout << "{\n";
        std::cout << "  \"model\": \"" << model_path << "\",\n";
        std::cout << "  \"prefill_tokens\": " << actual_prefill << ",\n";
        std::cout << "  \"decode_tokens\": " << cfg.max_tokens << ",\n";
        std::cout << "  \"no_evict\": " << (no_evict ? "true" : "false") << ",\n";
        std::cout << "  \"rows\": [\n";
        for (size_t i = 0; i < rows.size(); ++i) {
            const auto& row = rows[i];
            std::cout << "    {\n";
            std::cout << "      \"config\": \"" << row.config << "\",\n";
            std::cout << "      \"threads\": " << row.t << ",\n";
            std::cout << "      \"r\": " << (row.r < 0 ? "null" : std::to_string(row.r)) << ",\n";
            std::cout << "      \"K\": " << (row.K < 0 ? "null" : std::to_string(row.K)) << ",\n";
            std::cout << "      \"wall_s\": " << row.wall_s << ",\n";
            std::cout << "      \"tok_s\": " << tok_per_s(actual_prefill, cfg.max_tokens, row.wall_s) << ",\n";
            std::cout << "      \"vm_rss_mb\": " << (row.rss_bytes / (1024 * 1024)) << ",\n";
            std::cout << "      \"vm_hwm_mb\": " << (row.hwm_bytes / (1024 * 1024)) << ",\n";
            std::cout << "      \"prefill_rss_mb\": " << (row.prefill_rss / (1024 * 1024)) << ",\n";
            std::cout << "      \"prefill_hwm_mb\": " << (row.prefill_hwm / (1024 * 1024)) << ",\n";
            std::cout << "      \"failed\": " << (row.failed ? "true" : "false") << ",\n";
            std::cout << "      \"fallback\": " << (row.fallback ? "true" : "false") << "\n";
            std::cout << "    }" << (i + 1 < rows.size() ? "," : "") << "\n";
        }
        std::cout << "  ],\n";

        // Summary: fastest and smallest RSS among non-failed sharded configs
        // (exclude the un-sharded baseline, which is only a reference point).
        const sweep_row_t* fastest = nullptr;
        const sweep_row_t* smallest = nullptr;
        for (const auto& row : rows) {
            if (row.failed || row.fallback || row.config == "unshard") continue;
            if (fastest == nullptr || row.wall_s < fastest->wall_s) fastest = &row;
            if (smallest == nullptr || row.rss_bytes < smallest->rss_bytes) smallest = &row;
        }
        std::cout << "  \"summary\": {\n";
        if (fastest != nullptr) {
            std::cout << "    \"fastest\": {\"threads\": " << fastest->t
                      << ", \"r\": " << fastest->r
                      << ", \"K\": " << fastest->K
                      << ", \"wall_s\": " << fastest->wall_s << "},\n";
        }
        if (smallest != nullptr) {
            std::cout << "    \"smallest_rss\": {\"threads\": " << smallest->t
                      << ", \"r\": " << smallest->r
                      << ", \"K\": " << smallest->K
                      << ", \"vm_rss_mb\": " << (smallest->rss_bytes / (1024 * 1024)) << "}\n";
        }
        std::cout << "  }\n";
        std::cout << "}\n";
    } else {
        std::cout << "\nGizmo Sweep: " << model_path << "\n";
        std::cout << "  prefill tokens=" << actual_prefill
                  << "  decode tokens=" << cfg.max_tokens
                  << "  no_evict=" << (no_evict ? "true" : "false") << "\n\n";
        std::cout << " config  |  t |  r |  K | wall (s) |  tok/s | VmRSS (MB) | VmHWM (MB) | prefill RSS (MB) | prefill HWM (MB)\n"
                  << "---------+----+----+----+----------+--------+------------+------------+------------------+------------------\n";
        for (const auto& row : rows) {
            std::cout << std::left << std::setw(8) << row.config << " | " << std::right;
            if (row.r < 0) {
                std::cout << std::setw(3) << row.t << " | "
                          << std::setw(3) << "-" << " | "
                          << std::setw(3) << "-" << " |";
            } else {
                std::cout << std::setw(3) << row.t << " | "
                          << std::setw(3) << row.r << " | "
                          << std::setw(3) << row.K << " |";
            }
            if (row.failed || row.fallback) {
                if (row.fallback) {
                    std::cout << std::fixed << std::setprecision(2)
                              << std::setw(9) << row.wall_s << " | "
                              << std::setw(6) << tok_per_s(actual_prefill, cfg.max_tokens, row.wall_s) << " | "
                              << std::setw(10) << (row.rss_bytes / (1024 * 1024)) << " | "
                              << std::setw(10) << (row.hwm_bytes / (1024 * 1024)) << " | "
                              << std::setw(16) << "-" << " | "
                              << std::setw(16) << "-" << "\n";
                } else {
                    std::cout << std::setw(9) << "FAILED" << " | "
                              << std::setw(6) << "-" << " | "
                              << std::setw(10) << "-" << " | "
                              << std::setw(10) << "-" << " | "
                              << std::setw(16) << "-" << " | "
                              << std::setw(16) << "-" << "\n";
                }
                continue;
            }
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(9) << row.wall_s << " | "
                      << std::setw(6) << tok_per_s(actual_prefill, cfg.max_tokens, row.wall_s) << " | "
                      << std::setw(10) << (row.rss_bytes / (1024 * 1024)) << " | "
                      << std::setw(10) << (row.hwm_bytes / (1024 * 1024)) << " | ";
            if (row.prefill_rss > 0) {
                std::cout << std::setw(16) << (row.prefill_rss / (1024 * 1024)) << " | "
                          << std::setw(16) << (row.prefill_hwm / (1024 * 1024)) << "\n";
            } else {
                std::cout << std::setw(16) << "-" << " | "
                          << std::setw(16) << "-" << "\n";
            }
        }
        std::cout << "\n";
        std::cout << "Note: VmHWM is process-wide and monotonic. After the un-sharded\n"
                  << "baseline runs, the sharded HWM values are capped by the baseline\n"
                  << "peak. Use VmRSS and prefill RSS for per-config memory comparison.\n";
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

// Poll a local HTTP endpoint until it reports ready or the deadline expires.
// `path` is appended to http://host:port (e.g. "/v1/health" or "/health").
static bool wait_for_server_ready(const std::string& host, int32_t port,
                                    const std::string& path,
                                    int32_t timeout_seconds) {
    const std::string url = "http://" + host + ":" + std::to_string(port) + path;
    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::seconds(timeout_seconds);
    while (std::chrono::steady_clock::now() < deadline) {
        // Use curl via system() so we do not add a dependency on libcurl.
        const std::string cmd = "curl -s --max-time 2 " + url + " > /dev/null 2>&1";
        if (std::system(cmd.c_str()) == 0) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return false;
}

// Return a host string suitable for a client to connect to. When the server
// is bound to 0.0.0.0, clients should connect via localhost.
static std::string client_host(const std::string& bind_host) {
    if (bind_host == "0.0.0.0") return "127.0.0.1";
    return bind_host;
}

static bool command_exists(const char* name) {
    const std::string cmd = std::string("command -v ") + name + " > /dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
}

static std::string find_claude_executable() {
    if (command_exists("claude"))        return "claude";
    if (command_exists("claude-code"))   return "claude-code";
    if (command_exists("claude-desktop")) return "claude-desktop";
    return "";
}

// Start a child process. Returns the PID on success, -1 on failure.
// The command is executed directly (execvp) so shell quoting is not needed.
static pid_t start_child_process(const std::string& exe,
                                  const std::vector<std::string>& args) {
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(exe.c_str()));
    for (const auto& a : args) {
        argv.push_back(const_cast<char*>(a.c_str()));
    }
    argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0) {
        return -1;
    }
    if (pid == 0) {
        // Child: redirect stdout/stderr to our own so server logs appear inline.
        // We intentionally do not detach so Gizmo can terminate the process.
        execvp(exe.c_str(), argv.data());
        std::perror("execvp failed");
        _exit(127);
    }
    return pid;
}

// Send SIGTERM, then SIGKILL if the process is still alive after a short wait.
static void terminate_child(pid_t pid) {
    if (pid <= 0) return;
    kill(pid, SIGTERM);
    for (int i = 0; i < 50; ++i) {
        if (kill(pid, 0) != 0) return; // process is gone
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    kill(pid, SIGKILL);
}

// Launch the Claude Code CLI pointed at `base_url` with `model_id`.
// If the CLI is not installed, prints the manual connection command and
// returns after the user interrupts us. `server_is_running` and
// `stop_requested` are polled so the fallback exits cleanly.
static int launch_claude_client(
    const std::string& base_url,
    const std::string& model_id,
    std::function<bool()> server_is_running,
    std::function<bool()> stop_requested
) {
    const std::string env_prefix =
        "OPENAI_BASE_URL=" + base_url + " "
        "OPENAI_API_KEY=gizmo ";

    const std::string launch_cmd = find_claude_executable();
    std::cout << "  endpoint:    " + base_url + "\n";
    std::cout << "  model id:    " + model_id + "\n";
    std::cout.flush();

    if (launch_cmd.empty()) {
        std::cout << "\nClaude Code CLI not found in PATH.\n"
                  << "To connect Claude Code to this server, run in another terminal:\n\n"
                  << "  " << env_prefix << "claude\n\n"
                  << "Or set the model explicitly with:\n\n"
                  << "  " << env_prefix
                  << "OPENAI_MODEL=" << model_id << " claude\n\n"
                  << "Press Ctrl+C to stop the server.\n";
        std::cout.flush();

        while (server_is_running() && !stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        return 0;
    }

    std::cout << "  launching:   " << launch_cmd << "\n";
    std::cout.flush();

    const std::string full_cmd = env_prefix + "OPENAI_MODEL=" + model_id + " " + launch_cmd;
    const int claude_exit = std::system(full_cmd.c_str());

    std::cout << "Claude exited (" << claude_exit << ").\n";
    return WIFEXITED(claude_exit) ? WEXITSTATUS(claude_exit) : 1;
}

// Start a background Gizmo server for the given model and settings, then
// launch Claude Code configured to talk to it over the OpenAI-compatible
// API. When Claude exits, the server is stopped.
static int do_launch_gizmo_claude(const gizmo::CliOptions& options) {
    gizmo::ServeSettings settings;
    settings.model_path = options.model;
    auto dirs = gizmo::build_search_dirs(gizmo::parse_colon_dirs(options.extra_model_dirs));
    settings.model_path_extra = gizmo::join_colon_dirs(dirs);
    settings.host = options.host;
    settings.port = options.port;
    settings.threads = options.server_threads;
    settings.request_timeout_seconds = options.request_timeout_seconds;
    settings.context_size = options.context_size;
    settings.log_file = options.log_file;
    settings.json_logs = options.json_logs;
    settings.cors = options.cors;
    settings.no_evict = options.no_evict;
    settings.verbose = options.verbose;
    settings.no_shard = options.no_shard;
    settings.resident_layers = options.resident_layers.empty() ? 8 : options.resident_layers[0];
    settings.row_size = options.row_size.empty() ? 1 : options.row_size[0];

    std::cout << "Starting Gizmo server for Claude on http://"
              << settings.host << ":" << settings.port << "\n";
    std::cout << "Model: " << options.model << "\n";
    if (options.context_size > 0) {
        std::cout << "Context size: " << options.context_size << "\n";
    } else {
        std::cout << "Context size: auto (RAM-aware)\n";
    }
    std::cout.flush();

    gizmo::InferenceEngine engine;
    engine.set_verbose(settings.verbose);
    engine.set_threads(settings.threads);
    engine.set_context_size(settings.context_size);
    if (!engine.initialize(settings.model_path, /*layer_shard_lazy=*/!settings.no_shard)) {
        std::cerr << "\nFailed to initialize inference engine for: " << settings.model_path << "\n";
        return 1;
    }

    if (!settings.no_shard) {
        const int32_t resident = settings.resident_layers > 0 ? settings.resident_layers : 8;
        const int32_t row = settings.row_size > 0 ? settings.row_size : 1;
        engine.enable_sharded_engine(resident, /*evict_weights=*/!settings.no_evict, row);
        if (!engine.is_sharded()) {
            std::cout << "Note: sharded engine was not enabled; falling back to llama_decode.\n";
        }
    }

    gizmo::ServerConfig config;
    config.host = settings.host;
    config.port = settings.port;
    config.threads = settings.threads;
    config.cors = settings.cors;
    config.request_timeout_seconds = settings.request_timeout_seconds;
    config.log_file = settings.log_file;
    config.json_logs = settings.json_logs;

    gizmo::HttpServer server(engine, config);
    server.install_signal_handlers(&server);
    server.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (!server.is_running()) {
        std::cerr << "\nFailed to start HTTP server on " << settings.host << ":" << settings.port << "\n";
        return 1;
    }

    std::cout << "Waiting for server to be ready...\n";
    std::cout.flush();
    if (!wait_for_server_ready(client_host(settings.host), settings.port, "/v1/health", /*timeout_seconds=*/30)) {
        std::cerr << "\nServer did not become ready within 30 seconds.\n";
        server.stop();
        return 1;
    }

    std::cout << "Server ready.\n";
    std::cout.flush();

    const std::string base_url = "http://" + client_host(settings.host) + ":" + std::to_string(settings.port) + "/v1";
    const int claude_exit = launch_claude_client(
        base_url,
        engine.get_model_id(),
        [&server]() { return server.is_running(); },
        [&server]() { return server.stop_requested(); }
    );

    std::cout << "Stopping Gizmo server...\n";
    server.stop();
    return claude_exit;
}

// Launch Claude Code connected to a Rux disk-KV backend. Rux serves
// HuggingFace models over an OpenAI-compatible API with the KV cache
// kept on disk, so long contexts fit on modest RAM.
static int do_launch_rux_claude(const gizmo::CliOptions& options) {
    if (options.model.empty()) {
        std::cerr << "Error: --model required (HuggingFace model id for backend=rux)\n"
                  << "Usage: gizmo launch claude --backend rux -m <hf-model-id> [--port <N>]\n";
        return 1;
    }

    // Locate the `rux` executable.
    std::string rux_exe = options.rux_path;
    if (rux_exe.empty()) {
        if (command_exists("rux")) {
            rux_exe = "rux";
        } else {
            const char* home = std::getenv("HOME");
            std::vector<std::string> candidates;
            if (home != nullptr) {
                candidates.emplace_back(std::string(home) + "/Desktop/Rux/Rux/venv/bin/rux");
                candidates.emplace_back(std::string(home) + "/.local/bin/rux");
            }
            candidates.emplace_back("/usr/local/bin/rux");
            for (const auto& c : candidates) {
                if (std::filesystem::exists(c) && std::filesystem::is_regular_file(c)) {
                    rux_exe = c;
                    break;
                }
            }
        }
    }
    if (rux_exe.empty()) {
        std::cerr << "Error: `rux` executable not found.\n"
                  << "Install Rux or pass --rux-path /path/to/rux\n";
        return 1;
    }

    const int32_t max_seq_len = options.context_size > 0 ? options.context_size : 262144;

    std::cout << "Starting Rux server for Claude on http://"
              << options.host << ":" << options.port << "\n";
    std::cout << "Backend: rux\n";
    std::cout << "Model:   " << options.model << "\n";
    std::cout << "Context: " << max_seq_len << " tokens (disk-backed KV)\n";
    std::cout.flush();

    std::vector<std::string> rux_args = {
        "serve",
        "--host", options.host,
        "--port", std::to_string(options.port),
        "--max-seq-len", std::to_string(max_seq_len),
        "--chunk-size", "128",
    };

    pid_t rux_pid = start_child_process(rux_exe, rux_args);
    if (rux_pid < 0) {
        std::cerr << "\nFailed to start Rux server.\n";
        return 1;
    }

    std::cout << "Waiting for Rux server to be ready...\n";
    std::cout.flush();
    if (!wait_for_server_ready(client_host(options.host), options.port, "/health", /*timeout_seconds=*/60)) {
        std::cerr << "\nRux server did not become ready within 60 seconds.\n";
        terminate_child(rux_pid);
        return 1;
    }

    std::cout << "Rux server ready.\n";
    std::cout.flush();

    const std::string base_url = "http://" + client_host(options.host) + ":" + std::to_string(options.port) + "/v1";
    std::atomic<bool> rux_alive{true};
    std::thread reaper([rux_pid, &rux_alive]() {
        int status = 0;
        pid_t waited = waitpid(rux_pid, &status, 0);
        // waitpid returns the PID when the child exits (for any reason) or -1 on
        // error. In either case the server is no longer usable, so signal the
        // client loop to stop waiting.
        rux_alive = false;
        (void)waited;
        (void)status;
    });

    const int claude_exit = launch_claude_client(
        base_url,
        options.model,
        [&rux_alive]() { return rux_alive.load(); },
        []() { return false; }
    );

    std::cout << "Stopping Rux server...\n";
    terminate_child(rux_pid);
    reaper.join();
    return claude_exit;
}

// Dispatch launch claude to the selected backend.
static int do_launch_claude(const gizmo::CliOptions& options) {
    if (options.backend == "rux") {
        return do_launch_rux_claude(options);
    }
    if (options.backend != "gizmo") {
        std::cerr << "Error: unknown backend '" << options.backend
                  << "'; supported backends are 'gizmo' (default) and 'rux'.\n";
        return 1;
    }
    return do_launch_gizmo_claude(options);
}

static int do_bench(const std::string& model_path, bool prefill_only, int32_t resident_layers, bool no_evict, int32_t row_size, int32_t threads, int32_t context_size) {
    // Build a fixed prompt template and tokenize it once. Then
    // for each (N, path) we re-detokenize the first N tokens
    // back to a string and pass it to engine.generate(). This
    // is a workaround for the engine's text-only API; the
    // sharded engine itself takes a token array directly.
    const std::string prompt_template =
        "The capital of France is Paris. "
        "The capital of Germany is Berlin. "
        "The capital of Spain is Madrid. "
        "The capital of Italy is Rome. "
        "The capital of Japan is Tokyo. ";
    std::string prompt_repeated;
    while (prompt_repeated.size() < 4096) {
        prompt_repeated += prompt_template;
    }

    // Load model once. The vocab accessor requires a loaded
    // model, so we initialize to get it. Use lazy mmap (no prefault)
    // so the sharded-path memory numbers reflect the real low-memory
    // behavior rather than a model that was already fully faulted in.
    gizmo::InferenceEngine engine;
    engine.set_threads(threads);
    engine.set_context_size(context_size);
    if (!engine.initialize(model_path, /*layer_shard_lazy=*/true)) {
        std::cerr << "Failed to load model for bench: " << model_path << "\n";
        return 1;
    }
    const struct llama_vocab * vocab =
        llama_model_get_vocab(engine.raw_model());

    int32_t n = 0;
    std::vector<llama_token> tokens = gizmo::tokenize_text(
        vocab, prompt_repeated,
        /*add_special=*/true, /*parse_special=*/true, &n);
    if (n <= 0) {
        std::cerr << "Failed to tokenize bench prompt\n";
        return 1;
    }
    std::cout << "[bench] tokenized " << n << " tokens of repeated prompt\n";

    // Sharded scheduler must be enabled to use the sharded path.
    // We disable it for the un-sharded runs by NOT calling
    // enable_sharded_engine, then enable it once for all sharded
    // runs. The engine is reused across N values.
    //
    // 3 N values span 2 orders of magnitude in prefill cost.
    // The original 5-value list (1, 4, 16, 64, 256) was too slow
    // on the sharded path: the sharded engine builds a separate
    // ggml_cgraph per block, so a 256-token prefill runs the
    // O(N^2) attention 36 times. The 3 values here still show
    // the prefill-scaling trend; the user can extrapolate to
    // longer prompts from the slope.
    //
    // max_decode_tokens=1 is the minimum: one decode so the model
    // produces a real token (validates the KV cache was warmed
    // correctly). The decode loop dominates wall time; trimming
    // it from 8 to 1 keeps the bench under ~3 min on Qwen3-8B
    // instead of >10 min.
    const std::vector<int> Ns = { 1, 16, 64 };
    const int max_decode_tokens = 1;

    std::vector<bench_row_t> rows;
    rows.reserve(Ns.size() * 2);

    // All un-sharded runs first (sharded_sched_ is null so the
    // sharded prefill is skipped in engine.generate). Then enable
    // the sharded scheduler and run all Ns again. The rows are
    // stored in two contiguous blocks (un-sharded then sharded);
    // print_bench_table pairs rows[i*2] with rows[i*2+1].
    for (int N : Ns) {
        const int use_n = std::min(N, (int)tokens.size());
        std::cout << "\n[bench] N=" << use_n << " un-sharded...\n";
        rows.push_back(run_one_bench(engine, tokens, use_n,
                                     /*sharded=*/false, max_decode_tokens,
                                     prefill_only));
    }
    // Now build the sharded scheduler once and reuse for all Ns.
    // Honor --resident-layers for the sliding-window eviction policy
    // (Phase 8). Default 3 keeps the active block and the next two in
    // RAM, amortizing re-fault cost at the cost of ~3x per-block RSS.
    const int32_t bench_resident = resident_layers > 0 ? resident_layers : 1;
    const int32_t bench_row_size = row_size > 0 ? row_size : 1;
    engine.enable_sharded_engine(bench_resident, /*evict_weights=*/!no_evict, bench_row_size);
    for (int N : Ns) {
        const int use_n = std::min(N, (int)tokens.size());
        std::cout << "[bench] N=" << use_n << " sharded...\n";
        rows.push_back(run_one_bench(engine, tokens, use_n,
                                     /*sharded=*/true, max_decode_tokens,
                                     prefill_only));
    }

    print_bench_table(rows, model_path, engine.n_layer(), engine.embedding_dim(), prefill_only);
    return 0;
}

int main(int argc, char* argv[]) {
    gizmo::CliParser parser;
    auto options = parser.parse(argc, argv);

    if (options.help || options.command == gizmo::CommandType::Help) {
        parser.print_help();
        return 0;
    }

    if (options.command == gizmo::CommandType::Unknown) {
        std::cerr << "Error: Unknown command\n\n";
        parser.print_help();
        return 1;
    }

    // Handle commands
    switch (options.command) {
        case gizmo::CommandType::Info: {
            std::cout << "Gizmo System Info\n";
            std::cout << "=================\n";

            // Get system memory info
            std::ifstream meminfo("/proc/meminfo");
            std::string line;
            while (std::getline(meminfo, line)) {
                if (line.find("MemTotal") == 0 || line.find("MemAvailable") == 0) {
                    std::cout << line << "\n";
                }
            }

            if (options.measure_ram) {
                auto rss = gizmo::read_vm_rss_bytes();
                auto hwm = gizmo::read_vm_hwm_bytes();
                std::cout << "\nGizmo Process\n";
                std::cout << "  VmRSS:  " << (rss / (1024 * 1024)) << " MB\n";
                std::cout << "  VmHWM:  " << (hwm / (1024 * 1024)) << " MB\n";
            }

            return 0;
        }

        case gizmo::CommandType::List: {
            std::cout << "Available models:\n";
            auto paths = gizmo::scan_for_ggufs(
                gizmo::build_search_dirs(gizmo::parse_colon_dirs(options.extra_model_dirs)));
            if (paths.empty()) {
                std::cout << "(No GGUF models found in ~/.local/share/gizmo/models, "
                          << "~/.lmstudio/models, ~/.ollama/models/blobs, or ./models)\n";
                return 0;
            }
            for (const auto& path : paths) {
                auto info = gizmo::ModelManager::parse_gguf_header(path);
                std::string display_name = info.name.empty() ? fs::path(path).stem().string() : info.name;
                std::cout << "  " << display_name;
                if (!info.arch.empty()) {
                    std::cout << " [" << info.arch << "]";
                }
                if (info.parameter_count > 0) {
                    double p = static_cast<double>(info.parameter_count) / 1e9;
                    std::cout << " " << std::fixed << std::setprecision(1) << p << "B params";
                }
                std::cout << "\n       " << path << "\n";
                if (info.total_layers > 0 || info.embedding_dim > 0 ||
                    info.vocab_size > 0 || info.context_length > 0) {
                    std::cout << "       layers=" << info.total_layers
                              << " embd=" << info.embedding_dim
                              << " vocab=" << info.vocab_size
                              << " ctx=" << info.context_length
                              << " size=" << info.size_bytes << " bytes\n";
                }
            }
            std::cout << "\nUse a model with: gizmo run -m <path> -p \"Hello\"\n";
            return 0;
        }

        case gizmo::CommandType::Chat: {
            if (options.model.empty()) {
                // Launch the interactive chat TUI with a model picker.
                auto dirs = gizmo::build_search_dirs(gizmo::parse_colon_dirs(options.extra_model_dirs));
                return gizmo::run_chat_tui("",
                                           gizmo::join_colon_dirs(dirs),
                                           options.max_tokens,
                                           options.threads.empty() ? 4 : options.threads[0],
                                           options.context_size);
            }
            return do_chat(options.model,
                           options.resident_layers.empty() ? 8 : options.resident_layers[0],
                           options.no_shard,
                           options.no_evict,
                           options.row_size.empty() ? 1 : options.row_size[0],
                           options.threads.empty() ? 4 : options.threads[0],
                           options.max_tokens,
                           options.context_size,
                           options.measure_ram,
                           options.measure_interval_ms,
                           options.verbose,
                           options.progress);
        }

        case gizmo::CommandType::Bench: {
            if (options.model.empty()) {
                std::cerr << "Error: --model required for bench\n";
                return 1;
            }
            return do_bench(options.model, options.prefill_only,
                            options.resident_layers.empty() ? 8 : options.resident_layers[0],
                            options.no_evict,
                            options.row_size.empty() ? 1 : options.row_size[0],
                            options.threads.empty() ? 4 : options.threads[0],
                            options.context_size);
        }

        case gizmo::CommandType::Sweep: {
            if (options.model.empty()) {
                std::cerr << "Error: --model required for sweep\n";
                return 1;
            }
            return do_sweep(options.model,
                            options.resident_layers,
                            options.row_size,
                            options.threads,
                            options.no_evict,
                            options.sweep_max_tokens,
                            options.sweep_prefill_tokens,
                            options.json_output);
        }

        case gizmo::CommandType::Validate: {
            if (options.model.empty()) {
                std::cerr << "Error: --model required for validate\n";
                return 1;
            }
            return do_validate(options.model,
                               options.resident_layers.empty() ? 8 : options.resident_layers[0],
                               options.no_evict,
                               options.row_size.empty() ? 1 : options.row_size[0],
                               options.threads.empty() ? 4 : options.threads[0],
                               options.verbose, options.progress,
                               options.json_output,
                               options.argmax_only,
                               options.max_diff_threshold,
                               options.mean_diff_threshold,
                               options.context_size);
        }

        case gizmo::CommandType::Download: {
            if (options.download_url.empty()) {
                std::cerr << "Error: download URL required.\n"
                          << "Usage: gizmo download <url>\n"
                          << "       gizmo download --url <url> [-m output.gguf]\n";
                return 1;
            }
            gizmo::ModelManager mgr;
            // options.model can optionally name the output file.
            bool ok = mgr.download(options.download_url, options.model);
            return ok ? 0 : 1;
        }

        case gizmo::CommandType::Tui: {
            gizmo::ServeSettings settings;
            settings.model_path = options.model;
            auto dirs = gizmo::build_search_dirs(gizmo::parse_colon_dirs(options.extra_model_dirs));
            settings.model_path_extra = gizmo::join_colon_dirs(dirs);
            settings.host = options.host;
            settings.port = options.port;
            settings.threads = options.server_threads;
            settings.request_timeout_seconds = options.request_timeout_seconds;
            settings.context_size = options.context_size;
            settings.log_file = options.log_file;
            settings.json_logs = options.json_logs;
            settings.cors = options.cors;
            settings.no_evict = options.no_evict;
            settings.verbose = options.verbose;
            settings.no_shard = options.no_shard;
            settings.resident_layers = options.resident_layers.empty() ? 8 : options.resident_layers[0];
            settings.row_size = options.row_size.empty() ? 1 : options.row_size[0];
            return gizmo::run_server_tui(settings);
        }

        case gizmo::CommandType::Serve:
            // fallthrough
        case gizmo::CommandType::Server: {
            if (options.model.empty()) {
                parser.print_help();
                return 0;
            }
            gizmo::ServeSettings settings;
            settings.model_path = options.model;
            auto dirs = gizmo::build_search_dirs(gizmo::parse_colon_dirs(options.extra_model_dirs));
            settings.model_path_extra = gizmo::join_colon_dirs(dirs);
            settings.host = options.host;
            settings.port = options.port;
            settings.threads = options.server_threads;
            settings.request_timeout_seconds = options.request_timeout_seconds;
            settings.context_size = options.context_size;
            settings.log_file = options.log_file;
            settings.json_logs = options.json_logs;
            settings.cors = options.cors;
            settings.no_evict = options.no_evict;
            settings.verbose = options.verbose;
            settings.no_shard = options.no_shard;
            settings.resident_layers = options.resident_layers.empty() ? 8 : options.resident_layers[0];
            settings.row_size = options.row_size.empty() ? 1 : options.row_size[0];
            return gizmo::run_server_headless(settings);
        }

        case gizmo::CommandType::Launch: {
            if (options.model.empty()) {
                std::cerr << "Error: --model required for launch\n"
                          << "Usage: gizmo launch claude -m <model.gguf> [--port <N>]\n";
                return 1;
            }
            if (options.launch_target.empty() || options.launch_target != "claude") {
                std::cerr << "Error: launch target must be 'claude' (got: "
                          << (options.launch_target.empty() ? "<empty>" : options.launch_target)
                          << ")\n";
                return 1;
            }
            return do_launch_claude(options);
        }

        case gizmo::CommandType::Run: {
            if (options.model.empty()) {
                std::cerr << "Error: --model required\n";
                return 1;
            }

            gizmo::InferenceConfig config;
            config.max_tokens = options.max_tokens > 0 ? options.max_tokens
                                 : (options.prompt.empty() ? 256 : 128);
            config.temperature = 0.8f;

            // Snapshot VmRSS before model load so we can show real
            // deltas. The "92% savings" line in earlier versions was
            // arithmetic on hard-coded constants; we now report what
            // actually happened.
            const size_t rss_before_load = gizmo::read_vm_rss_bytes();

            gizmo::InferenceEngine engine;
            engine.set_verbose(options.verbose);
            engine.set_progress(options.progress);
            engine.set_threads(options.threads.empty() ? 4 : options.threads[0]);
            engine.set_context_size(options.context_size);
            if (!engine.initialize(options.model, /*layer_shard_lazy=*/!options.no_shard)) {
                std::cerr << "Failed to initialize inference engine\n";
                return 1;
            }

            // Enable the per-block sharded engine for prefill and decode
            // when sharding was requested (the default).  With layer
            // sharding enabled the model is mmap'd without prefault, so
            // only the blocks that are actually executed are resident.  The
            // --no-evict flag skips madvise(MADV_DONTNEED) and keeps the
            // active resident set from growing beyond the configured window.
            // With --no-shard the full model is prefaulted at load time.
            if (!options.no_shard) {
                engine.enable_sharded_engine(
                    options.resident_layers.empty() ? 8 : options.resident_layers[0],
                    /*evict_weights=*/!options.no_evict,
                    /*row_size=*/options.row_size.empty() ? 1 : options.row_size[0]);
                if (!engine.is_sharded()) {
                    std::cout << "Note: sharded engine was not enabled; falling back to llama_decode.\n";
                }
            }

            const int32_t n_layer = engine.n_layer();
            std::cout << "Model: " << options.model << "\n";
            std::cout << "Total layers: " << n_layer << "\n";
            std::cout << "Embedding dim: " << engine.embedding_dim() << "\n";
            std::cout << "Vocab size: " << engine.vocab_size() << "\n";
            std::cout << "Context size: " << llama_n_ctx(engine.raw_context()) << "\n";

            // Resident layer count for display only. Real per-block
            // residency control requires a custom forward pass; we keep
            // the CLI flag for compatibility with the original design
            // and surface it in --help. See README "Status" section.
            int32_t requested_resident = options.resident_layers.empty() ? 0 : options.resident_layers[0];
            if (requested_resident <= 0) {
                requested_resident = options.layers > 0 ? options.layers : 8;
            }
            std::cout << "Requested resident blocks: " << requested_resident
                      << " (informational; full graph is computed per token)\n";

            const size_t rss_after_load = gizmo::read_vm_rss_bytes();
            std::cout << "VmRSS after model load: "
                      << (rss_after_load / (1024 * 1024)) << " MB"
                      << "  (delta: +"
                      << ((rss_after_load - rss_before_load) / (1024 * 1024)) << " MB)\n";

            // Background VmRSS measurement thread. Prints to stderr so it
            // doesn't interleave with stdout token stream.
            std::atomic<bool> stop_measure{false};
            std::thread measure_thread;
            if (options.measure_ram) {
                measure_thread = std::thread([&]() {
                    while (!stop_measure.load()) {
                        const size_t rss = gizmo::read_vm_rss_bytes();
                        const size_t hwm = gizmo::read_vm_hwm_bytes();
                        std::cerr << "[measure] VmRSS=" << (rss / (1024 * 1024))
                                  << " MB  VmHWM=" << (hwm / (1024 * 1024)) << " MB\n";
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(options.measure_interval_ms));
                    }
                });
            }

            if (!options.prompt.empty()) {
                std::cout << "\n--- Generating Response ---\n";
                std::cout << "Prompt: " << options.prompt << "\n\n";
                std::string response = engine.generate(options.prompt, config);
                std::cout << "\nResponse: " << response << "\n";
            } else {
                std::cout << "\nNo prompt provided. Use -p \"your prompt\" to generate text.\n";
            }

            stop_measure.store(true);
            if (measure_thread.joinable()) measure_thread.join();

            const size_t rss_after_gen = gizmo::read_vm_rss_bytes();
            const size_t hwm_after_gen = gizmo::read_vm_hwm_bytes();
            std::cout << "\nVmRSS after generation: "
                      << (rss_after_gen / (1024 * 1024)) << " MB\n";
            std::cout << "VmHWM (peak resident):  "
                      << (hwm_after_gen / (1024 * 1024)) << " MB\n";

            return 0;
        }

        default:
            std::cerr << "Error: Command not implemented\n";
            return 1;
    }
}
