#include "cli_parser.hpp"
#include "inference_engine.hpp"
#include "model_picker.hpp"
#include "proc_status.hpp"
#include "server.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

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

// Run a single (N, sharded?) measurement against the engine.
// Returns the wall time of generate() and the post-run memory
// stats. The engine is reset_for_next_run() before the call so
// repeated calls don't accumulate KV cache.
//
// When prefill_only is true, the sharded-path run uses
// engine.prefill_only_tokens() (no decode, no un-sharded
// prefill, no model-page re-fault). The un-sharded-path run
// still uses llama_decode (it's the baseline). This gives a
// clean wall-time comparison of "sharded prefill" vs
// "un-sharded prefill" without the current Phase 6
// double-prefill cost masking the sharded time.
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
    std::string prompt_str;
    {
        const struct llama_vocab * vocab =
            llama_model_get_vocab(engine.raw_model());
        std::vector<char> buf(64);
        for (int i = 0; i < n_tokens; ++i) {
            int n = llama_token_to_piece(vocab, tokens[i], buf.data(),
                                         buf.size(), /*lstrip=*/0,
                                         /*special=*/true);
            if (n > 0) prompt_str.append(buf.data(), n);
        }
    }
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
    // from right after the sharded prefill (before the un-sharded
    // prefill masked it). These are the real "sharded path" memory
    // numbers; the post-decode-loop rss/hwm above are dominated by
    // the un-sharded decode loop's llama_decode.
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
    // (captured right after the per-block engine finished, before
    // the un-sharded prefill re-faulted the model). This is the
    // real "sharded path" memory cost. The post-decode VmHWM is
    // dominated by the un-sharded llama_decode and not informative.
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
              << "\neach block is MADV_PAGEOUT.\n";
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

static int do_bench(const std::string& model_path, bool prefill_only, int32_t resident_layers, bool no_evict, int32_t row_size) {
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
    // model, so we initialize to get it.
    gizmo::InferenceEngine engine;
    if (!engine.initialize(model_path, /*layer_shard_lazy=*/false)) {
        std::cerr << "Failed to load model for bench: " << model_path << "\n";
        return 1;
    }
    const struct llama_vocab * vocab =
        llama_model_get_vocab(engine.raw_model());

    std::vector<llama_token> tokens(prompt_repeated.size() + 16);
    int n = llama_tokenize(vocab, prompt_repeated.c_str(),
                           prompt_repeated.size(),
                           tokens.data(), tokens.size(),
                           /*add_special=*/true, /*parse_special=*/true);
    if (n < 0) {
        std::cerr << "Failed to tokenize bench prompt\n";
        return 1;
    }
    tokens.resize(n);
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

static int do_serve(const std::string& model_path, const std::string& host, int32_t port, int32_t /*resident_layers*/, bool /*no_evict*/, int32_t /*row_size*/) {
    // Snapshot VmRSS before model load so we can show real deltas.
    const size_t rss_before_load = gizmo::read_vm_rss_bytes();

    gizmo::InferenceEngine engine;
    if (!engine.initialize(model_path, /*layer_shard_lazy=*/false)) {
        std::cerr << "Failed to initialize inference engine\n";
        return 1;
    }

    // NOTE: server mode currently uses the un-sharded (full-model) path.
    // The per-block sharded engine crashes during scheduler reservation on
    // this llama.cpp commit; fixing it is the next sharded-engine phase.
    // For now, `gizmo serve` loads the whole model into RAM, which is the
    // same behavior as `ollama serve` for most users.

    const size_t rss_after_load = gizmo::read_vm_rss_bytes();
    std::cout << "VmRSS after model load: "
              << (rss_after_load / (1024 * 1024)) << " MB"
              << "  (delta: +" << ((rss_after_load - rss_before_load) / (1024 * 1024)) << " MB)\n";

    gizmo::Server server(&engine, model_path);
    return server.run(host, port) ? 0 : 1;
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
            std::cout << "(No models downloaded yet)\n";
            return 0;
        }

        case gizmo::CommandType::Bench: {
            if (options.model.empty()) {
                std::cerr << "Error: --model required for bench\n";
                return 1;
            }
            return do_bench(options.model, options.prefill_only, options.resident_layers, options.no_evict, options.row_size);
        }

        case gizmo::CommandType::Serve: {
            std::string model_path = options.model;
            if (model_path.empty()) {
                model_path = gizmo::pick_model_interactive();
                if (model_path.empty()) {
                    std::cerr << "No model selected. Use -m <path> to specify one.\n";
                    return 1;
                }
            }
            return do_serve(model_path, options.host, options.port,
                            options.resident_layers, options.no_evict, options.row_size);
        }

        case gizmo::CommandType::Download: {
            if (options.url.empty()) {
                std::cerr << "Error: --url required for download\n";
                return 1;
            }
            std::cout << "Downloading model from: " << options.url << "\n";
            std::cout << "(Download functionality not yet implemented)\n";
            return 0;
        }

        case gizmo::CommandType::Run:
        case gizmo::CommandType::Chat: {
            if (options.model.empty()) {
                std::cerr << "Error: --model required\n";
                return 1;
            }

            gizmo::InferenceConfig config;
            config.max_tokens = options.prompt.empty() ? 256 : 128;
            config.temperature = 0.8f;

            // Snapshot VmRSS before model load so we can show real
            // deltas. The "92% savings" line in earlier versions was
            // arithmetic on hard-coded constants; we now report what
            // actually happened.
            const size_t rss_before_load = gizmo::read_vm_rss_bytes();

            gizmo::InferenceEngine engine;
            if (!engine.initialize(options.model, options.no_shard)) {
                std::cerr << "Failed to initialize inference engine\n";
                return 1;
            }

            // Enable the per-block sharded engine for the prefill step
            // when sharding was requested (the default). The sharded
            // engine is prefill-only in Phase 6; the per-token decode
            // loop continues to use llama_decode. Pass
            // options.resident_layers for forward compatibility with
            // the sliding-window work in a later phase.
            if (!options.no_shard) {
                engine.enable_sharded_engine(
                    options.resident_layers,
                    /*evict_weights=*/true,
                    /*row_size=*/options.row_size);
            }

            const int32_t n_layer = engine.n_layer();
            std::cout << "Model: " << options.model << "\n";
            std::cout << "Total layers: " << n_layer << "\n";
            std::cout << "Embedding dim: " << engine.embedding_dim() << "\n";
            std::cout << "Vocab size: " << engine.vocab_size() << "\n";

            // Resident layer count for display only. Real per-block
            // residency control requires a custom forward pass; we keep
            // the CLI flag for compatibility with the original design
            // and surface it in --help. See README "Status" section.
            int32_t requested_resident = options.resident_layers;
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
            } else if (options.command == gizmo::CommandType::Run) {
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
