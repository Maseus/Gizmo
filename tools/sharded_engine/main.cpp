// Per-block sharded inference engine for Gizmo.
//
// Three modes:
//
//   run <model.gguf> <prompt>
//                              -- run the multi-block engine for the
//                                 prompt, print the first 16 logits.
//
//   validate <model.gguf> <prompt> <baseline.bin>
//                              -- run the multi-block engine and
//                                 compare against the baseline logits
//                                 file. Returns PASS/FAIL.
//
//   gen <model.gguf> <prompt> <n_tokens>
//                              -- (out of scope for this iteration)
//                                 smoke test multi-token generation.
//
// The sharded engine runs all n_layer blocks sequentially via
// per-block ggml_cgraph, threading the residual stream between
// blocks via a persistent F32 carrier on the scheduler's CPU
// buffer. Uses the model's native Q4_K/Q6_K weights directly (no
// F32 dequant).
//
// Build: see tools/sharded_engine/CMakeLists.txt.

#include "llama.h"

#include "multi_block.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-batch.h"
#include "llama-memory.h"
#include "llama-memory-hybrid.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kMaxPromptTokens = 64;

// Build a llama_batch from already-tokenized prompt ids and init
// the memory context for that batch. Returns the memory context
// (cast from llama_memory_context_ptr) on success. For hybrid
// models (qwen3.5) this is a llama_memory_hybrid_context with both
// get_attn() and get_recr(); run_multi_block routes per-layer via
// hparams.is_recr(il).
static llama_memory_context_ptr make_mctx(
    llama_context *    lctx,
    const llama_model *model,
    const llama_token *tokens,
    int                n_tokens)
{
    llama_memory_t mem = llama_get_memory(lctx);
    if (mem == nullptr) {
        return nullptr;
    }

    const uint32_t n_seq_max = 1;  // prefill only; single-stream
    llama_batch batch = llama_batch_init(n_tokens, /*embd=*/0, n_seq_max);
    if (!batch.token) {
        return nullptr;
    }
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }

    llama_batch_allocr balloc(model->hparams.n_pos_per_embd());
    if (!balloc.init(batch, model->vocab, mem, model->hparams.n_embd,
                     n_seq_max, /*output_all=*/false)) {
        llama_batch_free(batch);
        return nullptr;
    }

    // Use llama_n_ubatch() (public) instead of poking at the private
    // llama_context::cparams field. For prefill with n_tokens <=
    // n_ubatch the split is trivial.
    const uint32_t n_ubatch = llama_n_ubatch(lctx);
    auto mctx = mem->init_batch(balloc, n_ubatch, /*output_all=*/false);
    llama_batch_free(batch);
    return mctx;
}

}  // namespace

static int do_run(const char* model_path, const char* prompt, bool evict) {
    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    // Keep weights in their original mmap'd format. CPU weight repacking
    // would allocate a full extra copy of every tensor and OOM on 27B
    // models before the sharded engine can run.
    mparams.use_extra_bufts = false;
    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::fprintf(stderr, "ERROR: failed to load model: %s\n", model_path);
        llama_backend_free();
        return 1;
    }
    std::printf("[sharded] model loaded: n_layer=%d n_embd=%d\n",
                llama_model_n_layer(model),
                llama_model_n_embd(model));

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 256;
    cparams.n_batch = 64;
    cparams.n_ubatch = 64;
    // Optional GIZMO_NTHREADS env override (default 4). Useful for
    // debugging parallel-FP non-determinism by forcing a single thread.
    int n_threads = 4;
    if (const char* e = std::getenv("GIZMO_NTHREADS")) {
        int v = std::atoi(e);
        if (v > 0) n_threads = v;
    }
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "ERROR: failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        std::fprintf(stderr, "ERROR: ggml_backend_cpu_init failed\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    ggml_backend_cpu_set_n_threads(cpu_backend, n_threads);
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        &cpu_backend, &cpu_buft, 1,
        /*graph_size=*/8192, /*parallel=*/false, /*op_offload=*/false);
    if (!sched) {
        std::fprintf(stderr, "ERROR: ggml_backend_sched_new failed\n");
        ggml_backend_free(cpu_backend);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::vector<llama_token> tokens(kMaxPromptTokens);
    int n_tokens = llama_tokenize(
        llama_model_get_vocab(model), prompt, std::strlen(prompt),
        tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        std::fprintf(stderr, "ERROR: tokenization failed\n");
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    std::printf("[sharded] prompt: \"%s\" -> %d tokens: [", prompt, n_tokens);
    for (int i = 0; i < n_tokens; ++i) std::printf(" %d", tokens[i]);
    std::printf(" ]\n");

    std::printf("[sharded] running multi-block forward pass%s\n",
                evict ? " (with madvise(MADV_DONTNEED) eviction)" : "");

    // Instantiate the hybrid memory context so per-layer SSM state
    // (r_l, s_l) and KV cache cells are real ggml tensors we can
    // ggml_cpy into. For pure-attention models the context is a
    // llama_kv_cache_context; for qwen3.5 it's a hybrid context. The
    // sharded engine routes per-layer via hparams.is_recr(il).
    llama_memory_context_ptr mctx_holder = make_mctx(
        ctx, (const llama_model *) model, tokens.data(), n_tokens);
    auto * mctx_hybrid = static_cast<llama_memory_hybrid_context *>(mctx_holder.get());

    multi_block_result_t res = run_multi_block(
        (const llama_model*)model, sched, tokens.data(), n_tokens, evict,
        /*resident_layers=*/1, /*row_size=*/1, mctx_hybrid,
        /*pos_first=*/0, /*verbose=*/true);

    if (res.final_logits.empty()) {
        std::fprintf(stderr, "ERROR: run_multi_block failed\n");
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // The sharded engine produces [n_vocab, n_tokens] logits. Print
    // the first 16 of the last-token column.
    const int n_vocab = (int)llama_vocab_n_tokens(llama_model_get_vocab(model));
    const size_t last_col_base = (size_t)(n_tokens - 1) * n_vocab;
    std::printf("[sharded] final logits (last token): first 16 = [");
    for (int i = 0; i < 16 && i < n_vocab; ++i) {
        std::printf(" %.4f", res.final_logits[last_col_base + i]);
    }
    std::printf(" ... ]\n");

    int best_id = 0;
    float best_val = res.final_logits[last_col_base];
    for (int i = 1; i < n_vocab; ++i) {
        const float v = res.final_logits[last_col_base + i];
        if (v > best_val) { best_val = v; best_id = i; }
    }
    std::printf("  argmax token id: %d (logit=%.4f)\n", best_id, best_val);

    // Optional baseline diff: if GIZMO_BASELINE is set to a binary file
    // produced by gizmo_argmax_dump (int32 n_vocab + float[n_vocab]),
    // compare column-wise and print max / mean abs diff + a top-K
    // residual ranking.
    if (const char* baseline_path = std::getenv("GIZMO_BASELINE")) {
        std::ifstream bf(baseline_path, std::ios::binary);
        if (!bf) {
            std::fprintf(stderr, "WARN: could not open GIZMO_BASELINE %s\n", baseline_path);
        } else {
            int32_t base_nv = 0;
            bf.read((char*)&base_nv, sizeof(base_nv));
            if (base_nv != n_vocab) {
                std::fprintf(stderr, "WARN: baseline n_vocab=%d != model n_vocab=%d\n",
                             base_nv, n_vocab);
            } else {
                std::vector<float> baseline((size_t)n_vocab);
                bf.read((char*)baseline.data(), (std::streamsize)(n_vocab * sizeof(float)));
                std::vector<float> diff((size_t)n_vocab);
                double sum_abs = 0.0;
                float max_abs = 0.0f;
                int n_diff_gt_005 = 0, n_diff_gt_05 = 0, n_diff_gt_2 = 0;
                for (int i = 0; i < n_vocab; ++i) {
                    const float d = res.final_logits[last_col_base + i] - baseline[i];
                    diff[i] = d;
                    const float ad = d < 0 ? -d : d;
                    sum_abs += ad;
                    if (ad > max_abs) max_abs = ad;
                    if (ad > 0.05f) ++n_diff_gt_005;
                    if (ad > 0.5f)  ++n_diff_gt_05;
                    if (ad > 2.0f)  ++n_diff_gt_2;
                }
                std::printf("[sharded] vs baseline: max_abs=%.4f mean_abs=%.4f "
                            ">0.05=%d >0.5=%d >2.0=%d (of %d vocab)\n",
                            max_abs, sum_abs / n_vocab,
                            n_diff_gt_005, n_diff_gt_05, n_diff_gt_2, n_vocab);
                // Top-K largest |diff| entries.
                std::vector<std::pair<float,int>> by_diff;
                by_diff.reserve((size_t)n_vocab);
                for (int i = 0; i < n_vocab; ++i) by_diff.push_back({diff[i] < 0 ? -diff[i] : diff[i], i});
                std::partial_sort(by_diff.begin(), by_diff.begin() + std::min<int>(20, n_vocab),
                                  by_diff.end(),
                                  [](auto &a, auto &b) { return a.first > b.first; });
                std::printf("[sharded] top-K |diff| vocab entries:\n");
                char piece_buf[256];
                for (int i = 0; i < std::min<int>(20, n_vocab); ++i) {
                    const int id = by_diff[i].second;
                    int pn = llama_token_to_piece(llama_model_get_vocab(model), id,
                                                  piece_buf, sizeof(piece_buf), 0, false);
                    if (pn < 0) pn = 0;
                    piece_buf[pn] = 0;
                    std::printf("  [%2d] id=%-7d shard=%.4f base=%.4f diff=%+.4f text=\"%s\"\n",
                                i, id,
                                res.final_logits[last_col_base + id],
                                baseline[id],
                                diff[id], piece_buf);
                }
            }
        }
    }

    // Top-K argmax dump for debugging SSM output
    const int top_k = 20;
    std::printf("[sharded] top-%d last-token logits:\n", top_k);
    std::vector<std::pair<float,int>> scored;
    for (int i = 0; i < n_vocab; ++i) {
        scored.push_back({res.final_logits[last_col_base + i], i});
    }
    std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                      [](auto &a, auto &b) { return a.first > b.first; });
    for (int i = 0; i < top_k; ++i) {
        std::printf("    [%2d] id=%-6d logit=% .4f\n", i,
                    scored[i].second, scored[i].first);
    }

    // Print resident memory before cleanup (eviction effect should be
    // visible here: the mmap'd weights are not in RSS).
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0 ||
            line.rfind("VmHWM:", 0) == 0) {
            std::printf("[sharded] %s\n", line.c_str());
        }
    }

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu_backend);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

static int do_validate(const char* model_path, const char* prompt,
                       const char* baseline_path, bool evict) {
    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    // Keep weights in their original mmap'd format. CPU weight repacking
    // would allocate a full extra copy of every tensor and OOM on 27B
    // models before the sharded engine can run.
    mparams.use_extra_bufts = false;
    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::fprintf(stderr, "ERROR: failed to load model: %s\n", model_path);
        llama_backend_free();
        return 1;
    }
    std::printf("[validate] model loaded: n_layer=%d n_embd=%d\n",
                llama_model_n_layer(model),
                llama_model_n_embd(model));

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 256;
    cparams.n_batch = 64;
    cparams.n_ubatch = 64;
    // Optional GIZMO_NTHREADS env override (default 4). Useful for
    // debugging parallel-FP non-determinism by forcing a single thread.
    int n_threads = 4;
    if (const char* e = std::getenv("GIZMO_NTHREADS")) {
        int v = std::atoi(e);
        if (v > 0) n_threads = v;
    }
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "ERROR: failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        std::fprintf(stderr, "ERROR: ggml_backend_cpu_init failed\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    ggml_backend_cpu_set_n_threads(cpu_backend, n_threads);
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        &cpu_backend, &cpu_buft, 1,
        /*graph_size=*/8192, /*parallel=*/false, /*op_offload=*/false);
    if (!sched) {
        std::fprintf(stderr, "ERROR: ggml_backend_sched_new failed\n");
        ggml_backend_free(cpu_backend);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Tokenize
    std::vector<llama_token> tokens(kMaxPromptTokens);
    int n_tokens = llama_tokenize(
        llama_model_get_vocab(model), prompt, std::strlen(prompt),
        tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        std::fprintf(stderr, "ERROR: tokenization failed\n");
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    std::printf("[validate] prompt: \"%s\" -> %d tokens\n", prompt, n_tokens);

    // Run sharded engine
    std::printf("[validate] running multi-block forward pass%s\n",
                evict ? " (with madvise(MADV_DONTNEED) eviction)" : "");

    // Instantiate hybrid memory context so per-layer SSM state and
    // KV cache cells are real ggml tensors. The validate binary
    // still compares logits numerically against the baseline file.
    llama_memory_context_ptr mctx_holder = make_mctx(
        ctx, (const llama_model *) model, tokens.data(), n_tokens);
    auto * mctx_hybrid = static_cast<llama_memory_hybrid_context *>(mctx_holder.get());

    multi_block_result_t res = run_multi_block(
        (const llama_model*)model, sched, tokens.data(), n_tokens, evict,
        /*resident_layers=*/1, /*row_size=*/1, mctx_hybrid,
        /*pos_first=*/0, /*verbose=*/true);
    if (res.final_logits.empty()) {
        std::fprintf(stderr, "ERROR: run_multi_block failed\n");
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Load baseline: int32 n_vocab, then float[n_vocab]
    std::ifstream bf(baseline_path, std::ios::binary);
    if (!bf) {
        std::fprintf(stderr, "ERROR: cannot open baseline: %s\n", baseline_path);
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    int32_t base_n_vocab = 0;
    bf.read((char*)&base_n_vocab, sizeof(base_n_vocab));
    const int n_vocab = (int)llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (base_n_vocab != n_vocab) {
        std::fprintf(stderr,
            "ERROR: baseline n_vocab=%d, model n_vocab=%d\n",
            base_n_vocab, n_vocab);
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    std::vector<float> baseline((size_t)n_vocab);
    bf.read((char*)baseline.data(), n_vocab * sizeof(float));
    if ((int)baseline.size() != n_vocab) {
        std::fprintf(stderr, "ERROR: short baseline read (%zu of %d)\n",
                     baseline.size(), n_vocab);
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Compare last-token logits vs baseline
    const size_t last_col_base = (size_t)(n_tokens - 1) * n_vocab;
    float max_abs = 0.0f;
    double sum_abs = 0.0;
    int within_1e_3 = 0;
    int within_1e_4 = 0;
    for (int i = 0; i < n_vocab; ++i) {
        const float a = res.final_logits[last_col_base + i];
        const float b = baseline[i];
        const float d = std::fabs(a - b);
        if (d > max_abs) max_abs = d;
        sum_abs += d;
        if (d < 1e-3f) ++within_1e_3;
        if (d < 1e-4f) ++within_1e_4;
    }
    const double mean_abs = sum_abs / n_vocab;
    const double frac_1e_3 = (double)within_1e_3 / n_vocab;
    const double frac_1e_4 = (double)within_1e_4 / n_vocab;
    const bool pass = (max_abs < 1e-3f) && (frac_1e_3 > 0.99);

    std::printf("[validate] compare: max_abs_diff=%.6f mean_abs_diff=%.6f\n",
                max_abs, mean_abs);
    std::printf("             frac_within_1e-3=%.4f frac_within_1e-4=%.4f -> %s\n",
                frac_1e_3, frac_1e_4,
                pass ? "PASS" : "FAIL");

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu_backend);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return pass ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <run|validate|gen> [--evict] <model.gguf> <args...>\n"
            "  --evict: madvise(MADV_DONTNEED) on each block's weights\n"
            "           after compute. Reduces RAM residency at the cost\n"
            "           of page-fault latency on the next access.\n",
            argv[0]);
        return 1;
    }
    const std::string mode = argv[1];

    // Parse optional --evict flag (must come right after the mode).
    bool evict = false;
    int arg_shift = 0;
    if (argc >= 3 && std::string(argv[2]) == "--evict") {
        evict = true;
        arg_shift = 1;
    }

    if (mode == "run" && argc == 4 + arg_shift) {
        return do_run(argv[2 + arg_shift], argv[3 + arg_shift], evict);
    }
    if (mode == "validate" && argc == 5 + arg_shift) {
        return do_validate(argv[2 + arg_shift], argv[3 + arg_shift],
                           argv[4 + arg_shift], evict);
    }
    std::fprintf(stderr, "ERROR: unknown mode or bad args\n");
    return 1;
}
