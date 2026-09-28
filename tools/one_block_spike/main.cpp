// Per-block forward pass spike for Gizmo.
//
// Two modes:
//
//   capture <model.gguf> <prompt>  -- run the full engine on a prompt,
//                                      dump the final logits to <out.bin>.
//                                      This is the "engine did the right
//                                      thing" baseline.
//
//   run <model.gguf> <prompt>
//                                   -- build a per-block ggml graph for
//                                      block 0 of the model and compare
//                                      against the pure-C++ reference in
//                                      reference.cpp. PASS criterion from
//                                      the breather plan: max_abs_diff
//                                      < 1e-3 and > 99% of elements
//                                      within 1e-3, per position.
//
// Both modes use the public llama.h API.  The capture mode is the
// only working part of this iteration.
//
// Build: see tools/one_block_spike/CMakeLists.txt.

#include "llama.h"

#include "reference.h"
#include "spike_graph.h"
#include "llama-model.h"
#include "llama-context.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kMaxPromptTokens = 64;

}  // namespace

static int do_capture(const char* model_path, const char* prompt, const char* out_logits_path) {
    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::fprintf(stderr, "ERROR: failed to load model: %s\n", model_path);
        llama_backend_free();
        return 1;
    }

    std::printf("[capture] model loaded\n");
    std::printf("  n_layer  = %d\n", llama_model_n_layer(model));
    std::printf("  n_embd   = %d\n", llama_model_n_embd(model));
    std::printf("  n_head   = %d\n", llama_model_n_head(model));
    std::printf("  n_head_kv= %d\n", llama_model_n_head_kv(model));
    std::printf("  n_swa    = %d\n", llama_model_n_swa(model));
    std::printf("  vocab    = %lld\n",
                (long long)llama_vocab_n_tokens(llama_model_get_vocab(model)));

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 256;
    cparams.n_batch = 64;
    cparams.n_ubatch = 64;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "ERROR: failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Tokenize
    std::vector<llama_token> tokens(kMaxPromptTokens);
    int n_tokens = llama_tokenize(
        llama_model_get_vocab(model),
        prompt,
        std::strlen(prompt),
        tokens.data(),
        tokens.size(),
        true, true);
    if (n_tokens < 0) {
        std::fprintf(stderr, "ERROR: tokenization failed\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    std::printf("[capture] prompt: \"%s\"\n", prompt);
    std::printf("  -> %d tokens: [", n_tokens);
    for (int i = 0; i < n_tokens; ++i) std::printf(" %d", tokens[i]);
    std::printf(" ]\n");

    // Prefill
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tokens)) != 0) {
        std::fprintf(stderr, "ERROR: prefill decode failed\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Capture logits
    float* logits = llama_get_logits_ith(ctx, -1);
    if (!logits) {
        std::fprintf(stderr, "ERROR: no logits available\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    const int32_t n_vocab = (int32_t)llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::printf("[capture] logits: first 16 = [");
    for (int i = 0; i < 16 && i < n_vocab; ++i) std::printf(" %.4f", logits[i]);
    std::printf(" ... ]\n");

    int best_id = 0;
    float best_val = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_val) { best_val = logits[i]; best_id = i; }
    }
    std::printf("  argmax token id: %d (logit=%.4f)\n", best_id, best_val);

    // Write logits
    {
        std::ofstream f(out_logits_path, std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "ERROR: cannot open %s\n", out_logits_path);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        f.write((const char*)&n_vocab, sizeof(n_vocab));
        f.write((const char*)logits, n_vocab * sizeof(float));
    }
    std::printf("[capture] wrote %d logits to %s\n", n_vocab, out_logits_path);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

// do_run: per-block forward pass spike. Loads the model, builds a
// per-block ggml_cgraph for block 0, runs it, and compares against
// the pure-C++ reference. Same F32 weights are used by both, so the
// comparison is bit-exact up to summation order.
static int do_run(const char* model_path, const char* prompt) {
    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::fprintf(stderr, "ERROR: failed to load model: %s\n", model_path);
        llama_backend_free();
        return 1;
    }
    std::printf("[run] model loaded: n_layer=%d n_embd=%d n_head=%d n_head_kv=%d n_swa=%d\n",
                llama_model_n_layer(model),
                llama_model_n_embd(model),
                llama_model_n_head(model),
                llama_model_n_head_kv(model),
                llama_model_n_swa(model));

    // Build a context for its scheduler (used by the reference to
    // dequantize Q4_K weights via ggml_cpy, and by the spike graph
    // to allocate/compute). We never call llama_decode on it.
    auto cparams = llama_context_default_params();
    cparams.n_ctx = 256;
    cparams.n_batch = 64;
    cparams.n_ubatch = 64;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;
    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "ERROR: failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Make a fresh scheduler for the spike/reference. We can't reuse
    // ctx->get_sched() because llama.cpp's graph_reserve pre-allocates
    // it for its own use; we need our own scheduler for cpy+spike graphs.
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        std::fprintf(stderr, "ERROR: ggml_backend_cpu_init failed\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    ggml_backend_cpu_set_n_threads(cpu_backend, 1);
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
    if (n_tokens < 1) {
        std::fprintf(stderr, "ERROR: empty prompt\n");
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    std::printf("[run] prompt: \"%s\" -> %d tokens: [", prompt, n_tokens);
    for (int i = 0; i < n_tokens; ++i) std::printf(" %d", tokens[i]);
    std::printf(" ]\n");

    // ---- Reference: pure C++ math for block 0 over the tokens ----
    // First, force-load the block-0 weights (dequantized to F32).
    // The spike then reuses the same F32 arrays as inputs to its
    // ggml_cgraph, so the spike and reference compute with bit-
    // identical weights and the comparison is apples-to-apples.
    const ref_block_weights_t rw = ensure_ref_weights_loaded(model, sched);

    // Build F32 weight tensors in a metadata-only ggml_context.
    // ggml_set_input marks them as leaves so the spike graph will
    // treat them as inputs (no allocation by the scheduler for
    // these specific tensors' bodies); we allocate and fill them
    // ourselves via ggml_backend_sched_alloc_graph + ggml_backend_
    // tensor_set after the spike graph is built.
    ggml_init_params fw_iparams = {
        /* .mem_size   = */ 16 * ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * fw_ctx = ggml_init(fw_iparams);
    if (!fw_ctx) {
        std::fprintf(stderr, "ERROR: ggml_init failed for F32 weights ctx\n");
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    auto make_f32_tensor = [&](int64_t ne0, int64_t ne1) {
        int64_t ne[4] = { ne0, ne1, 1, 1 };
        ggml_tensor * t = ggml_new_tensor(fw_ctx, GGML_TYPE_F32, 4, ne);
        ggml_set_input(t);
        return t;
    };

    const int n_embd       = rw.n_embd;
    const int n_embd_head  = rw.n_embd_head;
    const int n_head       = rw.n_head;
    const int n_head_kv    = rw.n_head_kv;
    const int n_ff         = rw.n_ff;

    spike_f32_weights_t f32w{};
    f32w.attn_norm   = make_f32_tensor(n_embd, 1);
    f32w.attn_q_norm = make_f32_tensor(n_embd_head, 1);
    f32w.attn_k_norm = make_f32_tensor(n_embd_head, 1);
    f32w.ffn_norm    = make_f32_tensor(n_embd, 1);
    f32w.wq          = make_f32_tensor(n_embd, n_embd_head * n_head);
    f32w.wk          = make_f32_tensor(n_embd, n_embd_head * n_head_kv);
    f32w.wv          = make_f32_tensor(n_embd, n_embd_head * n_head_kv);
    f32w.wo          = make_f32_tensor(n_embd_head * n_head, n_embd);
    f32w.ffn_gate    = make_f32_tensor(n_embd, n_ff);
    f32w.ffn_up      = make_f32_tensor(n_embd, n_ff);
    f32w.ffn_down    = make_f32_tensor(n_ff,   n_embd);

    std::vector<float> ref_out((size_t)llama_model_n_embd(model) * n_tokens);
    compute_block_0_reference(model, sched, tokens.data(), n_tokens,
                              ref_out.data(), /*pos_offset=*/0);

    std::printf("[run] reference output (pos=0): first 16 = [");
    for (int i = 0; i < 16 && i < llama_model_n_embd(model); ++i) {
        std::printf(" %.4f", ref_out[i]);
    }
    std::printf(" ... ]\n");
    if (n_tokens > 1) {
        const int n_embd = llama_model_n_embd(model);
        std::printf("[run] reference output (pos=1): first 16 = [");
        for (int i = 0; i < 16 && i < n_embd; ++i) {
            std::printf(" %.4f", ref_out[n_embd + i]);
        }
        std::printf(" ... ]\n");
    }

    // Sanity check: no NaN, range reasonable.
    bool has_nan = false;
    float min_v = ref_out[0], max_v = ref_out[0];
    for (float v : ref_out) {
        if (std::isnan(v) || std::isinf(v)) has_nan = true;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }
    std::printf("[run] reference: has_nan=%d range=[%.3f, %.3f]\n",
                has_nan ? 1 : 0, min_v, max_v);
    if (has_nan) {
        std::fprintf(stderr, "ERROR: reference produced NaN/Inf; bailing\n");
        ggml_free(fw_ctx);
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // ---- Spike: per-block ggml_cgraph ----
    std::printf("[run] building spike graph for block 0 (n_tokens=%d)\n", n_tokens);
    spike_block_t sb = build_block_0_graph(model, n_tokens, /*pos_offset=*/0, &f32w);
    if (!sb.ctx || !sb.gf) {
        std::fprintf(stderr, "ERROR: build_block_0_graph returned empty graph\n");
        ggml_free(fw_ctx);
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    std::printf("[run] graph built: %d nodes\n", ggml_graph_n_nodes(sb.gf));

    // Allocate the graph. After this, the F32 weight input tensors
    // have backing storage on the scheduler's CPU buffer.
    if (!ggml_backend_sched_alloc_graph(sched, sb.gf)) {
        std::fprintf(stderr, "ERROR: sched_alloc_graph failed for spike graph\n");
        ggml_free(sb.ctx);
        ggml_free(fw_ctx);
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Copy the dequantized F32 weight data into the F32 weight
    // tensors' buffers.
    auto copy_in = [&](ggml_tensor * t, const float * src) {
        const size_t bytes = (size_t)ggml_nelements(t) * sizeof(float);
        ggml_backend_tensor_set(t, src, 0, bytes);
    };
    copy_in(f32w.attn_norm,   rw.attn_norm);
    copy_in(f32w.attn_q_norm, rw.attn_q_norm);
    copy_in(f32w.attn_k_norm, rw.attn_k_norm);
    copy_in(f32w.ffn_norm,    rw.ffn_norm);
    copy_in(f32w.wq,          rw.wq);
    copy_in(f32w.wk,          rw.wk);
    copy_in(f32w.wv,          rw.wv);
    copy_in(f32w.wo,          rw.wo);
    copy_in(f32w.ffn_gate,    rw.ffn_gate);
    copy_in(f32w.ffn_up,      rw.ffn_up);
    copy_in(f32w.ffn_down,    rw.ffn_down);

    // Fill input leaves: token_ids [n_tokens], positions [n_tokens].
    ggml_backend_tensor_set(sb.token_ids_in, tokens.data(), 0,
                            n_tokens * sizeof(llama_token));
    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; ++i) positions[i] = i;
    ggml_backend_tensor_set(sb.positions_in, positions.data(), 0,
                            n_tokens * sizeof(int32_t));

    ggml_backend_sched_graph_compute(sched, sb.gf);
    std::printf("[run] spike graph computed\n");

    // Read output. F32 [n_embd, n_tokens] row-major.
    const int n_embd_out = llama_model_n_embd(model);
    std::vector<float> spike_out((size_t)n_embd_out * n_tokens);
    ggml_backend_tensor_get(sb.out, spike_out.data(), 0,
                            (size_t)n_embd_out * n_tokens * sizeof(float));

    std::printf("[run] spike output (pos=0): first 16 = [");
    for (int i = 0; i < 16 && i < n_embd_out; ++i) {
        std::printf(" %.4f", spike_out[i]);
    }
    std::printf(" ... ]\n");
    if (n_tokens > 1) {
        std::printf("[run] spike output (pos=1): first 16 = [");
        for (int i = 0; i < 16 && i < n_embd_out; ++i) {
            std::printf(" %.4f", spike_out[n_embd_out + i]);
        }
        std::printf(" ... ]\n");
    }

    // Sanity check: no NaN, range reasonable.
    bool spike_nan = false;
    float spike_min = spike_out[0], spike_max = spike_out[0];
    for (float v : spike_out) {
        if (std::isnan(v) || std::isinf(v)) spike_nan = true;
        if (v < spike_min) spike_min = v;
        if (v > spike_max) spike_max = v;
    }
    std::printf("[run] spike: has_nan=%d range=[%.3f, %.3f]\n",
                spike_nan ? 1 : 0, spike_min, spike_max);
    if (spike_nan) {
        std::fprintf(stderr, "ERROR: spike produced NaN/Inf; bailing\n");
        ggml_free(sb.ctx);
        ggml_free(fw_ctx);
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Compare pos-1 output (most informative; pos-0 also useful).
    // PASS criterion from the breather plan: max abs diff < 1e-3 AND
    // > 99% of elements within 1e-3, per position. Both implementations
    // use the same F32 weights (dequantized once by the reference and
    // copied into the spike's input tensors), so the comparison is
    // bit-exact up to summation order in the matmuls.
    auto compare_pos = [&](int pos) {
        const size_t base = (size_t)pos * n_embd_out;
        float max_diff = 0.0f, sum_abs = 0.0f;
        int n_within_1e_3 = 0;
        for (int i = 0; i < n_embd_out; ++i) {
            const float a = ref_out[base + i];
            const float b = spike_out[base + i];
            const float d = std::fabs(a - b);
            if (d > max_diff) max_diff = d;
            sum_abs += d;
            if (d < 1e-3f) ++n_within_1e_3;
        }
        const float mean_abs = sum_abs / n_embd_out;
        const float frac = (float)n_within_1e_3 / n_embd_out;
        const bool pass = (max_diff < 1e-3f) && (frac > 0.99f);
        std::printf("[run] compare pos=%d: max_abs_diff=%.6f mean_abs_diff=%.6f "
                    "frac_within_1e-3=%.4f -> %s\n",
                    pos, max_diff, mean_abs, frac, pass ? "PASS" : "FAIL");
        return pass;
    };
    int n_positions_pass = 0;
    float global_max = 0.0f;
    for (int pos = 0; pos < n_tokens; ++pos) {
        if (compare_pos(pos)) ++n_positions_pass;
    }
    // Re-derive global_max from the per-pos pass (cheap; recompute).
    for (int pos = 0; pos < n_tokens; ++pos) {
        const size_t base = (size_t)pos * n_embd_out;
        for (int i = 0; i < n_embd_out; ++i) {
            const float d = std::fabs(ref_out[base + i] - spike_out[base + i]);
            if (d > global_max) global_max = d;
        }
    }
    const bool overall_pass = (n_positions_pass == n_tokens);
    std::printf("[run] %d/%d positions PASS, global max_abs_diff=%.6f -> %s\n",
                n_positions_pass, n_tokens, global_max,
                overall_pass ? "OVERALL: PASS" : "OVERALL: FAIL");
    if (!overall_pass) {
        std::fprintf(stderr, "ERROR: spike comparison failed\n");
        ggml_free(sb.ctx);
        ggml_free(fw_ctx);
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    ggml_free(sb.ctx);
    ggml_free(fw_ctx);
    ggml_backend_sched_reset(sched);
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu_backend);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

static void usage(const char* prog) {
    std::fprintf(stderr,
                 "Usage:\n"
                 "  %s capture <model.gguf> <prompt> <out_logits.bin>\n"
                 "  %s run <model.gguf> <prompt>\n",
                 prog, prog);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }
    const std::string mode = argv[1];
    if (mode == "capture") {
        if (argc != 5) { usage(argv[0]); return 1; }
        return do_capture(argv[2], argv[3], argv[4]);
    } else if (mode == "run") {
        if (argc != 4) { usage(argv[0]); return 1; }
        return do_run(argv[2], argv[3]);
    } else {
        usage(argv[0]);
        return 1;
    }
}
