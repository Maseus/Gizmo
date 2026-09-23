// Pure C++ reference for one qwen3 transformer block (block 0).
//
// Used as the oracle for the per-block forward pass spike. Mirrors
// the math in llama.cpp/src/models/qwen3.cpp:71-142 (block 0 only)
// but in pure C++ (no ggml ops) so we can compare spike output to
// an independent implementation.
//
// Block-0 weights are dequantized to F32 once on first call and
// cached. Memory: ~640 MB for qwen3:4b block 0.
//
// Dequantization path: llama.cpp's load step repacks Q4_K weights
// into an 8x8 tile layout (q4_K_8x8_q8_K) on the AMX/CPU backend.
// A raw read of `tensor->data` would see the repacked bytes, which
// the standard `dequantize_row_q4_K` cannot interpret. We use a
// one-shot `ggml_cpy(q4k_tensor, f32_dup)` graph through the
// scheduler, which dequantizes through the same backend code path
// that inference uses, producing standard F32 values.

#include "reference.h"
#include "llama-model.h"
#include "llama-hparams.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-quants.h"
#include "ggml-common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// Cached, dequantized F32 weights for block 0. Computed once.
struct BlockWeightsF32 {
    bool ready = false;

    std::vector<float> attn_norm;
    std::vector<float> wq;
    std::vector<float> wk;
    std::vector<float> wv;
    std::vector<float> wo;
    std::vector<float> attn_q_norm;
    std::vector<float> attn_k_norm;
    std::vector<float> ffn_norm;
    std::vector<float> ffn_gate;
    std::vector<float> ffn_up;
    std::vector<float> ffn_down;

    // Shapes
    int n_embd       = 0;
    int n_embd_head  = 0;
    int n_head       = 0;
    int n_head_kv    = 0;
    int n_embd_gqa   = 0;
    int n_ff         = 0;
    int n_rot        = 0;
    int n_ctx_orig   = 0;
    float freq_base  = 10000.0f;
    float eps        = 1e-5f;
    int rope_type    = 0;
    int n_vocab      = 0;
};

BlockWeightsF32 g_b0;

// Dequantize a ggml tensor to F32 via ggml_cpy through the scheduler.
// Returns a std::vector<float> of size ggml_nelements(t).
std::vector<float> dequantize_via_sched(ggml_backend_sched_t sched,
                                        const ggml_tensor * src,
                                        const char * tag) {
    const int64_t n = ggml_nelements(src);
    if (n == 0) {
        std::fprintf(stderr, "[ref]   skip %s (empty)\n", tag);
        return {};
    }
    std::fprintf(stderr, "[ref]   dequant %s: n=%lld type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
                 tag, (long long)n, ggml_type_name(src->type),
                 (long long)src->ne[0], (long long)src->ne[1],
                 (long long)src->ne[2], (long long)src->ne[3],
                 src->nb[0], src->nb[1], src->nb[2], src->nb[3]);
    std::fflush(stderr);

    ggml_init_params iparams = {
        /* .mem_size   = */ 32 * ggml_tensor_overhead() + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(iparams);
    if (!ctx) {
        std::fprintf(stderr, "ERROR: ggml_init failed for dequant %s\n", tag);
        std::abort();
    }

    ggml_tensor * dst = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, src->ne);
    if (!dst) {
        std::fprintf(stderr, "ERROR: ggml_new_tensor failed for dequant %s\n", tag);
        std::abort();
    }

    ggml_tensor * op = ggml_cpy(ctx, /*a*/ const_cast<ggml_tensor *>(src), dst);
    if (!op) {
        std::fprintf(stderr, "ERROR: ggml_cpy failed for dequant %s\n", tag);
        std::abort();
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, op);

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        std::fprintf(stderr, "ERROR: sched_alloc_graph failed for dequant %s\n", tag);
        std::abort();
    }
    ggml_backend_sched_graph_compute(sched, gf);

    std::vector<float> out((size_t)n);
    ggml_backend_tensor_get(dst, out.data(), 0, (size_t)n * sizeof(float));

    ggml_free(ctx);
    ggml_backend_sched_reset(sched);
    std::fprintf(stderr, "[ref]   dequant %s: done (%.2f MB)\n",
                 tag, (float)(n * 4) / (1024.0f * 1024.0f));
    return out;
}

// Dequantize the embedding rows for specific token ids via raw byte
// read + standard Q4_K/Q6_K dequant. Skips ggml graph machinery
// entirely. Works because llama.cpp's CPU repack is disabled in
// the spike build, so the model's Q*_K weights are in standard
// layout in CPU memory. Returns F32 [n_embd, n_tokens], row-major.
std::vector<float> dequant_tok_embd_rows(const ggml_tensor * src,
                                          const int32_t * token_ids,
                                          int n_tokens) {
    const int64_t n_embd = src->ne[0];
    const int64_t n_blocks_per_row = n_embd / QK_K;  // 2560/256 = 10 for qwen3
    const size_t row_bytes = (size_t)n_blocks_per_row * ggml_type_size(src->type);
    std::fprintf(stderr, "[ref]   dequant tok_embd rows: n_tokens=%d n_embd=%lld "
                 "type=%s n_blocks_per_row=%lld row_bytes=%zu\n",
                 n_tokens, (long long)n_embd, ggml_type_name(src->type),
                 (long long)n_blocks_per_row, row_bytes);
    std::fflush(stderr);

    // Read the rows we need from the backend buffer. Since repack is
    // disabled, the source buffer is a standard CPU buffer that
    // supports get_tensor.
    std::vector<uint8_t> buf((size_t)n_tokens * row_bytes);
    // Read each row individually (rows are at src->nb[1] stride).
    for (int t = 0; t < n_tokens; ++t) {
        const size_t row_off = (size_t)token_ids[t] * src->nb[1];
        ggml_backend_tensor_get(src, buf.data() + (size_t)t * row_bytes,
                                row_off, row_bytes);
    }

    // Dequantize each row.
    std::vector<float> out((size_t)n_embd * n_tokens);
    const ggml_type_traits * tr = ggml_get_type_traits(src->type);
    if (!tr || !tr->to_float) {
        std::fprintf(stderr, "ERROR: no dequant for tok_embd type %s\n",
                     ggml_type_name(src->type));
        std::abort();
    }
    for (int t = 0; t < n_tokens; ++t) {
        tr->to_float(buf.data() + (size_t)t * row_bytes,
                     out.data() + (size_t)t * n_embd,
                     n_embd);
    }

    std::fprintf(stderr, "[ref]   dequant tok_embd rows: done (%.2f MB)\n",
                 (float)(n_embd * n_tokens * 4) / (1024.0f * 1024.0f));
    return out;
}

void matmul(const float * W, const float * x, int n_out, int n_in, float * y) {
    for (int i = 0; i < n_out; ++i) {
        float acc = 0.0f;
        const float * row = W + (int64_t)i * n_in;
        for (int j = 0; j < n_in; ++j) {
            acc += row[j] * x[j];
        }
        y[i] = acc;
    }
}

void rmsnorm(const float * x, const float * gamma, int n_embd, int n_tokens, float eps, float * y) {
    for (int t = 0; t < n_tokens; ++t) {
        const float * xt = x + (int64_t)t * n_embd;
        float *       yt = y + (int64_t)t * n_embd;
        double sum = 0.0;
        for (int d = 0; d < n_embd; ++d) {
            sum += (double)xt[d] * (double)xt[d];
        }
        const float mean = (float)(sum / n_embd);
        const float scale = 1.0f / std::sqrt(mean + eps);
        for (int d = 0; d < n_embd; ++d) {
            yt[d] = xt[d] * scale;
        }
        if (gamma) {
            for (int d = 0; d < n_embd; ++d) {
                yt[d] *= gamma[d];
            }
        }
    }
}

void rmsnorm_per_head(const float * x, const float * gamma,
                      int n_embd_head, int n_head, int n_tokens,
                      float eps, float * y) {
    const int64_t per_token = (int64_t)n_embd_head * n_head;
    for (int t = 0; t < n_tokens; ++t) {
        for (int h = 0; h < n_head; ++h) {
            const float * xh = x + (int64_t)t * per_token + (int64_t)h * n_embd_head;
            float *       yh = y + (int64_t)t * per_token + (int64_t)h * n_embd_head;
            double sum = 0.0;
            for (int d = 0; d < n_embd_head; ++d) {
                sum += (double)xh[d] * (double)xh[d];
            }
            const float mean = (float)(sum / n_embd_head);
            const float scale = 1.0f / std::sqrt(mean + eps);
            for (int d = 0; d < n_embd_head; ++d) {
                yh[d] = xh[d] * scale * gamma[d];
            }
        }
    }
}

void rope_neox(const float * x, int n_embd_head, int n_head, int n_tokens,
               int n_rot, int pos, float freq_base,
               float * out) {
    // Neox half-rotation: rotate (x[i], x[i + n_rot/2]) with
    // cos/sin of theta_i. Same convention as ggml's rope_ext
    // (ops.cpp:6063 rotate_pairs with n_offset = n_dims/2).
    std::vector<float> cos_theta(n_rot / 2);
    std::vector<float> sin_theta(n_rot / 2);
    for (int i = 0; i < n_rot / 2; ++i) {
        const float theta = 1.0f / std::pow(freq_base, (float)(2 * i) / (float)n_rot);
        cos_theta[i] = std::cos((float)pos * theta);
        sin_theta[i] = std::sin((float)pos * theta);
    }
    const int n_half = n_rot / 2;
    const int64_t per_token = (int64_t)n_embd_head * n_head;
    for (int t = 0; t < n_tokens; ++t) {
        for (int h = 0; h < n_head; ++h) {
            const float * xh = x + (int64_t)t * per_token + (int64_t)h * n_embd_head;
            float *       oh = out + (int64_t)t * per_token + (int64_t)h * n_embd_head;
            for (int i = 0; i < n_half; ++i) {
                const float x0 = xh[i];
                const float x1 = xh[i + n_half];
                oh[i]        = x0 * cos_theta[i] - x1 * sin_theta[i];
                oh[i + n_half] = x0 * sin_theta[i] + x1 * cos_theta[i];
            }
            for (int d = n_rot; d < n_embd_head; ++d) {
                oh[d] = xh[d];
            }
        }
    }
}

void attention(const float * Q, const float * K, const float * V,
               int n_embd_head, int n_head_q, int n_head_kv, int n_tokens,
               float scale, float * out) {
    const int64_t per_token_q = (int64_t)n_embd_head * n_head_q;
    const int64_t per_token_k = (int64_t)n_embd_head * n_head_kv;
    for (int t = 0; t < n_tokens; ++t) {
        for (int hq = 0; hq < n_head_q; ++hq) {
            const int hk = hq * n_head_kv / n_head_q;
            const float * qh = Q + (int64_t)t * per_token_q + (int64_t)hq * n_embd_head;
            float *       oh = out + (int64_t)t * per_token_q + (int64_t)hq * n_embd_head;
            std::vector<float> scores((size_t)n_tokens);
            for (int tk = 0; tk < n_tokens; ++tk) {
                const float * kh = K + (int64_t)tk * per_token_k + (int64_t)hk * n_embd_head;
                double s = 0.0;
                for (int d = 0; d < n_embd_head; ++d) s += (double)qh[d] * (double)kh[d];
                scores[tk] = (float)s * scale;
            }
            float max_s = scores[0];
            for (int tk = 1; tk < n_tokens; ++tk) max_s = std::max(max_s, scores[tk]);
            float sum_exp = 0.0f;
            for (int tk = 0; tk < n_tokens; ++tk) {
                scores[tk] = std::exp(scores[tk] - max_s);
                sum_exp   += scores[tk];
            }
            const float inv = 1.0f / sum_exp;
            for (int tk = 0; tk < n_tokens; ++tk) scores[tk] *= inv;
            for (int d = 0; d < n_embd_head; ++d) oh[d] = 0.0f;
            for (int tk = 0; tk < n_tokens; ++tk) {
                const float * vh = V + (int64_t)tk * per_token_k + (int64_t)hk * n_embd_head;
                const float w = scores[tk];
                for (int d = 0; d < n_embd_head; ++d) {
                    oh[d] += w * vh[d];
                }
            }
        }
    }
}

void silu_inplace(float * x, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

void ensure_block_0_weights_loaded(const llama_model * model, ggml_backend_sched_t sched) {
    if (g_b0.ready) return;
    const auto & layer = model->layers[0];

    g_b0.n_embd      = (int)model->hparams.n_embd;
    g_b0.n_embd_head = (int)model->hparams.n_embd_head_v();
    g_b0.n_head      = (int)model->hparams.n_head();
    g_b0.n_head_kv   = (int)model->hparams.n_head_kv();
    g_b0.n_embd_gqa  = g_b0.n_embd_head * g_b0.n_head_kv;
    g_b0.n_ff        = (int)model->hparams.n_ff();
    g_b0.n_rot       = (int)model->hparams.n_rot();
    g_b0.n_ctx_orig  = (int)model->hparams.n_ctx_train;
    g_b0.freq_base   = model->hparams.rope_freq_base_train;
    g_b0.eps         = model->hparams.f_norm_rms_eps;
    g_b0.rope_type   = (int)model->hparams.rope_type;
    g_b0.n_vocab     = (int)ggml_nelements(model->tok_embd) / g_b0.n_embd;

    std::printf("[ref] block 0 shapes: n_embd=%d n_head=%d n_head_kv=%d n_embd_head=%d n_ff=%d n_rot=%d\n",
                g_b0.n_embd, g_b0.n_head, g_b0.n_head_kv, g_b0.n_embd_head,
                g_b0.n_ff, g_b0.n_rot);
    std::printf("[ref] rms_eps=%.2e freq_base=%.0f rope_type=%d\n",
                g_b0.eps, g_b0.freq_base, g_b0.rope_type);

    g_b0.attn_norm   = dequantize_via_sched(sched, layer.attn_norm,   "attn_norm");
    g_b0.wq          = dequantize_via_sched(sched, layer.wq,          "wq");
    g_b0.wk          = dequantize_via_sched(sched, layer.wk,          "wk");
    g_b0.wv          = dequantize_via_sched(sched, layer.wv,          "wv");
    g_b0.wo          = dequantize_via_sched(sched, layer.wo,          "wo");
    g_b0.attn_q_norm = dequantize_via_sched(sched, layer.attn_q_norm, "attn_q_norm");
    g_b0.attn_k_norm = dequantize_via_sched(sched, layer.attn_k_norm, "attn_k_norm");
    g_b0.ffn_norm    = dequantize_via_sched(sched, layer.ffn_norm,    "ffn_norm");
    g_b0.ffn_gate    = dequantize_via_sched(sched, layer.ffn_gate,    "ffn_gate");
    g_b0.ffn_up      = dequantize_via_sched(sched, layer.ffn_up,      "ffn_up");
    g_b0.ffn_down    = dequantize_via_sched(sched, layer.ffn_down,    "ffn_down");

    g_b0.ready = true;
    std::printf("[ref] block 0 weights loaded: %.1f MB dequantized\n",
                (g_b0.attn_norm.size() + g_b0.wq.size() + g_b0.wk.size() +
                 g_b0.wv.size() + g_b0.wo.size() + g_b0.attn_q_norm.size() +
                 g_b0.attn_k_norm.size() + g_b0.ffn_norm.size() +
                 g_b0.ffn_gate.size() + g_b0.ffn_up.size() + g_b0.ffn_down.size())
                 * 4.0f / (1024.0f * 1024.0f));
}

}  // namespace

ref_block_weights_t ensure_ref_weights_loaded(
    const llama_model * model,
    ggml_backend_sched_t sched
) {
    ensure_block_0_weights_loaded(model, sched);
    ref_block_weights_t r{};
    r.attn_norm   = g_b0.attn_norm.data();
    r.attn_q_norm = g_b0.attn_q_norm.data();
    r.attn_k_norm = g_b0.attn_k_norm.data();
    r.ffn_norm    = g_b0.ffn_norm.data();
    r.wq          = g_b0.wq.data();
    r.wk          = g_b0.wk.data();
    r.wv          = g_b0.wv.data();
    r.wo          = g_b0.wo.data();
    r.ffn_gate    = g_b0.ffn_gate.data();
    r.ffn_up      = g_b0.ffn_up.data();
    r.ffn_down    = g_b0.ffn_down.data();
    r.n_embd      = g_b0.n_embd;
    r.n_embd_head = g_b0.n_embd_head;
    r.n_head      = g_b0.n_head;
    r.n_head_kv   = g_b0.n_head_kv;
    r.n_ff        = g_b0.n_ff;
    r.n_rot       = g_b0.n_rot;
    r.freq_base   = g_b0.freq_base;
    r.eps         = g_b0.eps;
    r.rope_type   = g_b0.rope_type;
    r.n_ctx_train = g_b0.n_ctx_orig;
    r.n_vocab     = g_b0.n_vocab;
    return r;
}

std::vector<float> ref_dequant_embedding_rows(
    const llama_model * model,
    const int32_t *     token_ids,
    int                 n_tokens
) {
    return dequant_tok_embd_rows(model->tok_embd, token_ids, n_tokens);
}

void compute_block_0_reference(
    const llama_model *       model,
    ggml_backend_sched_t      sched,
    const int32_t *           token_ids,
    int                       n_tokens,
    float *                   out,
    int                       pos_offset
) {
    ensure_block_0_weights_loaded(model, sched);

    const int n_embd       = g_b0.n_embd;
    const int n_embd_head  = g_b0.n_embd_head;
    const int n_head       = g_b0.n_head;
    const int n_head_kv    = g_b0.n_head_kv;
    const int n_embd_gqa   = g_b0.n_embd_gqa;
    const int n_ff         = g_b0.n_ff;
    const int n_rot        = g_b0.n_rot;
    const float freq_base  = g_b0.freq_base;
    const float eps        = g_b0.eps;

    // 1. Embedding lookup via raw byte read + standard Q*_K dequant.
    // Skips ggml graph machinery. Works because the spike build has
    // GGML_CPU_REPACK=OFF, so the model's Q*_K weights are in standard
    // layout in CPU memory.
    std::vector<float> x = dequant_tok_embd_rows(model->tok_embd, token_ids, n_tokens);  // [n_embd, n_tokens] row-major

    // DEBUG: dump embedding first 8 elements for comparison against
    // the sharded engine's ggml path.
    std::fprintf(stderr, "[ref-db] pos0 emb first 8 = [");
    for (int i = 0; i < 8 && i < n_embd; ++i) std::fprintf(stderr, " %.4f", x[i]);
    std::fprintf(stderr, " ]\n");

    // 2. Pre-attention norm.
    std::vector<float> x_norm((size_t)n_embd * n_tokens);
    rmsnorm(x.data(), g_b0.attn_norm.data(), n_embd, n_tokens, eps, x_norm.data());

    // DEBUG: dump x_norm first 8 elements
    std::fprintf(stderr, "[ref-db] pos0 xn first 8  = [");
    for (int i = 0; i < 8 && i < n_embd; ++i) std::fprintf(stderr, " %.4f", x_norm[i]);
    std::fprintf(stderr, " ]\n");

    // 3. Q/K/V projections.
    // Q has n_head * n_embd_head output channels (= 32*128 = 4096 for
    // qwen3:4b), not n_embd. The matmul must write the full Q output
    // so the per-head RMSNorm/RoPE can iterate over all heads.
    const int n_q_out = n_embd_head * n_head;
    std::vector<float> q2d((size_t)n_q_out * n_tokens);
    std::vector<float> k2d((size_t)n_embd_gqa * n_tokens);
    std::vector<float> v2d((size_t)n_embd_gqa * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        matmul(g_b0.wq.data(), x_norm.data() + (int64_t)t * n_embd,
               n_q_out, n_embd, q2d.data() + (int64_t)t * n_q_out);
        matmul(g_b0.wk.data(), x_norm.data() + (int64_t)t * n_embd,
               n_embd_gqa, n_embd, k2d.data() + (int64_t)t * n_embd_gqa);
        matmul(g_b0.wv.data(), x_norm.data() + (int64_t)t * n_embd,
               n_embd_gqa, n_embd, v2d.data() + (int64_t)t * n_embd_gqa);
    }

    // 4. Q/K per-head RMSNorm.
    std::vector<float> q3d = q2d;
    std::vector<float> k3d = k2d;
    rmsnorm_per_head(q3d.data(), g_b0.attn_q_norm.data(),
                     n_embd_head, n_head, n_tokens, eps, q3d.data());
    rmsnorm_per_head(k3d.data(), g_b0.attn_k_norm.data(),
                     n_embd_head, n_head_kv, n_tokens, eps, k3d.data());

    // DEBUG: dump q3d first 8 (post-per-head-norm, pre-RoPE) for
    // comparison against sharded engine's Qpost intermediate.
    std::fprintf(stderr, "[ref-db] pos0 Qpost first 8 = [");
    for (int i = 0; i < 8 && i < n_embd_head * n_head; ++i) std::fprintf(stderr, " %.4f", q3d[i]);
    std::fprintf(stderr, " ]\n");

    // 5. RoPE for Q and K, one position at a time.
    std::vector<float> q_roped = q3d;
    std::vector<float> k_roped = k3d;
    for (int t = 0; t < n_tokens; ++t) {
        std::vector<float> q_t((size_t)n_embd_head * n_head);
        std::vector<float> k_t((size_t)n_embd_head * n_head_kv);
        for (int h = 0; h < n_head; ++h) {
            for (int d = 0; d < n_embd_head; ++d) {
                q_t[(int64_t)h * n_embd_head + d] =
                    q_roped[(int64_t)d + (int64_t)t * n_embd_head * n_head + (int64_t)h * n_embd_head];
            }
        }
        for (int h = 0; h < n_head_kv; ++h) {
            for (int d = 0; d < n_embd_head; ++d) {
                k_t[(int64_t)h * n_embd_head + d] =
                    k_roped[(int64_t)d + (int64_t)t * n_embd_head * n_head_kv + (int64_t)h * n_embd_head];
            }
        }
        rope_neox(q_t.data(), n_embd_head, n_head, 1, n_rot, pos_offset + t,
                  freq_base, q_t.data());
        rope_neox(k_t.data(), n_embd_head, n_head_kv, 1, n_rot, pos_offset + t,
                  freq_base, k_t.data());
        for (int h = 0; h < n_head; ++h) {
            for (int d = 0; d < n_embd_head; ++d) {
                q_roped[(int64_t)d + (int64_t)t * n_embd_head * n_head + (int64_t)h * n_embd_head] =
                    q_t[(int64_t)h * n_embd_head + d];
            }
        }
        for (int h = 0; h < n_head_kv; ++h) {
            for (int d = 0; d < n_embd_head; ++d) {
                k_roped[(int64_t)d + (int64_t)t * n_embd_head * n_head_kv + (int64_t)h * n_embd_head] =
                    k_t[(int64_t)h * n_embd_head + d];
            }
        }
    }

    // 6. Attention.
    std::vector<float> attn_out((size_t)n_embd_head * n_head * n_tokens);
    const float kq_scale = 1.0f / std::sqrt((float)n_embd_head);
    attention(q_roped.data(), k_roped.data(), v2d.data(),
              n_embd_head, n_head, n_head_kv, n_tokens,
              kq_scale, attn_out.data());

    // DEBUG: dump attn_out first 8 (post-flash, pre-WO) for comparison
    // against sharded engine's attn intermediate.
    std::fprintf(stderr, "[ref-db] pos0 attn  first 8 = [");
    for (int i = 0; i < 8 && i < (int)(n_embd_head * n_head); ++i) std::fprintf(stderr, " %.4f", attn_out[i]);
    std::fprintf(stderr, " ]\n");

    // 7. WO projection. attn_out is already [n_embd_head*n_head, n_tokens]
    // row-major, which is the layout WO expects (n_embd_head*n_head = 4096
    // inputs, n_embd = 2560 outputs). No reshape needed.
    std::vector<float> wo_out((size_t)n_embd * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        matmul(g_b0.wo.data(), attn_out.data() + (int64_t)t * n_embd_head * n_head,
               n_embd, n_embd_head * n_head, wo_out.data() + (int64_t)t * n_embd);
    }

    // 8. First residual.
    std::vector<float> ffn_inp((size_t)n_embd * n_tokens);
    for (size_t i = 0; i < ffn_inp.size(); ++i) {
        ffn_inp[i] = x[i] + wo_out[i];
    }

    // 9. Pre-FFN norm.
    std::vector<float> ffn_normed((size_t)n_embd * n_tokens);
    rmsnorm(ffn_inp.data(), g_b0.ffn_norm.data(), n_embd, n_tokens, eps, ffn_normed.data());

    // 10. SwiGLU FFN.
    std::vector<float> gate_buf((size_t)n_ff * n_tokens);
    std::vector<float> up_buf((size_t)n_ff * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        matmul(g_b0.ffn_gate.data(), ffn_normed.data() + (int64_t)t * n_embd,
               n_ff, n_embd, gate_buf.data() + (int64_t)t * n_ff);
        matmul(g_b0.ffn_up.data(), ffn_normed.data() + (int64_t)t * n_embd,
               n_ff, n_embd, up_buf.data() + (int64_t)t * n_ff);
    }
    silu_inplace(gate_buf.data(), (int)gate_buf.size());
    for (size_t i = 0; i < gate_buf.size(); ++i) {
        gate_buf[i] *= up_buf[i];
    }
    std::vector<float> ffn_out((size_t)n_embd * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        matmul(g_b0.ffn_down.data(), gate_buf.data() + (int64_t)t * n_ff,
               n_embd, n_ff, ffn_out.data() + (int64_t)t * n_embd);
    }

    // 11. Final residual.
    for (size_t i = 0; i < ffn_out.size(); ++i) {
        out[i] = ffn_inp[i] + ffn_out[i];
    }
}
