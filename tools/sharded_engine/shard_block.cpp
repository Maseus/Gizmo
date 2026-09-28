// Per-block ggml_cgraph for any qwen3/qwen3.5 block. See shard_block.h.
//
// Mirrors llama.cpp/src/models/qwen3.cpp:71-142 for one il (qwen3 path)
// and llama.cpp/src/models/qwen35.cpp:254-333 (qwen3.5 full-attn path).
//
// Phase 10/11: per-block dispatcher routes by block type:
//   - `is_recr(il) == true`        -> build_block_graph_ssm_stub (Phase 12)
//   - `is_recr(il) == false` and
//     `layer.ffn_norm != nullptr`  -> qwen3 builder (unchanged)
//   - `is_recr(il) == false` and
//     `layer.ffn_norm == nullptr` and
//     `layer.attn_post_norm != nullptr` -> qwen3.5 full-attn builder
//     (joint QG projection, MRoPE-4, attn_post_norm, gate*sigmoid)

#include "shard_block.h"
#include "llama-model.h"
#include "llama-hparams.h"
#include "llama-kv-cache.h"
#include "llama-memory.h"
#include "llama-memory-hybrid.h"
#include "llama-kv-cache.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>

namespace {
// See multi_block.cpp for the same helper. Returns the attention KV cache
// context whether the top-level memory context is a plain llama_kv_cache_context
// or a llama_memory_hybrid_context wrapping one.
llama_kv_cache_context * get_kv_ctx(llama_memory_context_i * mctx) {
    if (mctx == nullptr) {
        return nullptr;
    }
    auto * hybrid = dynamic_cast<llama_memory_hybrid_context *>(mctx);
    if (hybrid != nullptr) {
        return const_cast<llama_kv_cache_context *>(hybrid->get_attn());
    }
    return dynamic_cast<llama_kv_cache_context *>(mctx);
}

llama_memory_hybrid_context * get_hybrid_ctx(llama_memory_context_i * mctx) {
    return dynamic_cast<llama_memory_hybrid_context *>(mctx);
}
} // namespace

// Prepare a V tensor from the KV cache for ggml_flash_attn_ext, mirroring
// llama.cpp/src/llama-graph.cpp::build_attn_mha. The KV cache stores V
// transposed for some models (v_trans = true), which is detectable from the
// stride relationship nb[1] > nb[2] on the cache view returned by get_v().
// Without the conditional transpose the sharded attention output diverges from
// llama_decode on multi-token prompts for models like Qwen3.5.
static inline ggml_tensor * flash_attn_prep_v(
    ggml_context * ctx,
    ggml_tensor *  v_cache_view) {
    // Permute cache view [D/H, H_kv, n_kv, n_seq] -> [D/H, n_kv, H_kv, n_seq]
    ggml_tensor * v = ggml_permute(ctx, v_cache_view, 0, 2, 1, 3);
    if (v_cache_view->nb[1] > v_cache_view->nb[2]) {
        // v_trans path: native build_attn_mha transposes once more.
        v = ggml_transpose(ctx, v);
    }
    return ggml_cont(ctx, v);
}

ggml_tensor * build_block_graph_into(
    const llama_model *         model,
    ggml_context *              ctx,
    ggml_cgraph *               gf,
    int                         il,
    int                         n_tokens,
    ggml_tensor *               inpL_source,
    ggml_tensor *               token_ids,
    ggml_tensor *               positions,
    ggml_tensor *               kq_mask,
    ggml_tensor *               k_idxs,
    ggml_tensor *               v_idxs,
    llama_memory_context_i *    mctx
) {
    // Bisection aid: GIZMO_FORCE_STUB=1 forces every block (both SSM and
    // full-attn) through the SSM stub. The sharded engine becomes a
    // pure residual passthrough, isolating any per-block numerical
    // divergence to the per-block builders themselves.
    static const int s_force_stub = []() {
        const char * e = std::getenv("GIZMO_FORCE_STUB");
        if (e && std::atoi(e) != 0) {
            std::fprintf(stderr,
                "[sharded] GIZMO_FORCE_STUB=1: forcing all blocks through "
                "the SSM passthrough stub (residual only)\n");
            return 1;
        }
        return 0;
    }();
    if (s_force_stub) {
        return build_block_graph_ssm_stub_for_test(
            model, ctx, gf, il, n_tokens,
            inpL_source, token_ids, positions,
            kq_mask, k_idxs, v_idxs, mctx);
    }

    // Phase 10: dispatch by block type. SSM (recurrent / gated delta
    // net) blocks go to a stub for now; Phase 12 ports the real
    // builder from llama.cpp/src/models/qwen35.cpp::build_layer_attn_linear.
    // Full-attention blocks use the qwen3 builder below; Phase 11
    // upgrades it to qwen3.5 layout (joint QG, MRoPE-4, attn_post_norm).
    if (model->hparams.is_recr((uint32_t)il)) {
        // Phase 12: the SSM (recurrent / gated delta net) builder
        // needs the recurrent half of the hybrid memory context.
        // The stub path is retained for fallback testing only —
        // flip it on with GIZMO_SSM_STUB=1.
        auto * hybrid_ctx = get_hybrid_ctx(mctx);
        if (hybrid_ctx == nullptr) {
            std::fprintf(stderr, "ERROR: SSM block %d requires a hybrid memory context\n", il);
            return nullptr;
        }
        if (std::getenv("GIZMO_SSM_STUB") != nullptr) {
            return build_block_graph_ssm_stub_for_test(
                model, ctx, gf, il, n_tokens,
                inpL_source, token_ids, positions,
                kq_mask, k_idxs, v_idxs, mctx);
        }
        return build_block_graph_into_ssm_qwen35(
            model, ctx, gf, il, n_tokens,
            inpL_source, token_ids, positions,
            kq_mask, k_idxs, v_idxs,
            // get_recr() returns const; const_cast because the SSM
            // builder needs a non-const handle to write back into the
            // recurrent memory's F32 tensors via ggml_cpy. Same
            // pattern used in multi_block.cpp for kv_ctx.
            const_cast<llama_memory_recurrent_context *>(hybrid_ctx->get_recr()));
    }

    const auto & layer = model->layers[il];

    // Phase 11: qwen3.5 full-attn layers have a different schema
    // (attn_post_norm instead of ffn_norm, fused QG projection, MRoPE-4).
    // Detect by checking both: qwen3 populates `ffn_norm` and leaves
    // `attn_post_norm` null; qwen3.5 full-attn layers do the opposite.
    if (layer.ffn_norm == nullptr) {
        if (layer.attn_post_norm != nullptr) {
            auto * hybrid_ctx = get_hybrid_ctx(mctx);
            if (hybrid_ctx == nullptr) {
                std::fprintf(stderr, "ERROR: qwen3.5-style block %d requires a hybrid memory context\n", il);
                return nullptr;
            }
            return build_block_graph_into_attn_qwen35(
                model, ctx, gf, il, n_tokens,
                inpL_source, token_ids, positions,
                kq_mask, k_idxs, v_idxs, mctx);
        }
        // Some other new-arch layer we don't know yet. Emit a one-shot
        // error and return nullptr so the row driver aborts cleanly.
        static thread_local int s_warned_layer = -1;
        if (s_warned_layer != il) {
            std::fprintf(stderr,
                "ERROR: sharded engine: layer %d is full-attention but "
                "has neither ffn_norm (qwen3) nor attn_post_norm "
                "(qwen3.5). Unknown layout; aborting prefill.\n",
                il);
            s_warned_layer = il;
        }
        return nullptr;
    }
    const int n_head      = (int)model->hparams.n_head();
    const int n_head_kv   = (int)model->hparams.n_head_kv();
    const int n_embd_head = (int)model->hparams.n_embd_head_v();
    const int n_rot       = (int)model->hparams.n_rot();
    const int n_ctx_orig  = (int)model->hparams.n_ctx_train;
    const int rope_type   = (int)model->hparams.rope_type;
    const float freq_base = model->hparams.rope_freq_base_train;
    const float eps       = model->hparams.f_norm_rms_eps;

    // mark weights as inputs; their data is in the model mmap
    ggml_set_input(layer.attn_norm);
    ggml_set_input(layer.attn_q_norm);
    ggml_set_input(layer.attn_k_norm);
    ggml_set_input(layer.ffn_norm);
    ggml_set_input(layer.wq);
    ggml_set_input(layer.wk);
    ggml_set_input(layer.wv);
    ggml_set_input(layer.wo);
    ggml_set_input(layer.ffn_gate);
    ggml_set_input(layer.ffn_up);
    ggml_set_input(layer.ffn_down);

    // mark caller-managed shared row inputs (idempotent across blocks)
    ggml_set_input(kq_mask);
    ggml_set_input(positions);
    if (k_idxs != nullptr) ggml_set_input(k_idxs);
    if (v_idxs != nullptr) ggml_set_input(v_idxs);

    ggml_tensor * inpL;
    if (inpL_source == nullptr) {
        // first block of prefill: lookup embeddings via token_ids
        ggml_set_input(token_ids);
        inpL = ggml_get_rows(ctx, model->tok_embd, token_ids);
    } else {
        ggml_set_input(inpL_source);
        inpL = inpL_source;
    }

    // pre-attention norm
    ggml_tensor * x_norm = ggml_rms_norm(ctx, inpL, eps);
    x_norm = ggml_mul(ctx, x_norm, layer.attn_norm);

    // Q/K/V projections
    ggml_tensor * Qcur = ggml_mul_mat(ctx, layer.wq, x_norm);
    Qcur = ggml_reshape_3d(ctx, Qcur, n_embd_head, n_head, n_tokens);

    ggml_tensor * Kcur = ggml_mul_mat(ctx, layer.wk, x_norm);
    Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_tokens);

    ggml_tensor * Vcur = ggml_mul_mat(ctx, layer.wv, x_norm);
    Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_tokens);

    // per-head Q/K RMSNorm
    Qcur = ggml_rms_norm(ctx, Qcur, eps);
    Qcur = ggml_mul(ctx, Qcur, layer.attn_q_norm);

    Kcur = ggml_rms_norm(ctx, Kcur, eps);
    Kcur = ggml_mul(ctx, Kcur, layer.attn_k_norm);

    // RoPE
    Qcur = ggml_rope_ext(
            ctx, Qcur, positions, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base,
            1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    Kcur = ggml_rope_ext(
            ctx, Kcur, positions, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base,
            1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Phase 7 (if mctx is non-null): write K/V into the cache cells
    // at the slot indices init_batch assigned. K_cur/V_cur are still
    // in the [n_embd_head, n_head, n_tokens] layout here, which is
    // what cpy_k/cpy_v expect. The qwen3 KV cache is F16, so cast
    // K_cur/V_cur to F16 before cpy_k/cpy_v to match storage type.
    llama_kv_cache_context * kv_ctx = get_kv_ctx(mctx);
    if (kv_ctx != nullptr) {
        ggml_tensor * Kcur_f16 = (Kcur->type == GGML_TYPE_F16) ? Kcur : ggml_cast(ctx, Kcur, GGML_TYPE_F16);
        ggml_tensor * Vcur_f16 = (Vcur->type == GGML_TYPE_F16) ? Vcur : ggml_cast(ctx, Vcur, GGML_TYPE_F16);
        ggml_build_forward_expand(gf, kv_ctx->cpy_k(ctx, Kcur_f16, k_idxs, il));
        ggml_build_forward_expand(gf, kv_ctx->cpy_v(ctx, Vcur_f16, v_idxs, il));
    }

    // Phase 7 (if mctx is non-null): read K/V back from the cache for
    // the attention op. Cache views are pre-permute layout; the
    // permute step below produces [n_embd_head, n_tokens, n_head].
    if (kv_ctx != nullptr) {
        Kcur = kv_ctx->get_k(ctx, il);
        Vcur = kv_ctx->get_v(ctx, il);
    } else {
        // No-mctx path (Phase 6).
        if (Kcur->type == GGML_TYPE_F32) {
            Kcur = ggml_cast(ctx, Kcur, GGML_TYPE_F16);
        }
        if (Vcur->type == GGML_TYPE_F32) {
            Vcur = ggml_cast(ctx, Vcur, GGML_TYPE_F16);
        }
    }

    // flash attn wants Q/K/V permuted into the layout native build_attn_mha
    // produces. Q/K are always [D, n_tokens, n_head, 1]; V depends on v_trans.
    Qcur = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));
    Kcur = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));
    Vcur = flash_attn_prep_v(ctx, Vcur);

    const float kq_scale = 1.0f / std::sqrt((float)n_embd_head);
    ggml_tensor * attn_out = ggml_flash_attn_ext(
            ctx, Qcur, Kcur, Vcur, kq_mask,
            kq_scale, /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
    ggml_prec_set_acc(attn_out, GGML_PREC_F32);

    attn_out = ggml_reshape_2d(ctx, attn_out, n_embd_head * n_head, n_tokens);

    // WO projection + first residual
    ggml_tensor * wo_out = ggml_mul_mat(ctx, layer.wo, attn_out);
    ggml_tensor * ffn_inp = ggml_add(ctx, wo_out, inpL);

    // pre-FFN norm
    ggml_tensor * ffn_normed = ggml_rms_norm(ctx, ffn_inp, eps);
    ffn_normed = ggml_mul(ctx, ffn_normed, layer.ffn_norm);

    // SwiGLU FFN
    ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, ffn_normed);
    ggml_tensor * up   = ggml_mul_mat(ctx, layer.ffn_up,   ffn_normed);
    gate = ggml_silu(ctx, gate);
    ggml_tensor * glu  = ggml_mul(ctx, gate, up);
    ggml_tensor * ffn_out = ggml_mul_mat(ctx, layer.ffn_down, glu);

    // final residual
    ggml_tensor * result = ggml_add(ctx, ffn_out, ffn_inp);

    ggml_build_forward_expand(gf, result);
    return result;
}

shard_block_t build_shard_block_graph(
    const llama_model *            model,
    int                            il,
    int                            n_tokens,
    ggml_tensor *                  inpL_carrier,
    ggml_tensor *                  kq_mask,
    llama_memory_context_i *       mctx
) {
    shard_block_t out{};

    ggml_init_params iparams = {
        /* .mem_size   = */ 128 * ggml_tensor_overhead() + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(iparams);
    if (!ctx) {
        return out;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);

    // per-block inputs (the single-block wrapper allocates these
    // itself; the row driver pre-allocates them once and reuses).
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(positions);
    out.positions_in = positions;

    ggml_tensor * token_ids = nullptr;
    if (inpL_carrier == nullptr) {
        token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(token_ids);
        out.token_ids_in = token_ids;
    } else {
        out.residual_in = inpL_carrier;
    }

    ggml_tensor * k_idxs = nullptr;
    ggml_tensor * v_idxs = nullptr;
    if (mctx != nullptr) {
        llama_kv_cache_context * kv_ctx = get_kv_ctx(mctx);
        const llama_ubatch & ubatch = kv_ctx->get_ubatch();
        k_idxs = kv_ctx->build_input_k_idxs(ctx, ubatch);
        v_idxs = kv_ctx->build_input_v_idxs(ctx, ubatch);
        out.k_idxs_in = k_idxs;
        out.v_idxs_in = v_idxs;
    }

    ggml_tensor * out_residual = build_block_graph_into(
        model, ctx, gf, il, n_tokens,
        inpL_carrier, token_ids, positions, kq_mask,
        k_idxs, v_idxs, mctx);

    if (out_residual == nullptr) {
        ggml_free(ctx);
        out.ctx = nullptr;
        out.gf  = nullptr;
        return out;
    }

    out.ctx = ctx;
    out.gf  = gf;
    out.out = out_residual;
    return out;
}

// Phase 10 SSM stub. Phase 12 replaces this with the real
// gated-delta-net builder. For now, emit a single warning per layer
// (not per call) and propagate the input residual unchanged so the
// downstream row can still complete. The output is numerically
// wrong for that layer, but the sharded engine runs end-to-end
// without crashing.
//
// Selected via GIZMO_SSM_STUB=1; default Phase 12 uses the real
// gated-delta-net builder.
//
// For layer 0 (first block of the whole prefill), inpL_source is
// nullptr because the row driver looks up embeddings via token_ids.
// In that case we do the lookup ourselves (mirroring the qwen3
// full-attn path's "inpL_source == nullptr" branch) so the stub's
// return value is a real F32 [n_embd, n_tokens] tensor the next
// block can read.
//
// `gf` is intentionally unused here: the stub adds no nodes to the
// cgraph. The signature must match build_block_graph_into.
ggml_tensor * build_block_graph_ssm_stub_for_test(
    const llama_model *            model,
    ggml_context *                 ctx,
    ggml_cgraph *                  gf,
    int                            il,
    int                            /*n_tokens*/,
    ggml_tensor *                  inpL_source,
    ggml_tensor *                  token_ids,
    ggml_tensor *                  /*positions*/,
    ggml_tensor *                  /*kq_mask*/,
    ggml_tensor *                  /*k_idxs*/,
    ggml_tensor *                  /*v_idxs*/,
    llama_memory_context_i *       /*mctx*/
) {
    static thread_local int s_warned_layer = -1;
    if (s_warned_layer != il) {
        std::fprintf(stderr,
            "WARNING: SSM stub at layer %d (Phase 12 work). The sharded "
            "engine will pass the input residual through unchanged for "
            "this layer; outputs are numerically wrong from here on.\n",
            il);
        s_warned_layer = il;
    }

    ggml_tensor * passthrough;
    if (inpL_source != nullptr) {
        ggml_set_input(inpL_source);
        passthrough = inpL_source;
    } else {
        // First block of prefill: do the embedding lookup so the
        // returned tensor is a real F32 [n_embd, n_tokens] the next
        // block can read. Mirrors build_block_graph_into's first
        // branch (qwen35.cpp's hybrid model has tok_embd of shape
        // [n_embd, n_vocab] same as qwen3).
        ggml_set_input(token_ids);
        passthrough = ggml_get_rows(ctx, model->tok_embd, token_ids);
    }
    // Add the passthrough as a leaf of the row cgraph so the
    // scheduler considers it as an output of this block. Downstream
    // blocks will read from it as if it were a real residual.
    ggml_build_forward_expand(gf, passthrough);
    return passthrough;
}

// Phase 11: qwen3.5 full-attention block builder. Mirrors
// llama.cpp/src/models/qwen35.cpp:254-333 (build_layer_attn) and
// :155-203 (graph::graph's outer residual structure for this block),
// translated into the per-block append shape that build_block_graph_into
// uses.
//
// Key differences from the qwen3 builder:
//   1. `wq` outputs [Q | G] interleaved per head — wq's row count is
//      n_embd_head * 2, not n_embd_head.
//   2. MRoPE-4 — ggml_rope_multi with hparams.rope_sections[4],
//      not the 1-D ggml_rope_ext.
//   3. attn_post_norm replaces ffn_norm at the FFN-input slot. The
//      outer graph does:  attn_out -> residual add -> attn_post_norm
//      -> FFN -> second residual add. There is no separate pre-FFN norm.
//   4. gate*sigmoid after attention, before wo.
//   5. n_head != n_head_kv is allowed (GQA via qwen3.5's wk/wv strides).
//
// KV cache handling: full-attn layers in qwen3.5 use a normal KV cache
// (the ISWA split is at the *memory* layer not the block). The sharded
// engine's per-block cpy_k/cpy_v path takes the F16 cast from the
// qwen3 builder as-is.
//
// Returns the block's final residual (F32 [n_embd, n_tokens]).
ggml_tensor * build_block_graph_into_attn_qwen35(
    const llama_model *         model,
    ggml_context *              ctx,
    ggml_cgraph *               gf,
    int                         il,
    int                         n_tokens,
    ggml_tensor *               inpL_source,
    ggml_tensor *               token_ids,
    ggml_tensor *               positions,
    ggml_tensor *               kq_mask,
    ggml_tensor *               k_idxs,
    ggml_tensor *               v_idxs,
    llama_memory_context_i *    mctx
) {
    const auto & layer = model->layers[il];

    // Safety net: dispatcher only routes qwen3.5 full-attn layers
    // here, so attn_post_norm must be present and ffn_norm must be
    // null. If either invariant is violated, fail loudly rather than
    // touching a null tensor.
    GGML_ASSERT(layer.attn_post_norm != nullptr);
    GGML_ASSERT(layer.ffn_norm       == nullptr);

    const int n_head      = (int)model->hparams.n_head();
    const int n_head_kv   = (int)model->hparams.n_head_kv();
    const int n_embd_head = (int)model->hparams.n_embd_head_v();
    const int n_rot       = (int)model->hparams.n_rot();
    const int n_ctx_orig  = (int)model->hparams.n_ctx_train;
    const int rope_type   = (int)model->hparams.rope_type;
    const float freq_base = model->hparams.rope_freq_base_train;
    const float eps       = model->hparams.f_norm_rms_eps;
    const float kq_scale  = 1.0f / std::sqrt((float)n_embd_head);

    // MRoPE-4 sections (default qwen3.5: {11, 11, 10, 0}).
    int sections[4];
    for (int i = 0; i < 4; ++i) {
        sections[i] = model->hparams.rope_sections[i];
    }

    // mark weights as inputs.
    ggml_set_input(layer.attn_norm);
    ggml_set_input(layer.attn_q_norm);
    ggml_set_input(layer.attn_k_norm);
    ggml_set_input(layer.attn_post_norm);
    ggml_set_input(layer.wq);
    ggml_set_input(layer.wk);
    ggml_set_input(layer.wv);
    ggml_set_input(layer.wo);
    ggml_set_input(layer.ffn_gate);
    ggml_set_input(layer.ffn_up);
    ggml_set_input(layer.ffn_down);

    // mark caller-managed shared row inputs.
    ggml_set_input(kq_mask);
    ggml_set_input(positions);
    if (k_idxs != nullptr) ggml_set_input(k_idxs);
    if (v_idxs != nullptr) ggml_set_input(v_idxs);

    ggml_tensor * inpL;
    if (inpL_source == nullptr) {
        ggml_set_input(token_ids);
        inpL = ggml_get_rows(ctx, model->tok_embd, token_ids);
    } else {
        ggml_set_input(inpL_source);
        inpL = inpL_source;
    }

    // Pre-attention norm (same slot as qwen3).
    ggml_tensor * x_norm = ggml_rms_norm(ctx, inpL, eps);
    x_norm = ggml_mul(ctx, x_norm, layer.attn_norm);

    // Bisection probe: tag intermediate tensors with names matching
    // the sharded probe prefixes (GIZMO_PROBE_PREFIXES). The eval
    // callback in multi_block.cpp captures the last-token column of
    // each. Compare against reference dumps via diff_per_layer to
    // localize divergence within a layer.
    static const int s_probe_attn = []() {
        const char * e = std::getenv("GIZMO_PROBE_PREFIXES");
        return (e && std::strstr(e, "shard_attn_residual-")) ? 1 : 0;
    }();
    static const int s_probe_post_norm = []() {
        const char * e = std::getenv("GIZMO_PROBE_PREFIXES");
        return (e && std::strstr(e, "shard_attn_post_norm-")) ? 1 : 0;
    }();
    static const int s_probe_ffn_out = []() {
        const char * e = std::getenv("GIZMO_PROBE_PREFIXES");
        return (e && std::strstr(e, "shard_ffn_out-")) ? 1 : 0;
    }();
    static const int s_probe_inpL = []() {
        const char * e = std::getenv("GIZMO_PROBE_PREFIXES");
        return (e && std::strstr(e, "shard_inpL-")) ? 1 : 0;
    }();
    static const int s_probe_xnorm = []() {
        const char * e = std::getenv("GIZMO_PROBE_PREFIXES");
        return (e && std::strstr(e, "shard_x_norm-")) ? 1 : 0;
    }();
    static const int s_probe_attn_raw = []() {
        const char * e = std::getenv("GIZMO_PROBE_PREFIXES");
        return (e && std::strstr(e, "shard_attn_raw-")) ? 1 : 0;
    }();
    static const int s_probe_attn_gated = []() {
        const char * e = std::getenv("GIZMO_PROBE_PREFIXES");
        return (e && std::strstr(e, "shard_attn_gated-")) ? 1 : 0;
    }();
    static const int s_probe_qcur = []() {
        const char * e = std::getenv("GIZMO_PROBE_PREFIXES");
        return (e && std::strstr(e, "shard_qcur_normed-")) ? 1 : 0;
    }();
    static const int s_probe_kcur = []() {
        const char * e = std::getenv("GIZMO_PROBE_PREFIXES");
        return (e && std::strstr(e, "shard_kcur_normed-")) ? 1 : 0;
    }();
    static const int s_probe_kcur_cached = []() {
        const char * e = std::getenv("GIZMO_PROBE_PREFIXES");
        return (e && std::strstr(e, "shard_kcur_cached-")) ? 1 : 0;
    }();
    static const int s_probe_vcur_cached = []() {
        const char * e = std::getenv("GIZMO_PROBE_PREFIXES");
        return (e && std::strstr(e, "shard_vcur_cached-")) ? 1 : 0;
    }();
    static const int s_probe_kcur_rope = []() {
        const char * e = std::getenv("GIZMO_PROBE_PREFIXES");
        return (e && std::strstr(e, "shard_kcur_rope-")) ? 1 : 0;
    }();
    static const int s_probe_vcur_pre_rope = []() {
        const char * e = std::getenv("GIZMO_PROBE_PREFIXES");
        return (e && std::strstr(e, "shard_vcur_pre-")) ? 1 : 0;
    }();
    static const int s_probe_wo = []() {
        const char * e = std::getenv("GIZMO_PROBE_PREFIXES");
        return (e && std::strstr(e, "shard_wo_out-")) ? 1 : 0;
    }();
    static const int s_probe_qkv_raw = []() {
        const char * e = std::getenv("GIZMO_PROBE_PREFIXES");
        return (e && (std::strstr(e, "shard_kcur_raw-") || std::strstr(e, "shard_qfull_raw-") || std::strstr(e, "shard_vcur_raw-"))) ? 1 : 0;
    }();
    if (s_probe_inpL)      ggml_format_name(inpL,    "shard_inpL-%d", il);
    if (s_probe_xnorm)     ggml_format_name(x_norm,  "shard_x_norm-%d", il);

    // Joint QG projection: wq's row count is 2 * n_embd_head (Q+G
    // interleaved per head); Q+G split is done after the matmul via
    // stride-2 views.
    // Phase 12 fix: apply _s scaling tensors when present (mirrors
    // build_lora_mm in llama-graph.cpp:1514-1542).
    ggml_tensor * Qcur_full_mat = ggml_mul_mat(ctx, layer.wq, x_norm);
    if (layer.wq_s) Qcur_full_mat = ggml_mul(ctx, Qcur_full_mat, layer.wq_s);
    ggml_tensor * Qcur_full = ggml_reshape_3d(ctx, Qcur_full_mat, n_embd_head * 2, n_head, n_tokens);
    ggml_tensor * Kcur_mat = ggml_mul_mat(ctx, layer.wk, x_norm);
    if (layer.wk_s) Kcur_mat = ggml_mul(ctx, Kcur_mat, layer.wk_s);
    ggml_tensor * Kcur = ggml_reshape_3d(ctx, Kcur_mat, n_embd_head, n_head_kv, n_tokens);
    ggml_tensor * Vcur_mat = ggml_mul_mat(ctx, layer.wv, x_norm);
    if (layer.wv_s) Vcur_mat = ggml_mul(ctx, Vcur_mat, layer.wv_s);
    ggml_tensor * Vcur = ggml_reshape_3d(ctx, Vcur_mat, n_embd_head, n_head_kv, n_tokens);
    if (s_probe_qkv_raw) {
        ggml_format_name(Kcur_mat, "shard_kcur_raw-%d", il);
        // Tag the matmul output (matches the ref tag in
        // qwen35.cpp::build_qkv with reshape=false, where
        // Qcur_full IS the raw matmul output of the joint QG
        // projection). Both ref and shard probes read the last
        // column of the same F32 [8192, 5] tensor.
        ggml_format_name(Qcur_full_mat, "shard_qfull_raw-%d", il);
        ggml_format_name(Vcur_mat, "shard_vcur_raw-%d", il);
    }

    // Q view: stride-2 from Qcur_full, offset 0.
    ggml_tensor * Qcur = ggml_view_3d(
        ctx, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        /*offset=*/0);
    Qcur = ggml_rms_norm(ctx, Qcur, eps);
    Qcur = ggml_mul(ctx, Qcur, layer.attn_q_norm);

    // K norm (no gate attached).
    Kcur = ggml_rms_norm(ctx, Kcur, eps);
    Kcur = ggml_mul(ctx, Kcur, layer.attn_k_norm);
    if (s_probe_qcur)  ggml_format_name(Qcur, "shard_qcur_normed-%d", il);
    if (s_probe_kcur)  ggml_format_name(Kcur, "shard_kcur_normed-%d", il);

    // Gate view: same stride, offset half a head. qwen35.cpp calls
    // ggml_cont_2d on it so the subsequent ggml_mul with attn_out
    // is contiguous; we mirror that.
    ggml_tensor * gate = ggml_view_3d(
        ctx, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx, gate, n_embd_head * n_head, n_tokens);

    // MRoPE-4 instead of 1-D ggml_rope_ext. Note: qwen35.cpp does
    // NOT reshape Vcur before RoPE; reshape is only applied to Q and
    // K. Vcur was already 3d above so no extra reshape needed.
    Qcur = ggml_rope_multi(
        ctx, Qcur, positions, nullptr,
        n_rot, sections, rope_type, n_ctx_orig, freq_base,
        1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    Kcur = ggml_rope_multi(
        ctx, Kcur, positions, nullptr,
        n_rot, sections, rope_type, n_ctx_orig, freq_base,
        1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    if (s_probe_kcur_rope) ggml_format_name(Kcur, "shard_kcur_rope-%d", il);
    if (s_probe_vcur_pre_rope) ggml_format_name(Vcur, "shard_vcur_pre-%d", il);

    // KV cache wiring (same path as qwen3: cast to F16, cpy_k/cpy_v,
    // then read back via get_k/get_v if mctx present).
    if (mctx != nullptr) {
        llama_kv_cache_context * kv_ctx = get_kv_ctx(mctx);
        ggml_tensor * Kcur_f16 = (Kcur->type == GGML_TYPE_F16) ? Kcur : ggml_cast(ctx, Kcur, GGML_TYPE_F16);
        ggml_tensor * Vcur_f16 = (Vcur->type == GGML_TYPE_F16) ? Vcur : ggml_cast(ctx, Vcur, GGML_TYPE_F16);
        ggml_build_forward_expand(gf, kv_ctx->cpy_k(ctx, Kcur_f16, k_idxs, il));
        ggml_build_forward_expand(gf, kv_ctx->cpy_v(ctx, Vcur_f16, v_idxs, il));
    }
    if (mctx != nullptr) {
        llama_kv_cache_context * kv_ctx = get_kv_ctx(mctx);
        Kcur = kv_ctx->get_k(ctx, il);
        Vcur = kv_ctx->get_v(ctx, il);
    } else {
        if (Kcur->type == GGML_TYPE_F32) {
            Kcur = ggml_cast(ctx, Kcur, GGML_TYPE_F16);
        }
        if (Vcur->type == GGML_TYPE_F32) {
            Vcur = ggml_cast(ctx, Vcur, GGML_TYPE_F16);
        }
    }
    if (s_probe_kcur_cached) ggml_format_name(Kcur, "shard_kcur_cached-%d", il);
    if (s_probe_vcur_cached) ggml_format_name(Vcur, "shard_vcur_cached-%d", il);

    // Flash attn wants Q/K/V permuted into the layout native build_attn_mha
    // produces. Q/K are always [D, n_tokens, n_head, 1]; V depends on v_trans.
    Qcur = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));
    Kcur = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));
    Vcur = flash_attn_prep_v(ctx, Vcur);

    ggml_tensor * attn_out = ggml_flash_attn_ext(
        ctx, Qcur, Kcur, Vcur, kq_mask,
        kq_scale, /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
    ggml_prec_set_acc(attn_out, GGML_PREC_F32);

    attn_out = ggml_reshape_2d(ctx, attn_out, n_embd_head * n_head, n_tokens);

    // qwen3.5 difference: gate*sigmoid before wo.
    ggml_tensor * gate_sig = ggml_sigmoid(ctx, gate);
    ggml_tensor * attn_out_pre_gate = attn_out;
    attn_out = ggml_mul(ctx, attn_out, gate_sig);

    // WO + first residual.
    ggml_tensor * wo_out = ggml_mul_mat(ctx, layer.wo, attn_out);
    if (s_probe_wo) ggml_format_name(wo_out, "shard_wo_out-%d", il);
    ggml_tensor * ffn_residual = ggml_add(ctx, wo_out, inpL);

    // qwen3.5 difference: attn_post_norm replaces ffn_norm at the
    // FFN-input slot. The norm is applied to ffn_residual (post
    // attention + post residual), NOT to a separate pre-FFN norm.
    ggml_tensor * post_normed = ggml_rms_norm(ctx, ffn_residual, eps);
    post_normed = ggml_mul(ctx, post_normed, layer.attn_post_norm);

    // SwiGLU FFN (no additional pre-FFN norm -- that's the qwen3.5
    // point).
    ggml_tensor * ffn_gate_out = ggml_mul_mat(ctx, layer.ffn_gate, post_normed);
    ggml_tensor * ffn_up_out   = ggml_mul_mat(ctx, layer.ffn_up,   post_normed);
    ffn_gate_out = ggml_silu(ctx, ffn_gate_out);
    ggml_tensor * glu = ggml_mul(ctx, ffn_gate_out, ffn_up_out);
    ggml_tensor * ffn_out = ggml_mul_mat(ctx, layer.ffn_down, glu);

    // Second residual: ffn_out + ffn_residual (matches qwen35.cpp:195).
    ggml_tensor * result = ggml_add(ctx, ffn_out, ffn_residual);

    if (s_probe_attn)      ggml_format_name(ffn_residual, "shard_attn_residual-%d", il);
    if (s_probe_post_norm) ggml_format_name(post_normed,  "shard_attn_post_norm-%d", il);
    if (s_probe_ffn_out)   ggml_format_name(ffn_out,       "shard_ffn_out-%d", il);
    if (s_probe_attn_raw)  ggml_format_name(attn_out_pre_gate, "shard_attn_raw-%d", il);
    if (s_probe_attn_gated) ggml_format_name(attn_out, "shard_attn_gated-%d", il);

    ggml_build_forward_expand(gf, result);
    return result;
}

// Phase 12: qwen3.5 gated delta net (recurrent / linear-attention)
// block builder. Direct port of
// llama.cpp/src/models/qwen35.cpp::build_layer_attn_linear
// (lines 335-468) plus the outer residual block flow from
// qwen35.cpp:155-203.
//
// The sharded engine instantiates the hybrid memory context once
// per prefill and threads `recr_ctx` (the recurrent half) into
// every SSM block. The block uses `recr_ctx->get_r_l(il)` to read
// the conv state row at `kv_head`, `recr_ctx->get_s_l(il)` to read
// the recurrent state row at `kv_head`, and writes the updated
// states back via ggml_cpy nodes that get ggml_build_forward_expand'd
// into the same row cgraph.
//
// The conv_state / recurrent_state helper functions are inlined
// here as static helpers rather than introducing a new dependency
// on `models.h` / `llm_build_delta_net_base` -- this file already
// touches ggml + llama-memory only.
//
// The default cparams (fused_gdn_ar=false, fused_gdn_ch=false,
// n_rs_seq=0) routes through `build_delta_net_chunking` for
// n_seq_tokens > 1 and `build_delta_net_autoregressive` for
// n_seq_tokens == 1, both pure ggml ops.

// GDN L2 normalization per head: rms_norm with eps/n scale, where
// n is the head_k_dim. Equivalent to x / sqrt(sum(x^2) + eps/n).
// Static inline: same definition as llama.cpp/src/models/models.h:14.
static inline ggml_tensor * build_gdn_l2_norm_qwen35(
    ggml_context * ctx, ggml_tensor * x, float eps) {
    const int64_t n = x->ne[0];
    return ggml_scale(ctx, ggml_rms_norm(ctx, x, eps / (float)n),
                      1.0f / std::sqrt((float)n));
}

// Conv state row view: shape (n_embd_r, n_seqs) starting at row
// `head` of the per-layer recurrent memory buffer. For the
// sharded engine's prefill path we bypass the llama-graph
// build_rs/s_copy machinery and just take a direct view at the
// kv_head row -- same data, simpler code. The corresponding
// `ggml_cpy` writes back land in the same buffer (offset = head *
// row_count * sizeof(F32)).
static inline ggml_tensor * conv_states_row_view(
    ggml_context * ctx, ggml_tensor * conv_states_all,
    int64_t row_count, int64_t head, int64_t n_seqs) {
    return ggml_view_2d(ctx, conv_states_all,
                        row_count, n_seqs,
                        conv_states_all->nb[1],
                        head * row_count * sizeof(float));
}

static inline ggml_tensor * recurrent_state_view(
    ggml_context * ctx, ggml_tensor * ssm_states_all,
    int64_t head, int64_t n_seqs) {
    const int64_t n_embd_s = ssm_states_all->ne[0];
    return ggml_view_2d(ctx, ssm_states_all,
                        n_embd_s, n_seqs,
                        ssm_states_all->nb[1],
                        head * n_embd_s * sizeof(float));
}

ggml_tensor * build_block_graph_into_ssm_qwen35(
    const llama_model *               model,
    ggml_context *                    ctx,
    ggml_cgraph *                     gf,
    int                               il,
    int                               n_tokens,
    ggml_tensor *                     inpL_source,
    ggml_tensor *                     token_ids,
    ggml_tensor *                     /*positions*/,
    ggml_tensor *                     /*kq_mask*/,
    ggml_tensor *                     /*k_idxs*/,
    ggml_tensor *                     /*v_idxs*/,
    llama_memory_recurrent_context *  recr_ctx
) {
    const auto & layer = model->layers[il];

    GGML_ASSERT(ggml_is_contiguous(layer.ssm_conv1d));

    // hparams for this layer. qwen3.5 9B: n_embd=4096, ssm_d_inner=2048,
    // ssm_n_group=16, ssm_d_state=128, ssm_dt_rank=32, ssm_d_conv=4.
    const int64_t n_embd        = (int64_t)model->hparams.n_embd;
    const int64_t d_inner       = (int64_t)model->hparams.ssm_d_inner;
    const int64_t head_k_dim    = (int64_t)model->hparams.ssm_d_state;
    const int64_t num_k_heads   = (int64_t)model->hparams.ssm_n_group;
    const int64_t num_v_heads   = (int64_t)model->hparams.ssm_dt_rank;
    const int64_t head_v_dim    = d_inner / num_v_heads;
    const float   eps           = model->hparams.f_norm_rms_eps;

    // Phase 12 sharded engine runs prefill in a single sequence (n_seqs=1)
    // with n_seq_tokens = n_tokens (no ubatch splitting). For Phase 12
    // that's the only path we support; multi-stream is a future phase.
    const int64_t n_seqs       = 1;
    const int64_t n_seq_tokens = n_tokens;

    // Mark SSM weights as inputs (data is in the model mmap).
    ggml_set_input(layer.attn_norm);
    ggml_set_input(layer.attn_post_norm);
    ggml_set_input(layer.wqkv);
    ggml_set_input(layer.wqkv_gate);
    ggml_set_input(layer.ssm_beta);
    ggml_set_input(layer.ssm_alpha);
    ggml_set_input(layer.ssm_dt);
    ggml_set_input(layer.ssm_a);
    ggml_set_input(layer.ssm_conv1d);
    ggml_set_input(layer.ssm_norm);
    ggml_set_input(layer.ssm_out);
    ggml_set_input(layer.ffn_gate);
    ggml_set_input(layer.ffn_up);
    ggml_set_input(layer.ffn_down);
    if (layer.wqkv_s)      ggml_set_input(layer.wqkv_s);
    if (layer.wqkv_gate_s) ggml_set_input(layer.wqkv_gate_s);
    if (layer.ssm_beta_s)  ggml_set_input(layer.ssm_beta_s);
    if (layer.ssm_alpha_s) ggml_set_input(layer.ssm_alpha_s);
    if (layer.ssm_out_s)   ggml_set_input(layer.ssm_out_s);

    // Embedding lookup on the very first block of the whole prefill.
    ggml_tensor * inpL;
    if (inpL_source == nullptr) {
        ggml_set_input(token_ids);
        inpL = ggml_get_rows(ctx, model->tok_embd, token_ids);
    } else {
        ggml_set_input(inpL_source);
        inpL = inpL_source;
    }

    // Outer block flow: attn_norm at head, residual, attn_post_norm,
    // FFN, second residual. The GDN inner body replaces cur with
    // a new tensor; the residual adds it back to the saved inpSA.
    ggml_tensor * inpSA = inpL;

    // attn_norm (RMSNorm with layer.attn_norm weight).
    ggml_tensor * cur = ggml_rms_norm(ctx, inpL, eps);
    cur = ggml_mul(ctx, cur, layer.attn_norm);

    // Bisection probe tags (mirror of the attn builder). The eval
    // callback in multi_block.cpp captures the last-token column of
    // each tag and writes a per-slot binary dump. Comparison against
    // the reference (via gizmo_ref_per_layer_dump + diff_per_layer)
    // localizes any divergence to a specific il. Tags are emitted on
    // the same points the attn builder uses, so the diff_per_layer
    // toolchain is shared across both block types.
    static const int s_probe_attn = []() {
        const char * e = std::getenv("GIZMO_PROBE_PREFIXES");
        return (e && std::strstr(e, "shard_attn_residual-")) ? 1 : 0;
    }();
    static const int s_probe_post_norm = []() {
        const char * e = std::getenv("GIZMO_PROBE_PREFIXES");
        return (e && std::strstr(e, "shard_attn_post_norm-")) ? 1 : 0;
    }();
    static const int s_probe_ffn_out = []() {
        const char * e = std::getenv("GIZMO_PROBE_PREFIXES");
        return (e && std::strstr(e, "shard_ffn_out-")) ? 1 : 0;
    }();

    // ---- begin GDN inner body (port of build_layer_attn_linear) ----

    // qkv_mixed: wqkv input projection. Shape after reshape_3d:
    //   [qkv_dim, n_seq_tokens, n_seqs] where qkv_dim =
    //   head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads.
    // Mirrors qwen35.cpp's build_lora_mm: mul_mat then * wqkv_s (F32).
    // For Q4_K_M weights wqkv_s is a scalar {1} F32 carrying the
    // global dequant scale; multiplying it back restores the
    // reference output magnitudes.
    ggml_tensor * qkv_mixed = ggml_mul_mat(ctx, layer.wqkv, cur);
    if (layer.wqkv_s) qkv_mixed = ggml_mul(ctx, qkv_mixed, layer.wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx, qkv_mixed,
                                head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads,
                                n_seq_tokens, n_seqs);

    // z (gate): wqkv_gate input projection. Shape [head_v_dim * num_v_heads,
    // n_seq_tokens, n_seqs]. Same scale treatment as wqkv.
    ggml_tensor * z = ggml_mul_mat(ctx, layer.wqkv_gate, cur);
    if (layer.wqkv_gate_s) z = ggml_mul(ctx, z, layer.wqkv_gate_s);

    // beta = sigmoid(mlp(ssm_beta)(cur)) shape [1, num_v_heads, n_seq_tokens, n_seqs]
    ggml_tensor * beta = ggml_mul_mat(ctx, layer.ssm_beta, cur);
    if (layer.ssm_beta_s) beta = ggml_mul(ctx, beta, layer.ssm_beta_s);
    beta = ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx, beta);

    // alpha = softplus(mlp(ssm_alpha)(cur) + ssm_dt) * ssm_a,
    // shape [num_v_heads, n_seq_tokens, n_seqs] then reshaped to
    // [1, num_v_heads, n_seq_tokens, n_seqs] for the gate. ssm_dt and
    // ssm_a are rank-1 [num_v_heads] F32 tensors; ggml broadcasts them
    // over the trailing dim of alpha on add / mul (same pattern as
    // qwen35.cpp:369,373). Same scale treatment as wqkv.
    ggml_tensor * alpha = ggml_mul_mat(ctx, layer.ssm_alpha, cur);
    if (layer.ssm_alpha_s) alpha = ggml_mul(ctx, alpha, layer.ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx, alpha, num_v_heads, n_seq_tokens, n_seqs);
    alpha = ggml_add(ctx, alpha, layer.ssm_dt);
    alpha = ggml_softplus(ctx, alpha);
    alpha = ggml_mul(ctx, alpha, layer.ssm_a);
    ggml_tensor * gate = ggml_reshape_4d(ctx, alpha, 1, num_v_heads, n_seq_tokens, n_seqs);

    // Per-layer recurrent memory tensors. F32 [n_embd_r, mem_size] and
    // [n_embd_s, mem_size] for conv state and recurrent state
    // respectively. View into the row at `kv_head`.
    ggml_tensor * conv_states_all = recr_ctx->get_r_l(il);
    ggml_tensor * ssm_states_all  = recr_ctx->get_s_l(il);
    // Mark as inputs so the scheduler sees them as graph sources;
    // they live in the recurrent memory's ggml_context (not `ctx`),
    // and views of them are read at conv/state build time.
    ggml_set_input(conv_states_all);
    ggml_set_input(ssm_states_all);
    const int64_t kv_head         = (int64_t)recr_ctx->get_head();
    const int64_t mem_size        = (int64_t)recr_ctx->get_size();
    (void) mem_size;  // reserved for n_rs_seq > 0 future path
    const int64_t n_embd_r        = (int64_t)model->hparams.n_embd_r();
    const int64_t n_embd_s        = (int64_t)model->hparams.n_embd_s();

    ggml_tensor * conv_kernel      = layer.ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * num_k_heads * head_k_dim;

    // conv_input = concat([conv_kernel_size-1 last rows of conv state],
    //                      qkv_mixed_transposed).
    // conv_states row view, shape (n_embd_r, n_seqs), reshape to
    // (kernel_size-1, channels, n_seqs).
    ggml_tensor * conv_states_row = conv_states_row_view(
        ctx, conv_states_all, n_embd_r, kv_head, n_seqs);
    ggml_tensor * conv_states =
        ggml_reshape_3d(ctx, conv_states_row,
                        conv_kernel_size - 1, conv_channels, n_seqs);

    ggml_tensor * qkv_mixed_transposed = ggml_transpose(ctx, qkv_mixed);
    ggml_tensor * conv_input = ggml_concat(ctx, conv_states, qkv_mixed_transposed, 0);

    // Write back the trailing (kernel_size-1) channels into the
    // recurrent memory row at kv_head.
    {
        const int64_t s_idx    = conv_input->ne[0] - conv_states->ne[0];
        const int64_t row_size = (conv_kernel_size - 1) * conv_channels;
        ggml_tensor * conv_state_last =
            ggml_view_3d(ctx, conv_input,
                         conv_kernel_size - 1, conv_channels, n_seqs,
                         conv_input->nb[1], conv_input->nb[2],
                         (size_t)s_idx * ggml_element_size(conv_input));
        ggml_tensor * conv_state_update =
            ggml_view_2d(ctx, conv_states_all,
                         row_size, n_seqs,
                         conv_states_all->nb[1],
                         (size_t)(kv_head * row_size) * ggml_element_size(conv_states_all));
        ggml_build_forward_expand(gf,
            ggml_cpy(ctx, conv_state_last, conv_state_update));
    }

    // Initial state view: shape (n_embd_s, n_seqs) at row kv_head.
    ggml_tensor * state_view = recurrent_state_view(
        ctx, ssm_states_all, kv_head, n_seqs);
    ggml_tensor * state = ggml_reshape_4d(ctx, state_view,
                                          head_v_dim, head_v_dim, num_v_heads, n_seqs);

    // Run the conv. ggml_ssm_conv expects a contiguous kernel
    // (already asserted at the top of this function).
    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx, conv_input, conv_kernel);
    ggml_tensor * conv_silu = ggml_silu(ctx, conv_output_proper);

    // Slice Q, K, V from conv_silu. Layout in conv_silu:
    //   [head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads,
    //    n_seq_tokens, n_seqs]
    const int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    const size_t  nb1_qkv  = qkv_dim * ggml_element_size(conv_silu);

    ggml_tensor * q_conv = ggml_view_4d(ctx, conv_silu,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        (size_t)head_k_dim * ggml_element_size(conv_silu),
        nb1_qkv,
        (size_t)n_seq_tokens * nb1_qkv,
        0);

    ggml_tensor * k_conv = ggml_view_4d(ctx, conv_silu,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        (size_t)head_k_dim * ggml_element_size(conv_silu),
        nb1_qkv,
        (size_t)n_seq_tokens * nb1_qkv,
        (size_t)(head_k_dim * num_k_heads) * ggml_element_size(conv_silu));

    ggml_tensor * v_conv = ggml_view_4d(ctx, conv_silu,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        (size_t)head_v_dim * ggml_element_size(conv_silu),
        nb1_qkv,
        (size_t)n_seq_tokens * nb1_qkv,
        (size_t)(2 * head_k_dim * num_k_heads) * ggml_element_size(conv_silu));

    q_conv = build_gdn_l2_norm_qwen35(ctx, q_conv, eps);
    k_conv = build_gdn_l2_norm_qwen35(ctx, k_conv, eps);

    // GQA: q/k keep H_k = num_k_heads. ggml_gated_delta_net handles
    // the v_head % k_head mapping internally (iv1 % neq1).
    // Fused GDN path matches reference llama.cpp when
    // cparams.fused_gdn_ch == true (and fused_gdn_ar for n_tokens=1).
    // The op handles all chunks internally; its output packs the
    // attention result [S_v, H_v, n_tokens, n_seqs] followed by the
    // final state [S_v, S_v, H_v, n_seqs] in a single result tensor.
    ggml_tensor * gdn_result = ggml_gated_delta_net(
        ctx, q_conv, k_conv, v_conv, gate, beta, state, /*K=*/1);
    ggml_tensor * output = ggml_view_4d(ctx, gdn_result,
            head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            (size_t)head_v_dim * ggml_element_size(gdn_result),
            (size_t)head_v_dim * num_v_heads * ggml_element_size(gdn_result),
            (size_t)head_v_dim * num_v_heads * n_seq_tokens * ggml_element_size(gdn_result),
            0);
    ggml_tensor * new_state = ggml_view_4d(ctx, gdn_result,
            head_v_dim, head_v_dim, num_v_heads, n_seqs,
            (size_t)head_v_dim * ggml_element_size(gdn_result),
            (size_t)head_v_dim * head_v_dim * ggml_element_size(gdn_result),
            (size_t)head_v_dim * head_v_dim * num_v_heads * ggml_element_size(gdn_result),
            (size_t)head_v_dim * num_v_heads * n_seq_tokens * n_seqs * ggml_element_size(gdn_result));

    // Write the new recurrent state back into the recurrent memory row
    // at kv_head.
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx, new_state,
            ggml_view_2d(ctx, ssm_states_all, n_embd_s, n_seqs,
                ssm_states_all->nb[1],
                (size_t)(kv_head * n_embd_s) * ggml_element_size(ssm_states_all))));

    // gated norm: rms_norm(output) * silu(z_2d) where z_2d has head
    // shape [head_v_dim, num_v_heads, n_seq_tokens, n_seqs].
    ggml_tensor * z_2d = ggml_reshape_4d(ctx, z,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor * attn_out_norm = ggml_rms_norm(ctx, output, eps);
    attn_out_norm = ggml_mul(ctx, attn_out_norm, layer.ssm_norm);
    ggml_tensor * z_silu = ggml_silu(ctx, z_2d);
    attn_out_norm = ggml_mul(ctx, attn_out_norm, z_silu);

    // Final output projection + reshape to [n_embd, n_tokens].
    ggml_tensor * final_output = ggml_reshape_3d(ctx, attn_out_norm,
        head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cur = ggml_mul_mat(ctx, layer.ssm_out, final_output);
    if (layer.ssm_out_s) cur = ggml_mul(ctx, cur, layer.ssm_out_s);
    cur = ggml_reshape_2d(ctx, cur, n_embd, n_seq_tokens * n_seqs);

    // ---- end GDN inner body ----

    // Outer residual: cur + inpSA.
    ggml_tensor * ffn_residual = ggml_add(ctx, cur, inpSA);

    if (s_probe_attn) ggml_format_name(ffn_residual, "shard_attn_residual-%d", il);

    // attn_post_norm replaces ffn_norm at the FFN-input slot.
    ggml_tensor * post_normed = ggml_rms_norm(ctx, ffn_residual, eps);
    post_normed = ggml_mul(ctx, post_normed, layer.attn_post_norm);

    // SwiGLU FFN.
    ggml_tensor * ffn_gate_out = ggml_mul_mat(ctx, layer.ffn_gate, post_normed);
    ggml_tensor * ffn_up_out   = ggml_mul_mat(ctx, layer.ffn_up,   post_normed);
    ffn_gate_out = ggml_silu(ctx, ffn_gate_out);
    ggml_tensor * glu = ggml_mul(ctx, ffn_gate_out, ffn_up_out);
    ggml_tensor * ffn_out = ggml_mul_mat(ctx, layer.ffn_down, glu);

    // Second residual: ffn_out + ffn_residual (matches qwen35.cpp:195).
    ggml_tensor * result = ggml_add(ctx, ffn_out, ffn_residual);

    if (s_probe_post_norm) ggml_format_name(post_normed, "shard_attn_post_norm-%d", il);
    if (s_probe_ffn_out)   ggml_format_name(ffn_out,       "shard_ffn_out-%d", il);

    ggml_build_forward_expand(gf, result);
    return result;
}
