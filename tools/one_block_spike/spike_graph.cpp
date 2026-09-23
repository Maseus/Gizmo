// Per-block ggml_cgraph for qwen3 block 0.
//
// Builds a small ggml_cgraph that runs exactly one qwen3 transformer
// block (block 0) for n_tokens input tokens. The graph consumes
// pre-dequantized F32 weight tensors (no Q4_K/Q6_K dequant happens
// at compute time) and produces the F32 residual stream output.
//
// Mirrors llama.cpp/src/models/qwen3.cpp:71-142 (block 0 only) but
// uses a single graph with no KV cache, so the math is simpler:
//   embedding -> RMSNorm -> Q/K/V matmul -> per-head Q/K RMSNorm ->
//   RoPE -> flash_attn -> WO -> residual -> RMSNorm -> SwiGLU FFN ->
//   residual.
//
// Compared to the reference in reference.cpp, the math is bit-
// identical (both use the same dequantized F32 weights and the
// same F32 matmul arithmetic, including the half-rotation Neox
// RoPE convention that ggml uses).

#include "spike_graph.h"
#include "llama-model.h"
#include "llama-hparams.h"
#include "ggml.h"

spike_block_t build_block_0_graph(
    const llama_model *          model,
    int                          n_tokens,
    int                          pos_offset,
    const spike_f32_weights_t *  f32w
) {
    spike_block_t out{};

    const auto & layer = model->layers[0];
    const int n_head       = (int)model->hparams.n_head();
    const int n_head_kv    = (int)model->hparams.n_head_kv();
    const int n_embd_head  = (int)model->hparams.n_embd_head_v();
    const int n_rot        = (int)model->hparams.n_rot();
    const int n_ctx_orig   = (int)model->hparams.n_ctx_train;
    const int rope_type    = (int)model->hparams.rope_type;
    const float freq_base  = model->hparams.rope_freq_base_train;
    const float eps        = model->hparams.f_norm_rms_eps;
    (void)pos_offset;
    (void)layer;

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

    // Input leaves.
    ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(token_ids);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(positions);

    // Mark all consumed weight tensors as inputs so the scheduler
    // does not try to allocate them (their data lives in the
    // caller's storage).
    ggml_set_input(f32w->attn_norm);
    ggml_set_input(f32w->attn_q_norm);
    ggml_set_input(f32w->attn_k_norm);
    ggml_set_input(f32w->ffn_norm);
    ggml_set_input(f32w->wq);
    ggml_set_input(f32w->wk);
    ggml_set_input(f32w->wv);
    ggml_set_input(f32w->wo);
    ggml_set_input(f32w->ffn_gate);
    ggml_set_input(f32w->ffn_up);
    ggml_set_input(f32w->ffn_down);

    // 1. Embedding lookup.
    ggml_tensor * inpL = ggml_get_rows(ctx, model->tok_embd, token_ids);

    // 2. Pre-attention norm.
    ggml_tensor * x_norm = ggml_rms_norm(ctx, inpL, eps);
    x_norm = ggml_mul(ctx, x_norm, f32w->attn_norm);

    // 3. Q/K/V projections.
    ggml_tensor * Qcur = ggml_mul_mat(ctx, f32w->wq, x_norm);
    Qcur = ggml_reshape_3d(ctx, Qcur, n_embd_head, n_head, n_tokens);

    ggml_tensor * Kcur = ggml_mul_mat(ctx, f32w->wk, x_norm);
    Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_tokens);

    ggml_tensor * Vcur = ggml_mul_mat(ctx, f32w->wv, x_norm);
    Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_tokens);

    // 4. Per-head Q/K RMSNorm.
    Qcur = ggml_rms_norm(ctx, Qcur, eps);
    Qcur = ggml_mul(ctx, Qcur, f32w->attn_q_norm);

    Kcur = ggml_rms_norm(ctx, Kcur, eps);
    Kcur = ggml_mul(ctx, Kcur, f32w->attn_k_norm);

    // 5. RoPE.
    Qcur = ggml_rope_ext(
            ctx, Qcur, positions, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base,
            1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    Kcur = ggml_rope_ext(
            ctx, Kcur, positions, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base,
            1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // 6. Flash attention. Permute Q/K/V to flash_attn's expected
    // layout [n_embd, n_batch, n_head, 1].
    Qcur = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));
    Kcur = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));
    Vcur = ggml_cont(ctx, ggml_permute(ctx, Vcur, 0, 2, 1, 3));

    const float kq_scale = 1.0f / std::sqrt((float)n_embd_head);
    ggml_tensor * attn_out = ggml_flash_attn_ext(
            ctx, Qcur, Kcur, Vcur, /*mask=*/nullptr,
            kq_scale, /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);

    // F32 accumulation for the attention output.
    ggml_prec_set_acc(attn_out, GGML_PREC_F32);

    // Reshape to 2D for WO mul_mat.
    attn_out = ggml_reshape_2d(ctx, attn_out, n_embd_head * n_head, n_tokens);

    // 7. WO projection.
    ggml_tensor * wo_out = ggml_mul_mat(ctx, f32w->wo, attn_out);

    // 8. First residual.
    ggml_tensor * ffn_inp = ggml_add(ctx, wo_out, inpL);

    // 9. Pre-FFN norm.
    ggml_tensor * ffn_normed = ggml_rms_norm(ctx, ffn_inp, eps);
    ffn_normed = ggml_mul(ctx, ffn_normed, f32w->ffn_norm);

    // 10. SwiGLU FFN.
    ggml_tensor * gate = ggml_mul_mat(ctx, f32w->ffn_gate, ffn_normed);
    ggml_tensor * up   = ggml_mul_mat(ctx, f32w->ffn_up,   ffn_normed);
    gate = ggml_silu(ctx, gate);
    ggml_tensor * glu  = ggml_mul(ctx, gate, up);
    ggml_tensor * ffn_out = ggml_mul_mat(ctx, f32w->ffn_down, glu);

    // 11. Final residual.
    ggml_tensor * result = ggml_add(ctx, ffn_out, ffn_inp);

    ggml_build_forward_expand(gf, result);

    out.ctx           = ctx;
    out.gf            = gf;
    out.out           = result;
    out.token_ids_in  = token_ids;
    out.positions_in  = positions;
    return out;
}
