// Post-tail graph: rms_norm -> mul_mat(output) -> logits.
// Mirrors llama.cpp/src/models/qwen3.cpp:144-156 and
// llama.cpp/src/models/qwen35.cpp:206-222.

#include "tail_graph.h"
#include "llama-model.h"
#include "llama-hparams.h"
#include "ggml.h"

tail_graph_t build_tail_graph(
    const llama_model *      model,
    int                      n_tokens,
    ggml_tensor *            residual_carrier
) {
    tail_graph_t out{};

    const float eps = model->hparams.f_norm_rms_eps;
    (void)n_tokens;

    ggml_init_params iparams = {
        /* .mem_size   = */ 64 * ggml_tensor_overhead() + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(iparams);
    if (!ctx) {
        return out;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);

    // residual_in is the F32 [n_embd, n_tokens] from the last block
    ggml_set_input(residual_carrier);
    out.residual_in = residual_carrier;

    // mark weights as inputs (data in model mmap)
    ggml_set_input(model->output_norm);
    ggml_set_input(model->output);
    if (model->output_s) ggml_set_input(model->output_s);

    // output norm
    ggml_tensor * cur = ggml_rms_norm(ctx, residual_carrier, eps);
    cur = ggml_mul(ctx, cur, model->output_norm);

    // lm_head: mul_mat(output, cur) -> [n_vocab, n_tokens]
    cur = ggml_mul_mat(ctx, model->output, cur);
    // Native graphs use build_lora_mm, which applies model.output_s
    // when present (Q4_K_M tied output weights may carry a scalar
    // dequant scale). Apply the same scale here so the sharded tail
    // matches llama_decode logits.
    if (model->output_s) {
        cur = ggml_mul(ctx, cur, model->output_s);
    }

    ggml_build_forward_expand(gf, cur);

    out.ctx = ctx;
    out.gf  = gf;
    out.out = cur;
    return out;
}
