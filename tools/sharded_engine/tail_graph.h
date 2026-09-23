// Post-tail graph: rms_norm -> mul_mat(output) -> logits.
//
// Mirrors llama.cpp/src/models/qwen3.cpp:144-156 (the tail of the
// qwen3 model). Takes the final block's residual as input and
// produces F32[n_vocab] logits (or F32[n_vocab, n_tokens] for the
// full sequence if requested).

#pragma once

#include "ggml.h"

struct llama_model;

struct tail_graph_t {
    ggml_context * ctx;            // owned, free with ggml_free
    ggml_cgraph  * gf;             // owned by ctx
    ggml_tensor  * out;            // F32 [n_vocab, n_tokens] logits
    ggml_tensor  * residual_in;    // F32 [n_embd, n_tokens], set_input (caller-managed)
};

// Build the post-tail graph: rms_norm -> mul_mat(model.output).
//
// `residual_carrier` is the F32 [n_embd, n_tokens] tensor produced
// by the last block (n_layer-1). Marked as input; the caller fills
// it before compute.
//
// Output: F32 [n_vocab, n_tokens] logits (or F32 [n_vocab] for the
// last token, since output weight is [n_embd, n_vocab] the mul_mat
// yields [n_vocab, n_tokens]).
//
// The graph is built with no_alloc=true. The caller invokes
// ggml_backend_sched_alloc_graph(sched, gf) before computing.
tail_graph_t build_tail_graph(
    const llama_model *  model,
    int                  n_tokens,
    ggml_tensor *        residual_carrier
);
