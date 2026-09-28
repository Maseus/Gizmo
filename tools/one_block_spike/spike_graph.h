// Per-block ggml_cgraph builder for qwen3 block 0.
//
// Builds a small ggml_cgraph that runs exactly one qwen3 transformer
// block (block 0) for `n_tokens` input tokens. The graph uses
// F32 weight tensors (pre-dequantized by the caller) and produces
// the F32 residual stream output.
//
// Used as the spike implementation that gets compared against the
// pure-C++ reference in reference.cpp. Both use the same F32 weight
// arrays, so the comparison is bit-exact up to summation order.

#pragma once

#include "ggml.h"

struct llama_model;

struct spike_block_t {
    ggml_context * ctx;            // owned, free with ggml_free
    ggml_cgraph  * gf;             // owned by ctx
    ggml_tensor  * out;            // final residual, F32 [n_embd, n_tokens]
    ggml_tensor  * token_ids_in;   // I32 [n_tokens]
    ggml_tensor  * positions_in;   // I32 [n_tokens]
};

// F32 weight tensors for block 0, in the same layout the reference
// uses (dequantized from the model's Q4_K/Q6_K weights). Caller
// owns the underlying storage. The graph only reads from them.
struct spike_f32_weights_t {
    ggml_tensor * attn_norm;       // F32 [n_embd]
    ggml_tensor * attn_q_norm;     // F32 [n_embd_head]
    ggml_tensor * attn_k_norm;     // F32 [n_embd_head]
    ggml_tensor * ffn_norm;        // F32 [n_embd]
    ggml_tensor * wq;              // F32 [n_embd, n_embd_head * n_head]
    ggml_tensor * wk;              // F32 [n_embd, n_embd_head * n_head_kv]
    ggml_tensor * wv;              // F32 [n_embd, n_embd_head * n_head_kv]
    ggml_tensor * wo;              // F32 [n_embd_head * n_head, n_embd]
    ggml_tensor * ffn_gate;        // F32 [n_embd, n_ff]
    ggml_tensor * ffn_up;          // F32 [n_embd, n_ff]
    ggml_tensor * ffn_down;        // F32 [n_ff, n_embd]
};

spike_block_t build_block_0_graph(
    const llama_model *          model,
    int                          n_tokens,
    int                          pos_offset,
    const spike_f32_weights_t *  f32w
);
