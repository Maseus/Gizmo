// Pure C++ reference implementation of one qwen3 block (block 0).
//
// This is the oracle for the per-block forward pass spike. It does
// the same math as the ggml_cgraph that spike_graph.cpp builds, but
// in pure C++ (no ggml ops). The two should produce the same F32
// output within Q4_K quantization noise (~5% relative).
//
// Block-0 weights are dequantized to F32 once on first call and
// cached for reuse. Memory cost ~640 MB for qwen3:4b block 0.

#pragma once

#include <cstdint>
#include <vector>
#include "ggml.h"
#include "ggml-backend.h"

struct llama_model;

// F32-dequantized weight tensors for block 0, plus a couple of
// scalar views. Caller does NOT own the underlying storage; the
// reference's static cache does. Use compute_block_0_reference()
// first to trigger dequant, then read these pointers.
struct ref_block_weights_t {
    const float * attn_norm;
    const float * attn_q_norm;
    const float * attn_k_norm;
    const float * ffn_norm;
    const float * wq;
    const float * wk;
    const float * wv;
    const float * wo;
    const float * ffn_gate;
    const float * ffn_up;
    const float * ffn_down;
    int n_embd;
    int n_embd_head;
    int n_head;
    int n_head_kv;
    int n_ff;
    int n_rot;
    float freq_base;
    float eps;
    int rope_type;
    int n_ctx_train;
    int n_vocab;
};

// Force dequantization of block-0 weights so callers can read them.
// Idempotent. Returns the (now-populated) weight table.
ref_block_weights_t ensure_ref_weights_loaded(
    const llama_model * model,
    ggml_backend_sched_t sched
);

// Dequantize the embedding rows for the given token ids. Output
// is F32 row-major, shape [n_embd, n_tokens]. Caller owns the
// returned vector. Uses raw byte read + standard Q*_K dequant
// (no ggml graph machinery).
std::vector<float> ref_dequant_embedding_rows(
    const llama_model * model,
    const int32_t * token_ids,
    int n_tokens
);

// Computes block 0's residual output for `n_tokens` input tokens.
// `token_ids` is the array of token ids (length n_tokens).
// `out` is the F32 output buffer, row-major, shape [n_embd, n_tokens].
// `pos_offset` is the position of token_ids[0] in the sequence.
void compute_block_0_reference(
    const llama_model *       model,
    ggml_backend_sched_t      sched,
    const int32_t *           token_ids,
    int                       n_tokens,
    float *                   out,
    int                       pos_offset = 0
);
