// Per-block ggml_cgraph builder for any qwen3 block.
//
// Like the spike's spike_graph.h, but uses the model's native Q4_K
// weights directly (no F32 dequant). The sharded engine runs all
// n_layer blocks sequentially, threading the residual stream between
// them via a persistent F32 carrier on the scheduler's CPU buffer.
//
// Mirrors llama.cpp/src/models/qwen3.cpp:71-142 for a single `il`.
//
// Phase 9 row-graph mode: build_block_graph_into appends one block
// to an existing (ctx, gf), so K consecutive blocks can share one
// cgraph and amortize sched_alloc_graph / sched_reset overhead.

#pragma once

#include "ggml.h"

struct llama_model;
struct llama_memory_hybrid_context;
struct llama_memory_recurrent_context;

struct shard_block_t {
    ggml_context * ctx;            // owned, free with ggml_free
    ggml_cgraph  * gf;             // owned by ctx
    ggml_tensor  * out;            // final residual of this block, F32 [n_embd, n_tokens]
    ggml_tensor  * token_ids_in;   // I32 [n_tokens], only used by il=0
    ggml_tensor  * positions_in;   // I32 [n_tokens]
    ggml_tensor  * residual_in;    // F32 [n_embd, n_tokens], set_input (caller-managed)
    ggml_tensor  * kq_mask_in;     // F16 [n_tokens, n_tokens], causal mask
    ggml_tensor  * k_idxs_in;      // I64 [n_tokens], KV cache cell indices (Phase 7 only)
    ggml_tensor  * v_idxs_in;      // I64 [n_tokens] (or [n_tokens*n_embd_v_gqa]), V cell indices
};

// Append block `il`'s nodes into an existing (ctx, gf). Returns the
// block's final residual tensor (F32 [n_embd, n_tokens]) so the next
// block in the row can read from it directly without a host copy.
//
// `inpL_source`: F32 [n_embd, n_tokens] or nullptr. If null, the
// graph looks up embeddings via `ggml_get_rows(model->tok_embd,
// token_ids)` — pass `token_ids` in that case.
//
// `token_ids`: I32 [n_tokens] or nullptr. Pass nullptr when
// `inpL_source` is non-null, or when this block is not the FIRST
// block of the whole prefill.
//
// `positions`: I32 [n_tokens], required. Caller-managed (the row
// driver allocates once per row; the single-block wrapper
// allocates per call).
//
// `kq_mask`, `k_idxs`, `v_idxs`: caller-managed tensors shared
// across the row. `ggml_set_input` is idempotent so the row driver
// can mark them once.
//
// `mctx`: same semantics as build_shard_block_graph.
ggml_tensor * build_block_graph_into(
    const llama_model *             model,
    ggml_context *                  ctx,
    ggml_cgraph *                   gf,
    int                             il,
    int                             n_tokens,
    ggml_tensor *                   inpL_source,
    ggml_tensor *                   token_ids,
    ggml_tensor *                   positions,
    ggml_tensor *                   kq_mask,
    ggml_tensor *                   k_idxs,
    ggml_tensor *                   v_idxs,
    llama_memory_hybrid_context *   mctx
);

// Convenience wrapper: build a per-block ggml_cgraph for block
// `il` of `model`. Equivalent to a single-block row.
shard_block_t build_shard_block_graph(
    const llama_model *             model,
    int                             il,
    int                             n_tokens,
    ggml_tensor *                   inpL_carrier,
    ggml_tensor *                   kq_mask,
    llama_memory_hybrid_context *   mctx
);

// Phase 10/12 stub for SSM (gated delta net / recurrent) blocks.
// Returns the input residual unchanged so the row driver can still
// chain. Default behavior (Phase 12) uses the real builder; set
// GIZMO_SSM_STUB=1 to force this passthrough instead.
//
// `model` MUST be non-null. `il` is the absolute layer index.
//
// Signature matches `build_block_graph_into` minus the recurrent
// context (the stub does not touch state).
ggml_tensor * build_block_graph_ssm_stub_for_test(
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
    llama_memory_hybrid_context * mctx
);

// Phase 11: qwen3.5 full-attention block builder. The 4 differences
// from qwen3, as documented in docs/qwen35-sharded-engine.md §2:
//   1. joint QG projection (wq outputs [Q | G] interleaved per head)
//   2. MRoPE-4 via ggml_rope_multi (hparams.rope_sections[4])
//   3. attn_post_norm replaces ffn_norm at the FFN-input slot
//   4. sigmoid(gate_view) * attn_out before wo
//
// Same signature as build_block_graph_into so the dispatcher can
// route to it for full-attn layers that have attn_post_norm and not
// ffn_norm.
ggml_tensor * build_block_graph_into_attn_qwen35(
    const llama_model *             model,
    ggml_context *                  ctx,
    ggml_cgraph *                   gf,
    int                             il,
    int                             n_tokens,
    ggml_tensor *                   inpL_source,
    ggml_tensor *                   token_ids,
    ggml_tensor *                   positions,
    ggml_tensor *                   kq_mask,
    ggml_tensor *                   k_idxs,
    ggml_tensor *                   v_idxs,
    llama_memory_hybrid_context *   mctx
);

// Phase 12: qwen3.5 gated delta net (SSM / recurrent) block builder.
// Port of llama.cpp/src/models/qwen35.cpp::build_layer_attn_linear,
// plus the outer residual structure from qwen35.cpp::graph::graph
// (attn_norm at the head, post-attn residual, attn_post_norm, FFN,
// second residual).
//
// Distinct from the attn builder because the memory context it
// needs is the recurrent half of the hybrid memory
// (`llama_memory_recurrent_context`), accessed via the dispatcher
// in shard_block.cpp calling `hybrid_ctx->get_recr()`.
//
// The signature mirrors build_block_graph_into for slot uniformity;
// positions/kq_mask/k_idxs/v_idxs are unused on the SSM path.
ggml_tensor * build_block_graph_into_ssm_qwen35(
    const llama_model *               model,
    ggml_context *                    ctx,
    ggml_cgraph *                     gf,
    int                               il,
    int                               n_tokens,
    ggml_tensor *                     inpL_source,
    ggml_tensor *                     token_ids,
    ggml_tensor *                     positions,
    ggml_tensor *                     kq_mask,
    ggml_tensor *                     k_idxs,
    ggml_tensor *                     v_idxs,
    llama_memory_recurrent_context *  recr_ctx
);
