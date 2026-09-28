// Multi-block driver: runs all n_layer blocks of the model
// sequentially, threading the residual stream between them via
// a persistent F32 carrier on the scheduler's CPU buffer.

#pragma once

#include <cstdint>
#include <vector>

#include "ggml-backend.h"
#include "llama-memory.h"

struct llama_model;

struct multi_block_result_t {
    std::vector<float> final_logits;  // F32[n_vocab * n_tokens]
    std::vector<float> residual;      // F32[n_embd * n_tokens] (last block's output)
};

// Run all n_layer blocks of `model` for the given `token_ids` and
// return the final logits + residual.
//
// The driver chains K consecutive blocks into one ggml_cgraph
// (Phase 9 "row-graph amortization"), so sched_alloc_graph /
// sched_reset / sched_graph_compute are called N/K times instead
// of N times. With K=1 this is identical to the prior per-block
// path; with K=N all blocks live in one cgraph.
//
// `row_size`: number of blocks per cgraph. Clamped to [1, n_layer].
// Values larger than ~55 require ggml_new_graph_custom with a
// larger size (the driver handles this automatically via
// ggml_new_graph_custom).
//
// If `evict_weights` is true, after each row's compute the driver
// calls madvise(MADV_DONTNEED) on each block in the row whose
// weights are no longer needed.
//
// `resident_layers` controls the eviction window (per-block
// within each row): a block `il` is evicted right after its row's
// compute only when `il + resident_layers < current_il + 1`.
//
// If `mctx` is non-null, each block writes K/V into the KV cache
// via mctx->cpy_k/cpy_v and reads them back via mctx->get_k/get_v.
// k_idxs/v_idxs are built once per row (slot offsets are
// independent of il).
multi_block_result_t run_multi_block(
    const llama_model *             model,
    ggml_backend_sched_t            sched,
    const int32_t *                 token_ids,
    int                             n_tokens,
    bool                            evict_weights,
    int                             resident_layers,
    int                             row_size,
    llama_memory_context_i *        mctx,
    int                             pos_first = 0,
    bool                            verbose = false,
    bool                            progress = false,
    const char *                    stage = "prefill"
);
