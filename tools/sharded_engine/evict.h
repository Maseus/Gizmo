// Page-aligned posix_madvise(MADV_DONTNEED) helper for sharded engine.
//
// Each block's weights live in the model's mmap. After a block's
// compute, its weights are never read again in this run, so we can
// safely MADV_DONTNEED the pages and let the OS free them. The
// pages will be re-faulted from disk if accessed again (e.g., the
// next run, or a different model that maps the same file).
//
// For prefill-only use, the eviction is safe and immediate: the
// next block's compute faults its own pages back in. For
// token-by-token generation, KV cache writes happen every step,
// so blocks' K/V tensors must NOT be evicted between decode
// steps. That's a future work item; this helper only evicts
// weight tensors, not KV cache.

#pragma once

#include "llama-model.h"

#include <cstddef>

namespace sharded_evict {

// Evict every weight tensor in `layer` via MADV_DONTNEED.
// Null tensors are skipped. Returns the number of bytes evicted
// (sum of ggml_nbytes for each non-null tensor, page-aligned up).
size_t evict_block_weights(const llama_layer & layer);

// Convenience: evict the layers' weights for a contiguous range
// of block indices. Returns total bytes evicted.
size_t evict_range(const llama_model & model, int il_first, int il_last);

}  // namespace sharded_evict
