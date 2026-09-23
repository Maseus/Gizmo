// Page-aligned weight-eviction helper. See evict.h.
//
// We use Linux's `MADV_PAGEOUT` (madvise(2)) rather than the
// POSIX `posix_madvise(MADV_DONTNEED)` because on Linux, the
// POSIX version is implemented as a lazy hint equivalent to
// `MADV_FREE`: it only marks pages as freeable, and the kernel
// doesn't actually reclaim them until memory pressure occurs.
// `MADV_PAGEOUT` (Linux 5.4+) immediately pages out the range —
// for our file-backed mmap of a read-only GGUF, that means the
// clean pages are dropped straight out of the page cache and the
// process's resident set drops to match.

#include "evict.h"
#include "ggml.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdint>

namespace {

// Page size is constant for the process lifetime.
size_t page_size() {
    static const size_t s = (size_t)sysconf(_SC_PAGESIZE);
    return s > 0 ? s : 4096;
}

// Evict a single tensor's storage. Returns the number of bytes
// evicted (page-aligned up from the tensor's byte size), or 0 if
// the tensor is null or has no data.
size_t evict_tensor(const ggml_tensor * t, const char * tag) {
    if (!t || !t->data) {
        return 0;
    }
    const size_t nbytes = ggml_nbytes(t);
    if (nbytes == 0) {
        return 0;
    }
    const size_t ps = page_size();
    const uintptr_t addr = (uintptr_t)t->data;
    const uintptr_t aligned_addr = addr & ~(uintptr_t)(ps - 1);
    // The eviction covers [aligned_addr, aligned_addr + aligned_len).
    // aligned_len must reach past the end of the tensor's bytes.
    const size_t slack = (size_t)(addr - aligned_addr);
    const size_t aligned_len = (slack + nbytes + ps - 1) & ~(ps - 1);

    if (madvise((void *)aligned_addr, aligned_len, MADV_PAGEOUT) != 0) {
        std::fprintf(stderr, "[evict] WARN: madvise(MADV_PAGEOUT) on %s (%zu bytes) failed: errno=%d\n",
                     tag, nbytes, errno);
        return 0;
    }
    return aligned_len;
}

}  // namespace

namespace sharded_evict {

size_t evict_block_weights(const llama_layer & layer) {
    size_t total = 0;
    total += evict_tensor(layer.attn_norm,       "attn_norm");
    total += evict_tensor(layer.attn_norm_b,     "attn_norm_b");
    total += evict_tensor(layer.attn_q_norm,     "attn_q_norm");
    total += evict_tensor(layer.attn_q_norm_b,   "attn_q_norm_b");
    total += evict_tensor(layer.attn_k_norm,     "attn_k_norm");
    total += evict_tensor(layer.attn_k_norm_b,   "attn_k_norm_b");
    total += evict_tensor(layer.ffn_norm,        "ffn_norm");
    total += evict_tensor(layer.ffn_norm_b,      "ffn_norm_b");
    total += evict_tensor(layer.wq,              "wq");
    total += evict_tensor(layer.wk,              "wk");
    total += evict_tensor(layer.wv,              "wv");
    total += evict_tensor(layer.wo,              "wo");
    total += evict_tensor(layer.wo_b,            "wo_b");
    total += evict_tensor(layer.ffn_gate,        "ffn_gate");
    total += evict_tensor(layer.ffn_up,          "ffn_up");
    total += evict_tensor(layer.ffn_down,        "ffn_down");
    total += evict_tensor(layer.ffn_gate_b,      "ffn_gate_b");
    total += evict_tensor(layer.ffn_up_b,        "ffn_up_b");
    total += evict_tensor(layer.ffn_down_b,      "ffn_down_b");
    return total;
}

size_t evict_range(const llama_model & model, int il_first, int il_last) {
    if (il_first < 0) il_first = 0;
    if (il_last >= (int)model.layers.size()) il_last = (int)model.layers.size() - 1;
    size_t total = 0;
    for (int il = il_first; il <= il_last; ++il) {
        total += evict_block_weights(model.layers[il]);
    }
    return total;
}

}  // namespace sharded_evict
