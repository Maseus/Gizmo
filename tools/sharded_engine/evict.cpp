// Page-aligned weight-eviction helper. See evict.h.
//
// We use Linux's `madvise(MADV_DONTNEED)` (not the POSIX wrapper)
// for file-backed mmap regions.  On Linux, `madvise(MADV_DONTNEED)`
// immediately invalidates the PTEs for a file-backed mapping and the
// pages are dropped from the process RSS; the next read faults them
// back in from the GGUF file.  This is exactly what the per-block
// sharded engine needs.
//
// Note: `posix_madvise(POSIX_MADV_DONTNEED)` is *not* equivalent on
// Linux; glibc implements it as a lazy hint similar to `MADV_FREE`, so
// pages stay resident until memory pressure.  We therefore use the raw
// `madvise` syscall directly.

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

    if (madvise((void *)aligned_addr, aligned_len, MADV_DONTNEED) != 0) {
        std::fprintf(stderr, "[evict] WARN: madvise(MADV_DONTNEED) on %s (%zu bytes) failed: errno=%d\n",
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
    // qwen3.5 SSM (gated delta net) weights. Null on non-SSM layers
    // (qwen3 has none of these); evict_tensor short-circuits.
    total += evict_tensor(layer.wqkv,            "wqkv");
    total += evict_tensor(layer.wqkv_s,          "wqkv_s");
    total += evict_tensor(layer.wqkv_gate,       "wqkv_gate");
    total += evict_tensor(layer.wqkv_gate_s,     "wqkv_gate_s");
    total += evict_tensor(layer.attn_post_norm,  "attn_post_norm");
    total += evict_tensor(layer.ssm_conv1d,      "ssm_conv1d");
    total += evict_tensor(layer.ssm_dt,          "ssm_dt");
    total += evict_tensor(layer.ssm_a,           "ssm_a");
    total += evict_tensor(layer.ssm_beta,        "ssm_beta");
    total += evict_tensor(layer.ssm_beta_s,      "ssm_beta_s");
    total += evict_tensor(layer.ssm_alpha,       "ssm_alpha");
    total += evict_tensor(layer.ssm_alpha_s,     "ssm_alpha_s");
    total += evict_tensor(layer.ssm_norm,        "ssm_norm");
    total += evict_tensor(layer.ssm_out,         "ssm_out");
    total += evict_tensor(layer.ssm_out_s,       "ssm_out_s");
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
