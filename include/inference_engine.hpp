#ifndef GIZMO_INFERENCE_ENGINE_HPP
#define GIZMO_INFERENCE_ENGINE_HPP

#include <string>
#include <cstdint>
#include <cstddef>
#include <functional>

#include "llama.h"
#include "ggml-backend.h"

namespace gizmo {

struct InferenceConfig {
    int32_t context_size = 512;
    int32_t max_tokens = 256;
    float temperature = 0.8f;
    float top_p = 0.95f;
    int32_t top_k = 40;
    float repeat_penalty = 1.1f;
};

struct TokenResult {
    std::string text;
    int32_t token_id;
    float probability;
};

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    // Initialize with model path. layer_shard_lazy=true asks llama.cpp to
    // mark per-block tensors as TENSOR_READ_LAZY so they page-fault on
    // first decode instead of pre-loading at construction time. Pass false
    // when --no-shard is set.
    bool initialize(const std::string& model_path, bool layer_shard_lazy);

    // Legacy overload: layer_shard_lazy defaults to true.
    bool initialize(const std::string& model_path, int32_t layers_ignored);

    // After initialize, call this to enable the per-block sharded engine
    // for the prefill step. When enabled, generate() and generate_token()
    // use run_multi_block() for the prompt forward pass, then fall back
    // to llama_decode() for the per-token decode loop. This is the
    // Phase 6 wiring — the sharded engine is prefill-only today; the
    // per-token decode step is un-sharded. `resident_layers` is plumbed
    // through for forward compatibility (a future phase will implement
    // a sliding window of N resident blocks). `evict_weights` enables
    // madvise(MADV_PAGEOUT) on each block's weights after compute.
    //
    // `row_size` (Phase 9): number of consecutive blocks chained into a
    // single ggml_cgraph. K=1 preserves the byte-exact Phase 8 per-block
    // behavior (regression baseline). Larger K reduces scheduler overhead
    // (sched_alloc_graph / sched_reset / sched_graph_compute calls) by a
    // factor of K. Clamped to [1, n_layer].
    void enable_sharded_engine(
        int32_t resident_layers,
        bool    evict_weights,
        int32_t row_size = 1
    );

    // Generate text from a prompt
    std::string generate(const std::string& prompt, const InferenceConfig& config);

    // Generate text from a prompt, calling callback for each decoded token.
    // If callback returns false, generation stops early. Returns the total
    // number of generated tokens. Used by the HTTP server for SSE streaming.
    int generate_stream(
        const std::string& prompt,
        const InferenceConfig& config,
        std::function<bool(const std::string& token)> callback
    );

    // Generate a single token
    TokenResult generate_token(const std::string& context, const InferenceConfig& config);

    // Run only the sharded prefill on a tokenized prompt and
    // return. Does NOT populate the un-sharded KV cache, so no
    // decode loop can follow. Useful for benchmarks that want
    // to measure the sharded prefill in isolation (without the
    // current Phase 6 double-prefill cost masking the wall time).
    // Returns 0 on success, -1 on failure.
    int prefill_only(const std::string& prompt);

    // As prefill_only(string), but takes already-tokenized
    // input. The caller owns the tokens and the engine copies
    // the values it needs internally.
    int prefill_only_tokens(const llama_token* tokens, int32_t n_tokens);

    // Reset the engine for a fresh forward pass. Clears the
    // un-sharded KV cache and resets the sharded scheduler's
    // per-block scratch. Use between runs that share the same
    // model load (e.g. benchmarks). The sharded engine's
    // carrier is re-created on the next generate() call.
    void reset_for_next_run();

    // Check if engine is ready
    bool is_ready() const;

    // Get model info
    std::string get_model_info() const;

    // Raw accessors for the sharded wrapper / external introspection.
    llama_model*   raw_model()   const { return llama_model_;   }
    llama_context* raw_context() const { return llama_context_; }

    // Model metadata (returns 0 if not initialized).
    int32_t n_layer()        const;
    int32_t embedding_dim()  const;
    int32_t vocab_size()     const;

    // Sharding status (for diagnostics / --measure-ram).
    bool is_sharded() const { return use_sharded_engine_; }

    // VmRSS captured at the end of the most recent sharded prefill
    // (right after run_multi_block returns, before the un-sharded
    // prefill re-faults model pages). 0 if no sharded prefill has
    // run yet, or if sharding is disabled. Used by the bench to
    // report the steady-state cost of the sharded path without the
    // KV-cache-warming prefill masking it.
    size_t last_sharded_prefill_rss() const {
        return last_sharded_prefill_rss_;
    }

    // VmHWM captured at the same point. Higher than RSS if pages
    // were touched during compute and then evicted. 0 if no
    // sharded prefill has run.
    size_t last_sharded_prefill_hwm() const {
        return last_sharded_prefill_hwm_;
    }

private:
    bool initialized_;
    std::string model_path_;
    int32_t layers_loaded_;
    llama_context* llama_context_;
    llama_model*   llama_model_;

    // Sharded-engine state. Built in enable_sharded_engine(), freed
    // in the destructor. Both pointers may be null when the un-sharded
    // path is selected (--no-shard).
    bool                 use_sharded_engine_ = false;
    bool                 sharded_evict_weights_ = false;
    int32_t              sharded_resident_layers_ = 1;  // unused in Phase 6
    int32_t              sharded_row_size_ = 1;         // Phase 9 row-graph K
    ggml_backend_t       sharded_cpu_backend_ = nullptr;
    ggml_backend_sched_t sharded_sched_       = nullptr;
    int32_t              sharded_n_threads_   = 4;

    // VmRSS / VmHWM captured at the end of the most recent sharded
    // prefill. Set by do_prefill() after run_multi_block returns.
    size_t last_sharded_prefill_rss_ = 0;
    size_t last_sharded_prefill_hwm_ = 0;
};

} // namespace gizmo

#endif // GIZMO_INFERENCE_ENGINE_HPP
