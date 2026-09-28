#ifndef GIZMO_INFERENCE_ENGINE_HPP
#define GIZMO_INFERENCE_ENGINE_HPP

#include <string>
#include <vector>
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
    int32_t seed = -1;  // -1 = random seed from llama_sampler_init_dist
    std::vector<std::string> stop;  // optional stop strings/sequences
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

    // Initialize with model path. layer_shard_lazy=true disables mmap
    // prefetch so weight pages fault on first use instead of during load,
    // keeping the initial RSS low for sharded inference. Pass false when
    // --no-shard is set.
    bool initialize(const std::string& model_path, bool layer_shard_lazy);

    // Legacy overload: layer_shard_lazy defaults to true.
    bool initialize(const std::string& model_path, int32_t layers_ignored);

    // After initialize, call this to enable the per-block sharded engine
    // for both prefill and decode. When enabled, generate() uses
    // run_multi_block() for the prompt forward pass and for each generated
    // token, falling back to llama_decode() only if the sharded path fails.
    // `resident_layers` is plumbed through for forward compatibility (a future
    // phase will implement a sliding window of N resident blocks).
    // `evict_weights` enables madvise(MADV_DONTNEED) on each block's weights
    // after compute.
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

    // Control sharded-engine diagnostic output. When false (default) the
    // per-block runner and inference engine keep stdout/stderr quiet except
    // for user-facing output and errors. Set true to restore the Phase 8
    // per-row progress and argmax logging.
    void set_verbose(bool verbose) { verbose_ = verbose; }

    // Show a concise in-place progress indicator during sharded prefill
    // and decode. Independent of verbose_; useful for large models where
    // the user wants feedback without the full per-row diagnostic dump.
    void set_progress(bool progress) { progress_ = progress; }

    // Set the number of CPU threads for both the native llama.cpp context
    // (llama_set_n_threads) and the sharded CPU backend. Must be called
    // before enable_sharded_engine() to take effect on the sharded path;
    // called at runtime it updates the native context threads.
    void set_threads(int32_t threads);

    // Generate text from a prompt. If `quiet` is true, token output
    // and progress messages are suppressed (used by sweep/bench).
    std::string generate(const std::string& prompt, const InferenceConfig& config, bool quiet = false);

    // Generate text from a prompt, calling `callback` for every emitted
    // token. The callback receives the detokenized text piece and the raw
    // token id. Used by the HTTP server for streaming (SSE) responses.
    // Returns false if generation failed before the first token.
    // If `should_cancel` returns true the decode loop is aborted early; this
    // is used by the HTTP server to enforce per-request timeouts.
    bool generate_stream(
        const std::string& prompt,
        const InferenceConfig& config,
        std::function<void(const std::string& token_text, int32_t token_id)> callback,
        std::function<bool()> should_cancel = nullptr
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

    // Run a prefill (sharded if enabled, otherwise un-sharded)
    // and return the last-token logits in `out_logits`. Does not
    // run sampling or decode. Clears the KV cache first so the
    // call is independent of prior state. Used by the validate
    // command to compare sharded vs un-sharded outputs.
    //
    // Returns 0 on success, -1 on failure.
    int validate_prefill(
        const std::string& prompt,
        std::vector<float>& out_logits
    );

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

    // Concise model identifier suitable for OpenAI-compatible responses.
    std::string get_model_id() const;

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
    bool                 verbose_ = false;
    bool                 progress_ = false;
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
