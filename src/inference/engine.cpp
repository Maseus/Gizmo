#include "inference_engine.hpp"
#include "proc_status.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <iostream>
#include <vector>
#include <unistd.h>

#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "multi_block.h"
#include "llama-model.h"
#include "llama-arch.h"

namespace gizmo {

namespace {

// llama_backend_init / llama_backend_free are global. Multiple InferenceEngine
// instances may exist simultaneously (e.g. validate creates baseline + sharded),
// so keep a process-wide refcount and only call free when the last engine
// is destroyed.
std::atomic<int>& backend_refcount() {
    static std::atomic<int> count{0};
    return count;
}

// Two-pass tokenization helper: returns the exact token vector for `text`,
// resizing automatically when the fixed buffer would overflow. Returns an
// empty vector and sets *out_n negative on error.
std::vector<llama_token> tokenize_text(
    const llama_vocab* vocab,
    const std::string& text,
    bool add_special,
    bool parse_special,
    int32_t* out_n = nullptr
) {
    std::vector<llama_token> tokens;
    if (out_n != nullptr) *out_n = 0;

    int32_t n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                               nullptr, 0, add_special, parse_special);
    if (n == 0) {
        if (out_n != nullptr) *out_n = 0;
        return tokens;
    }
    if (n < 0) {
        // llama_tokenize returns the negative required count when the output
        // buffer (including a zero-length nullptr buffer) is too small.
        n = -n;
    }

    tokens.resize(static_cast<size_t>(n));
    int32_t n_check = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                                    tokens.data(), n, add_special, parse_special);
    if (n_check < 0) {
        if (out_n != nullptr) *out_n = n_check;
        tokens.clear();
        return tokens;
    }
    if (out_n != nullptr) *out_n = n_check;
    return tokens;
}

// The sharded per-block engine is currently validated for the pure-attention
// Qwen3 architecture. Qwen3.5 has a hybrid recurrent/attention layout whose
// sharded path is still experimental; it must use native llama_decode until
// recurrent-state parity is established. Other architectures also fall back
// to native llama_decode.
bool is_sharded_arch_supported(const llama_model* model) {
    if (model == nullptr) {
        return false;
    }
    switch (model->arch) {
        case LLM_ARCH_QWEN3:
        case LLM_ARCH_QWEN35:
            // Per-block sharded engine is validated for Qwen3's pure-attention
            // blocks and has been extended to Qwen3.5's hybrid layout (full
            // attention + gated-delta-net SSM). Models using these arches run
            // through the per-block engine when sharding is enabled.
            return true;
        // Qwen3Next/VL and all MoE variants are not yet validated with the
        // per-block sharded engine.
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VL:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
            return false;
        default:
            return false;
    }
}

} // namespace

InferenceEngine::InferenceEngine()
    : initialized_(false)
    , layers_loaded_(0)
    , llama_context_(nullptr)
    , llama_model_(nullptr) {
}

void InferenceEngine::set_threads(int32_t threads) {
    if (threads < 1) threads = 1;
    sharded_n_threads_ = threads;
    if (llama_context_ != nullptr) {
        llama_set_n_threads(llama_context_, threads, threads);
    }
}

InferenceEngine::~InferenceEngine() {
    // Tear down sharded-engine resources first. The scheduler's
    // buffer pool may reference allocations owned by llama_context_,
    // so it must be freed before the context.
    if (sharded_sched_ != nullptr) {
        ggml_backend_sched_free(sharded_sched_);
        sharded_sched_ = nullptr;
    }
    if (sharded_cpu_backend_ != nullptr) {
        ggml_backend_free(sharded_cpu_backend_);
        sharded_cpu_backend_ = nullptr;
    }
    if (llama_context_ != nullptr) {
        llama_free(llama_context_);
        llama_context_ = nullptr;
    }
    if (llama_model_ != nullptr) {
        llama_model_free(llama_model_);
        llama_model_ = nullptr;
    }

    // Balance the llama_backend_init() from initialize(). Only free the
    // global backend when the last InferenceEngine instance is destroyed.
    if (backend_refcount().fetch_sub(1) == 1) {
        llama_backend_free();
    }
}

bool InferenceEngine::initialize(const std::string& model_path, int32_t /*layers_ignored*/) {
    // Legacy overload: sharding enabled by default.
    return initialize(model_path, /*layer_shard_lazy=*/true);
}

bool InferenceEngine::initialize(const std::string& model_path, bool layer_shard_lazy) {
    model_path_ = model_path;

    if (verbose_) {
        std::cout << "Initializing llama.cpp inference engine...\n";
        std::cout << "Model path: " << model_path << "\n";
        std::cout << "Layer sharding: requested "
                  << (layer_shard_lazy ? "true" : "false")
                  << " (note: per-block residency control requires enable_sharded_engine)\n";
    }

    llama_backend_init();

    // Take a global backend reference. Released (and possibly freed) by a
    // RAII guard on every early-return path and by the destructor on success.
    backend_refcount().fetch_add(1);
    struct BackendRefGuard {
        bool active = true;
        ~BackendRefGuard() {
            if (active && backend_refcount().fetch_sub(1) == 1) {
                llama_backend_free();
            }
        }
        void disarm() { active = false; }
    };
    BackendRefGuard ref_guard;

    struct llama_model_params model_params = llama_model_default_params();

    // CPU-only build: n_gpu_layers=0 means everything on CPU RAM. Sharding
    // happens at decode time via posix_madvise on the mmap region, not at
    // load time.
    model_params.n_gpu_layers = 0;
    // Disable CPU weight repacking (Q4_K -> Q4_K_8x8). Repacking creates a
    // full in-RAM copy of every quantized weight tensor; for qwen3.8:27b
    // (~17 GB) that doubles peak RSS and OOM-kills the load before the
    // sharded engine can manage residency. Keeping the original mmap'd
    // Q4_K format lets the per-block engine evict pages as intended.
    model_params.use_extra_bufts = false;

    // layer_shard_lazy = do not pre-fault weight pages during load.  The
    // weights are still mmap'd, but MAP_POPULATE / POSIX_MADV_WILLNEED are
    // skipped, so the initial HWM is only metadata + working buffers.  The
    // per-block engine then faults in just the layers it needs and evicts
    // them with MADV_DONTNEED.  This is different from TENSOR_READ_LAZY:
    // we keep the non-lazy mmap path so tensors remain in the file-backed
    // mapping and can genuinely be dropped.
    model_params.layer_shard_lazy = layer_shard_lazy;
    if (layer_shard_lazy) {
        model_params.load_mode  = LLAMA_LOAD_MODE_MMAP;     // must use mmap
        model_params.lazy_mode  = LLAMA_LAZY_MODE_OFF;      // avoid memcpy path
    }

    llama_model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (llama_model_ == nullptr) {
        std::cerr << "Error: Failed to load model: " << model_path << "\n";
        return false;
    }

    const struct llama_vocab* vocab = llama_model_get_vocab(llama_model_);
    if (verbose_) {
        std::cout << "Model loaded successfully!\n";
        std::cout << "Total layers in model: " << llama_model_n_layer(llama_model_) << "\n";
        std::cout << "Embedding dimension: " << llama_model_n_embd(llama_model_) << "\n";
        std::cout << "Vocab size: " << llama_vocab_n_tokens(vocab) << "\n";
    }

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = context_size_ > 0 ? context_size_ : 4096;
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 512;
    ctx_params.n_threads = sharded_n_threads_;
    ctx_params.n_threads_batch = sharded_n_threads_;

    llama_context_ = llama_init_from_model(llama_model_, ctx_params);
    if (llama_context_ == nullptr) {
        std::cerr << "Error: Failed to create llama context\n";
        llama_model_free(llama_model_);
        llama_model_ = nullptr;
        return false;
    }

    // Initialization succeeded; the destructor will release the backend ref.
    ref_guard.disarm();
    initialized_ = true;
    if (verbose_) {
        std::cout << "Context initialized. Ready for inference.\n";
    }
    return true;
}

void InferenceEngine::enable_sharded_engine(
    int32_t resident_layers,
    bool    evict_weights,
    int32_t row_size
) {
    if (!initialized_ || llama_model_ == nullptr) {
        std::cerr << "enable_sharded_engine: model not initialized\n";
        return;
    }

    if (!is_sharded_arch_supported(llama_model_)) {
        if (verbose_) {
            std::cout << "Sharded engine disabled: model architecture is not "
                      << "qwen3/qwen3.5 (falling back to llama_decode).\n";
        }
        // Make sure the flag stays false so generate() uses native llama_decode.
        use_sharded_engine_ = false;
        return;
    }

    // Free any previously-created sharded resources so the sweep can
    // reconfigure resident_layers / row_size between runs without leaking
    // schedulers/backends.
    if (sharded_sched_ != nullptr) {
        ggml_backend_sched_free(sharded_sched_);
        sharded_sched_ = nullptr;
    }
    if (sharded_cpu_backend_ != nullptr) {
        ggml_backend_free(sharded_cpu_backend_);
        sharded_cpu_backend_ = nullptr;
    }

    use_sharded_engine_       = true;
    sharded_resident_layers_  = resident_layers > 0 ? resident_layers : 1;
    sharded_evict_weights_    = evict_weights;
    // Phase 9 row-graph K. Clamp to >= 1; let run_multi_block clamp to n_layer.
    sharded_row_size_         = row_size > 0 ? row_size : 1;

    sharded_cpu_backend_ = ggml_backend_cpu_init();
    if (sharded_cpu_backend_ == nullptr) {
        std::cerr << "enable_sharded_engine: ggml_backend_cpu_init failed; "
                     "falling back to un-sharded path\n";
        use_sharded_engine_ = false;
        return;
    }
    ggml_backend_cpu_set_n_threads(sharded_cpu_backend_, sharded_n_threads_);

    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    sharded_sched_ = ggml_backend_sched_new(
        &sharded_cpu_backend_, &cpu_buft, /*n_backends=*/1,
        /*graph_size=*/8192, /*parallel=*/false, /*op_offload=*/false);
    if (sharded_sched_ == nullptr) {
        std::cerr << "enable_sharded_engine: ggml_backend_sched_new failed; "
                     "falling back to un-sharded path\n";
        ggml_backend_free(sharded_cpu_backend_);
        sharded_cpu_backend_ = nullptr;
        use_sharded_engine_ = false;
        return;
    }

    if (verbose_) {
        std::cout << "Sharded engine enabled (resident_layers="
                  << sharded_resident_layers_
                  << ", evict_weights=" << (sharded_evict_weights_ ? "true" : "false")
                  << ", row_size=" << sharded_row_size_
                  << ", threads=" << sharded_n_threads_ << ")\n";
    }
}

// Internal helper: prefill the prompt through the sharded engine.
//
// When sharded is enabled (Phase 7), the sharded engine writes K/V
// directly into llama_context_'s KV cache, and the per-token
// decode loop runs on that warm cache with no second prefill.
// prefill_unsharded=false skips the un-sharded fallback path
// (used by prefill_only_tokens for memory-only measurements).
static int do_prefill(
    llama_context *       ctx,
    const llama_model *   model,
    ggml_backend_sched_t  sched,
    llama_token *         tokens,
    int32_t               n_tokens,
    bool                  evict_weights,
    int32_t               resident_layers,
    int32_t               row_size,
    size_t *              out_sharded_rss = nullptr,
    size_t *              out_sharded_hwm = nullptr,
    bool                  prefill_unsharded = true,
    bool                  verbose = false,
    bool                  progress = false
) {
    const int n_vocab = (int)llama_vocab_n_tokens(llama_model_get_vocab(model));

    if (sched != nullptr) {
        // Phase 7: allocate KV cache cells for the full prompt before
        // running the sharded prefill. The per-block graphs use
        // mctx->cpy_k/cpy_v to write K/V into the cache and
        // mctx->get_k/get_v to read them back for attention. After the
        // sharded prefill completes, the cache holds the K/V values,
        // so the un-sharded decode loop can pick up without a second
        // prefill.
        llama_memory_context_t mctx =
            llama_kv_cache_init_for_batch(ctx, n_tokens);
        if (mctx == nullptr) {
            std::cerr << "Error: llama_kv_cache_init_for_batch failed\n";
            return -1;
        }
        // Commit the slot allocation BEFORE run_multi_block. This
        // updates llama_kv_cache_context::n_kv to the per-ubatch
        // padded slot size, so get_k()/get_v() return views with
        // n_kv matching the kq_mask width. Without this the views
        // span the full cache and flash_attn reads uninitialised
        // K/V at positions past n_tokens.
        mctx->apply();
        multi_block_result_t res = run_multi_block(
            model, sched, tokens, n_tokens, evict_weights,
            resident_layers, row_size, mctx,
            /*pos_first=*/0, verbose, progress, "prefill");
        if (res.final_logits.empty()) {
            std::cerr << "Error: sharded prefill failed\n";
            llama_memory_context_free(mctx);
            return -1;
        }
        // Capture VmRSS/HWM at the end of the sharded prefill. The
        // bench reports these as the steady-state cost of the sharded
        // path with no un-sharded re-fault (Phase 7).
        if (out_sharded_rss != nullptr) {
            *out_sharded_rss = gizmo::read_vm_rss_bytes();
        }
        if (out_sharded_hwm != nullptr) {
            *out_sharded_hwm = gizmo::read_vm_hwm_bytes();
        }
        // The sharded result IS used downstream in Phase 7 — the
        // logits feed llama_sampler_sample in the decode loop (we
        // no longer run the un-sharded prefill below). We log the
        // argmax so it can be compared against the un-sharded
        // path.
        const size_t last_col = (size_t)(n_tokens - 1) * n_vocab;
        int best_id = 0;
        float best_val = res.final_logits[last_col];
        for (int i = 1; i < n_vocab; ++i) {
            const float v = res.final_logits[last_col + i];
            if (v > best_val) {
                best_val = v;
                best_id = i;
            }
        }
        if (verbose) {
            std::cout << "[sharded] prefill done; argmax=" << best_id
                      << " (logit=" << best_val << ")\n";
        }
        // Phase 7: the sharded prefill populated the KV cache. Skip
        // the un-sharded prefill to avoid re-faulting the model.
        prefill_unsharded = false;
        // Copy sharded prefill's last-token logits into
        // llama_context's output buffer; llama_sampler_sample reads
        // from there during the decode loop.
        float * logits_buf = llama_get_logits(ctx);
        if (logits_buf != nullptr) {
            std::memcpy(logits_buf, res.final_logits.data() + last_col,
                        (size_t)n_vocab * sizeof(float));
        }
        llama_memory_context_free(mctx);
    }
    if (!prefill_unsharded) {
        // Sharded-prefill-only path OR Phase 7's "sharded wrote K/V
        // already, skip the un-sharded prefill" path. Either way,
        // KV cache is populated and the decode loop can run.
        return 0;
    }
    // Un-sharded path: prefill in chunks that fit the context's batch
    // limit. llama_decode asserts when a batch exceeds n_batch.
    const int32_t n_batch_max = llama_n_batch(ctx) > 0 ? llama_n_batch(ctx) : 512;
    for (int32_t offset = 0; offset < n_tokens; offset += n_batch_max) {
        const int32_t chunk = std::min(n_batch_max, n_tokens - offset);
        if (llama_decode(ctx, llama_batch_get_one(tokens + offset, chunk)) != 0) {
            std::cerr << "Error: Failed to decode prompt\n";
            return -1;
        }
    }
    {
        float * logits = llama_get_logits(ctx);
        if (logits != nullptr) {
            int best_id = 0;
            float best_val = logits[0];
            for (int i = 1; i < n_vocab; ++i) {
                if (logits[i] > best_val) {
                    best_val = logits[i];
                    best_id = i;
                }
            }
            if (verbose) {
                std::cout << "[un-sharded] prefill done; argmax=" << best_id
                          << " (logit=" << best_val << ")\n";
            }
        }
    }
    return 0;
}

// Internal helper: run a single generated token through the sharded
// engine. The KV cache cell is allocated by llama_kv_cache_init_for_decode,
// which appends the token at the position after the existing prompt.
// After compute the logits are copied into llama_context's output buffer
// so the next llama_sampler_sample call can consume them.
static int do_decode_sharded(
    llama_context *       ctx,
    const llama_model *   model,
    ggml_backend_sched_t  sched,
    llama_token           token_id,
    llama_pos             pos,
    bool                  evict_weights,
    int32_t               resident_layers,
    int32_t               row_size,
    bool                  verbose = false,
    bool                  progress = false
) {
    if (sched == nullptr) {
        return -1;
    }

    llama_memory_context_t mctx = llama_kv_cache_init_for_decode(ctx, token_id);
    if (mctx == nullptr) {
        std::cerr << "Error: llama_kv_cache_init_for_decode failed\n";
        return -1;
    }

    // Commit the slot allocation so the KV cache sees the new token
    // at the right position and run_multi_block's get_n_kv reflects it.
    if (!mctx->apply()) {
        std::cerr << "Error: sharded decode mctx apply failed\n";
        llama_memory_context_free(mctx);
        return -1;
    }

    int32_t token_ids_arr[1] = { token_id };
    multi_block_result_t res = run_multi_block(
        model, sched, token_ids_arr, /*n_tokens=*/1,
        evict_weights, resident_layers, row_size, mctx,
        /*pos_first=*/(int)pos, verbose, progress, "decode");

    if (res.final_logits.empty()) {
        std::cerr << "Error: sharded decode failed\n";
        llama_memory_context_free(mctx);
        return -1;
    }

    const int n_vocab = (int)llama_vocab_n_tokens(llama_model_get_vocab(model));
    float * logits_buf = llama_get_logits(ctx);
    if (logits_buf != nullptr) {
        std::memcpy(logits_buf, res.final_logits.data(),
                    (size_t)n_vocab * sizeof(float));
    }

    // Log argmax for parity checks against un-sharded decode.
    int best_id = 0;
    float best_val = res.final_logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (res.final_logits[i] > best_val) { best_val = res.final_logits[i]; best_id = i; }
    }
    if (verbose) {
        std::cout << "[sharded] decode token done; argmax=" << best_id
                  << " (logit=" << best_val << ")\n";
    }

    llama_memory_context_free(mctx);
    return 0;
}

void InferenceEngine::reset_for_next_run() {
    // Reset the memory context so the next generate()/prefill starts
    // from a truly empty state. For pure-attention models this only
    // clears KV cells; for hybrid recurrent models (qwen3.5) it also
    // zeros the SSM state buffers. Passing data=true is required on
    // recurrent models because leftover R/S state from a previous
    // prompt leaks into the next one and breaks parity with a fresh
    // llama_decode run.
    if (llama_context_ != nullptr) {
        llama_memory_t mem = llama_get_memory(llama_context_);
        if (mem != nullptr) {
            llama_memory_clear(mem, /*data=*/true);
        }
    }
}

bool InferenceEngine::generate_stream(
    const std::string& prompt,
    const InferenceConfig& config,
    std::function<void(const std::string& token_text, int32_t token_id)> callback,
    std::function<bool()> should_cancel
) {
    if (!initialized_ || llama_context_ == nullptr) {
        return false;
    }

    const struct llama_vocab* vocab = llama_model_get_vocab(llama_model_);

    int32_t n_tokens = 0;
    std::vector<llama_token> tokens = tokenize_text(vocab, prompt, true, true, &n_tokens);
    if (n_tokens <= 0) {
        std::cerr << "Error: Failed to tokenize prompt\n";
        return false;
    }

    // Honor per-request context-size requests by rebuilding the context if
    // the requested size differs from the current one. Callers typically
    // reset the KV cache before generation, so this is safe.
    if (config.context_size > 0 &&
        static_cast<int32_t>(llama_n_ctx(llama_context_)) != config.context_size) {
        llama_free(llama_context_);
        llama_context_ = nullptr;
        struct llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = config.context_size;
        ctx_params.n_batch = 512;
        ctx_params.n_ubatch = 512;
        ctx_params.n_threads = sharded_n_threads_;
        ctx_params.n_threads_batch = sharded_n_threads_;
        llama_context_ = llama_init_from_model(llama_model_, ctx_params);
        if (llama_context_ == nullptr) {
            std::cerr << "Error: Failed to recreate llama context for context_size="
                      << config.context_size << "\n";
            return false;
        }
    }

    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler* sampler = llama_sampler_chain_init(sparams);

    const int32_t n_vocab = static_cast<int32_t>(llama_vocab_n_tokens(vocab));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(config.top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(config.top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(n_vocab, 64,
        config.repeat_penalty > 0.0f ? config.repeat_penalty : 1.0f, 0.0f, 0.0f));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(config.temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(config.seed >= 0 ? config.seed : 42));

    // Prefill. Sharded path runs the per-block engine, which writes
    // K/V directly into the KV cache. Non-sharded path runs the
    // full-model llama_decode prefill.
    size_t sharded_rss = 0;
    size_t sharded_hwm = 0;
    if (do_prefill(llama_context_, (const llama_model*)llama_model_,
                   sharded_sched_, tokens.data(), n_tokens,
                   sharded_evict_weights_, sharded_resident_layers_,
                   sharded_row_size_,
                   &sharded_rss, &sharded_hwm,
                   /*prefill_unsharded=*/true, verbose_, progress_) != 0) {
        llama_sampler_free(sampler);
        return false;
    }
    // Stash the sharded prefill's steady-state memory so the bench
    // (and other callers) can read it after generate() returns.
    // 0 means sharding was disabled or hasn't run.
    last_sharded_prefill_rss_ = sharded_rss;
    last_sharded_prefill_hwm_ = sharded_hwm;

    int32_t n_predict = config.max_tokens;
    int32_t n_cur = 0;
    int32_t n_past = n_tokens;  // KV cache already holds the prompt
    std::string generated_text;
    bool cancelled = false;

    while (n_cur < n_predict) {
        if (should_cancel && should_cancel()) {
            if (verbose_) {
                std::cout << "Generation cancelled by caller\n";
            }
            cancelled = true;
            break;
        }

        if (progress_ && !verbose_ && isatty(STDOUT_FILENO)) {
            std::printf("\r[decode] token %2d/%d", n_cur + 1, n_predict);
            std::fflush(stdout);
        }

        llama_token new_token_id = llama_sampler_sample(sampler, llama_context_, -1);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            if (verbose_) {
                std::cout << "End of generation token reached\n";
            }
            break;
        }

        std::string token_text;
        char piece[256];
        int32_t piece_len = llama_token_to_piece(vocab, new_token_id, piece, sizeof(piece), 0, false);
        if (piece_len < 0) {
            // Buffer too small: resize and retry.
            std::vector<char> big_piece(-piece_len + 1, '\0');
            piece_len = llama_token_to_piece(vocab, new_token_id, big_piece.data(),
                                              static_cast<int32_t>(big_piece.size()), 0, false);
            if (piece_len > 0) {
                token_text.assign(big_piece.data(), static_cast<size_t>(piece_len));
            }
        } else if (piece_len > 0) {
            piece[piece_len] = '\0';
            token_text.assign(piece, static_cast<size_t>(piece_len));
        }

        if (!token_text.empty()) {
            generated_text += token_text;
            if (callback) {
                callback(token_text, new_token_id);
            }
            // Check stop sequences against the accumulated output.
            bool should_stop = false;
            for (const auto& s : config.stop) {
                if (s.empty()) continue;
                if (generated_text.size() >= s.size() &&
                    generated_text.compare(generated_text.size() - s.size(), s.size(), s) == 0) {
                    should_stop = true;
                    break;
                }
            }
            if (should_stop) {
                if (verbose_) {
                    std::cout << "Stop sequence matched\n";
                }
                break;
            }
        }

        if (use_sharded_engine_) {
            if (do_decode_sharded(llama_context_, (const llama_model*)llama_model_,
                                  sharded_sched_, new_token_id, (llama_pos)n_past,
                                  sharded_evict_weights_, sharded_resident_layers_,
                                  sharded_row_size_, verbose_, progress_) != 0) {
                std::cerr << "Warning: sharded decode failed, falling back to llama_decode\n";
                if (llama_decode(llama_context_, llama_batch_get_one(&new_token_id, 1)) != 0) {
                    std::cerr << "Error: un-sharded decode also failed\n";
                    llama_sampler_free(sampler);
                    return false;
                }
            }
        } else {
            if (llama_decode(llama_context_, llama_batch_get_one(&new_token_id, 1)) != 0) {
                std::cerr << "Error: Failed to decode generated token\n";
                llama_sampler_free(sampler);
                return false;
            }
            if (verbose_) {
                float * logits = llama_get_logits(llama_context_);
                if (logits != nullptr) {
                    int best_id = 0;
                    float best_val = logits[0];
                    for (int i = 1; i < n_vocab; ++i) {
                        if (logits[i] > best_val) {
                            best_val = logits[i];
                            best_id = i;
                        }
                    }
                    std::cout << "[un-sharded] decode token done; argmax=" << best_id
                              << " (logit=" << best_val << ")\n";
                }
            }
        }

        ++n_past;
        n_cur++;
    }

    if (progress_ && !verbose_ && isatty(STDOUT_FILENO)) {
        std::printf("\r%-40s\r", "");
        std::fflush(stdout);
    }

    llama_sampler_free(sampler);
    return !cancelled;
}

std::string InferenceEngine::generate(const std::string& prompt, const InferenceConfig& config, bool quiet) {
    if (!initialized_ || llama_context_ == nullptr) {
        return "";
    }

    if (!quiet) {
        std::cout << "Generating response for prompt: " << prompt.substr(0, 50) << "...\n";
    }

    if (verbose_ && !quiet) {
        // Tokenize once just to print the count; generate_stream will tokenize again.
        const struct llama_vocab* vocab = llama_model_get_vocab(llama_model_);
        int32_t n = 0;
        (void)tokenize_text(vocab, prompt, true, true, &n);
        std::cout << "Tokenized prompt: " << (n > 0 ? n : 0) << " tokens\n";
    }

    std::string result;
    bool ok = generate_stream(prompt, config, [&result, quiet](const std::string& token_text, int32_t) {
        result += token_text;
        if (!quiet) {
            std::cout << token_text << std::flush;
        }
    });

    if (!ok) {
        return "[Decode error]";
    }

    if (!quiet) {
        std::cout << "\n";
    }
    return result;
}

TokenResult InferenceEngine::generate_token(const std::string& context, const InferenceConfig& config) {
    TokenResult result;
    if (!initialized_ || llama_context_ == nullptr) {
        return result;
    }

    const struct llama_vocab* vocab = llama_model_get_vocab(llama_model_);

    int32_t n_tokens = 0;
    std::vector<llama_token> tokens = tokenize_text(vocab, context, true, true, &n_tokens);
    if (n_tokens <= 0) {
        return result;
    }

    if (do_prefill(llama_context_, (const llama_model*)llama_model_,
                   sharded_sched_, tokens.data(), n_tokens,
                   sharded_evict_weights_, sharded_resident_layers_,
                   sharded_row_size_,
                   &last_sharded_prefill_rss_,
                   &last_sharded_prefill_hwm_,
                   /*prefill_unsharded=*/true, verbose_, progress_) != 0) {
        return result;
    }

    struct llama_sampler* sampler = llama_sampler_init_temp(config.temperature);
    llama_token new_token = llama_sampler_sample(sampler, llama_context_, -1);

    result.token_id = new_token;
    char piece[256];
    int32_t piece_len = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, false);
    if (piece_len > 0) {
        piece[piece_len] = '\0';
        result.text = piece;
    }
    llama_sampler_free(sampler);

    return result;
}

int InferenceEngine::prefill_only(const std::string& prompt) {
    if (!initialized_ || llama_context_ == nullptr) {
        return -1;
    }
    if (sharded_sched_ == nullptr) {
        std::cerr << "prefill_only: sharded engine not enabled; "
                     "call enable_sharded_engine() first\n";
        return -1;
    }
    const struct llama_vocab* vocab = llama_model_get_vocab(llama_model_);
    int32_t n_tokens = 0;
    std::vector<llama_token> tokens = tokenize_text(vocab, prompt, true, true, &n_tokens);
    if (n_tokens <= 0) {
        std::cerr << "prefill_only: tokenization failed\n";
        return -1;
    }
    return prefill_only_tokens(tokens.data(), n_tokens);
}

int InferenceEngine::prefill_only_tokens(const llama_token* tokens, int32_t n_tokens) {
    if (!initialized_ || llama_context_ == nullptr) {
        return -1;
    }
    if (sharded_sched_ == nullptr) {
        std::cerr << "prefill_only_tokens: sharded engine not enabled; "
                     "call enable_sharded_engine() first\n";
        return -1;
    }
    // prefill_unsharded=false: skip the un-sharded prefill so the
    // sharded prefill stands alone for memory measurement. rss/hwm
    // are captured into the engine members for last_sharded_prefill_*().
    if (do_prefill(llama_context_, (const llama_model*)llama_model_,
                   sharded_sched_, const_cast<llama_token*>(tokens), n_tokens,
                   sharded_evict_weights_, sharded_resident_layers_,
                   sharded_row_size_,
                   &last_sharded_prefill_rss_,
                   &last_sharded_prefill_hwm_,
                   /*prefill_unsharded=*/false, verbose_, progress_) != 0) {
        return -1;
    }
    return 0;
}

int InferenceEngine::validate_prefill(
    const std::string& prompt,
    std::vector<float>& out_logits
) {
    if (!initialized_ || llama_context_ == nullptr || llama_model_ == nullptr) {
        return -1;
    }

    reset_for_next_run();

    const struct llama_vocab* vocab = llama_model_get_vocab(llama_model_);
    int32_t n_tokens = 0;
    std::vector<llama_token> tokens = tokenize_text(vocab, prompt, true, true, &n_tokens);
    if (n_tokens <= 0) {
        if (verbose_) {
            std::cerr << "validate_prefill: tokenization failed\n";
        }
        return -1;
    }

    const int n_vocab = (int)llama_vocab_n_tokens(vocab);
    out_logits.assign((size_t)n_vocab, 0.0f);

    if (sharded_sched_ != nullptr) {
        // Sharded path: run the per-block engine and copy its last-token
        // logits back to the caller. Skip the un-sharded fallback.
        if (do_prefill(llama_context_, (const llama_model*)llama_model_,
                       sharded_sched_, tokens.data(), n_tokens,
                       sharded_evict_weights_, sharded_resident_layers_,
                       sharded_row_size_,
                       nullptr, nullptr,
                       /*prefill_unsharded=*/false, verbose_, progress_) != 0) {
            return -1;
        }
        // do_prefill already copied sharded last-token logits into
        // llama_context_'s output buffer when sharded_sched_ is non-null.
        const float* buf = llama_get_logits(llama_context_);
        if (buf == nullptr) {
            return -1;
        }
        std::memcpy(out_logits.data(), buf, (size_t)n_vocab * sizeof(float));
    } else {
        // Un-sharded baseline path: run native llama_decode and read the
        // last-token logits directly. The caller must load the model with
        // --no-shard (layer_shard_lazy=false) to use this branch.
        if (llama_decode(llama_context_, llama_batch_get_one(tokens.data(), n_tokens)) != 0) {
            if (verbose_) {
                std::cerr << "validate_prefill: llama_decode failed\n";
            }
            return -1;
        }
        const float* logits = llama_get_logits(llama_context_);
        if (logits == nullptr) {
            return -1;
        }
        // llama_batch_get_one leaves batch.logits == nullptr, which tells
        // llama_decode to compute logits for the last token only. The
        // returned buffer is therefore just n_vocab floats.
        std::memcpy(out_logits.data(), logits, (size_t)n_vocab * sizeof(float));
    }

    return 0;
}

bool InferenceEngine::is_ready() const {
    return initialized_;
}

std::string InferenceEngine::get_model_info() const {
    if (llama_model_ == nullptr) {
        return "Model not loaded";
    }
    const struct llama_vocab* vocab = llama_model_get_vocab(llama_model_);
    std::string info = "Model: " + model_path_;
    info += ", Layers: " + std::to_string(llama_model_n_layer(llama_model_));
    info += ", Embd: " + std::to_string(llama_model_n_embd(llama_model_));
    info += ", Vocab: " + std::to_string(llama_vocab_n_tokens(vocab));
    return info;
}

std::string InferenceEngine::get_model_id() const {
    if (model_path_.empty()) {
        return "gizmo-model";
    }
    // Return the basename without the .gguf extension.
    size_t slash = model_path_.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? model_path_ : model_path_.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    if (dot != std::string::npos) {
        name = name.substr(0, dot);
    }
    return name;
}

int32_t InferenceEngine::n_layer() const {
    if (llama_model_ == nullptr) return 0;
    return llama_model_n_layer(llama_model_);
}

int32_t InferenceEngine::embedding_dim() const {
    if (llama_model_ == nullptr) return 0;
    return llama_model_n_embd(llama_model_);
}

int32_t InferenceEngine::vocab_size() const {
    if (llama_model_ == nullptr) return 0;
    return llama_vocab_n_tokens(llama_model_get_vocab(llama_model_));
}

} // namespace gizmo
