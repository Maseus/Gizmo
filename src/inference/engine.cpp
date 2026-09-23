#include "inference_engine.hpp"
#include "proc_status.hpp"

#include <cstring>
#include <iostream>
#include <vector>

#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "multi_block.h"
#include "llama-model.h"

namespace gizmo {

InferenceEngine::InferenceEngine()
    : initialized_(false)
    , layers_loaded_(0)
    , llama_context_(nullptr)
    , llama_model_(nullptr) {
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
}

bool InferenceEngine::initialize(const std::string& model_path, int32_t /*layers_ignored*/) {
    // Legacy overload: sharding enabled by default.
    return initialize(model_path, /*layer_shard_lazy=*/true);
}

bool InferenceEngine::initialize(const std::string& model_path, bool layer_shard_lazy) {
    model_path_ = model_path;

    std::cout << "Initializing llama.cpp inference engine...\n";
    std::cout << "Model path: " << model_path << "\n";
    std::cout << "Layer sharding: requested "
              << (layer_shard_lazy ? "true" : "false")
              << " (note: per-block residency control requires enable_sharded_engine)\n";

    llama_backend_init();

    struct llama_model_params model_params = llama_model_default_params();

    // CPU-only build: n_gpu_layers=0 means everything on CPU RAM. Sharding
    // happens at decode time via posix_madvise on the mmap region, not at
    // load time. The layer_shard_lazy flag is accepted but currently a
    // no-op (see llama-model.cpp for why we don't use TENSOR_READ_LAZY).
    model_params.n_gpu_layers = 0;
    (void) layer_shard_lazy;

    llama_model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (llama_model_ == nullptr) {
        std::cerr << "Error: Failed to load model: " << model_path << "\n";
        llama_backend_free();
        return false;
    }

    const struct llama_vocab* vocab = llama_model_get_vocab(llama_model_);
    std::cout << "Model loaded successfully!\n";
    std::cout << "Total layers in model: " << llama_model_n_layer(llama_model_) << "\n";
    std::cout << "Embedding dimension: " << llama_model_n_embd(llama_model_) << "\n";
    std::cout << "Vocab size: " << llama_vocab_n_tokens(vocab) << "\n";

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 512;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;

    llama_context_ = llama_init_from_model(llama_model_, ctx_params);
    if (llama_context_ == nullptr) {
        std::cerr << "Error: Failed to create llama context\n";
        llama_model_free(llama_model_);
        llama_model_ = nullptr;
        llama_backend_free();
        return false;
    }

    initialized_ = true;
    std::cout << "Context initialized. Ready for inference.\n";
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

    std::cout << "Sharded engine enabled (resident_layers="
              << sharded_resident_layers_
              << ", evict_weights=" << (sharded_evict_weights_ ? "true" : "false")
              << ", row_size=" << sharded_row_size_
              << ", threads=" << sharded_n_threads_ << ")\n";
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
    bool                  prefill_unsharded = true
) {
    if (sched != nullptr) {
        // Phase 7: allocate KV cache cells for the prefill batch
        // before running the sharded prefill. The per-block graphs
        // use mctx->cpy_k/cpy_v to write K/V into the cache and
        // mctx->get_k/get_v to read them back for attention. After
        // the sharded prefill completes, the cache holds the K/V
        // values, so the un-sharded decode loop can pick up without
        // a second prefill.
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
        if (mctx != nullptr) mctx->apply();
        multi_block_result_t res = run_multi_block(
            model, sched, tokens, n_tokens, evict_weights,
            resident_layers, row_size, mctx);
        if (res.final_logits.empty()) {
            std::cerr << "Error: sharded prefill failed\n";
            if (mctx != nullptr) llama_memory_context_free(mctx);
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
        const int n_vocab = (int)llama_vocab_n_tokens(llama_model_get_vocab(model));
        const size_t last_col = (size_t)(n_tokens - 1) * n_vocab;
        int best_id = 0;
        float best_val = res.final_logits[last_col];
        for (int i = 1; i < n_vocab; ++i) {
            const float v = res.final_logits[last_col + i];
            if (v > best_val) { best_val = v; best_id = i; }
        }
        std::cout << "[sharded] prefill done; argmax=" << best_id
                  << " (logit=" << best_val << ")\n";
        // Phase 7: the sharded prefill populated the KV cache. Skip
        // the un-sharded prefill to avoid re-faulting the model.
        if (mctx != nullptr) {
            prefill_unsharded = false;
        }
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
    // Un-sharded path: prefill via the full llama_decode. The
    // sharded path no longer reaches here in Phase 7.
    if (llama_decode(ctx, llama_batch_get_one(tokens, n_tokens)) != 0) {
        std::cerr << "Error: Failed to decode prompt\n";
        return -1;
    }
    {
        const int n_vocab = (int)llama_vocab_n_tokens(llama_model_get_vocab(model));
        float * logits = llama_get_logits(ctx);
        if (logits != nullptr) {
            int best_id = 0;
            float best_val = logits[0];
            for (int i = 1; i < n_vocab; ++i) {
                if (logits[i] > best_val) { best_val = logits[i]; best_id = i; }
            }
            std::cout << "[un-sharded] prefill done; argmax=" << best_id
                      << " (logit=" << best_val << ")\n";
        }
    }
    return 0;
}

void InferenceEngine::reset_for_next_run() {
    // Clear the un-sharded KV cache so the next generate() starts
    // from an empty context. The sharded engine's per-block
    // scratch is freed automatically by ggml_backend_sched_reset
    // at the start of each block; we don't reset between runs
    // because the scheduler is reused.
    if (llama_context_ != nullptr) {
        llama_memory_t mem = llama_get_memory(llama_context_);
        if (mem != nullptr) {
            llama_memory_clear(mem, /*data=*/true);
        }
    }
}

std::string InferenceEngine::generate(const std::string& prompt, const InferenceConfig& config) {
    if (!initialized_ || llama_context_ == nullptr) {
        return "";
    }

    std::cout << "Generating response for prompt: " << prompt.substr(0, 50) << "...\n";

    const struct llama_vocab* vocab = llama_model_get_vocab(llama_model_);

    std::vector<llama_token> tokens(512);
    int32_t n_tokens = llama_tokenize(
        vocab,
        prompt.c_str(),
        prompt.size(),
        tokens.data(),
        tokens.size(),
        true,
        true
    );

    if (n_tokens < 0) {
        std::cerr << "Error: Failed to tokenize prompt\n";
        return "[Tokenization error]";
    }

    std::cout << "Tokenized prompt: " << n_tokens << " tokens\n";

    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler* sampler = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(config.top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(config.top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(config.temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));

    // Prefill. Sharded path runs the per-block engine first (and
    // logs its result), then runs the un-sharded prefill to warm
    // the KV cache. Non-sharded path runs only the un-sharded
    // prefill, as before.
    size_t sharded_rss = 0;
    size_t sharded_hwm = 0;
    if (do_prefill(llama_context_, (const llama_model*)llama_model_,
                   sharded_sched_, tokens.data(), n_tokens,
                   sharded_evict_weights_, sharded_resident_layers_,
                   sharded_row_size_,
                   &sharded_rss, &sharded_hwm) != 0) {
        llama_sampler_free(sampler);
        return "[Decode error]";
    }
    // Stash the sharded prefill's steady-state memory so the bench
    // (and other callers) can read it after generate() returns.
    // 0 means sharding was disabled or hasn't run.
    last_sharded_prefill_rss_ = sharded_rss;
    last_sharded_prefill_hwm_ = sharded_hwm;

    std::string result;
    int32_t n_predict = config.max_tokens;
    int32_t n_cur = 0;

    while (n_cur < n_predict) {
        llama_token new_token_id = llama_sampler_sample(sampler, llama_context_, -1);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            std::cout << "End of generation token reached\n";
            break;
        }

        char piece[256];
        int32_t piece_len = llama_token_to_piece(vocab, new_token_id, piece, sizeof(piece), 0, false);
        if (piece_len > 0) {
            piece[piece_len] = '\0';
            result += piece;
            std::cout << piece << std::flush;
        }

        if (llama_decode(llama_context_, llama_batch_get_one(&new_token_id, 1)) != 0) {
            std::cerr << "Error: Failed to decode generated token\n";
            break;
        }

        n_cur++;
    }

    llama_sampler_free(sampler);
    std::cout << "\n";
    return result;
}

int InferenceEngine::generate_stream(
    const std::string& prompt,
    const InferenceConfig& config,
    std::function<bool(const std::string& token)> callback
) {
    if (!initialized_ || llama_context_ == nullptr) {
        return -1;
    }

    const struct llama_vocab* vocab = llama_model_get_vocab(llama_model_);

    std::vector<llama_token> tokens(512);
    int32_t n_tokens = llama_tokenize(
        vocab,
        prompt.c_str(),
        prompt.size(),
        tokens.data(),
        tokens.size(),
        true,
        true
    );

    if (n_tokens < 0) {
        std::cerr << "Error: Failed to tokenize prompt\n";
        return -1;
    }

    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler* sampler = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(config.top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(config.top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(config.temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));

    size_t sharded_rss = 0;
    size_t sharded_hwm = 0;
    if (do_prefill(llama_context_, (const llama_model*)llama_model_,
                   sharded_sched_, tokens.data(), n_tokens,
                   sharded_evict_weights_, sharded_resident_layers_,
                   sharded_row_size_,
                   &sharded_rss, &sharded_hwm) != 0) {
        llama_sampler_free(sampler);
        return -1;
    }
    last_sharded_prefill_rss_ = sharded_rss;
    last_sharded_prefill_hwm_ = sharded_hwm;

    int generated = 0;
    int32_t n_predict = config.max_tokens;
    while (generated < n_predict) {
        llama_token new_token_id = llama_sampler_sample(sampler, llama_context_, -1);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        char piece[256];
        int32_t piece_len = llama_token_to_piece(vocab, new_token_id, piece, sizeof(piece), 0, false);
        if (piece_len > 0) {
            piece[piece_len] = '\0';
            if (!callback(std::string(piece))) {
                break;
            }
        }

        if (llama_decode(llama_context_, llama_batch_get_one(&new_token_id, 1)) != 0) {
            std::cerr << "Error: Failed to decode generated token\n";
            break;
        }

        ++generated;
    }

    llama_sampler_free(sampler);
    return generated;
}

TokenResult InferenceEngine::generate_token(const std::string& context, const InferenceConfig& config) {
    TokenResult result;
    if (!initialized_ || llama_context_ == nullptr) {
        return result;
    }

    const struct llama_vocab* vocab = llama_model_get_vocab(llama_model_);

    std::vector<llama_token> tokens(512);
    int32_t n_tokens = llama_tokenize(
        vocab,
        context.c_str(),
        context.size(),
        tokens.data(),
        tokens.size(),
        true,
        true
    );

    if (n_tokens < 0) {
        return result;
    }

    if (do_prefill(llama_context_, (const llama_model*)llama_model_,
                   sharded_sched_, tokens.data(), n_tokens,
                   sharded_evict_weights_, sharded_resident_layers_,
                   sharded_row_size_,
                   &last_sharded_prefill_rss_,
                   &last_sharded_prefill_hwm_) != 0) {
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
    std::vector<llama_token> tokens(prompt.size() + 16);
    int32_t n_tokens = llama_tokenize(
        vocab, prompt.c_str(), prompt.size(),
        tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
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
                   /*prefill_unsharded=*/false) != 0) {
        return -1;
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
