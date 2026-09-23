// Multi-block driver. See multi_block.h.

#include "multi_block.h"

#include "ggml.h"
#include "ggml-backend.h"

#include "evict.h"
#include "shard_block.h"
#include "tail_graph.h"

#include "llama-model.h"
#include "llama-hparams.h"
#include "llama-kv-cache.h"
#include "llama-memory.h"
#include "llama-memory-hybrid.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

multi_block_result_t run_multi_block(
    const llama_model *             model,
    ggml_backend_sched_t            sched,
    const int32_t *                 token_ids,
    int                             n_tokens,
    bool                            evict_weights,
    int                             resident_layers,
    int                             row_size,
    llama_memory_context_i *        mctx
) {
    multi_block_result_t result{};
    const int n_layer = (int)model->hparams.n_layer();
    const int n_embd   = (int)model->hparams.n_embd;
    const int n_vocab  = (int)model->vocab.n_tokens();

    // Clamp row_size to [1, n_layer]. With row_size >= n_layer the
    // whole model fits in one cgraph; with row_size == 1 we get
    // byte-exact per-block behavior (Phase 8 baseline).
    const int K = std::max(1, std::min(row_size > 0 ? row_size : 1, n_layer));
    const int n_rows = (n_layer + K - 1) / K;

    // Phase 10: hybrid models are no longer rejected upfront. The
    // per-block dispatcher in shard_block.cpp routes full-attention
    // layers through the qwen3 builder and SSM (recurrent / gated
    // delta net) layers through a stub. Phase 12 implements SSM;
    // until then, SSM blocks pass the input residual through
    // unchanged and emit a one-shot warning per layer.
    int n_full = 0, n_recr = 0;
    int first_recr_il = -1;
    for (int il = 0; il < n_layer; ++il) {
        if (model->hparams.is_recr((uint32_t)il)) {
            ++n_recr;
            if (first_recr_il < 0) first_recr_il = il;
        } else {
            ++n_full;
        }
    }
    if (n_recr > 0) {
        std::fprintf(stderr,
            "WARNING: sharded engine: model has %d full-attention + "
            "%d recurrent (SSM) layers (first recurrent: layer %d). "
            "Full-attention layers work correctly; SSM layers are "
            "stubbed (input residual passed through unchanged) until "
            "Phase 12 lands.\n",
            n_full, n_recr, first_recr_il);
    }

    // ---- Carrier context: a CPU-buft-backed ctx holding the F32
    // residual carrier tensor AND the per-row shared input tensors
    // (positions, token_ids_row0, kq_mask). Allocating on the CPU
    // buft lets us ggml_backend_tensor_set their data BEFORE
    // sched_alloc_graph, then mark them as inputs in each row's
    // ctx (ggml_set_input is idempotent — same tensor pointer
    // works across rows).
    ggml_init_params carrier_iparams = {
        /* .mem_size   = */ 4 * ggml_tensor_overhead() + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * carrier_ctx = ggml_init(carrier_iparams);
    if (!carrier_ctx) {
        std::fprintf(stderr, "ERROR: failed to init carrier context\n");
        return result;
    }

    int64_t carrier_ne[4] = { n_embd, n_tokens, 1, 1 };
    ggml_tensor * carrier = ggml_new_tensor(carrier_ctx, GGML_TYPE_F32, 4, carrier_ne);
    ggml_set_input(carrier);

    // Causal attention mask: F16 [n_kv, n_tokens, 1, 1].
    int32_t n_kv_mask = n_tokens;
    if (mctx != nullptr) {
        // mctx is the llama_memory_context_i interface pointer. For
        // hybrid models (qwen3.5) the underlying object is a
        // llama_memory_hybrid_context; for pure-attn models (qwen3)
        // it's a llama_kv_cache_context. dynamic_cast discriminates
        // the two safely: returns nullptr for non-hybrid contexts, in
        // which case we fall back to n_kv_mask = n_tokens.
        auto * hybrid_ctx = dynamic_cast<llama_memory_hybrid_context *>(mctx);
        if (hybrid_ctx != nullptr) {
            // hybrid_ctx->get_attn() returns const; const_cast so we can
            // call non-const kv_ctx->get_n_kv(). Safe: we never mutate
            // the KV cache from this read-only probe.
            auto * kv_ctx = const_cast<llama_kv_cache_context *>(hybrid_ctx->get_attn());
            if (kv_ctx != nullptr) {
                n_kv_mask = (int32_t) kv_ctx->get_n_kv();
            }
        }
    }
    int64_t mask_ne[4] = { n_kv_mask, n_tokens, 1, 1 };
    ggml_tensor * kq_mask = ggml_new_tensor(carrier_ctx, GGML_TYPE_F16, 4, mask_ne);
    ggml_set_input(kq_mask);

    // Positions tensor: I32, shape depends on whether the model uses
    // MRoPE (qwen3.5/4-axis) or plain 1-D RoPE (qwen3). For MRoPE,
    // ggml_rope_multi asserts b->ne[0] == a->ne[2] * 4 since each
    // token contributes one position ID per RoPE section. We store
    // the IDs interleaved per-token: [t0_s0, t0_s1, t0_s2, t0_s3,
    // t1_s0, ...]. For text-only prefill with no vision/video axes,
    // each token's 4 sections get the same scalar position i.
    const uint32_t n_pos_per_embd = model->hparams.n_pos_per_embd();
    ggml_tensor * positions = ggml_new_tensor_1d(carrier_ctx, GGML_TYPE_I32, (int64_t)n_tokens * n_pos_per_embd);
    ggml_set_input(positions);

    // token_ids tensor: only used by the row containing il=0 (block 0
    // of the first row reads embeddings from model->tok_embd). Created
    // here so we can fill it once before the per-row loop.
    ggml_tensor * token_ids_in = ggml_new_tensor_1d(carrier_ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(token_ids_in);

    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_t carrier_buf =
        ggml_backend_alloc_ctx_tensors_from_buft(carrier_ctx, cpu_buft);
    if (!carrier_buf) {
        std::fprintf(stderr, "ERROR: failed to allocate carrier on CPU\n");
        return result;
    }

    // Fill causal mask once. Layout: ggml row-major (ne[0]=cols=n_kv,
    // ne[1]=rows=n_tokens). Columns j >= n_tokens masked -inf.
    {
        const uint16_t neg_inf_f16 = 0xFC00u;
        std::vector<uint16_t> mask_data((size_t)n_kv_mask * n_tokens, neg_inf_f16);
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j <= i && j < n_tokens; ++j) {
                mask_data[(size_t)i * n_kv_mask + j] = 0;
            }
        }
        ggml_backend_tensor_set(kq_mask, mask_data.data(), 0,
                                (size_t)n_kv_mask * n_tokens * sizeof(uint16_t));
    }

    // Fill positions and token_ids once. For MRoPE each token gets
    // n_pos_per_embd consecutive IDs; for plain RoPE it gets one.
    std::vector<int32_t> positions_data((size_t)n_tokens * n_pos_per_embd);
    for (int i = 0; i < n_tokens; ++i) {
        for (uint32_t s = 0; s < n_pos_per_embd; ++s) {
            positions_data[(size_t)i * n_pos_per_embd + s] = i;
        }
    }
    ggml_backend_tensor_set(positions,    positions_data.data(), 0,
                            (size_t)n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(token_ids_in, token_ids,           0,
                            (size_t)n_tokens * sizeof(int32_t));

    // Host staging buffer for residual copies (only needed between rows).
    std::vector<float> residual_staging((size_t)n_embd * n_tokens);

    // ---- Per-row graph driver. ----
    for (int row = 0; row < n_rows; ++row) {
        const int il_start = row * K;
        const int K_this   = std::min(K, n_layer - il_start);

        // Per-row ctx + cgraph sized for K_this blocks. Each block
        // contributes ~37 nodes to the cgraph, so we budget 64 per
        // block (headroom for ops that may share). For K=1, this is
        // 64 nodes, well below the original per-block allocation that
        // used ggml_graph_overhead() (=2048). For K=36 we get 2304
        // nodes — comfortably under the 8192-node scheduler capacity.
        // SSM (gated delta net) blocks use ~290 nodes, full-attn
        // blocks use ~34; budget per-block is 512 to leave headroom.
        const size_t max_nodes  = (size_t)(512 * K_this);
        const size_t ctx_bytes  = (size_t)(256 * K_this) * ggml_tensor_overhead()
                                + ggml_graph_overhead_custom(max_nodes, false);
        ggml_init_params row_iparams = {
            ctx_bytes, nullptr, true,
        };
        ggml_context * row_ctx = ggml_init(row_iparams);
        if (!row_ctx) {
            std::fprintf(stderr, "ERROR: failed to init row_ctx for row %d (K=%d)\n",
                         row, K_this);
            return result;
        }
        ggml_cgraph * row_gf = ggml_new_graph_custom(row_ctx, max_nodes, false);

        ggml_backend_sched_reset(sched);

        // positions / token_ids_in / kq_mask / carrier are owned by
        // carrier_ctx (CPU-buft-backed, pre-filled). Their tensor
        // pointers are passed straight into build_block_graph_into;
        // ggml_set_input is idempotent so re-marking across rows is
        // harmless.

        ggml_tensor * k_idxs_row = nullptr;
        ggml_tensor * v_idxs_row = nullptr;
        auto * hybrid_ctx_row = dynamic_cast<llama_memory_hybrid_context *>(mctx);
        // k_idxs/v_idxs are only allocated by sched_alloc_graph if
        // some block in the row references them (full-attention blocks
        // wire them into kv_ctx->cpy_k/cpy_v via ggml_build_forward_expand).
        // SSM rows never touch them, so we skip construction there -
        // building them unconditionally and then calling
        // set_input_k_idxs on an unallocated buffer triggers
        // GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer)).
        bool row_needs_kv = false;
        for (int j = 0; j < K_this; ++j) {
            if (!model->hparams.is_recr((uint32_t)(il_start + j))) {
                row_needs_kv = true;
                break;
            }
        }
        if (hybrid_ctx_row != nullptr && row_needs_kv) {
            auto * kv_ctx = const_cast<llama_kv_cache_context *>(hybrid_ctx_row->get_attn());
            if (kv_ctx != nullptr) {
                const llama_ubatch & ubatch = kv_ctx->get_ubatch();
                k_idxs_row = kv_ctx->build_input_k_idxs(row_ctx, ubatch);
                v_idxs_row = kv_ctx->build_input_v_idxs(row_ctx, ubatch);
                // set_input_k_idxs / set_input_v_idxs called AFTER alloc below.
            }
        }

        // Chain K_this blocks. Block 0 of the row reads from tok_embd
        // (il_start==0) or the carrier; block j>0 reads block j-1's out.
        ggml_tensor * cur_inpL = (il_start == 0) ? nullptr : carrier;
        ggml_tensor * out_residual = nullptr;
        for (int j = 0; j < K_this; ++j) {
            const int il = il_start + j;
            // token_ids_in is only meaningful for the first block of
            // the whole prefill (il==0). Pass it only there.
            ggml_tensor * tok_ids_for_block =
                (j == 0 && il_start == 0) ? token_ids_in : nullptr;
            ggml_tensor * r = build_block_graph_into(
                model, row_ctx, row_gf,
                il, n_tokens,
                cur_inpL, tok_ids_for_block, positions, kq_mask,
                k_idxs_row, v_idxs_row, hybrid_ctx_row);
            if (r == nullptr) {
                std::fprintf(stderr, "ERROR: build_block_graph_into failed for il=%d\n", il);
                ggml_free(row_ctx);
                return result;
            }
            cur_inpL = r;
            out_residual = r;
        }

        // Allocate and run. Schedules the whole row's worth of nodes in one call.
        if (!ggml_backend_sched_alloc_graph(sched, row_gf)) {
            std::fprintf(stderr, "ERROR: sched_alloc_graph failed for row %d (n_nodes=%d)\n",
                         row, ggml_graph_n_nodes(row_gf));
            ggml_free(row_ctx);
            return result;
        }

        // Fill k_idxs/v_idxs now that they have backing storage.
        if (hybrid_ctx_row != nullptr && row_needs_kv && k_idxs_row != nullptr) {
            auto * kv_ctx = const_cast<llama_kv_cache_context *>(hybrid_ctx_row->get_attn());
            if (kv_ctx != nullptr) {
                const llama_ubatch & ubatch = kv_ctx->get_ubatch();
                kv_ctx->set_input_k_idxs(k_idxs_row, &ubatch);
                kv_ctx->set_input_v_idxs(v_idxs_row, &ubatch);
            }
        }

        const enum ggml_status ec = ggml_backend_sched_graph_compute(sched, row_gf);
        if (ec != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "ERROR: graph_compute failed for row %d status=%d\n",
                         row, (int)ec);
            ggml_free(row_ctx);
            return result;
        }

        // Copy the row's last block's residual into the carrier for the next row.
        if (row < n_rows - 1) {
            const size_t residual_bytes = (size_t)n_embd * n_tokens * sizeof(float);
            ggml_backend_tensor_get(out_residual, residual_staging.data(), 0, residual_bytes);
            ggml_backend_tensor_set(carrier, residual_staging.data(), 0, residual_bytes);
        }

        // Optional per-row residual dump: prints the first 8 floats of the
        // last token's residual at the row boundary. Useful for localizing
        // where the sharded engine diverges from the reference. Off by
        // default; enable via GIZMO_DUMP_RESIDUAL=1.
        static const int dump_env = []() {
            const char* e = std::getenv("GIZMO_DUMP_RESIDUAL");
            int v = (e && std::atoi(e) != 0) ? 1 : 0;
            std::fprintf(stderr, "[sharded] GIZMO_DUMP_RESIDUAL=%d\n", v);
            return v;
        }();
        if (dump_env) {
            const int last_tok = n_tokens - 1;
            const size_t offset_bytes = (size_t)last_tok * n_embd * sizeof(float);
            std::vector<float> last_token_residual((size_t)n_embd);
            ggml_backend_tensor_get(out_residual, last_token_residual.data(),
                                    offset_bytes, (size_t)n_embd * sizeof(float));
            std::printf("[sharded] residual dump row=%d il_start=%d last_tok first 8:",
                        row, il_start);
            for (int i = 0; i < 8 && i < n_embd; ++i) {
                std::printf(" %.4f", last_token_residual[i]);
            }
            std::printf("\n");
            std::fflush(stdout);
        }

        // Per-block eviction within the row. For each block il in
        // [il_start, il_start+K_this), evict `il - resident_layers`
        // if it has fallen out of the residency window.
        size_t total_evicted = 0;
        if (evict_weights) {
            for (int j = 0; j < K_this; ++j) {
                const int il = il_start + j;
                if (il == n_layer - 1) {
                    total_evicted += sharded_evict::evict_block_weights(model->layers[il]);
                } else {
                    const int out_of_window = il - resident_layers;
                    if (out_of_window >= 0) {
                        total_evicted += sharded_evict::evict_block_weights(model->layers[out_of_window]);
                    }
                }
            }
        }

        std::printf("[sharded] row %d/%d (il=%d..%d, n_nodes=%d, K=%d, resident=%d%s)\n",
                    row, n_rows, il_start, il_start + K_this - 1,
                    ggml_graph_n_nodes(row_gf), K_this, resident_layers,
                    evict_weights ? "" : ", no-evict");
        if (evict_weights && (row % 2 == 1 || row == n_rows - 1)) {
            std::ifstream f("/proc/self/status");
            std::string line;
            while (std::getline(f, line)) {
                if (line.rfind("VmRSS:", 0) == 0) {
                    std::printf("[sharded]   %s\n", line.c_str());
                    break;
                }
            }
        }
        if (evict_weights && total_evicted > 0) {
            std::printf("[sharded]   page-out: %.1f MB\n",
                        (double)total_evicted / (1024.0 * 1024.0));
        }

        ggml_free(row_ctx);
    }

    // ---- Tail: rms_norm + lm_head, using the carrier as input.
    ggml_backend_sched_reset(sched);
    tail_graph_t tg = build_tail_graph(model, n_tokens, carrier);
    if (!tg.ctx || !tg.gf) {
        std::fprintf(stderr, "ERROR: build_tail_graph failed\n");
        return result;
    }
    if (!ggml_backend_sched_alloc_graph(sched, tg.gf)) {
        std::fprintf(stderr, "ERROR: sched_alloc_graph failed for tail graph\n");
        return result;
    }
    const enum ggml_status tg_ec = ggml_backend_sched_graph_compute(sched, tg.gf);
    if (tg_ec != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ERROR: graph_compute failed for tail graph status=%d\n", (int)tg_ec);
        return result;
    }

    result.final_logits.resize((size_t)n_vocab * n_tokens);
    result.residual   .resize((size_t)n_embd   * n_tokens);

    const size_t logits_bytes  = (size_t)n_vocab * n_tokens * sizeof(float);
    const size_t residual_bytes = (size_t)n_embd * n_tokens * sizeof(float);
    ggml_backend_tensor_get(tg.out, result.final_logits.data(), 0, logits_bytes);
    ggml_backend_tensor_get(carrier, result.residual.data(), 0, residual_bytes);

    ggml_free(carrier_ctx);

    return result;
}
