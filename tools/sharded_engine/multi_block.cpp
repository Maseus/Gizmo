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

namespace {
// The memory context returned by llama_kv_cache_init_for_batch/decode is
// either a plain llama_kv_cache_context (pure-attention models like Qwen3)
// or a llama_memory_hybrid_context wrapping an attention KV cache plus a
// recurrent cache (hybrid models like Qwen3.5). This helper returns the
// underlying attention KV cache context in both cases.
llama_kv_cache_context * get_kv_ctx(llama_memory_context_i * mctx) {
    if (mctx == nullptr) {
        return nullptr;
    }
    auto * hybrid = dynamic_cast<llama_memory_hybrid_context *>(mctx);
    if (hybrid != nullptr) {
        return const_cast<llama_kv_cache_context *>(hybrid->get_attn());
    }
    return dynamic_cast<llama_kv_cache_context *>(mctx);
}
} // namespace
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// Per-block probe state (file-scope so the eval callback can fill it).
// When GIZMO_PROBE_PREFIXES="prefix1,prefix2,..." is set, each tensor
// whose name starts with any registered prefix gets its last-token
// column captured. Each prefix maps to its own output file via
// GIZMO_PROBE_FILE_PREFIX. The format mirrors ref_per_layer_dump so
// the sharded outputs can be diffed directly against the reference.
//
// Header: int32[3] = {count, n_tokens, n_embd}
// Per-layer: int32 il, int32 n_embd, float[n_embd] (last-tok column)
namespace {
constexpr int kMaxProbeLayers = 256;

// Strip trailing '-' from prefix for cleaner filenames.
static std::string strip_trailing_dash(const std::string & s) {
    if (!s.empty() && s.back() == '-') return s.substr(0, s.size() - 1);
    return s;
}

// Filenames mirror the reference dumps (e.g. "/tmp/ref_attn_residual.bin")
// regardless of whether the on-graph tensor prefix is "shard_attn_residual-"
// or just "attn_residual-". Drop a leading "shard_" so the slot prefix used
// for eval-name match stays distinct from the reference prefix (avoiding
// duplicate captures if both engines run in the same process), while the
// output filename stays unprefixed / `ref_`-prefixed consistent with the
// reference dump tool.
static std::string strip_leading_shard(const std::string & s) {
    constexpr const char * lead = "shard_";
    constexpr size_t llen = 6;
    if (s.size() > llen && std::strncmp(s.c_str(), lead, llen) == 0) {
        return s.substr(llen);
    }
    return s;
}

// Per-block probe: captures every tensor whose name starts with any
// GIZMO_PROBE_PREFIX (comma-separated). Each prefix maps to its own
// output file path via GIZMO_PROBE_FILE_<PREFIX_UPPER>. Used for
// bisecting qwen3.5 SSM drift between sharded and reference paths:
// we capture sharded's attn_residual-{il}, attn_post_norm-{il}, ffn_out-{il}
// and l_out-{il} and diff against reference dumps.
struct ProbeSlot {
    std::string                prefix;     // e.g. "shard_attn_residual-"
    std::string                path;       // e.g. "/tmp/shard_attn_residual.bin"
    std::vector<std::vector<float>> cols;  // [count][n_embd]
    // n_embd_for_il[i] = n_embd of row i when s_probe_all_cols is on.
    // Stored separately because all-cols mode allocates cols as
    // cols[i] of size n_cols * n_embd_for_il[i].
    std::vector<int>           n_embd_for_il;
    int                        count = 0;
    int                        n_embd = 0;
};
static std::vector<ProbeSlot> g_probe_slots;

static void init_probe_slots(int /*n_tokens*/) {
    const char * e = std::getenv("GIZMO_PROBE_PREFIXES");
    if (!e || !*e) return;
    const char * pe = std::getenv("GIZMO_PROBE_FILE_PREFIX");
    const std::string file_prefix = pe ? pe : "/tmp/";
    std::string s = e;
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos) end = s.size();
        std::string pref = s.substr(start, end - start);
        if (!pref.empty()) {
            ProbeSlot ps;
            ps.prefix = pref;
            ps.path   = file_prefix + strip_trailing_dash(strip_leading_shard(pref)) + ".bin";
            g_probe_slots.push_back(std::move(ps));
        }
        start = end + 1;
    }
    for (auto & ps : g_probe_slots) {
        std::fprintf(stderr, "[sharded] probe slot prefix='%s' file='%s'\n",
                     ps.prefix.c_str(), ps.path.c_str());
    }
}

// Match `prefix` against `name` AND reject any name whose full-name
// ending is exactly " (view)". The qwen3.5 ref graph (qwen35.cpp:275,
// :289) creates ggml_view_3d sub-tensors of Qcur_full that inherit
// the auto-suffix "<src->name> (view)". Those views share the
// matmul-output prefix (e.g. "Qcur_full-3") and would otherwise
// overwrite the slot's column capture with smaller buffers.
// User code that does explicitly rename the view ("Qcur_reshaped",
// "gate_reshaped") does not collide.
static bool prefix_match_no_view(const char * name, const std::string & prefix) {
    const size_t plen = prefix.size();
    if (std::strncmp(name, prefix.c_str(), plen) != 0) return false;
    // Match "<prefix><il> (view)" where the trailing 7 chars are
    // " (view)" (space + paren-wrapped "view"). ggml_format_name
    // appends " (view)" to the source name verbatim.
    static const char view_suffix[] = " (view)";
    constexpr size_t view_len = sizeof(view_suffix) - 1;  // 7
    const size_t nlen = std::strlen(name);
    if (nlen >= view_len &&
        std::memcmp(name + (nlen - view_len), view_suffix, view_len) == 0) {
        return false;
    }
    return true;
}

bool probe_eval_cb(ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (!t || t->name[0] == '\0') return false;
    if (ask) {
        for (auto & ps : g_probe_slots) {
            if (prefix_match_no_view(t->name, ps.prefix)) {
                return true;
            }
        }
        return false;
    }
    if (g_probe_slots.empty()) return true;
    // Find a matching slot.
    ProbeSlot * slot = nullptr;
    for (auto & ps : g_probe_slots) {
        if (prefix_match_no_view(t->name, ps.prefix)) {
            slot = &ps;
            break;
        }
    }
    if (slot == nullptr) return true;
    // Extract il from the trailing "-{il}" suffix.
    const char * dash = std::strrchr(t->name, '-');
    if (!dash) return true;
    int il = std::atoi(dash + 1);
    if (il < 0 || il >= kMaxProbeLayers) return true;
    const int64_t n_embd = t->ne[0];
    const int64_t n_cols = t->ne[1] * t->ne[2] * t->ne[3];
    const size_t elem_size = ggml_element_size(t);
    // GIZMO_PROBE_ALL_COLS=1 dumps every column (default dumps only
    // last column to keep per-block files small). All-columns mode
    // is used to bisect column 0 vs column 4 divergence on multi-token
    // matmul outputs (e.g. Qcur_full).
    static const int s_probe_all_cols = []() {
        const char * e = std::getenv("GIZMO_PROBE_ALL_COLS");
        return (e && std::atoi(e) != 0) ? 1 : 0;
    }();
    const int64_t last_col = s_probe_all_cols ? (n_cols - 1) : (n_cols - 1);
    const int64_t first_col = s_probe_all_cols ? 0 : (n_cols - 1);
    if ((int)slot->cols.size() <= il) slot->cols.resize(il + 1);
    if (s_probe_all_cols) {
        // Allocate n_cols * n_embd. Header will be {count, n_cols, n_embd};
        // per-row layout: {il, n_embd, cols[0..n_cols-1]}.
        const size_t total = (size_t)n_cols * (size_t)n_embd;
        if ((int)slot->cols[il].size() != (int)total) {
            slot->cols[il].assign(total, 0.0f);
            if ((int)slot->n_embd_for_il.size() <= il) slot->n_embd_for_il.resize(il + 1, 0);
            slot->n_embd_for_il[il] = (int)n_embd;
            if (slot->n_embd == 0) slot->n_embd = (int) n_embd;
        }
        // Scratch buffer holds one column of raw bytes (F16/BF16/etc.)
        // before F32 conversion. Reused across columns in this loop.
        std::vector<uint8_t> scratch((size_t)n_embd * elem_size);
        for (int64_t c = first_col; c <= last_col; ++c) {
            const size_t offset = (size_t) c * (size_t) n_embd * elem_size;
            const size_t size   = (size_t) n_embd * elem_size;
            ggml_backend_tensor_get(t, scratch.data(), offset, size);
            float * dst = slot->cols[il].data() + c * n_embd;
            if (t->type == GGML_TYPE_F32) {
                std::memcpy(dst, scratch.data(), size);
            } else if (t->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(scratch.data()), dst, n_embd);
            } else if (t->type == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(scratch.data()), dst, n_embd);
            } else {
                std::fprintf(stderr, "[sharded] probe: unsupported tensor type %d for %s\n",
                             (int)t->type, t->name);
            }
        }
    } else {
        const size_t offset = (size_t) last_col * (size_t) n_embd * elem_size;
        const size_t size   = (size_t) n_embd * elem_size;
        if ((int)slot->cols[il].size() != (int)n_embd) {
            slot->cols[il].assign((size_t)n_embd, 0.0f);
        }
        std::vector<uint8_t> scratch((size_t)n_embd * elem_size);
        ggml_backend_tensor_get(t, scratch.data(), offset, size);
        if (t->type == GGML_TYPE_F32) {
            std::memcpy(slot->cols[il].data(), scratch.data(), size);
        } else if (t->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(scratch.data()),
                                  slot->cols[il].data(), n_embd);
        } else if (t->type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(scratch.data()),
                                  slot->cols[il].data(), n_embd);
        } else {
            std::fprintf(stderr, "[sharded] probe: unsupported tensor type %d for %s\n",
                         (int)t->type, t->name);
        }
    }
    if (il + 1 > slot->count) slot->count = il + 1;
    if (slot->n_embd == 0) slot->n_embd = (int) n_embd;
    return true;  // keep scheduler going
}

static void write_probe_slot(const ProbeSlot & ps, int n_tokens) {
    // Count only the non-empty rows (eval callback may fire for some
    // layer indices but never for others, e.g. SSM layers don't emit
    // attn_residual tensors).
    int emitted = 0;
    for (const auto & c : ps.cols) if (!c.empty()) ++emitted;
    if (emitted == 0) return;
    FILE * fp = std::fopen(ps.path.c_str(), "wb");
    if (!fp) {
        std::fprintf(stderr, "ERROR: cannot open %s for probe write\n",
                     ps.path.c_str());
        return;
    }
    // When GIZMO_PROBE_ALL_COLS is set, the per-row buffer length
    // is n_tokens * n_embd_for_il[il]. Header n_embd stays the
    // per-column n_embd so diff_per_layer column-offset math still
    // works (it reads at offsets (col)*n_embd from the start of each
    // row).
    static const int s_probe_all_cols = []() {
        const char * e = std::getenv("GIZMO_PROBE_ALL_COLS");
        return (e && std::atoi(e) != 0) ? 1 : 0;
    }();
    int32_t header[3] = { emitted, n_tokens, ps.n_embd };
    std::fwrite(header, sizeof(int32_t), 3, fp);
    // Emit rows in ascending il order using the actual layer index,
    // not the slot offset, so the diff tool can address rows by il.
    std::vector<int> emitted_ils;
    for (int il = 0; il < (int)ps.cols.size(); ++il) {
        if (!ps.cols[il].empty()) emitted_ils.push_back(il);
    }
    for (int il : emitted_ils) {
        std::fwrite(&il,        sizeof(int32_t), 1, fp);
        // All-cols mode: per-row floats = n_tokens * n_embd_per_col.
        // Last-col mode: per-row floats = n_embd_per_col. Write the
        // row's actual on-disk width so the reader knows how much to
        // unpack (the previous shape kept header n_embd at the per-col
        // value while writing n_tokens*n_embd bytes -- readers using
        // struct.unpack('{n}f') would over-read or fail).
        int32_t ne = s_probe_all_cols
            ? (int32_t) ps.cols[il].size()
            : ps.n_embd;
        std::fwrite(&ne,        sizeof(int32_t), 1, fp);
        const size_t bytes = (size_t)ne * sizeof(float);
        std::fwrite(ps.cols[il].data(), 1, bytes, fp);
    }
    std::fclose(fp);
    std::fprintf(stderr, "[sharded] dumped %d entries to %s (n_embd=%d)\n",
                 emitted, ps.path.c_str(), ps.n_embd);
}
} // namespace

multi_block_result_t run_multi_block(
    const llama_model *             model,
    ggml_backend_sched_t            sched,
    const int32_t *                 token_ids,
    int                             n_tokens,
    bool                            evict_weights,
    int                             resident_layers,
    int                             row_size,
    llama_memory_context_i *        mctx,
    int                             pos_first,
    bool                            verbose
) {
    multi_block_result_t result{};
    const int n_layer = (int)model->hparams.n_layer();
    const int n_embd   = (int)model->hparams.n_embd;
    const int n_vocab  = (int)model->vocab.n_tokens();

    // Per-block probe: GIZMO_DUMP_BLOCK=/path/to/out.bin
    // Bisection probe: GIZMO_PROBE_PREFIXES="shard_attn_residual-,shard_attn_post_norm-,shard_ffn_out-,shard_l_out-"
    // sets up capture slots. GIZMO_PROBE_FILE_PREFIX="/tmp/" controls
    // output path. Each prefix gets its own binary file with the same
    // format as ref_per_layer_dump so the two can be diffed.
    init_probe_slots(n_tokens);

    // Clamp row_size to [1, n_layer]. With row_size >= n_layer the
    // whole model fits in one cgraph; with row_size == 1 we get
    // byte-exact per-block behavior (Phase 8 baseline).
    const int K = std::max(1, std::min(row_size > 0 ? row_size : 1, n_layer));
    const int n_rows = (n_layer + K - 1) / K;

    // Phase 10: hybrid models are no longer rejected upfront. The
    // per-block dispatcher in shard_block.cpp routes full-attention
    // layers through the qwen3.5 attn builder and SSM (recurrent /
    // gated delta net) layers through the Phase 12 GDN port of
    // build_layer_attn_linear. Phase 12 status: SSM blocks produce
    // numerical output close to the reference but with a small drift
    // that propagates through downstream layers; final argmax is not
    // yet identical to llama_decode. GIZMO_SSM_STUB=1 falls back to
    // the passthrough stub for diagnostic isolation.
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
    if (verbose && n_recr > 0) {
        std::fprintf(stderr,
            "[sharded] model has %d full-attention + %d recurrent (SSM) "
            "layers (first recurrent: layer %d). Phase 12: GDN builder "
            "active; argmax parity with llama_decode is in progress.\n",
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
    // Width is the number of KV cells visible to this batch. For a
    // prefill this equals n_tokens; for a decode step it equals the
    // full cache length (n_past + 1). The underlying KV context knows
    // that width after apply().
    int32_t n_kv_mask = n_tokens;
    if (mctx != nullptr) {
        auto * kv_ctx = get_kv_ctx(mctx);
        if (kv_ctx != nullptr) {
            n_kv_mask = (int32_t) kv_ctx->get_n_kv();
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
    // ne[1]=rows=n_tokens). Query at absolute position pos_first+i can
    // attend to KV cells at positions j <= pos_first+i. For a normal
    // prefill pos_first=0 and n_kv_mask=n_tokens, so this reduces to the
    // familiar upper-triangular mask. For a decode step (n_tokens=1,
    // pos_first=n_past, n_kv_mask=n_past+1) the single query sees the
    // entire cache history.
    {
        const uint16_t neg_inf_f16 = 0xFC00u;
        std::vector<uint16_t> mask_data((size_t)n_kv_mask * n_tokens, neg_inf_f16);
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv_mask && j <= pos_first + i; ++j) {
                mask_data[(size_t)i * n_kv_mask + j] = 0;
            }
        }
        ggml_backend_tensor_set(kq_mask, mask_data.data(), 0,
                                (size_t)n_kv_mask * n_tokens * sizeof(uint16_t));
    }

    // Fill positions and token_ids once. For MRoPE (qwen3.5/4-axis,
    // n_pos_per_embd == 4) the layout must be section-major, matching
    // llama-graph.cpp::llm_graph_input_pos::set_input:
    //   [pos_0..pos_N for dim0, pos_0..pos_N for dim1,
    //    pos_0..pos_N for dim2, 0..0 for dim3].
    // For plain RoPE (n_pos_per_embd == 1) this collapses to the
    // usual [pos_first, ..., pos_first+n_tokens-1].
    std::vector<int32_t> positions_data((size_t)n_tokens * n_pos_per_embd);
    for (uint32_t s = 0; s < n_pos_per_embd; ++s) {
        for (int i = 0; i < n_tokens; ++i) {
            // MRoPE last section is padding (zeros); plain RoPE has a
            // single section that must carry the real position.
            const bool is_last_section = (s == n_pos_per_embd - 1);
            if (n_pos_per_embd == 1) {
                positions_data[(size_t)s * n_tokens + i] = pos_first + i;
            } else {
                positions_data[(size_t)s * n_tokens + i] = is_last_section ? 0 : (pos_first + i);
            }
        }
    }
    ggml_backend_tensor_set(positions,    positions_data.data(), 0,
                            (size_t)n_tokens * n_pos_per_embd * sizeof(int32_t));
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

        // Per-block probe: install eval callback when any probe slot
        // is configured. The callback fires on tensors matching any
        // registered prefix — naming happens in the per-block builder
        // (e.g. shard_attn_residual-{il} in build_block_graph_into_attn_qwen35).
        if (!g_probe_slots.empty()) {
            ggml_backend_sched_set_eval_callback(sched, probe_eval_cb, nullptr);
        }

        ggml_backend_sched_reset(sched);

        // positions / token_ids_in / kq_mask / carrier are owned by
        // carrier_ctx (CPU-buft-backed, pre-filled). Their tensor
        // pointers are passed straight into build_block_graph_into;
        // ggml_set_input is idempotent so re-marking across rows is
        // harmless.

        ggml_tensor * k_idxs_row = nullptr;
        ggml_tensor * v_idxs_row = nullptr;
        llama_kv_cache_context * kv_ctx_row = get_kv_ctx(mctx);
        // k_idxs/v_idxs are only allocated by sched_alloc_graph if
        // some block in the row references them (full-attention blocks
        // wire them into kv_ctx->cpy_k/cpy_v via ggml_build_forward_expand).
        // SSM rows never touch them, so we skip construction there -
        // building them unconditionally and then calling
        // set_input_k_idxs on an unallocated buffer triggers
        // GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer)).
        // When GIZMO_FORCE_STUB=1 is set in shard_block.cpp, every block
        // is routed through the SSM stub regardless of is_recr(il). The
        // stub never references k_idxs/v_idxs in its cgraph, so sched_alloc_graph
        // will leave those tensor buffers null. Treating those rows as
        // "needs KV" would call set_input_k_idxs on a null buffer and trip
        // GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer)).
        static const int s_force_stub = []() {
            const char * e = std::getenv("GIZMO_FORCE_STUB");
            return (e && std::atoi(e) != 0) ? 1 : 0;
        }();
        bool row_needs_kv = false;
        if (!s_force_stub) {
            for (int j = 0; j < K_this; ++j) {
                if (!model->hparams.is_recr((uint32_t)(il_start + j))) {
                    row_needs_kv = true;
                    break;
                }
            }
        }
        if (kv_ctx_row != nullptr && row_needs_kv) {
            const llama_ubatch & ubatch = kv_ctx_row->get_ubatch();
            k_idxs_row = kv_ctx_row->build_input_k_idxs(row_ctx, ubatch);
            v_idxs_row = kv_ctx_row->build_input_v_idxs(row_ctx, ubatch);
            // set_input_k_idxs / set_input_v_idxs called AFTER alloc below.
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
                k_idxs_row, v_idxs_row, mctx);
            if (r == nullptr) {
                std::fprintf(stderr, "ERROR: build_block_graph_into failed for il=%d\n", il);
                ggml_free(row_ctx);
                return result;
            }
            // Per-block probe: tag the output of every block so the
            // scheduler's eval callback can identify it. Format mirrors
            // the reference probe's "l_out-{il}" naming, just with a
            // different prefix to avoid collisions.
            for (const auto & ps : g_probe_slots) {
                if (ps.prefix == "shard_l_out-") {
                    ggml_format_name(r, "shard_l_out-%d", il);
                }
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
        if (kv_ctx_row != nullptr && row_needs_kv && k_idxs_row != nullptr) {
            const llama_ubatch & ubatch = kv_ctx_row->get_ubatch();
            kv_ctx_row->set_input_k_idxs(k_idxs_row, &ubatch);
            kv_ctx_row->set_input_v_idxs(v_idxs_row, &ubatch);
        }

        const enum ggml_status ec = ggml_backend_sched_graph_compute(sched, row_gf);
        if (ec != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "ERROR: graph_compute failed for row %d status=%d\n",
                         row, (int)ec);
            ggml_free(row_ctx);
            return result;
        }

        // Copy the row's last block's residual into the carrier.  The
        // next row reads from the carrier, and the final tail graph
        // also reads from it, so we must always copy even for the last
        // row.  Previously the tail graph was accidentally using the
        // second-to-last row's residual on models with n_rows > 1.
        const size_t residual_bytes = (size_t)n_embd * n_tokens * sizeof(float);
        ggml_backend_tensor_get(out_residual, residual_staging.data(), 0, residual_bytes);
        ggml_backend_tensor_set(carrier, residual_staging.data(), 0, residual_bytes);

        // Optional per-row residual dump: prints the first 8 floats of the
        // last token's residual at the row boundary. Useful for localizing
        // where the sharded engine diverges from the reference. Off by
        // default; enable via GIZMO_DUMP_RESIDUAL=1.
        //
        // GIZMO_DUMP_RESIDUAL=2: full-residual L2 norm + checksum for
        // divergence detection across runs (same input -> same norm).
        static const int dump_env = []() {
            const char* e = std::getenv("GIZMO_DUMP_RESIDUAL");
            int v = (e && std::atoi(e) != 0) ? 1 : 0;
            if (e && (std::atoi(e) == 2)) v = 2;
            return v;
        }();
        if (verbose && dump_env) {
            static bool dump_env_announced = false;
            if (!dump_env_announced) {
                dump_env_announced = true;
                std::fprintf(stderr, "[sharded] GIZMO_DUMP_RESIDUAL=%d\n", dump_env);
            }
        }
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
            if (dump_env >= 2) {
                double sum2 = 0.0;
                uint32_t hash = 0u;
                for (int i = 0; i < n_embd; ++i) {
                    const float v = last_token_residual[i];
                    sum2 += (double)v * (double)v;
                    // FNV-1a over the 4 bytes of the float.
                    const uint8_t * bytes = (const uint8_t *)&v;
                    for (int b = 0; b < 4; ++b) {
                        hash ^= (uint32_t)bytes[b];
                        hash *= 0x01000193u;
                    }
                }
                std::printf("  L2=%.6f hash=%08x", std::sqrt(sum2), hash);
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

        if (verbose) {
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

    // Per-block probe: write captured tensors (one file per slot).
    for (const auto & ps : g_probe_slots) {
        write_probe_slot(ps, n_tokens);
    }
    g_probe_slots.clear();

    ggml_free(carrier_ctx);

    return result;
}
