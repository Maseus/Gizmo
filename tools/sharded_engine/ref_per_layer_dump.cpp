// Reference per-layer residual dump.
//
// Runs llama_decode on the prompt, hooks ggml_backend_sched_set_eval_callback
// to copy the last-token column of each tagged tensor as soon as the
// scheduler has computed it (before the buffer gets reused by the next
// layer's op).
//
// Each tag name comes from the graph-level `cb(...)` calls in
// qwen35.cpp (e.g. "attn_norm", "attn_residual", "attn_post_norm",
// "ffn_out", "l_out", "attn_pregate", "attn_gated"). The context's
// graph_get_cb wrapper calls ggml_format_name(cur, "%s-%d", name, il)
// so the scheduler-fired eval callback can recover both the slot name
// and the layer index from t->name.
//
// Multiple slots are supported via comma-separated
// GIZMO_REF_TENSOR="attn_residual-,attn_post_norm-,ffn_out-,l_out-".
// Each slot writes its own output file
// `<GIZMO_PROBE_FILE_PREFIX><strip("shard_")><slot>-<il...>.bin` --
// filename format mirrors the sharded probe so diff_per_layer can
// compare them directly. Default slot prefix list and behavior (single
// prefix "l_out-") matches the original tool.
//
// Output format (binary):
//   header: int32[3] = {count, n_tokens, n_embd}
//   per-layer: int32 il, int32 n_embd, float[n_embd] (last-tok column)
//
// Usage:
//   ref_per_layer_dump <model.gguf> <prompt> <out.bin>

#include "llama.h"
#include "llama-context.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr int MAX_LAYERS = 256;
constexpr int MAX_SLOTS  = 16;

struct ProbeSlot {
    std::string                     prefix;   // e.g. "l_out-"
    std::string                     path;     // e.g. "/tmp/ref_l_out.bin"
    std::vector<std::vector<float>> cols;     // [count][n_embd]
    std::vector<int>                n_embd_for_il;
    int                             count = 0;
    int                             n_embd = 0;
};

std::vector<ProbeSlot> g_slots;
int                    g_n_tokens = 0;

std::string strip_trailing_dash(const std::string & s) {
    if (!s.empty() && s.back() == '-') return s.substr(0, s.size() - 1);
    return s;
}

// Mirror sharded probe file naming: drop a leading "shard_" so the
// filename stays unprefixed regardless of on-graph prefix.
std::string strip_leading_shard(const std::string & s) {
    constexpr const char * lead = "shard_";
    constexpr size_t llen = 6;
    if (s.size() > llen && std::strncmp(s.c_str(), lead, llen) == 0) {
        return s.substr(llen);
    }
    return s;
}

void init_slots(const char * out_path_default) {
    const char * e = std::getenv("GIZMO_REF_TENSOR");
    std::string env = (e && *e) ? e : "l_out-";
    const char * pe = std::getenv("GIZMO_PROBE_FILE_PREFIX");
    std::string file_prefix = (pe && *pe) ? pe : "/tmp/";

    size_t start = 0;
    while (start < env.size() && g_slots.size() < MAX_SLOTS) {
        size_t end = env.find(',', start);
        if (end == std::string::npos) end = env.size();
        std::string pref = env.substr(start, end - start);
        if (!pref.empty()) {
            ProbeSlot ps;
            ps.prefix = pref;
            const std::string & file_pref = (g_slots.size() == 0 && !std::getenv("GIZMO_REF_TENSOR"))
                ? std::string(out_path_default)
                : (file_prefix + strip_trailing_dash(strip_leading_shard(pref)) + ".bin");
            ps.path = file_pref;
            g_slots.push_back(std::move(ps));
        }
        start = end + 1;
    }
    for (const auto & ps : g_slots) {
        std::fprintf(stderr, "[ref] slot prefix='%s' file='%s'\n",
                     ps.prefix.c_str(), ps.path.c_str());
    }
}

// Skip captures for tensors whose name is `<prefix><il> (view)`. The
// qwen3.5 ref graph creates ggml_view_3d sub-tensors of Qcur_full
// (qwen35.cpp:275, :289) that inherit the auto-suffix
// "<src->name> (view)". They share the matmul-output prefix and
// would otherwise overwrite the slot's column capture with smaller
// buffers. User code that renames the view does not collide.
static bool prefix_match_no_view(const char * name, const std::string & prefix, size_t * match_len_out) {
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
    if (match_len_out) *match_len_out = plen;
    return true;
}

bool probe_eval_cb(ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (ask) return true;
    if (!t || t->name[0] == '\0') return true;
    if (g_slots.empty()) return true;
    // Find a matching slot. When multiple slots have nested prefixes
    // (e.g. "attn_post_norm-" is a prefix of nothing else in this
    // tool's usage, but we keep behavior symmetric with the sharded
    // probe), pick the LONGEST matching prefix to win.
    const ProbeSlot * slot = nullptr;
    size_t best_len = 0;
    for (const auto & ps : g_slots) {
        size_t ml = 0;
        if (prefix_match_no_view(t->name, ps.prefix, &ml)) {
            if (ml > best_len) {
                slot = &ps;
                best_len = ml;
            }
        }
    }
    if (slot == nullptr) return true;
    const char * dash = std::strrchr(t->name, '-');
    int il;
    if (!dash) {
        il = -1;
    } else {
        il = std::atoi(dash + 1);
        if (il < -1 || il >= MAX_LAYERS) return true;
    }

    // Mirror the sharded probe: stash in a slot-owned column.
    // Use a fixed offset for the tail tensors (il == -1) because the
    // storage vector is indexed by non-negative integers.
    const int store_idx = (il == -1) ? (MAX_LAYERS - 1) : il;
    ProbeSlot & s = const_cast<ProbeSlot &>(*slot);
    const int64_t n_embd = t->ne[0];
    const int64_t n_cols = t->ne[1] * t->ne[2] * t->ne[3];
    // GIZMO_PROBE_ALL_COLS=1 dumps all columns (default: last only).
    static const int s_all_cols = []() {
        const char * e = std::getenv("GIZMO_PROBE_ALL_COLS");
        return (e && std::atoi(e) != 0) ? 1 : 0;
    }();
    if ((int)s.cols.size() <= store_idx) s.cols.resize(store_idx + 1);
    if (s_all_cols) {
        const size_t total = (size_t)n_cols * (size_t)n_embd;
        if ((int)s.cols[store_idx].size() != (int)total) {
            s.cols[store_idx].assign(total, 0.0f);
        }
        for (int64_t c = 0; c < n_cols; ++c) {
            const size_t offset = (size_t) c * (size_t) n_embd * sizeof(float);
            const size_t size   = (size_t) n_embd * sizeof(float);
            ggml_backend_tensor_get(t,
                s.cols[store_idx].data() + c * n_embd,
                offset, size);
        }
        if (s.n_embd_for_il.size() <= (size_t)store_idx) s.n_embd_for_il.resize(store_idx + 1, 0);
        s.n_embd_for_il[store_idx] = (int)(n_cols * n_embd);
    } else {
        const int64_t last_col = n_cols - 1;
        const size_t offset = (size_t) last_col * (size_t) n_embd * sizeof(float);
        const size_t size   = (size_t) n_embd * sizeof(float);
        if ((int)s.cols[store_idx].size() != (int)n_embd) s.cols[store_idx].assign((size_t)n_embd, 0.0f);
        ggml_backend_tensor_get(t, s.cols[store_idx].data(), offset, size);
    }
    if (store_idx + 1 > s.count) s.count = store_idx + 1;
    if (s.n_embd == 0) s.n_embd = (int) n_embd;
    return true;
}

void write_slot(const ProbeSlot & s, int n_tokens) {
    int emitted = 0;
    for (const auto & c : s.cols) if (!c.empty()) ++emitted;
    if (emitted == 0) return;
    FILE * fp = std::fopen(s.path.c_str(), "wb");
    if (!fp) {
        std::fprintf(stderr, "ERROR: cannot open %s for ref probe write\n",
                     s.path.c_str());
        return;
    }
    int32_t header[3] = { emitted, n_tokens, s.n_embd };
    std::fwrite(header, sizeof(int32_t), 3, fp);
    static const int s_all_cols = []() {
        const char * e = std::getenv("GIZMO_PROBE_ALL_COLS");
        return (e && std::atoi(e) != 0) ? 1 : 0;
    }();
    std::vector<int> emitted_ils;
    for (int il = 0; il < (int)s.cols.size(); ++il) {
        if (!s.cols[il].empty()) emitted_ils.push_back(il);
    }
    for (int il : emitted_ils) {
        std::fwrite(&il, sizeof(int32_t), 1, fp);
        int32_t ne = s_all_cols ? s.n_embd_for_il[il] : s.n_embd;
        std::fwrite(&ne, sizeof(int32_t), 1, fp);
        std::fwrite(s.cols[il].data(), sizeof(float), (size_t)ne, fp);
    }
    std::fclose(fp);
    std::fprintf(stderr, "[ref] dumped %d entries to %s (n_embd=%d)\n",
                 emitted, s.path.c_str(), s.n_embd);
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> <prompt> <out.bin>\n"
            "  env GIZMO_REF_TENSOR: comma-separated tensor name prefixes "
            "(default 'l_out-')\n"
            "  env GIZMO_PROBE_FILE_PREFIX: file output prefix (default "
            "'/tmp/'). The slot name (with leading 'shard_' stripped and "
            "trailing '-' stripped) is appended.\n",
            argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* prompt = argv[2];
    const char* out_path = argv[3];

    init_slots(out_path);

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    // Disable CPU weight repacking so the reference dump can load 27B
    // models without an extra in-RAM copy of every tensor.
    mparams.use_extra_bufts = false;
    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) { std::fprintf(stderr, "load model failed\n"); return 1; }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 256;
    cparams.n_batch = 64;
    cparams.n_ubatch = 64;
    int n_threads = 1;
    if (const char * e = std::getenv("GIZMO_NTHREADS")) {
        int v = std::atoi(e);
        if (v > 0) n_threads = v;
    }
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.cb_eval = probe_eval_cb;
    cparams.cb_eval_user_data = nullptr;
    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) { std::fprintf(stderr, "ctx init failed\n"); return 1; }

    std::vector<llama_token> toks(64);
    int n = llama_tokenize(llama_model_get_vocab(model), prompt,
                           std::strlen(prompt),
                           toks.data(), toks.size(), true, true);
    if (n < 0) { std::fprintf(stderr, "tok fail\n"); return 1; }
    std::printf("[ref] prompt: \"%s\" n_tokens=%d\n", prompt, n);
    g_n_tokens = n;

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        batch.token[i] = toks[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "llama_decode failed\n"); return 1;
    }

    std::printf("[ref] captured %zu slots\n", g_slots.size());
    for (const auto & s : g_slots) {
        std::printf("  prefix=%s count=%d n_embd=%d\n",
                    s.prefix.c_str(), s.count, s.n_embd);
        write_slot(s, n);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
