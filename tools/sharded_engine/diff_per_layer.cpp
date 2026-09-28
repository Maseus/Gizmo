// Side-by-side diff of two per-layer residual dump files.
//
// Format (binary, identical to ref_per_layer_dump and the sharded
// engine's GIZMO_DUMP_BLOCK output):
//   header: int32[3] = {count, n_tokens, n_embd}
//   per-layer: int32 il, int32 n_embd, float[n_embd] (last-tok column)
//
// Unlike ref_per_layer_dump (which emits one row per layer starting
// from il=0), the sharded probe stores rows only for layers whose
// tensor was actually computed (e.g. full-attn layers with il in
// {3,7,11,...}). Rows are addressed by the in-file il value. Where a
// row is missing from one side, we report it as MISSING rather than
// silently zeroing it. Used for bisecting qwen3.5 SSM drift between
// the sharded engine and the reference.
//
// Usage:
//   diff_per_layer <sharded.bin> <reference.bin> [tolerance]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

struct DumpFile {
    int32_t count    = 0;
    int32_t n_tokens = 0;
    int32_t n_embd   = 0;
    // key = il (in-file), value = column data
    std::map<int32_t, std::vector<float>> cols;
};

static bool load_dump(const char * path, DumpFile & df, const char * tag) {
    FILE * fp = std::fopen(path, "rb");
    if (!fp) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", path);
        return false;
    }
    int32_t header[3];
    if (std::fread(header, sizeof(int32_t), 3, fp) != 3) {
        std::fprintf(stderr, "ERROR: %s: short header\n", path);
        std::fclose(fp); return false;
    }
    df.count    = header[0];
    df.n_tokens = header[1];
    df.n_embd   = header[2];
    std::fprintf(stderr, "[%s] %s count=%d n_tokens=%d n_embd=%d\n",
                 tag, path, df.count, df.n_tokens, df.n_embd);
    for (int row = 0; row < df.count; ++row) {
        int32_t il_read, ne_read;
        if (std::fread(&il_read, sizeof(int32_t), 1, fp) != 1) {
            std::fprintf(stderr, "ERROR: %s: short il at row=%d\n",
                         path, row);
            std::fclose(fp); return false;
        }
        if (std::fread(&ne_read, sizeof(int32_t), 1, fp) != 1) {
            std::fprintf(stderr, "ERROR: %s: short ne at row=%d\n",
                         path, row);
            std::fclose(fp); return false;
        }
        if (ne_read != df.n_embd) {
            std::fprintf(stderr, "ERROR: %s: ne mismatch at row=%d "
                         "(got ne=%d want %d)\n",
                         path, row, ne_read, df.n_embd);
            std::fclose(fp); return false;
        }
        std::vector<float> col((size_t)df.n_embd);
        if (std::fread(col.data(), sizeof(float), df.n_embd, fp)
                != (size_t)df.n_embd) {
            std::fprintf(stderr, "ERROR: %s: short data at row=%d\n",
                         path, row);
            std::fclose(fp); return false;
        }
        df.cols[il_read] = std::move(col);
    }
    std::fclose(fp);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr,
            "usage: %s <sharded.bin> <reference.bin> [tolerance]\n", argv[0]);
        return 1;
    }
    const char * shard_path = argv[1];
    const char * ref_path   = argv[2];
    double tol = (argc == 4) ? std::atof(argv[3]) : 1e-2;

    DumpFile shard, ref;
    if (!load_dump(shard_path, shard, "shard")) return 1;
    if (!load_dump(ref_path,   ref,   "ref"))   return 1;

    if (shard.n_embd != ref.n_embd) {
        std::fprintf(stderr, "ERROR: n_embd mismatch shard=%d ref=%d\n",
                     shard.n_embd, ref.n_embd);
        return 1;
    }

    // Union of layer indices present in either file.
    int32_t min_il = INT32_MAX, max_il = INT32_MIN;
    for (auto const & kv : shard.cols) {
        min_il = std::min(min_il, kv.first);
        max_il = std::max(max_il, kv.first);
    }
    for (auto const & kv : ref.cols) {
        min_il = std::min(min_il, kv.first);
        max_il = std::max(max_il, kv.first);
    }
    if (min_il == INT32_MAX) {
        std::fprintf(stderr, "ERROR: no rows in either file\n");
        return 1;
    }

    int first_bad = -1;
    int worst_il  = -1;
    double worst_l2 = 0.0;
    std::printf("il  L2(shard)        L2(ref)          L2(diff)         "
                "max_abs_diff  status\n");
    for (int32_t il = min_il; il <= max_il; ++il) {
        bool in_shard = shard.cols.count(il) > 0;
        bool in_ref   = ref.cols.count(il) > 0;
        if (!in_shard && !in_ref) continue;
        if (!in_shard || !in_ref) {
            std::printf("%2d  %14s  %14s  %14s  %12s  %s\n",
                        il, "---", "---", "---", "---",
                        (in_shard ? "MISSING_REF" : "MISSING_SHARD"));
            continue;
        }
        const std::vector<float> & sv = shard.cols[il];
        const std::vector<float> & rv = ref.cols[il];
        double l2_sh = 0.0, l2_rf = 0.0, l2_df = 0.0, max_ad = 0.0;
        for (int i = 0; i < shard.n_embd; ++i) {
            const double s = sv[i];
            const double r = rv[i];
            const double d = s - r;
            l2_sh += s * s;
            l2_rf += r * r;
            l2_df += d * d;
            const double ad = d < 0 ? -d : d;
            if (ad > max_ad) max_ad = ad;
        }
        l2_sh = std::sqrt(l2_sh);
        l2_rf = std::sqrt(l2_rf);
        l2_df = std::sqrt(l2_df);
        const bool bad = l2_df > tol;
        if (bad && first_bad < 0) first_bad = il;
        if (l2_df > worst_l2) { worst_l2 = l2_df; worst_il = il; }
        std::printf("%2d  %14.6f  %14.6f  %14.6f  %12.6f  %s\n",
                    il, l2_sh, l2_rf, l2_df, max_ad,
                    bad ? "BAD" : "ok");
    }
    std::printf("\nfirst bad il (tol=%g): %s\n",
                tol, first_bad >= 0
                    ? (std::to_string(first_bad).c_str())
                    : "(none)");
    std::printf("worst  il: %d  L2(diff)=%.6f\n", worst_il, worst_l2);
    return first_bad >= 0 ? 2 : 0;
}
