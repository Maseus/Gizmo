// Dump full last-token logits to a binary file: int32 n_vocab followed
// by float[n_vocab]. Used for offline diff between sharded engine and
// llama_decode baseline.

#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <prompt> <out.bin>\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* prompt = argv[2];
    const char* out_path = argv[3];

    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    // Disable CPU weight repacking so the baseline can load 27B models
    // without an extra in-RAM copy of every tensor.
    mparams.use_extra_bufts = false;
    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) { std::fprintf(stderr, "load model failed\n"); return 1; }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 256;
    cparams.n_batch = 64;
    cparams.n_ubatch = 64;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;
    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) { std::fprintf(stderr, "ctx init failed\n"); return 1; }

    std::vector<llama_token> toks(64);
    int n = llama_tokenize(llama_model_get_vocab(model), prompt, std::strlen(prompt),
                           toks.data(), toks.size(), true, true);
    if (n < 0) { std::fprintf(stderr, "tok fail\n"); return 1; }

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
        std::fprintf(stderr, "llama_decode failed\n");
        return 1;
    }

    const float* logits = llama_get_logits(ctx);
    const int n_vocab = (int)llama_vocab_n_tokens(llama_model_get_vocab(model));

    // Dump last column only.
    std::vector<float> last_col(n_vocab);
    std::memcpy(last_col.data(), &logits[(size_t)(n - 1) * n_vocab], n_vocab * sizeof(float));

    FILE* fp = std::fopen(out_path, "wb");
    if (!fp) { std::fprintf(stderr, "open %s failed\n", out_path); return 1; }
    int32_t nv = (int32_t)n_vocab;
    std::fwrite(&nv, sizeof(nv), 1, fp);
    std::fwrite(last_col.data(), sizeof(float), n_vocab, fp);
    std::fclose(fp);

    int best = 0; float bv = last_col[0];
    for (int i = 1; i < n_vocab; ++i) if (last_col[i] > bv) { bv = last_col[i]; best = i; }
    char piece_buf[256];
    int pn = llama_token_to_piece(llama_model_get_vocab(model), best,
                                   piece_buf, sizeof(piece_buf), 0, false);
    if (pn < 0) pn = 0;
    piece_buf[pn] = 0;
    std::printf("prompt: \"%s\" tokens=%d argmax_id=%d text=\"%s\"\n",
                prompt, n, best, piece_buf);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
