// Reference argmax probe: runs llama_decode on a prompt and prints
// the argmax of the last-token logits. Used as the baseline for
// validating the sharded engine output.

#include "llama.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <prompt>\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* prompt = argv[2];

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
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
        batch.logits[i] = (i == n - 1) ? 1 : 0;
    }
    batch.n_tokens = n;

    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "llama_decode failed\n");
        return 1;
    }

    const float* logits = llama_get_logits(ctx);
    const int n_vocab = (int)llama_vocab_n_tokens(llama_model_get_vocab(model));
    int best_id = 0;
    float best_val = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_val) { best_val = logits[i]; best_id = i; }
    }
    std::printf("prompt: \"%s\" -> %d tokens\n", prompt, n);
    for (int i = 0; i < n; ++i) std::printf("  [%d] %d\n", i, toks[i]);
    char piece_buf[256];
    int pn = llama_token_to_piece(llama_model_get_vocab(model), best_id,
                                   piece_buf, sizeof(piece_buf), 0, false);
    if (pn < 0) pn = 0; piece_buf[pn] = 0;
    std::printf("argmax: id=%d logit=%.4f text=\"%s\"\n", best_id, best_val, piece_buf);

    // Top-K for context.
    std::vector<std::pair<float,int>> scored;
    for (int i = 0; i < n_vocab; ++i) scored.push_back({logits[i], i});
    std::partial_sort(scored.begin(), scored.begin() + 8, scored.end(),
                      [](auto& a, auto& b){ return a.first > b.first; });
    std::printf("top-8:\n");
    for (int i = 0; i < 8; ++i) {
        int id = scored[i].second;
        pn = llama_token_to_piece(llama_model_get_vocab(model), id,
                                   piece_buf, sizeof(piece_buf), 0, false);
        if (pn < 0) pn = 0; piece_buf[pn] = 0;
        std::printf("  [%d] id=%d logit=%.4f text=\"%s\"\n",
                    i, id, scored[i].first, piece_buf);
    }
    std::printf("logits[last] first 16:");
    for (int i = 0; i < 16; ++i) std::printf(" %.4f", logits[i]);
    std::printf("\n");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
