// Reference residual dump: runs llama_decode and prints the
// intermediate residual stream (not exposed by llama_decode itself,
// so we use the public API: read final logits and verify).
// Actually the residual stream is NOT exposed publicly; instead
// we read all 5 input tokens' positions and verify logits match.
// This is just a placeholder for now.
#include "llama.h"
#include <cstdio>
int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <prompt>\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto* m = llama_model_load_from_file(argv[1], mp);
    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 64;
    cp.n_ubatch = 64;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    auto* ctx = llama_init_from_model(m, cp);
    auto* v = llama_model_get_vocab(m);

    std::vector<llama_token> toks(64);
    int n = llama_tokenize(v, argv[2], std::strlen(argv[2]),
                           toks.data(), toks.size(), true, true);
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        batch.token[i] = toks[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = n;
    llama_decode(ctx, batch);

    // Public API does not expose intermediate residual.
    // Print final logits first 16 + argmax.
    const float* logits = llama_get_logits(ctx);
    int nv = llama_vocab_n_tokens(v);
    std::printf("reference_logits_first_16:");
    for (int i = 0; i < 16; ++i) std::printf(" %.4f", logits[i]);
    std::printf("\n");

    int best = 0; float bv = logits[0];
    for (int i = 1; i < nv; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
    char pb[64]; int pn = llama_token_to_piece(v, best, pb, sizeof(pb), 0, false);
    if (pn < 0) pn = 0; pb[pn] = 0;
    std::printf("argmax=%d text=\"%s\"\n", best, pb);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(m);
    llama_backend_free();
    return 0;
}
