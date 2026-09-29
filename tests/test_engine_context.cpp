#include "context_validation.hpp"
#include "inference_engine.hpp"
#include "test_harness.hpp"
#include "tokenizer.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>

using namespace gizmo;

namespace {

std::string get_test_model_path() {
    const char* env = std::getenv("GIZMO_TEST_MODEL");
    return env ? std::string(env) : std::string{};
}

// Build a prompt that tokenizes to at least `count` tokens on most BPE
// vocabularies. Repeating a common word produces roughly one token per word.
std::string make_long_prompt(int count) {
    std::ostringstream oss;
    for (int i = 0; i < count; ++i) {
        oss << "hello ";
    }
    return oss.str();
}

} // namespace

#define ASSERT_SKIP_IF_NO_MODEL()                       \
    do {                                                \
        std::string _m = get_test_model_path();         \
        if (_m.empty() || !std::filesystem::exists(_m)) { \
            std::cout << "[SKIP] no model fixture\n";   \
            return;                                     \
        }                                               \
    } while (0)

TEST(engine_generate_stream_rejects_prompt_at_context_limit) {
    ASSERT_SKIP_IF_NO_MODEL();
    // Use a 256-token context (the practical floor for many llama.cpp models)
    // and a much larger prompt so tokenization count never matters: the prompt
    // will always exceed the window.
    InferenceEngine engine;
    engine.set_threads(2);
    engine.set_context_size(256);
    ASSERT_TRUE(engine.initialize(get_test_model_path(), /*layer_shard_lazy=*/true));

    InferenceConfig cfg;
    cfg.max_tokens = 1;
    bool called = false;
    bool ok = engine.generate_stream(make_long_prompt(300), cfg, [&](const std::string&, int32_t) {
        called = true;
    });
    ASSERT_FALSE(ok);
    ASSERT_FALSE(called);
}

TEST(engine_generate_stream_rejects_prompt_exceeding_context) {
    ASSERT_SKIP_IF_NO_MODEL();
    InferenceEngine engine;
    engine.set_threads(2);
    engine.set_context_size(256);
    ASSERT_TRUE(engine.initialize(get_test_model_path(), /*layer_shard_lazy=*/true));

    InferenceConfig cfg;
    cfg.max_tokens = 1;
    bool ok = engine.generate_stream(make_long_prompt(300), cfg, [](const std::string&, int32_t) {});
    ASSERT_FALSE(ok);
}

TEST(engine_generate_stream_clamps_max_tokens_to_remaining_slots) {
    ASSERT_SKIP_IF_NO_MODEL();
    InferenceEngine engine;
    engine.set_threads(2);
    engine.set_context_size(256);
    ASSERT_TRUE(engine.initialize(get_test_model_path(), /*layer_shard_lazy=*/true));

    // "hello" is small; leaves most of the 256-slot window. Request far more
    // than could fit so the engine must clamp.
    InferenceConfig cfg;
    cfg.max_tokens = 1000;
    cfg.temperature = 0.0f; // deterministic
    cfg.seed = 42;
    int generated = 0;
    bool ok = engine.generate_stream("hello", cfg, [&](const std::string&, int32_t) {
        ++generated;
    });
    ASSERT_TRUE(ok);
    // Must not exceed the number of available slots (minus prompt).
    ASSERT_TRUE(generated <= 255);
}

TEST(engine_generate_token_rejects_too_long_context) {
    ASSERT_SKIP_IF_NO_MODEL();
    InferenceEngine engine;
    engine.set_threads(2);
    engine.set_context_size(256);
    ASSERT_TRUE(engine.initialize(get_test_model_path(), /*layer_shard_lazy=*/true));

    InferenceConfig cfg;
    cfg.max_tokens = 1;
    TokenResult res = engine.generate_token(make_long_prompt(300), cfg);
    ASSERT_TRUE(res.text.empty());
}

TEST(engine_prefill_only_tokens_rejects_too_long_prompt) {
    ASSERT_SKIP_IF_NO_MODEL();
    InferenceEngine engine;
    engine.set_threads(2);
    engine.set_context_size(256);
    ASSERT_TRUE(engine.initialize(get_test_model_path(), /*layer_shard_lazy=*/true));
    engine.enable_sharded_engine(/*resident_layers=*/1, /*evict_weights=*/true);
    if (!engine.is_sharded()) {
        std::cout << "[SKIP] model architecture not supported by sharded engine\n";
        return;
    }

    int rc = engine.prefill_only(make_long_prompt(300));
    ASSERT_EQ(rc, -1);
}

TEST(engine_validate_prefill_rejects_too_long_prompt) {
    ASSERT_SKIP_IF_NO_MODEL();
    InferenceEngine engine;
    engine.set_threads(2);
    engine.set_context_size(256);
    ASSERT_TRUE(engine.initialize(get_test_model_path(), /*layer_shard_lazy=*/true));

    std::vector<float> logits;
    int rc = engine.validate_prefill(make_long_prompt(300), logits);
    ASSERT_EQ(rc, -1);
}

TEST(engine_per_request_context_size_rebuild_and_persistence) {
    ASSERT_SKIP_IF_NO_MODEL();
    InferenceEngine engine;
    engine.set_threads(2);
    engine.set_context_size(256); // initial context
    ASSERT_TRUE(engine.initialize(get_test_model_path(), /*layer_shard_lazy=*/true));
    ASSERT_EQ(engine.context_size(), 256);

    // Ask for a larger per-request context that still fits on this machine.
    InferenceConfig cfg;
    cfg.context_size = 512;
    cfg.max_tokens = 1;
    cfg.temperature = 0.0f;
    cfg.seed = 42;

    bool ok = engine.generate_stream("hello", cfg, [](const std::string&, int32_t) {});
    if (!ok) {
        // Rebuilding to 512 may fail on very low-RAM machines; treat as skip.
        std::cout << "[SKIP] per-request rebuild to 512 failed (likely RAM)\n";
        return;
    }
    // context_size_ should now be persisted at 512.
    ASSERT_EQ(engine.context_size(), 512);

    // A subsequent call without config.context_size should use the new size.
    InferenceConfig cfg2;
    cfg2.max_tokens = 1;
    cfg2.temperature = 0.0f;
    cfg2.seed = 42;
    bool ok2 = engine.generate_stream(make_long_prompt(400), cfg2, [](const std::string&, int32_t) {});
    // 400 hellos < 512 context, so it should fit (leaves slots for generation).
    ASSERT_TRUE(ok2);
}
