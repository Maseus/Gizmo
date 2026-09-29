#include "context_validation.hpp"
#include "test_harness.hpp"

using namespace gizmo;

TEST(prompt_fits_context_basic) {
    ASSERT_TRUE(prompt_fits_context(1, 2));
    ASSERT_TRUE(prompt_fits_context(511, 512));
    ASSERT_FALSE(prompt_fits_context(512, 512)); // need 1 free slot
    ASSERT_FALSE(prompt_fits_context(513, 512));
    ASSERT_FALSE(prompt_fits_context(0, 512));   // non-positive prompt rejected
    ASSERT_FALSE(prompt_fits_context(10, 0));   // zero context rejected
    ASSERT_FALSE(prompt_fits_context(-5, 512)); // negative prompt rejected
}

TEST(clamp_generation_length_basic) {
    // 100 tokens in 128 context -> 28 free slots.
    ASSERT_EQ(clamp_generation_length(100, 128, 10), 10);
    ASSERT_EQ(clamp_generation_length(100, 128, 50), 28); // clamped to remaining
    ASSERT_EQ(clamp_generation_length(100, 128, 28), 28);

    // No room.
    ASSERT_EQ(clamp_generation_length(128, 128, 10), 0);
    ASSERT_EQ(clamp_generation_length(200, 128, 10), 0);

    // Edge cases.
    ASSERT_EQ(clamp_generation_length(127, 128, 100), 1); // exactly one slot
    ASSERT_EQ(clamp_generation_length(127, 128, 0), 0);  // non-positive max rejected
    ASSERT_EQ(clamp_generation_length(127, 128, -1), 0);
    ASSERT_EQ(clamp_generation_length(10, 0, 5), 0);    // zero context
    ASSERT_EQ(clamp_generation_length(-1, 128, 5), 0);  // negative prompt
}

TEST(clamp_generation_length_matches_generate_stream_policy) {
    // generate_stream rejects prompt equal to context and clamps max_tokens to
    // remaining slots. Verify the helper encodes exactly that policy.
    const int32_t n_ctx = 4096;
    const int32_t n_tokens = 4000;
    const int32_t slots_remaining = n_ctx - n_tokens;

    ASSERT_TRUE(prompt_fits_context(n_tokens, n_ctx));
    ASSERT_EQ(clamp_generation_length(n_tokens, n_ctx, 256), 96); // clamped
    ASSERT_EQ(clamp_generation_length(n_tokens, n_ctx, 96), 96);
    ASSERT_EQ(clamp_generation_length(n_tokens, n_ctx, 1), 1);

    // If max_tokens is huge we should still only fill the remaining slots.
    ASSERT_EQ(clamp_generation_length(n_tokens, n_ctx, INT32_MAX), slots_remaining);
}
