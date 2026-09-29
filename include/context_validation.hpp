#ifndef GIZMO_CONTEXT_VALIDATION_HPP
#define GIZMO_CONTEXT_VALIDATION_HPP

#include <cstdint>

namespace gizmo {

// Returns true if a prompt of `n_tokens` fits in a context of `n_ctx` with at
// least one slot left for generation. Non-positive values are rejected.
inline bool prompt_fits_context(int32_t n_tokens, int32_t n_ctx) {
    return n_ctx > 0 && n_tokens > 0 && n_tokens < n_ctx;
}

// Clamp the requested generation length to the number of free slots in the
// context window. Returns 0 if there is no room to generate anything (prompt
// does not fit, context is uninitialized, or max_tokens is non-positive).
inline int32_t clamp_generation_length(int32_t n_tokens, int32_t n_ctx, int32_t max_tokens) {
    if (n_ctx <= 0 || n_tokens < 0) {
        return 0;
    }
    if (n_tokens >= n_ctx) {
        return 0;
    }
    const int32_t slots_remaining = n_ctx - n_tokens;
    if (max_tokens <= 0) {
        return 0;
    }
    return max_tokens < slots_remaining ? max_tokens : slots_remaining;
}

} // namespace gizmo

#endif // GIZMO_CONTEXT_VALIDATION_HPP
