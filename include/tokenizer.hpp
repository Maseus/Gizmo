#ifndef GIZMO_TOKENIZER_HPP
#define GIZMO_TOKENIZER_HPP

#include <string>
#include <vector>
#include <cstdint>

#include "llama.h"

namespace gizmo {

// Two-pass tokenization helper: returns the exact token vector for `text`,
// resizing automatically when a fixed buffer would overflow. Returns an empty
// vector and sets *out_n negative on error.
std::vector<llama_token> tokenize_text(
    const llama_vocab* vocab,
    const std::string& text,
    bool add_special,
    bool parse_special,
    int32_t* out_n = nullptr
);

// Detokenize a single token into a UTF-8 string piece. Returns an empty string
// on error.
std::string token_to_piece(
    const llama_vocab* vocab,
    llama_token token,
    bool special = false
);

// Detokenize a token sequence into a UTF-8 string. Returns an empty string on
// error. This is useful when a caller needs to turn a controlled token array
// back into text for the engine's text-based API.
std::string tokens_to_string(
    const llama_vocab* vocab,
    const std::vector<llama_token>& tokens,
    bool special = false
);

} // namespace gizmo

#endif // GIZMO_TOKENIZER_HPP
