#include "tokenizer.hpp"

namespace gizmo {

std::vector<llama_token> tokenize_text(
    const llama_vocab* vocab,
    const std::string& text,
    bool add_special,
    bool parse_special,
    int32_t* out_n
) {
    std::vector<llama_token> tokens;
    if (out_n != nullptr) *out_n = 0;
    if (!vocab) {
        return tokens;
    }

    int32_t n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                               nullptr, 0, add_special, parse_special);
    if (n == 0) {
        if (out_n != nullptr) *out_n = 0;
        return tokens;
    }
    if (n < 0) {
        // llama_tokenize returns the negative required count when the output
        // buffer (including a zero-length nullptr buffer) is too small.
        n = -n;
    }

    tokens.resize(static_cast<size_t>(n));
    int32_t n_check = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                                    tokens.data(), n, add_special, parse_special);
    if (n_check < 0) {
        if (out_n != nullptr) *out_n = n_check;
        tokens.clear();
        return tokens;
    }
    if (out_n != nullptr) *out_n = n_check;
    return tokens;
}

std::string token_to_piece(
    const llama_vocab* vocab,
    llama_token token,
    bool special
) {
    if (!vocab) {
        return "";
    }
    char buf[256];
    int32_t len = llama_token_to_piece(vocab, token, buf, sizeof(buf), /*lstrip=*/0, special);
    if (len < 0) {
        std::vector<char> big(-len + 1, '\0');
        len = llama_token_to_piece(vocab, token, big.data(),
                                   static_cast<int32_t>(big.size()), /*lstrip=*/0, special);
        if (len > 0) {
            return std::string(big.data(), static_cast<size_t>(len));
        }
    } else if (len > 0) {
        return std::string(buf, static_cast<size_t>(len));
    }
    return "";
}

std::string tokens_to_string(
    const llama_vocab* vocab,
    const std::vector<llama_token>& tokens,
    bool special
) {
    if (!vocab || tokens.empty()) {
        return "";
    }
    std::string result;
    for (llama_token t : tokens) {
        result += token_to_piece(vocab, t, special);
    }
    return result;
}

} // namespace gizmo
