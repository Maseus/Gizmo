#ifndef GIZMO_CHAT_TEMPLATE_HPP
#define GIZMO_CHAT_TEMPLATE_HPP

#include "llama.h"

#include <string>
#include <utility>
#include <vector>

namespace gizmo {

// Apply the loaded model's built-in chat template to a list of messages.
// Returns an empty string if no template is available or application fails.
// `messages` is a list of (role, content) pairs (e.g. "system", "user",
// "assistant").  When `add_assistant` is true the formatted string ends
// with the assistant prefix ready for generation.
std::string apply_chat_template(
    const llama_model* model,
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_assistant);

} // namespace gizmo

#endif // GIZMO_CHAT_TEMPLATE_HPP
