#include "chat_template.hpp"

#include <string>
#include <vector>

namespace gizmo {

std::string apply_chat_template(
    const llama_model* model,
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_assistant
) {
    if (!model || messages.empty()) {
        return "";
    }

    const char* tmpl = llama_model_chat_template(model, /*name=*/nullptr);
    std::vector<llama_chat_message> chat;
    chat.reserve(messages.size());
    for (const auto& m : messages) {
        chat.push_back({m.first.c_str(), m.second.c_str()});
    }

    std::string buf(4096, '\0');
    int32_t needed = llama_chat_apply_template(
        tmpl, chat.data(), chat.size(), add_assistant,
        buf.data(), static_cast<int32_t>(buf.size()));
    if (needed < 0) {
        return "";
    }
    if (needed > static_cast<int32_t>(buf.size())) {
        buf.resize(static_cast<size_t>(needed) + 1);
        needed = llama_chat_apply_template(
            tmpl, chat.data(), chat.size(), add_assistant,
            buf.data(), static_cast<int32_t>(buf.size()));
    }
    if (needed <= 0) {
        return "";
    }
    return std::string(buf.data(), static_cast<size_t>(needed));
}

} // namespace gizmo
