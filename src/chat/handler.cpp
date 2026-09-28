#include "chat_handler.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace gizmo {

ChatHandler::ChatHandler()
    : system_prompt_("")
    , max_history_size_(100) {
}

ChatHandler::~ChatHandler() {
}

void ChatHandler::start_session(const std::string& system_prompt) {
    system_prompt_ = system_prompt;
    history_.clear();

    if (!system_prompt.empty()) {
        add_message("system", system_prompt);
    }
}

void ChatHandler::add_message(const std::string& role, const std::string& content) {
    Message msg;
    msg.role = role;
    msg.content = content;
    history_.push_back(msg);

    // Trim history if it exceeds max size
    if (history_.size() > max_history_size_) {
        // Keep system message at front, then remove oldest user/assistant messages
        size_t start_index = system_prompt_.empty() ? 0 : 1;
        size_t to_remove = history_.size() - max_history_size_;
        history_.erase(history_.begin() + start_index,
                       history_.begin() + start_index + to_remove);
    }
}

std::string ChatHandler::get_formatted_history() const {
    std::ostringstream oss;

    for (const auto& msg : history_) {
        if (msg.role == "system") {
            oss << "System: " << msg.content << "\n";
        } else if (msg.role == "user") {
            oss << "User: " << msg.content << "\n";
        } else if (msg.role == "assistant") {
            oss << "Assistant: " << msg.content << "\n";
        }
    }

    return oss.str();
}

std::deque<Message> ChatHandler::get_recent_messages(size_t count) const {
    std::deque<Message> recent;
    size_t start = history_.size() > count ? history_.size() - count : 0;

    for (size_t i = start; i < history_.size(); ++i) {
        recent.push_back(history_[i]);
    }

    return recent;
}

void ChatHandler::clear_history() {
    history_.clear();
    if (!system_prompt_.empty()) {
        add_message("system", system_prompt_);
    }
}

bool ChatHandler::save_to_file(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    // Simple format: role|content per line
    for (const auto& msg : history_) {
        file << msg.role << "|" << msg.content << "\n";
    }

    return true;
}

bool ChatHandler::load_from_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    history_.clear();
    system_prompt_.clear();

    std::string line;
    while (std::getline(file, line)) {
        size_t delim = line.find('|');
        if (delim != std::string::npos) {
            Message msg;
            msg.role = line.substr(0, delim);
            msg.content = line.substr(delim + 1);
            history_.push_back(msg);

            if (msg.role == "system") {
                system_prompt_ = msg.content;
            }
        }
    }

    return true;
}

ChatPreset ChatHandler::preset_default() {
    ChatPreset preset;
    preset.name = "Default";
    preset.system_prompt = "You are a helpful assistant.";
    preset.temperature = 0.8f;
    preset.max_tokens = 256;
    return preset;
}

ChatPreset ChatHandler::preset_coding() {
    ChatPreset preset;
    preset.name = "Coding Assistant";
    preset.system_prompt = "You are an expert programmer. Provide clear, well-commented code examples. Explain your reasoning.";
    preset.temperature = 0.2f;
    preset.max_tokens = 512;
    return preset;
}

ChatPreset ChatHandler::preset_creative() {
    ChatPreset preset;
    preset.name = "Creative Writing";
    preset.system_prompt = "You are a creative writing assistant. Help with storytelling, poetry, and imaginative content.";
    preset.temperature = 1.0f;
    preset.max_tokens = 384;
    return preset;
}

} // namespace gizmo
