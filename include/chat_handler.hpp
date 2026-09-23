#ifndef GIZMO_CHAT_HANDLER_HPP
#define GIZMO_CHAT_HANDLER_HPP

#include <string>
#include <vector>
#include <deque>

namespace gizmo {

struct Message {
    std::string role;  // "user", "assistant", "system"
    std::string content;
};

struct ChatPreset {
    std::string name;
    std::string system_prompt;
    float temperature;
    int32_t max_tokens;
};

class ChatHandler {
public:
    ChatHandler();
    ~ChatHandler();

    // Start a new chat session
    void start_session(const std::string& system_prompt = "");

    // Add a message to history
    void add_message(const std::string& role, const std::string& content);

    // Get formatted chat history for prompt
    std::string get_formatted_history() const;

    // Get recent messages (for context window)
    std::deque<Message> get_recent_messages(size_t count) const;

    // Clear chat history
    void clear_history();

    // Save chat to file
    bool save_to_file(const std::string& path) const;

    // Load chat from file
    bool load_from_file(const std::string& path);

    // Built-in presets
    static ChatPreset preset_default();
    static ChatPreset preset_coding();
    static ChatPreset preset_creative();

private:
    std::vector<Message> history_;
    std::string system_prompt_;
    size_t max_history_size_;
};

} // namespace gizmo

#endif // GIZMO_CHAT_HANDLER_HPP
