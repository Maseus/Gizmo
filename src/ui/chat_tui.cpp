#include "chat_tui.hpp"

#include "chat_template.hpp"
#include "inference_engine.hpp"
#include "model_discovery.hpp"
#include "proc_status.hpp"
#include "terminal_utils.hpp"
#include "util/string.hpp"

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace gizmo {

namespace {

// Ask for a custom GGUF path on the terminal (restores canonical mode).
std::string ask_custom_path() {
    show_cursor();
    clear_screen();
    std::cout << "\x1b[1;36mCustom model path\x1b[0m\n\n";
    std::cout << "Enter the full path to a .gguf file:\n> ";
    std::cout.flush();
    std::string path;
    if (!std::getline(std::cin, path)) return "";
    return path;
}

} // namespace

int run_chat_tui(const std::string& model_path,
                 const std::string& model_path_extra,
                 int32_t max_tokens,
                 int32_t threads) {
    const bool is_tty = isatty(STDIN_FILENO) && isatty(STDOUT_FILENO);

    // Resolve model path.
    std::string path = model_path;
    if (path.empty()) {
        auto extra_dirs = parse_colon_dirs(model_path_extra);
        auto models = discover_models(extra_dirs);
        // Append a custom-path option.
        DiscoveredModel custom;
        custom.source = "custom";
        custom.name = "<Enter custom GGUF path>";
        custom.path = "";
        custom.size_bytes = 0;
        models.push_back(custom);

        if (models.empty()) {
            path = ask_custom_path();
        } else {
            path = pick_model_interactive(models);
            if (path.empty()) {
                size_t custom_idx = models.size() - 1;
                if (custom_idx < models.size() && models[custom_idx].source == "custom") {
                    path = ask_custom_path();
                }
            }
        }
    }

    if (path.empty()) {
        std::cerr << "No model selected.\n";
        return 1;
    }

    // Load engine with a low-memory default profile.
    InferenceEngine engine;
    engine.set_threads(threads > 0 ? threads : 4);
    if (!engine.initialize(path, /*layer_shard_lazy=*/true)) {
        std::cerr << "Failed to initialize inference engine for: " << path << "\n";
        return 1;
    }
    engine.enable_sharded_engine(
        /*resident_layers=*/1,
        /*evict_weights=*/true,
        /*row_size=*/1);
    if (!engine.is_sharded()) {
        std::cout << "Note: sharded engine was not enabled; falling back to llama_decode.\n";
    }

    const struct llama_vocab* vocab = llama_model_get_vocab(engine.raw_model());

    // Conversation history.
    std::vector<std::pair<std::string, std::string>> messages;
    messages.push_back({"system", "You are a helpful assistant."});

    InferenceConfig cfg;
    cfg.max_tokens = max_tokens > 0 ? max_tokens : 128;
    cfg.temperature = 0.8f;

    if (!is_tty) {
        // Non-TTY fallback: simple line-based chat (same as gizmo chat).
        std::cout << "\nGizmo Chat: " << path << "\n";
        std::cout << "Type a message and press Enter. Commands: /quit, /reset, /clear\n";
        std::cout << "---------------------------------------------------------------\n";
        std::string input;
        while (true) {
            std::cout << "\nYou: ";
            std::cout.flush();
            if (!std::getline(std::cin, input)) break;
            if (input == "/quit" || input == "/exit") break;
            if (input == "/reset" || input == "/clear") {
                messages.clear();
                messages.push_back({"system", "You are a helpful assistant."});
                engine.reset_for_next_run();
                std::cout << "[chat] context cleared\n";
                continue;
            }
            if (input.empty()) continue;

            messages.push_back({"user", input});
            std::string prompt = gizmo::apply_chat_template(engine.raw_model(), messages, /*add_ass=*/true);
            if (prompt.empty()) {
                prompt.clear();
                for (const auto& m : messages) {
                    prompt += m.first + ": " + m.second + "\n\n";
                }
                prompt += "assistant: ";
            }

            std::cout << "Assistant: ";
            std::cout.flush();
            engine.reset_for_next_run();
            const auto t0 = std::chrono::steady_clock::now();
            std::string response = engine.generate(prompt, cfg, /*quiet=*/true);
            const auto t1 = std::chrono::steady_clock::now();
            messages.push_back({"assistant", response});
            std::cout << response;

            std::vector<llama_token> response_toks(response.size() + 16);
            int n_response_tokens = llama_tokenize(
                vocab, response.c_str(), response.size(),
                response_toks.data(), response_toks.size(),
                /*add_special=*/false, /*parse_special=*/false);
            if (n_response_tokens < 0) n_response_tokens = 0;

            const double wall_s = std::chrono::duration<double>(t1 - t0).count();
            const double tok_s = wall_s > 0.0 ? n_response_tokens / wall_s : 0.0;
            const size_t rss = read_vm_rss_bytes();
            const size_t hwm = read_vm_hwm_bytes();

            std::cout << "\n  [" << n_response_tokens << " tokens in "
                      << std::fixed << std::setprecision(2) << wall_s << " s => "
                      << std::setprecision(1) << tok_s << " tok/s; RSS "
                      << (rss / (1024 * 1024)) << " MB; HWM "
                      << (hwm / (1024 * 1024)) << " MB]\n";
        }
        std::cout << "\nChat ended.\n";
        return 0;
    }

    // TTY interactive chat TUI.
    RawMode raw;
    if (!raw.enable()) {
        std::cerr << "Failed to enable raw terminal mode.\n";
        return 1;
    }
    CursorGuard cursor;
    cursor.hide();

    std::string input;
    std::string current_response;
    bool generating = false;

    auto draw_header = [&]() {
        move_cursor(1, 1);
        clear_line();
        std::cout << "\x1b[1;36mGizmo Chat\x1b[0m  " << engine.get_model_info()
                  << "  |  /quit, /reset, /clear";
    };

    auto draw_history = [&]() {
        // Print all previous turns starting from row 3.
        int row = 3;
        for (size_t i = 0; i < messages.size(); ++i) {
            if (messages[i].first == "system") continue;
            move_cursor(row++, 1);
            clear_line();
            if (messages[i].first == "user") {
                std::cout << "\x1b[1;32mYou:\x1b[0m " << messages[i].second;
            } else if (messages[i].first == "assistant") {
                std::cout << "\x1b[1;34mAssistant:\x1b[0m " << messages[i].second;
            }
        }
    };

    auto draw_input_line = [&]() {
        move_cursor(22, 1);
        clear_line();
        std::cout << "\x1b[1;32mYou:\x1b[0m " << input;
    };

    auto draw_status = [&](const std::string& text) {
        move_cursor(24, 1);
        clear_line();
        std::cout << "\x1b[2m" << text << "\x1b[0m";
        std::cout.flush();
    };

    auto draw_current_assistant = [&]() {
        move_cursor(23, 1);
        clear_line();
        std::cout << "\x1b[1;34mAssistant:\x1b[0m " << current_response;
    };

    auto redraw = [&]() {
        clear_screen();
        draw_header();
        draw_history();
        if (generating) {
            draw_current_assistant();
        }
        draw_input_line();
        std::cout.flush();
    };

    redraw();

    auto generate_reply = [&]() {
        if (input.empty()) return;
        messages.push_back({"user", input});
        input.clear();
        redraw();

        std::string prompt = apply_chat_template(engine.raw_model(), messages, /*add_ass=*/true);
        if (prompt.empty()) {
            prompt.clear();
            for (const auto& m : messages) {
                prompt += m.first + ": " + m.second + "\n\n";
            }
            prompt += "assistant: ";
        }

        engine.reset_for_next_run();
        current_response.clear();
        generating = true;
        redraw();

        const auto t0 = std::chrono::steady_clock::now();
        int token_count = 0;

        auto on_token = [&](const std::string& token_text, int32_t /*token_id*/) {
            current_response += token_text;
            ++token_count;
            const auto t1 = std::chrono::steady_clock::now();
            const double wall_s = std::chrono::duration<double>(t1 - t0).count();
            const double tok_s = wall_s > 0.0 ? token_count / wall_s : 0.0;
            const size_t rss = read_vm_rss_bytes();
            const size_t hwm = read_vm_hwm_bytes();

            draw_current_assistant();
            std::ostringstream status;
            status << "tok/s=" << std::fixed << std::setprecision(1) << tok_s
                   << "  tokens=" << token_count
                   << "  RSS=" << (rss / (1024 * 1024)) << " MB"
                   << "  HWM=" << (hwm / (1024 * 1024)) << " MB";
            draw_status(status.str());
        };

        engine.generate_stream(prompt, cfg, on_token);
        generating = false;
        messages.push_back({"assistant", current_response});
        current_response.clear();
        redraw();
    };

    // Main input loop.
    while (true) {
        draw_input_line();
        char c = 0;
        ssize_t n = read(STDIN_FILENO, &c, 1);
        if (n <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        if (c == 3) {
            // Ctrl+C quits immediately.
            break;
        }
        if (c == 27) {
            // Discard a likely ANSI/arrow-key escape sequence.
            char discard[2] = {0, 0};
            (void)read(STDIN_FILENO, discard, 2);
            continue;
        }
        if (c == '\n' || c == '\r') {
            std::string trimmed = input;
            trimmed.erase(0, trimmed.find_first_not_of(" \t\r\n"));
            trimmed.erase(trimmed.find_last_not_of(" \t\r\n") + 1);
            if (trimmed == "/quit" || trimmed == "/exit") break;
            if (trimmed == "/reset" || trimmed == "/clear") {
                messages.clear();
                messages.push_back({"system", "You are a helpful assistant."});
                engine.reset_for_next_run();
                input.clear();
                redraw();
                continue;
            }
            generate_reply();
            continue;
        }
        if (c == 127 || c == '\b') {
            if (!input.empty()) input.pop_back();
        } else if (c >= 32 && c < 127) {
            input.push_back(c);
        }
        // Ignore arrow keys etc. for now.
    }

    clear_screen();
    std::cout << "Chat ended.\n";
    return 0;
}

} // namespace gizmo
