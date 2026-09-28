#include "tui.hpp"

#include "cli_parser.hpp"
#include "inference_engine.hpp"
#include "model_discovery.hpp"
#include "proc_status.hpp"
#include "server/server.hpp"
#include "terminal_utils.hpp"
#include "util/string.hpp"

#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>

// For IP discovery
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>

// For filesystem discovery
#include <dirent.h>
#include <sys/stat.h>

namespace gizmo {

namespace {

// ---------------------------------------------------------------------------
// Tiny filesystem helpers (avoid C++17 <filesystem> dependency on older toolchains).
// ---------------------------------------------------------------------------

bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// ---------------------------------------------------------------------------
// Shared model discovery + custom entry wrapper
// ---------------------------------------------------------------------------

std::vector<DiscoveredModel> discover_models_with_custom(const std::vector<std::string>& extra_dirs) {
    auto models = gizmo::discover_models(extra_dirs);
    // Append the custom-path option.
    DiscoveredModel custom;
    custom.source = "custom";
    custom.name = "<Enter custom GGUF path>";
    custom.path = "";
    custom.size_bytes = 0;
    models.push_back(custom);
    return models;
}

std::string ask_custom_path() {
    show_cursor();
    clear_screen();
    std::cout << "\x1b[1;36mCustom model path\x1b[0m\n\n";
    std::cout << "Enter the full path to a .gguf file:\n";
    std::string path;
    std::cout << "> ";
    std::cout.flush();
    if (!std::getline(std::cin, path)) return "";
    return path;
}

// ---------------------------------------------------------------------------
// Inference profile picker
// ---------------------------------------------------------------------------

struct InferenceProfile {
    std::string name;
    std::string description;
    bool no_shard = false;
    int resident_layers = 8;
    bool evict_weights = true;
};

std::vector<InferenceProfile> default_profiles() {
    return {
        { "Low memory",    "Sharded, keep 1 layer resident (~2 GB peak on 27B)", false, 1,  true },
        { "Balanced",      "Sharded, keep 8 layers resident (default)",          false, 8,  true },
        { "Fast",          "Load full model into RAM (no sharding)",             true,  8,  false },
        { "Custom",        "Enter a resident-layer count",                       false, 1,  true },
    };
}

int ask_custom_resident_layers() {
    show_cursor();
    clear_screen();
    std::cout << "\x1b[1;36mCustom resident layers\x1b[0m\n\n";
    std::cout << "How many transformer blocks should stay resident in RAM?\n";
    std::cout << "Use 1 for the lowest memory footprint, or a large number to disable eviction.\n";
    std::cout << "> ";
    std::cout.flush();
    std::string line;
    if (!std::getline(std::cin, line)) return 1;
    long v = std::strtol(line.c_str(), nullptr, 10);
    if (v < 1) v = 1;
    return static_cast<int>(v);
}

std::string profile_display_line(const InferenceProfile& p) {
    return p.name + "  -  " + p.description;
}

InferenceProfile pick_profile_noninteractive(const std::vector<InferenceProfile>& profiles) {
    std::cout << "Inference profiles:\n";
    for (size_t i = 0; i < profiles.size(); ++i) {
        std::cout << "  " << (i + 1) << ". " << profile_display_line(profiles[i]) << "\n";
    }
    std::cout << "Enter number (q to quit): ";
    std::cout.flush();
    std::string line;
    if (!std::getline(std::cin, line)) return {};
    if (line == "q" || line == "Q") return {};
    char* end = nullptr;
    long n = std::strtol(line.c_str(), &end, 10);
    if (end == line.c_str() || n < 1 || static_cast<size_t>(n) > profiles.size()) return {};
    return profiles[static_cast<size_t>(n) - 1];
}

InferenceProfile pick_profile_interactive(const std::vector<InferenceProfile>& profiles) {
    const bool is_tty = isatty(STDIN_FILENO) && isatty(STDOUT_FILENO);
    if (!is_tty) {
        return pick_profile_noninteractive(profiles);
    }

    RawMode raw;
    if (!raw.enable()) {
        return pick_profile_noninteractive(profiles);
    }
    CursorGuard cursor;
    cursor.hide();

    size_t selected = 0;
    auto draw = [&]() {
        clear_screen();
        std::cout << "\x1b[1;36mGizmo Server\x1b[0m - pick an inference profile\n\n";
        std::cout << "Use \xe2\x86\x91/\xe2\x86\x93 or type a number, Enter to confirm, q to quit.\n\n";
        for (size_t i = 0; i < profiles.size(); ++i) {
            const auto& p = profiles[i];
            bool active = (i == selected);
            if (active) std::cout << "\x1b[7m";
            std::cout << "  " << (i + 1) << ". " << profile_display_line(p);
            if (active) std::cout << "\x1b[0m";
            std::cout << "\n";
        }
        std::cout.flush();
    };

    std::string number_buffer;
    bool dirty = true;
    while (true) {
        if (dirty) {
            draw();
            dirty = false;
        }
        char c = 0;
        ssize_t n = read(STDIN_FILENO, &c, 1);
        if (n <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        if (c == 'q' || c == 'Q') {
            return {};
        }
        if (c == '\n' || c == '\r') {
            return profiles[selected];
        }
        if (c == '\x1b') {
            char seq[3] = {0};
            if (read(STDIN_FILENO, &seq[0], 1) <= 0) continue;
            if (read(STDIN_FILENO, &seq[1], 1) <= 0) continue;
            if (seq[0] == '[') {
                if (seq[1] == 'A' && selected > 0) { --selected; dirty = true; }
                else if (seq[1] == 'B' && selected + 1 < profiles.size()) { ++selected; dirty = true; }
            }
            continue;
        }
        if (c >= '0' && c <= '9') {
            number_buffer.push_back(c);
            long idx = std::strtol(number_buffer.c_str(), nullptr, 10);
            if (idx >= 1 && static_cast<size_t>(idx) <= profiles.size()) {
                selected = static_cast<size_t>(idx) - 1;
                dirty = true;
            }
        } else {
            number_buffer.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// IP discovery
// ---------------------------------------------------------------------------

std::vector<std::string> local_ipv4_addresses() {
    std::vector<std::string> out;
    struct ifaddrs* ifa = nullptr;
    if (getifaddrs(&ifa) != 0) return out;
    for (struct ifaddrs* cur = ifa; cur != nullptr; cur = cur->ifa_next) {
        if (!cur->ifa_addr) continue;
        if (cur->ifa_addr->sa_family != AF_INET) continue;
        void* addr = &(reinterpret_cast<struct sockaddr_in*>(cur->ifa_addr)->sin_addr);
        char buf[INET_ADDRSTRLEN];
        if (inet_ntop(AF_INET, addr, buf, sizeof(buf))) {
            std::string s(buf);
            if (s != "127.0.0.1") out.push_back(s);
        }
    }
    freeifaddrs(ifa);
    // Always include localhost as a fallback.
    if (out.empty()) out.push_back("127.0.0.1");
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Serve settings editor (interactive feature toggles)
// ---------------------------------------------------------------------------

struct ServeSettingsState {
    std::string host = "0.0.0.0";
    int port = 8080;
    int request_timeout_seconds = 300;
    bool cors = false;
    bool no_evict = false;
    bool verbose = false;
    int threads = 4;
};

// Editable fields for the settings screen. Order must match ServeSettingsState.
enum class SettingsField : int {
    Host,
    Port,
    RequestTimeout,
    Cors,
    NoEvict,
    Verbose,
    Threads,
    Start,
    Count
};

ServeSettingsState edit_settings_interactive(const ServeSettingsState& seed) {
    const bool is_tty = isatty(STDIN_FILENO) && isatty(STDOUT_FILENO);
    ServeSettingsState st = seed;
    if (!is_tty) {
        return st;
    }

    RawMode raw;
    if (!raw.enable()) {
        return st;
    }
    CursorGuard cursor;
    cursor.hide();

    int selected = 0;
    const int n_fields = static_cast<int>(SettingsField::Count);

    auto field_label = [](SettingsField f) -> std::string {
        switch (f) {
            case SettingsField::Host:           return "Host";
            case SettingsField::Port:           return "Port";
            case SettingsField::RequestTimeout: return "Request timeout (seconds)";
            case SettingsField::Cors:           return "CORS";
            case SettingsField::NoEvict:        return "Keep weights resident (no eviction)";
            case SettingsField::Verbose:        return "Verbose sharded-engine output";
            case SettingsField::Threads:        return "Threads";
            case SettingsField::Start:          return "[ Start server ]";
            default: return "";
        }
    };

    auto field_value = [&](SettingsField f) -> std::string {
        switch (f) {
            case SettingsField::Host:           return st.host;
            case SettingsField::Port:           return std::to_string(st.port);
            case SettingsField::RequestTimeout: return std::to_string(st.request_timeout_seconds);
            case SettingsField::Cors:           return st.cors ? "on" : "off";
            case SettingsField::NoEvict:        return st.no_evict ? "on" : "off";
            case SettingsField::Verbose:        return st.verbose ? "on" : "off";
            case SettingsField::Threads:        return std::to_string(st.threads);
            default: return "";
        }
    };

    std::string edit_buffer;
    bool editing = false;
    bool dirty = true;

    auto draw = [&]() {
        clear_screen();
        std::cout << "\x1b[1;36mGizmo Server\x1b[0m - configure features\n\n";
        std::cout << "Use \xe2\x86\x91/\xe2\x86\x93 to move, Enter to edit/toggle, q to quit.\n\n";
        for (int i = 0; i < n_fields; ++i) {
            SettingsField f = static_cast<SettingsField>(i);
            bool active = (i == selected);
            if (active && !editing) std::cout << "\x1b[7m";
            std::cout << "  " << field_label(f);
            if (f != SettingsField::Start) {
                std::cout << ": ";
                if (active && editing) {
                    std::cout << "\x1b[36m" << edit_buffer << "_\x1b[0m";
                } else {
                    std::cout << field_value(f);
                }
            }
            if (active && !editing) std::cout << "\x1b[0m";
            std::cout << "\n";
        }
        std::cout << "\n\x1b[2mCORS is required for browser-based frontends (OpenWebUI, Hermes Desktop).\x1b[0m\n";
        std::cout.flush();
    };

    auto apply_edit = [&]() {
        if (edit_buffer.empty()) return;
        SettingsField f = static_cast<SettingsField>(selected);
        if (f == SettingsField::Host) {
            st.host = edit_buffer;
        } else if (f == SettingsField::Port) {
            char* end = nullptr;
            long v = std::strtol(edit_buffer.c_str(), &end, 10);
            if (end != edit_buffer.c_str() && v > 0 && v < 65536) st.port = static_cast<int>(v);
        } else if (f == SettingsField::RequestTimeout) {
            char* end = nullptr;
            long v = std::strtol(edit_buffer.c_str(), &end, 10);
            if (end != edit_buffer.c_str() && v > 0 && v <= 86400) st.request_timeout_seconds = static_cast<int>(v);
        } else if (f == SettingsField::Threads) {
            char* end = nullptr;
            long v = std::strtol(edit_buffer.c_str(), &end, 10);
            if (end != edit_buffer.c_str() && v > 0 && v <= 256) st.threads = static_cast<int>(v);
        }
        edit_buffer.clear();
        editing = false;
    };

    while (true) {
        if (dirty) {
            draw();
            dirty = false;
        }
        char c = 0;
        ssize_t n = read(STDIN_FILENO, &c, 1);
        if (n <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        if (c == 'q' || c == 'Q') {
            return {};
        }
        if (editing) {
            if (c == '\n' || c == '\r') {
                apply_edit();
                dirty = true;
            } else if (c == 127 || c == '\b') {
                if (!edit_buffer.empty()) edit_buffer.pop_back();
                dirty = true;
            } else if (c >= 32 && c < 127) {
                edit_buffer.push_back(c);
                dirty = true;
            } else if (c == '\x1b') {
                // Cancel edit on escape.
                edit_buffer.clear();
                editing = false;
                dirty = true;
            }
            continue;
        }
        if (c == '\n' || c == '\r') {
            SettingsField f = static_cast<SettingsField>(selected);
            if (f == SettingsField::Start) {
                return st;
            }
            if (f == SettingsField::Cors || f == SettingsField::NoEvict || f == SettingsField::Verbose) {
                bool* target = nullptr;
                if (f == SettingsField::Cors) target = &st.cors;
                else if (f == SettingsField::NoEvict) target = &st.no_evict;
                else if (f == SettingsField::Verbose) target = &st.verbose;
                if (target) *target = !(*target);
                dirty = true;
            } else {
                edit_buffer = field_value(f);
                editing = true;
                dirty = true;
            }
            continue;
        }
        if (c == '\x1b') {
            char seq[3] = {0};
            if (read(STDIN_FILENO, &seq[0], 1) <= 0) continue;
            if (read(STDIN_FILENO, &seq[1], 1) <= 0) continue;
            if (seq[0] == '[') {
                if (seq[1] == 'A' && selected > 0) { --selected; dirty = true; }
                else if (seq[1] == 'B' && selected + 1 < n_fields) { ++selected; dirty = true; }
            }
            continue;
        }
    }
}

// ---------------------------------------------------------------------------
// Server TUI public implementation
// ---------------------------------------------------------------------------

class ServerTui {
public:
    explicit ServerTui(const ServeSettings& settings) : settings_(settings) {}

    int run() {
        std::string path = settings_.model_path;
        if (path.empty()) {
            auto extra_dirs = parse_colon_dirs(settings_.model_path_extra);
            auto models = discover_models_with_custom(extra_dirs);
            if (models.empty()) {
                path = ask_custom_path();
            } else {
                path = gizmo::pick_model_interactive(models);
                if (path.empty()) {
                    // The user selected the custom-path option.
                    size_t custom_idx = models.size() - 1; // appended last
                    if (custom_idx < models.size() && models[custom_idx].source == "custom") {
                        path = ask_custom_path();
                    }
                }
            }
        }
        if (path.empty() || !file_exists(path)) {
            std::cerr << "No model selected.\n";
            return 1;
        }

        auto profiles = default_profiles();
        InferenceProfile profile = pick_profile_interactive(profiles);
        if (profile.name.empty()) {
            std::cerr << "No inference profile selected.\n";
            return 1;
        }
        if (profile.name == "Custom") {
            profile.resident_layers = ask_custom_resident_layers();
        }

        ServeSettingsState st;
        st.host = settings_.host;
        st.port = settings_.port;
        st.request_timeout_seconds = settings_.request_timeout_seconds;
        st.cors = settings_.cors;
        st.no_evict = settings_.no_evict;
        st.verbose = settings_.verbose;
        st.threads = settings_.threads;
        st = edit_settings_interactive(st);
        if (st.host.empty()) {
            std::cerr << "Server configuration cancelled.\n";
            return 1;
        }

        return serve(path, profile, st);
    }

private:
    ServeSettings settings_;

    int serve(const std::string& model_path, const InferenceProfile& profile,
              const ServeSettingsState& st) {
        InferenceEngine engine;
        engine.set_verbose(st.verbose);
        engine.set_threads(st.threads);

        clear_screen();
        std::cout << "\x1b[1;36mGizmo Server\x1b[0m\n";
        std::cout << "Loading model: " << model_path << "\n";
        std::cout << "Profile: " << profile.name;
        if (!profile.no_shard) {
            std::cout << " (resident " << profile.resident_layers << " blocks)";
        }
        std::cout << "\n";
        std::cout << "CORS: " << (st.cors ? "enabled" : "disabled") << "\n";
        std::cout << "Listen: " << st.host << ":" << st.port << "\n";
        std::cout.flush();

        if (!engine.initialize(model_path, /*layer_shard_lazy=*/!profile.no_shard)) {
            std::cerr << "\nFailed to initialize inference engine for: " << model_path << "\n";
            return 1;
        }

        if (!profile.no_shard) {
            engine.enable_sharded_engine(
                profile.resident_layers,
                profile.evict_weights,
                /*row_size=*/1);
            if (!engine.is_sharded()) {
                std::cout << "Note: sharded engine was not enabled; falling back to llama_decode.\n";
            }
        }

        ServerConfig config;
        config.host = st.host;
        config.port = st.port;
        config.threads = st.threads;
        config.cors = st.cors;
        config.request_timeout_seconds = st.request_timeout_seconds;
        config.log_file = settings_.log_file;
        config.json_logs = settings_.json_logs;

        HttpServer server(engine, config);
        server.install_signal_handlers(&server);
        server.start();

        // Small delay to let the listener start before we read is_running().
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!server.is_running()) {
            std::cerr << "\nFailed to start HTTP server on " << st.host << ":" << st.port << "\n";
            return 1;
        }

        // Run dashboard in raw terminal mode.
        RawMode raw;
        bool raw_ok = raw.enable();
        CursorGuard cursor;
        cursor.hide();

        auto draw = [&]() {
            clear_screen();
            std::cout << "\x1b[1;36mGizmo Server\x1b[0m is running\n";
            std::cout << "Model:  " << engine.get_model_info() << "\n";
            std::cout << "Layers: " << engine.n_layer();
            if (engine.is_sharded()) {
                std::cout << "  |  Sharding: enabled  |  Resident: " << profile.resident_layers << "\n";
            } else {
                std::cout << "  |  Sharding: disabled (full model)\n";
            }
            std::cout << "Listen: " << st.host << ":" << st.port << "\n";

            std::vector<std::string> ips = local_ipv4_addresses();
            std::cout << "IPs:    ";
            for (size_t i = 0; i < ips.size(); ++i) {
                if (i) std::cout << ", ";
                std::cout << "http://" << ips[i] << ":" << st.port;
            }
            std::cout << "\n\n";

            size_t rss = read_vm_rss_bytes();
            size_t hwm = read_vm_hwm_bytes();
            std::cout << "Memory: RSS " << (rss / (1024 * 1024)) << " MB  |  HWM "
                      << (hwm / (1024 * 1024)) << " MB\n\n";

            auto logs = server.recent_requests(12);
            if (logs.empty()) {
                std::cout << "No requests yet. Server is ready.\n";
            } else {
                std::cout << "Recent requests:\n";
                for (auto it = logs.rbegin(); it != logs.rend(); ++it) {
                    std::cout << "  [" << it->time << "] "
                              << it->method << " " << it->path
                              << "  " << it->status;
                    if (it->tokens > 0) std::cout << "  tokens=" << it->tokens;
                    if (it->seconds > 0.0) {
                        std::cout << "  time=" << std::fixed << std::setprecision(2)
                                  << it->seconds << "s";
                    }
                    std::cout << "\n";
                }
            }

            std::cout << "\n\x1b[2mPress 'q' to stop the server and quit.\x1b[0m\n";
            std::cout.flush();
        };

        // Main dashboard loop: refresh periodically and check for 'q'.
        bool user_quit = false;
        while (server.is_running() && !user_quit && !server.stop_requested()) {
            draw();
            for (int i = 0; i < 20; ++i) { // ~2 second polling window
                if (!server.is_running() || server.stop_requested()) break;
                if (raw_ok) {
                    char c = 0;
                    if (read(STDIN_FILENO, &c, 1) == 1 && (c == 'q' || c == 'Q')) {
                        user_quit = true;
                        break;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        server.stop();
        clear_screen();
        std::cout << "Server stopped.\n";
        return 0;
    }
};

int run_server_tui(const ServeSettings& settings) {
    ServerTui tui(settings);
    return tui.run();
}

// ---------------------------------------------------------------------------
// Headless server startup
// ---------------------------------------------------------------------------

namespace {

int serve_headless(const ServeSettings& settings) {
    const std::string& model_path = settings.model_path;
    if (model_path.empty() || !file_exists(model_path)) {
        std::cerr << "Error: model not found: " << model_path << "\n";
        return 1;
    }

    InferenceEngine engine;
    engine.set_verbose(settings.verbose);
    engine.set_threads(settings.threads);

    std::cout << "Loading model: " << model_path << "\n";
    if (!engine.initialize(model_path, /*layer_shard_lazy=*/!settings.no_shard)) {
        std::cerr << "Failed to initialize inference engine for: " << model_path << "\n";
        return 1;
    }

    if (!settings.no_shard) {
        const int32_t resident = settings.resident_layers > 0 ? settings.resident_layers : 8;
        const int32_t row = settings.row_size > 0 ? settings.row_size : 1;
        engine.enable_sharded_engine(resident, /*evict_weights=*/!settings.no_evict, row);
        if (!engine.is_sharded()) {
            std::cout << "Note: sharded engine was not enabled; falling back to llama_decode.\n";
        }
    }

    ServerConfig config;
    config.host = settings.host;
    config.port = settings.port;
    config.threads = settings.threads;
    config.cors = settings.cors;
    config.request_timeout_seconds = settings.request_timeout_seconds;
    config.log_file = settings.log_file;
    config.json_logs = settings.json_logs;

    HttpServer server(engine, config);
    server.install_signal_handlers(&server);
    server.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (!server.is_running()) {
        std::cerr << "Failed to start HTTP server on " << settings.host << ":" << settings.port << "\n";
        return 1;
    }

    std::cout << "Gizmo server listening on http://" << settings.host << ":" << settings.port << "\n";
    std::cout << "Model id: " << engine.get_model_id() << "\n";
    if (settings.cors) {
        std::cout << "CORS enabled for browser frontends.\n";
    }
    std::cout << "Press Ctrl+C to stop.\n";

    while (server.is_running() && !server.stop_requested()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    server.stop();
    std::cout << "Server stopped.\n";
    return 0;
}

} // namespace

int run_server_headless(const ServeSettings& settings) {
    return serve_headless(settings);
}

} // namespace gizmo
