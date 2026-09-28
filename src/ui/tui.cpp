#include "tui.hpp"

#include "cli_parser.hpp"
#include "inference_engine.hpp"
#include "proc_status.hpp"
#include "server/server.hpp"

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

struct DiscoveredModel {
    std::string source;      // "ollama", "lmstudio", "custom"
    std::string name;
    std::string path;
    uint64_t    size_bytes = 0;
};

struct Path {
    std::string value;
    Path(const std::string& s) : value(s) {}
    Path operator/(const std::string& child) const {
        if (value.empty()) return Path(child);
        if (value.back() == '/') return Path(value + child);
        return Path(value + "/" + child);
    }
    std::string string() const { return value; }
    const char* c_str() const { return value.c_str(); }
};

bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool dir_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

uint64_t file_size(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) return static_cast<uint64_t>(st.st_size);
    return 0;
}

std::string home_dir() {
    const char* home = getenv("HOME");
    if (home) return home;
    return ".";
}

void recursive_gguf(const std::string& dir, std::vector<std::string>& out) {
    if (!dir_exists(dir)) return;
    DIR* d = opendir(dir.c_str());
    if (!d) return;
    while (struct dirent* ent = readdir(d)) {
        std::string name = ent->d_name;
        if (name == "." || name == "..") continue;
        std::string full = (Path(dir) / name).string();
        bool is_dir = (ent->d_type == DT_DIR) ||
                      (ent->d_type == DT_UNKNOWN && dir_exists(full));
        bool is_file = (ent->d_type == DT_REG) ||
                       (ent->d_type == DT_UNKNOWN && file_exists(full));
        if (is_dir) {
            recursive_gguf(full, out);
        } else if (is_file) {
            if (name.size() > 5 && name.compare(name.size() - 5, 5, ".gguf") == 0) {
                // Skip helper files that are not the main weights.
                std::string lower = name;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                if (lower.find("mmproj") != std::string::npos) continue;
                if (lower.find("mtp") != std::string::npos) continue;
                if (lower.find("embed") != std::string::npos) continue;
                if (lower.find("vision") != std::string::npos) continue;
                out.push_back(full);
            }
        }
    }
    closedir(d);
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

std::string format_bytes(uint64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = static_cast<double>(bytes);
    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        ++unit;
    }
    std::ostringstream oss;
    if (unit == 0) oss << bytes << " " << units[unit];
    else oss << std::fixed << std::setprecision(1) << size << " " << units[unit];
    return oss.str();
}

// ---------------------------------------------------------------------------
// Terminal control
// ---------------------------------------------------------------------------

// Forward declarations for terminal control used by CursorGuard.
void hide_cursor();
void show_cursor();

struct CursorGuard {
    bool hidden = false;
    void hide() { hide_cursor(); hidden = true; }
    ~CursorGuard() { if (hidden) show_cursor(); }
};

struct RawMode {
    termios old_tio;
    bool active = false;

    bool enable() {
        if (tcgetattr(STDIN_FILENO, &old_tio) != 0) return false;
        termios new_tio = old_tio;
        new_tio.c_lflag &= ~(ICANON | ECHO);
        new_tio.c_cc[VMIN] = 0;
        new_tio.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &new_tio) != 0) return false;
        active = true;
        return true;
    }

    void disable() {
        if (active) {
            tcsetattr(STDIN_FILENO, TCSANOW, &old_tio);
            active = false;
        }
    }

    ~RawMode() { disable(); }
};

void clear_screen() {
    std::cout << "\x1b[2J\x1b[H";
}

void hide_cursor() {
    std::cout << "\x1b[?25l";
}

void show_cursor() {
    std::cout << "\x1b[?25h";
}

// ---------------------------------------------------------------------------
// Model discovery
// ---------------------------------------------------------------------------

using json = nlohmann::json;

namespace {

// Recursively collect regular files under `dir`.
void recursive_files(const std::string& dir, std::vector<std::string>& out) {
    if (!dir_exists(dir)) return;
    DIR* d = opendir(dir.c_str());
    if (!d) return;
    while (struct dirent* ent = readdir(d)) {
        std::string name = ent->d_name;
        if (name == "." || name == "..") continue;
        std::string full = (Path(dir) / name).string();
        bool is_dir = (ent->d_type == DT_DIR) ||
                      (ent->d_type == DT_UNKNOWN && dir_exists(full));
        if (is_dir) {
            recursive_files(full, out);
        } else if (file_exists(full)) {
            out.push_back(full);
        }
    }
    closedir(d);
}

} // namespace

std::vector<DiscoveredModel> discover_ollama_models() {
    std::vector<DiscoveredModel> out;
    Path manifests = Path(home_dir()) / ".ollama/models/manifests";
    if (!dir_exists(manifests.string())) return out;

    std::vector<std::string> files;
    recursive_files(manifests.string(), files);

    for (const std::string& manifest_path : files) {
        std::ifstream f(manifest_path);
        if (!f) continue;
        json manifest;
        try {
            f >> manifest;
        } catch (...) { continue; }

        if (!manifest.contains("layers") || !manifest["layers"].is_array()) continue;
        std::string digest;
        for (const auto& layer : manifest["layers"]) {
            if (!layer.contains("mediaType")) continue;
            std::string mt = layer.value("mediaType", "");
            if (mt == "application/vnd.ollama.image.model") {
                digest = layer.value("digest", "");
                break;
            }
        }
        if (digest.empty()) continue;

        std::string blob_name = digest;
        std::replace(blob_name.begin(), blob_name.end(), ':', '-');
        Path blob_path = Path(home_dir()) / ".ollama/models/blobs" / blob_name;
        if (!file_exists(blob_path.string())) continue;

        // Display name: registry/namespace/model:tag from the relative path.
        std::string rel = manifest_path.substr(manifests.string().size());
        if (!rel.empty() && rel.front() == '/') rel = rel.substr(1);
        size_t colon = rel.find_last_of('/');
        std::string name = rel;
        if (colon != std::string::npos) {
            name = rel.substr(0, colon) + ":" + rel.substr(colon + 1);
        }

        DiscoveredModel m;
        m.source = "ollama";
        m.name = name;
        m.path = blob_path.string();
        m.size_bytes = file_size(m.path);
        out.push_back(m);
    }
    return out;
}

std::vector<DiscoveredModel> discover_lmstudio_models() {
    std::vector<DiscoveredModel> out;
    Path models_dir = Path(home_dir()) / ".lmstudio/models";
    std::vector<std::string> files;
    recursive_gguf(models_dir.string(), files);
    for (const auto& p : files) {
        DiscoveredModel m;
        m.source = "lmstudio";
        // Use the file name as the display name; strip directory noise.
        size_t slash = p.find_last_of('/');
        m.name = (slash == std::string::npos) ? p : p.substr(slash + 1);
        m.path = p;
        m.size_bytes = file_size(p);
        out.push_back(m);
    }
    return out;
}

} // namespace

std::vector<DiscoveredModel> discover_models() {
    std::vector<DiscoveredModel> out;
    auto ollama = discover_ollama_models();
    auto lmstudio = discover_lmstudio_models();
    out.reserve(ollama.size() + lmstudio.size() + 1);
    out.insert(out.end(), ollama.begin(), ollama.end());
    out.insert(out.end(), lmstudio.begin(), lmstudio.end());
    // Stable sort by source then name.
    std::sort(out.begin(), out.end(), [](const DiscoveredModel& a, const DiscoveredModel& b) {
        if (a.source != b.source) return a.source < b.source;
        return a.name < b.name;
    });
    // Append the custom-path option.
    DiscoveredModel custom;
    custom.source = "custom";
    custom.name = "<Enter custom GGUF path>";
    custom.path = "";
    custom.size_bytes = 0;
    out.push_back(custom);
    return out;
}

// ---------------------------------------------------------------------------
// Interactive model picker
// ---------------------------------------------------------------------------

namespace {

std::string pick_model_noninteractive(const std::vector<DiscoveredModel>& models) {
    std::cout << "Available models:\n";
    for (size_t i = 0; i < models.size(); ++i) {
        std::cout << "  " << (i + 1) << ". [" << models[i].source << "] "
                  << models[i].name;
        if (models[i].size_bytes > 0) std::cout << " (" << format_bytes(models[i].size_bytes) << ")";
        std::cout << "\n";
    }
    std::cout << "Enter number (q to quit): ";
    std::cout.flush();
    std::string line;
    if (!std::getline(std::cin, line)) return "";
    if (line == "q" || line == "Q") return "";
    char* end = nullptr;
    long n = std::strtol(line.c_str(), &end, 10);
    if (end == line.c_str() || n < 1 || static_cast<size_t>(n) > models.size()) return "";
    return models[static_cast<size_t>(n) - 1].path;
}

std::string pick_model_interactive(const std::vector<DiscoveredModel>& models) {
    if (models.empty()) return "";

    const bool is_tty = isatty(STDIN_FILENO) && isatty(STDOUT_FILENO);
    if (!is_tty) {
        return pick_model_noninteractive(models);
    }

    RawMode raw;
    if (!raw.enable()) {
        return pick_model_noninteractive(models);
    }
    CursorGuard cursor;
    cursor.hide();

    size_t selected = 0;

    auto draw = [&]() {
        clear_screen();
        std::cout << "\x1b[1;36mGizmo Server\x1b[0m - pick a model\n\n";
        std::cout << "Use ↑/↓ or type a number, Enter to confirm, q to quit.\n\n";
        for (size_t i = 0; i < models.size(); ++i) {
            const auto& m = models[i];
            bool active = (i == selected);
            if (active) std::cout << "\x1b[7m";
            std::cout << "  " << (i + 1) << ". ";
            if (m.source == "ollama") std::cout << "\x1b[1;33m[ollama]\x1b[0m";
            else if (m.source == "lmstudio") std::cout << "\x1b[1;34m[lmstudio]\x1b[0m";
            else std::cout << "\x1b[1;32m[custom]\x1b[0m";
            std::cout << " " << m.name;
            if (m.size_bytes > 0) std::cout << " (" << format_bytes(m.size_bytes) << ")";
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
            return "";
        }
        if (c == '\n' || c == '\r') {
            return models[selected].path;
        }
        if (c == '\x1b') {
            char seq[3] = {0};
            if (read(STDIN_FILENO, &seq[0], 1) <= 0) continue;
            if (read(STDIN_FILENO, &seq[1], 1) <= 0) continue;
            if (seq[0] == '[') {
                if (seq[1] == 'A' && selected > 0) { --selected; dirty = true; }
                else if (seq[1] == 'B' && selected + 1 < models.size()) { ++selected; dirty = true; }
            }
            continue;
        }
        if (c >= '0' && c <= '9') {
            number_buffer.push_back(c);
            // If the buffer forms a valid 1-based index, jump there.
            long idx = std::strtol(number_buffer.c_str(), nullptr, 10);
            if (idx >= 1 && static_cast<size_t>(idx) <= models.size()) {
                selected = static_cast<size_t>(idx) - 1;
                dirty = true;
            }
        } else {
            number_buffer.clear();
        }
    }
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
// Server TUI public implementation
// ---------------------------------------------------------------------------

class ServerTui {
public:
    ServerTui(const std::string& host, int port, bool cors)
        : host_(host), port_(port), cors_(cors) {}

    int run() {
        auto models = discover_models();
        if (models.empty()) {
            // No discovered models: go straight to custom path.
            std::string path = ask_custom_path();
            if (path.empty() || !file_exists(path)) {
                std::cerr << "No model selected.\n";
                return 1;
            }
            return serve(path);
        }

        std::string path = pick_model_interactive(models);
        if (path.empty()) {
            // The user selected the custom-path option.
            size_t custom_idx = models.size() - 1; // appended last
            if (custom_idx < models.size() && models[custom_idx].source == "custom") {
                path = ask_custom_path();
            }
        }
        if (path.empty() || !file_exists(path)) {
            std::cerr << "No model selected.\n";
            return 1;
        }
        return serve(path);
    }

private:
    std::string host_;
    int port_;
    bool cors_;

    int serve(const std::string& model_path) {
        // Load the model. For a stable API server we default to un-sharded
        // (full-model) inference so requests don't hit the experimental sharded
        // scheduler crashes.
        InferenceEngine engine;
        engine.set_threads(4);

        clear_screen();
        std::cout << "\x1b[1;36mGizmo Server\x1b[0m\n";
        std::cout << "Loading model: " << model_path << "\n";
        std::cout << "Layers: " << "..." << "\n";
        std::cout.flush();

        if (!engine.initialize(model_path, /*layer_shard_lazy=*/false)) {
            std::cerr << "\nFailed to initialize inference engine for: " << model_path << "\n";
            return 1;
        }

        ServerConfig config;
        config.host = host_;
        config.port = port_;
        config.threads = 4;
        config.cors = cors_;

        HttpServer server(engine, config);
        server.start();

        // Small delay to let the listener start before we read is_running().
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!server.is_running()) {
            std::cerr << "\nFailed to start HTTP server on " << host_ << ":" << port_ << "\n";
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
            std::cout << "Layers: " << engine.n_layer() << "  |  Sharding: disabled (API stable)\n";
            std::cout << "Listen: " << host_ << ":" << port_ << "\n";

            std::vector<std::string> ips = local_ipv4_addresses();
            std::cout << "IPs:    ";
            for (size_t i = 0; i < ips.size(); ++i) {
                if (i) std::cout << ", ";
                std::cout << "http://" << ips[i] << ":" << port_;
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
        while (server.is_running() && !user_quit) {
            draw();
            for (int i = 0; i < 20; ++i) { // ~2 second polling window
                if (!server.is_running()) break;
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

int run_server_tui(const std::string& host, int port, bool cors) {
    ServerTui tui(host, port, cors);
    return tui.run();
}

} // namespace gizmo
