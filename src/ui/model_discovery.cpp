#include "model_discovery.hpp"

#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <filesystem>
#include <sys/stat.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace gizmo {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

// Forward declarations for terminal control used by pick_model_interactive.
void clear_screen();
void hide_cursor();
void show_cursor();

std::string home_dir() {
    const char* home = getenv("HOME");
    if (home) return home;
    return ".";
}

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

// Recursively collect regular files under `dir`.
void recursive_files(const std::string& dir, std::vector<std::string>& out) {
    if (!dir_exists(dir)) return;
    DIR* d = opendir(dir.c_str());
    if (!d) return;
    while (struct dirent* ent = readdir(d)) {
        std::string name = ent->d_name;
        if (name == "." || name == "..") continue;
        std::string full = dir;
        if (!full.empty() && full.back() != '/') full += '/';
        full += name;
        bool is_dir = (ent->d_type == DT_DIR) ||
                      (ent->d_type == DT_UNKNOWN && dir_exists(full));
        bool is_file = (ent->d_type == DT_REG) ||
                       (ent->d_type == DT_UNKNOWN && file_exists(full));
        if (is_dir) {
            recursive_files(full, out);
        } else if (is_file) {
            out.push_back(full);
        }
    }
    closedir(d);
}

// Recursively collect GGUF files under `dir`, skipping helper files.
void recursive_gguf(const std::string& dir, std::vector<std::string>& out) {
    if (!dir_exists(dir)) return;
    DIR* d = opendir(dir.c_str());
    if (!d) return;
    while (struct dirent* ent = readdir(d)) {
        std::string name = ent->d_name;
        if (name == "." || name == "..") continue;
        std::string full = dir;
        if (!full.empty() && full.back() != '/') full += '/';
        full += name;
        bool is_dir = (ent->d_type == DT_DIR) ||
                      (ent->d_type == DT_UNKNOWN && dir_exists(full));
        bool is_file = (ent->d_type == DT_REG) ||
                       (ent->d_type == DT_UNKNOWN && file_exists(full));
        if (is_dir) {
            recursive_gguf(full, out);
        } else if (is_file) {
            if (name.size() > 5 && name.compare(name.size() - 5, 5, ".gguf") == 0) {
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

std::string filename_from_path(const std::string& path) {
    size_t slash = path.find_last_of('/');
    return (slash == std::string::npos) ? path : path.substr(slash + 1);
}

std::vector<DiscoveredModel> discover_ollama_models() {
    std::vector<DiscoveredModel> out;
    std::string base = home_dir();
    base += "/.ollama/models";
    fs::path manifests = fs::path(base) / "manifests";
    if (!fs::exists(manifests)) return out;

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
        fs::path blob_path = fs::path(base) / "blobs" / blob_name;
        if (!fs::exists(blob_path)) continue;

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
        m.size_bytes = fs::file_size(blob_path);
        out.push_back(m);
    }
    return out;
}

std::vector<DiscoveredModel> discover_lmstudio_models() {
    std::vector<DiscoveredModel> out;
    fs::path models_dir = fs::path(home_dir()) / ".lmstudio" / "models";
    std::vector<std::string> files;
    recursive_gguf(models_dir.string(), files);
    for (const auto& p : files) {
        DiscoveredModel m;
        m.source = "lmstudio";
        m.name = filename_from_path(p);
        m.path = p;
        m.size_bytes = file_size(p);
        out.push_back(m);
    }
    return out;
}

std::vector<DiscoveredModel> discover_user_models(const std::vector<std::string>& dirs) {
    std::vector<DiscoveredModel> out;
    for (const auto& dir : dirs) {
        if (dir.empty()) continue;
        std::vector<std::string> files;
        recursive_gguf(dir, files);
        for (const auto& p : files) {
            DiscoveredModel m;
            m.source = "user";
            m.name = filename_from_path(p);
            m.path = p;
            m.size_bytes = file_size(p);
            out.push_back(m);
        }
    }
    return out;
}

// Terminal control helpers.
void clear_screen() {
    std::cout << "\x1b[2J\x1b[H";
}

void hide_cursor() {
    std::cout << "\x1b[?25l";
}

void show_cursor() {
    std::cout << "\x1b[?25h";
}

} // namespace

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

std::vector<DiscoveredModel> discover_models(const std::vector<std::string>& extra_dirs) {
    std::vector<DiscoveredModel> out;
    auto ollama = discover_ollama_models();
    auto lmstudio = discover_lmstudio_models();
    auto user = discover_user_models(extra_dirs);

    out.reserve(ollama.size() + lmstudio.size() + user.size());
    out.insert(out.end(), ollama.begin(), ollama.end());
    out.insert(out.end(), lmstudio.begin(), lmstudio.end());
    out.insert(out.end(), user.begin(), user.end());

    std::sort(out.begin(), out.end(), [](const DiscoveredModel& a, const DiscoveredModel& b) {
        if (a.source != b.source) return a.source < b.source;
        return a.name < b.name;
    });
    return out;
}

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

    struct CursorGuard {
        bool hidden = false;
        void hide() { hide_cursor(); hidden = true; }
        ~CursorGuard() { if (hidden) show_cursor(); }
    };

    RawMode raw;
    if (!raw.enable()) {
        return pick_model_noninteractive(models);
    }
    CursorGuard cursor;
    cursor.hide();

    size_t selected = 0;

    auto draw = [&]() {
        clear_screen();
        std::cout << "\x1b[1;36mGizmo\x1b[0m - pick a model\n\n";
        std::cout << "Use \xe2\x86\x91/\xe2\x86\x93 or type a number, Enter to confirm, q to quit.\n\n";
        for (size_t i = 0; i < models.size(); ++i) {
            const auto& m = models[i];
            bool active = (i == selected);
            if (active) std::cout << "\x1b[7m";
            std::cout << "  " << (i + 1) << ". ";
            if (m.source == "ollama") std::cout << "\x1b[1;33m[ollama]\x1b[0m";
            else if (m.source == "lmstudio") std::cout << "\x1b[1;34m[lmstudio]\x1b[0m";
            else if (m.source == "user") std::cout << "\x1b[1;35m[user]\x1b[0m";
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

} // namespace gizmo
