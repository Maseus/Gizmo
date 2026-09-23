#include "model_picker.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// For isatty / fileno
#include <termios.h>
#include <unistd.h>

#include "json.hpp"

namespace gizmo {

namespace fs = std::filesystem;
using json = nlohmann::json;

static std::string home_dir() {
    const char* home = std::getenv("HOME");
    if (home) return home;
    return "";
}

static std::string human_size(uint64_t bytes) {
    const char* units[] = { "B", "KB", "MB", "GB", "TB" };
    int unit = 0;
    double size = static_cast<double>(bytes);
    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        ++unit;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << size << " " << units[unit];
    return oss.str();
}

static std::string filename_no_ext(const std::string& path) {
    fs::path p(path);
    return p.stem().string();
}

// --- Ollama discovery ---

static std::vector<DiscoveredModel> discover_ollama_models() {
    std::vector<DiscoveredModel> result;
    std::string base = home_dir() + "/.ollama/models";
    fs::path manifests = fs::path(base) / "manifests";
    fs::path blobs = fs::path(base) / "blobs";

    if (!fs::exists(manifests) || !fs::exists(blobs)) {
        return result;
    }

    try {
        for (const auto& entry : fs::recursive_directory_iterator(manifests)) {
            if (!entry.is_regular_file()) continue;

            std::ifstream f(entry.path());
            if (!f) continue;
            json manifest;
            try { f >> manifest; } catch (...) { continue; }

            if (!manifest.contains("layers")) continue;
            for (const auto& layer : manifest["layers"]) {
                std::string mt = layer.value("mediaType", "");
                if (mt != "application/vnd.ollama.image.model") continue;
                std::string digest = layer.value("digest", "");
                if (digest.empty()) continue;

                // digest is "sha256:abcdef..." -> blob file is blobs/sha256-abcdef...
                std::string blob_name = digest;
                std::replace(blob_name.begin(), blob_name.end(), ':', '-');
                fs::path blob_path = blobs / blob_name;
                if (!fs::exists(blob_path)) continue;

                DiscoveredModel m;
                m.source = "ollama";
                m.path = blob_path.string();
                m.size_bytes = fs::file_size(blob_path);

                // Build a display name from the manifest relative path.
                // e.g. manifests/registry.ollama.ai/library/qwen3/4b
                fs::path rel = fs::relative(entry.path(), manifests);
                std::string stem = rel.parent_path().string();
                if (!stem.empty() && stem.back() == '/') stem.pop_back();
                std::replace(stem.begin(), stem.end(), '/', ':');
                std::replace(stem.begin(), stem.end(), '\\', ':');
                m.name = stem + ":" + entry.path().stem().string();
                result.push_back(m);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: failed to scan Ollama models: " << e.what() << "\n";
    }
    return result;
}

// --- LM Studio discovery ---

static bool looks_like_llm(const std::string& path) {
    std::string lower = path;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    // Skip projector, embedding, and multi-modal files that share a folder.
    if (lower.find("mmproj") != std::string::npos) return false;
    if (lower.find("mtp") != std::string::npos) return false;
    if (lower.find("embed") != std::string::npos) return false;
    if (lower.find("vision") != std::string::npos) return false;
    return true;
}

static std::vector<DiscoveredModel> discover_lmstudio_models() {
    std::vector<DiscoveredModel> result;
    fs::path base = fs::path(home_dir()) / ".lmstudio" / "models";
    if (!fs::exists(base)) return result;

    try {
        for (const auto& publisher : fs::directory_iterator(base)) {
            if (!publisher.is_directory()) continue;
            for (const auto& model_dir : fs::recursive_directory_iterator(publisher.path())) {
                if (!model_dir.is_regular_file()) continue;
                const fs::path& p = model_dir.path();
                if (p.extension() != ".gguf") continue;
                if (!looks_like_llm(p.string())) continue;

                DiscoveredModel m;
                m.source = "lmstudio";
                m.path = p.string();
                m.size_bytes = fs::file_size(p);

                // Display name: "Publisher/Model/filename"
                fs::path rel = fs::relative(p, base);
                m.name = rel.parent_path().string();
                if (m.name.empty()) m.name = p.stem().string();
                result.push_back(m);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: failed to scan LM Studio models: " << e.what() << "\n";
    }
    return result;
}

std::vector<DiscoveredModel> discover_default_models() {
    auto ollama = discover_ollama_models();
    auto lmstudio = discover_lmstudio_models();
    std::vector<DiscoveredModel> all;
    all.reserve(ollama.size() + lmstudio.size());
    all.insert(all.end(), ollama.begin(), ollama.end());
    all.insert(all.end(), lmstudio.begin(), lmstudio.end());

    // Sort by source then by name for a predictable list.
    std::sort(all.begin(), all.end(), [](const DiscoveredModel& a, const DiscoveredModel& b) {
        if (a.source != b.source) return a.source < b.source;
        return a.name < b.name;
    });
    return all;
}

// --- Terminal picker helpers ---

static void clear_screen() {
    std::cout << "\033[2J\033[H" << std::flush;
}

static void hide_cursor() {
    std::cout << "\033[?25l" << std::flush;
}

static void show_cursor() {
    std::cout << "\033[?25h" << std::flush;
}

static int read_key() {
    int c = std::getchar();
    if (c == '\033') {
        int second = std::getchar();
        if (second == '[') {
            int seq = std::getchar();
            if (seq == 'A') return -1; // up
            if (seq == 'B') return -2; // down
        }
    }
    return c;
}

static void draw_menu(const std::vector<DiscoveredModel>& models, int selected) {
    clear_screen();
    std::cout << "\033[1;36mGizmo Model Picker\033[0m\n";
    std::cout << "Choose a model to serve (\033[33m↑/↓\033[0m move, \033[33mEnter\033[0m select, \033[33mq\033[0m quit):\n\n";

    int count = static_cast<int>(models.size());
    int start = std::max(0, selected - 10);
    int end = std::min(count, start + 21);
    if (end - start < 21 && count > 21) start = std::max(0, end - 21);

    for (int i = start; i < end; ++i) {
        const auto& m = models[i];
        std::string badge = m.source == "ollama" ? "\033[1;34mollama\033[0m" : "\033[1;35mlmstudio\033[0m";
        std::cout << std::left;
        if (i == selected) {
            std::cout << "\033[7m > " << std::setw(3) << (i + 1) << " "
                      << badge << "  " << std::setw(50) << m.name
                      << " " << std::setw(10) << human_size(m.size_bytes)
                      << "\033[0m\n";
        } else {
            std::cout << "   " << std::setw(3) << (i + 1) << " "
                      << badge << "  " << std::setw(50) << m.name
                      << " " << std::setw(10) << human_size(m.size_bytes) << "\n";
        }
    }
    std::cout << "\n";
}

std::string pick_model_interactive() {
    auto models = discover_default_models();
    if (models.empty()) {
        std::cerr << "No models found in default Ollama or LM Studio directories.\n"
                  << "Use `gizmo serve -m /path/to/model.gguf` to specify a model.\n";
        return "";
    }

    bool tty = isatty(fileno(stdin)) && isatty(fileno(stdout));
    if (!tty) {
        // Non-interactive fallback: just print a numbered list.
        std::cout << "Available models:\n";
        for (size_t i = 0; i < models.size(); ++i) {
            std::cout << "  " << (i + 1) << ". [" << models[i].source << "] "
                      << models[i].name << " (" << human_size(models[i].size_bytes) << ")\n";
        }
        std::cout << "Enter number: " << std::flush;
        size_t n = 0;
        if (!(std::cin >> n) || n == 0 || n > models.size()) return "";
        return models[n - 1].path;
    }

    // Switch stdin to raw-ish single-character mode for arrow keys.
    struct termios old_tio, new_tio;
    tcgetattr(STDIN_FILENO, &old_tio);
    new_tio = old_tio;
    new_tio.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_tio);

    hide_cursor();
    int selected = 0;
    int key = 0;
    draw_menu(models, selected);

    while ((key = read_key()) != 'q' && key != 'Q' && key != 3 && key != 27) {
        if (key == '\n' || key == '\r') {
            break;
        }
        if (key == -1) { // up
            if (selected > 0) --selected;
        } else if (key == -2) { // down
            if (selected + 1 < static_cast<int>(models.size())) ++selected;
        } else if (std::isdigit(key)) {
            // Direct number entry: read remaining digits and jump.
            std::string num(1, static_cast<char>(key));
            tcsetattr(STDIN_FILENO, TCSANOW, &old_tio); // restore for digits
            while (std::cin.peek() != EOF && std::isdigit(std::cin.peek())) {
                num.push_back(static_cast<char>(std::cin.get()));
            }
            tcsetattr(STDIN_FILENO, TCSANOW, &new_tio);
            int n = 0;
            try { n = std::stoi(num); } catch (...) { n = 0; }
            if (n >= 1 && n <= static_cast<int>(models.size())) {
                selected = n - 1;
            }
        }
        draw_menu(models, selected);
    }

    tcsetattr(STDIN_FILENO, TCSANOW, &old_tio);
    show_cursor();
    clear_screen();

    if (key == '\n' || key == '\r') {
        return models[selected].path;
    }
    return "";
}

} // namespace gizmo
