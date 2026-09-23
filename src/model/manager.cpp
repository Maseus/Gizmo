#include "model_manager.hpp"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>

namespace gizmo {

namespace fs = std::filesystem;

ModelManager::ModelManager() : models_dir_("") {
}

ModelManager::~ModelManager() {
}

bool ModelManager::initialize(const std::string& models_dir) {
    models_dir_ = models_dir;

    // Create models directory if it doesn't exist
    if (!fs::exists(models_dir_)) {
        fs::create_directories(models_dir_);
    }

    // Scan for existing models
    models_.clear();
    for (const auto& entry : fs::directory_iterator(models_dir_)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".gguf") {
                ModelInfo info = parse_gguf_header(entry.path().string());
                info.name = entry.path().stem().string();
                info.path = entry.path().string();
                models_.push_back(info);
            }
        }
    }

    return true;
}

std::vector<ModelInfo> ModelManager::list_models() const {
    return models_;
}

bool ModelManager::download(const std::string& url, const std::string& output_path) {
    // TODO: Implement HTTP download with progress
    // Could use libcurl or simple HTTP client
    (void)url;
    (void)output_path;
    return false;
}

ModelInfo* ModelManager::get_model(const std::string& name) {
    for (auto& model : models_) {
        if (model.name == name) {
            return &model;
        }
    }
    return nullptr;
}

bool ModelManager::remove(const std::string& name) {
    auto it = std::find_if(models_.begin(), models_.end(),
        [&name](const ModelInfo& m) { return m.name == name; });

    if (it != models_.end()) {
        // Delete file
        fs::remove(it->path);
        // Remove from list
        models_.erase(it);
        return true;
    }
    return false;
}

ModelInfo ModelManager::parse_gguf_header(const std::string& path) {
    ModelInfo info;
    info.name = "";
    info.path = path;
    info.format = "GGUF";
    info.size_bytes = 0;
    info.total_layers = 0;
    info.embedding_dim = 0;
    info.vocab_size = 0;

    // Get file size
    if (fs::exists(path)) {
        info.size_bytes = fs::file_size(path);
    }

    // TODO: Parse actual GGUF header to extract:
    // - tensor count
    // - layer count
    // - embedding dimension
    // - vocab size
    // This requires reading the binary GGUF format

    return info;
}

} // namespace gizmo
