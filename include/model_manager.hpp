#ifndef GIZMO_MODEL_MANAGER_HPP
#define GIZMO_MODEL_MANAGER_HPP

#include <string>
#include <vector>
#include <cstdint>

namespace gizmo {

struct ModelInfo {
    std::string name;
    std::string path;
    std::string format;  // GGUF version
    std::string arch;    // architecture, e.g. "qwen3"
    size_t size_bytes;
    int64_t parameter_count = 0;
    int32_t total_layers;
    int32_t embedding_dim;
    int32_t vocab_size;
    int32_t context_length = 0;  // trained context window from GGUF metadata
};

class ModelManager {
public:
    ModelManager();
    ~ModelManager();

    // Initialize with models directory
    bool initialize(const std::string& models_dir);

    // List all available models
    std::vector<ModelInfo> list_models() const;

    // Download a model from URL
    bool download(const std::string& url, const std::string& output_path);

    // Get model info by name
    ModelInfo* get_model(const std::string& name);

    // Delete a model
    bool remove(const std::string& name);

    // Parse GGUF file header
    static ModelInfo parse_gguf_header(const std::string& path);

private:
    std::string models_dir_;
    std::vector<ModelInfo> models_;
};

} // namespace gizmo

#endif // GIZMO_MODEL_MANAGER_HPP
