#ifndef GIZMO_MODEL_PICKER_HPP
#define GIZMO_MODEL_PICKER_HPP

#include <string>
#include <vector>
#include <cstdint>

namespace gizmo {

// One discoverable model entry from a default model source.
struct DiscoveredModel {
    std::string source;      // "ollama" or "lmstudio"
    std::string name;      // display name
    std::string path;      // absolute path to the GGUF file
    uint64_t size_bytes = 0;
};

// Scan default Ollama and LM Studio model directories.
std::vector<DiscoveredModel> discover_default_models();

// Interactive terminal picker. Returns empty string if user cancels.
std::string pick_model_interactive();

} // namespace gizmo

#endif // GIZMO_MODEL_PICKER_HPP
