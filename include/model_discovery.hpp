#ifndef GIZMO_MODEL_DISCOVERY_HPP
#define GIZMO_MODEL_DISCOVERY_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace gizmo {

// One discoverable model entry from a default model source.
struct DiscoveredModel {
    std::string source;      // "ollama", "lmstudio", "custom", "user"
    std::string name;        // display name
    std::string path;        // absolute path to the GGUF file
    uint64_t    size_bytes = 0;
};

// Build the ordered list of directories to scan, merging:
//   1) explicit extra_dirs
//   2) the GIZMO_MODEL_PATH environment variable
//   3) built-in defaults (~/.local/share/gizmo/models, ~/.lmstudio/models,
//      ~/.ollama/models/blobs, ./models)
// Duplicates are removed while preserving order.
std::vector<std::string> build_search_dirs(
    const std::vector<std::string>& extra_dirs = {});

// Scan the built-in (and optional extra) model directories for GGUF files,
// skipping helper files such as mmproj, mtp, embedding, and vision tensors.
// Results are sorted and de-duplicated.
std::vector<std::string> scan_for_ggufs(
    const std::vector<std::string>& extra_dirs = {});

// Scan common model directories (and any extra directories supplied by the
// caller) for GGUF files. Skips helper files such as mmproj, mtp, embedding,
// and vision tensors. A trailing "custom" entry is not appended here; callers
// that want it can add one after calling this function.
std::vector<DiscoveredModel> discover_models(
    const std::vector<std::string>& extra_dirs = {});

// Format a byte count as human-readable string (B/KB/MB/GB/TB).
std::string format_bytes(uint64_t bytes);

// Interactive terminal model picker. Returns the selected model path, or an
// empty string if the user cancels. Falls back to a numbered list when stdin
// or stdout is not a TTY.
std::string pick_model_interactive(const std::vector<DiscoveredModel>& models);

// Non-interactive numbered list fallback. Returns the selected path or "".
std::string pick_model_noninteractive(const std::vector<DiscoveredModel>& models);

} // namespace gizmo

#endif // GIZMO_MODEL_DISCOVERY_HPP
