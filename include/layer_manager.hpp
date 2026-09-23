#ifndef GIZMO_LAYER_MANAGER_HPP
#define GIZMO_LAYER_MANAGER_HPP

#include <cstdint>
#include <cstddef>
#include <utility>
#include <vector>

namespace gizmo {

struct LayerInfo {
    int32_t index;
    size_t size_bytes;
    bool loaded;
};

// A list of byte ranges (offset, length) within the GGUF mmap region that
// belong to a particular transformer block. Offsets are absolute from the
// base of the mmap; lengths are byte counts. Page-aligned by the producer.
using BlockRanges = std::vector<std::pair<size_t, size_t>>;

class LayerManager {
public:
    LayerManager();
    ~LayerManager();

    // Initialize with model layer count
    bool initialize(int32_t total_layers, size_t layer_size_bytes);

    // Load a range of layers into RAM
    bool load_layers(int32_t start_index, int32_t count);

    // Unload layers from RAM
    bool unload_layers(int32_t start_index, int32_t count);

    // Check if a specific layer is loaded
    bool is_layer_loaded(int32_t index) const;

    // Get currently loaded layer count
    int32_t get_loaded_count() const;

    // Get max layers that can be loaded based on memory limit
    int32_t calculate_max_layers(size_t available_ram_bytes, size_t layer_size_bytes) const;

    // Memory tracking
    size_t get_current_memory_usage() const;

    // Per-block byte ranges (mmap-relative offsets + lengths), set by the
    // sharded engine after it walks the model's tensor map. Used by the
    // madvise loop; not used by load_layers / unload_layers.
    const BlockRanges& block_ranges(int layer_idx) const;
    void set_block_ranges(int layer_idx, BlockRanges r);

private:
    int32_t total_layers_;
    int32_t max_loaded_layers_;
    size_t layer_size_bytes_;
    size_t current_memory_usage_;
    std::vector<LayerInfo> layers_;
    std::vector<BlockRanges> block_ranges_;
};

} // namespace gizmo

#endif // GIZMO_LAYER_MANAGER_HPP
