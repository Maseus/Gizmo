#include "layer_manager.hpp"
#include <algorithm>

namespace gizmo {

LayerManager::LayerManager()
    : total_layers_(0)
    , max_loaded_layers_(0)
    , layer_size_bytes_(0)
    , current_memory_usage_(0) {
}

LayerManager::~LayerManager() {
    // Cleanup handled by unload
    unload_layers(0, total_layers_);
}

bool LayerManager::initialize(int32_t total_layers, size_t layer_size_bytes) {
    if (total_layers <= 0 || layer_size_bytes == 0) {
        return false;
    }

    total_layers_ = total_layers;
    layer_size_bytes_ = layer_size_bytes;
    max_loaded_layers_ = total_layers; // Default: all layers

    layers_.clear();
    layers_.reserve(total_layers);

    for (int32_t i = 0; i < total_layers; ++i) {
        layers_.push_back({i, layer_size_bytes, false});
    }

    block_ranges_.clear();
    block_ranges_.resize(total_layers);

    current_memory_usage_ = 0;
    return true;
}

bool LayerManager::load_layers(int32_t start_index, int32_t count) {
    if (start_index < 0 || start_index >= total_layers_) {
        return false;
    }

    int32_t actual_count = std::min(count, total_layers_ - start_index);

    for (int32_t i = start_index; i < start_index + actual_count; ++i) {
        if (!layers_[i].loaded) {
            layers_[i].loaded = true;
            current_memory_usage_ += layer_size_bytes_;
        }
    }

    return true;
}

bool LayerManager::unload_layers(int32_t start_index, int32_t count) {
    if (start_index < 0 || start_index >= total_layers_) {
        return false;
    }

    int32_t actual_count = std::min(count, total_layers_ - start_index);

    for (int32_t i = start_index; i < start_index + actual_count; ++i) {
        if (layers_[i].loaded) {
            layers_[i].loaded = false;
            current_memory_usage_ -= layer_size_bytes_;
        }
    }

    return true;
}

bool LayerManager::is_layer_loaded(int32_t index) const {
    if (index < 0 || index >= total_layers_) {
        return false;
    }
    return layers_[index].loaded;
}

int32_t LayerManager::get_loaded_count() const {
    int32_t count = 0;
    for (const auto& layer : layers_) {
        if (layer.loaded) {
            ++count;
        }
    }
    return count;
}

int32_t LayerManager::calculate_max_layers(size_t available_ram_bytes, size_t layer_size_bytes) const {
    if (layer_size_bytes == 0) {
        return 0;
    }
    // Use 80% of available RAM as safety margin
    size_t safe_ram = (available_ram_bytes * 80) / 100;
    int32_t max_layers = static_cast<int32_t>(safe_ram / layer_size_bytes);
    return std::min(max_layers, total_layers_);
}

size_t LayerManager::get_current_memory_usage() const {
    return current_memory_usage_;
}

const BlockRanges& LayerManager::block_ranges(int layer_idx) const {
    static const BlockRanges empty;
    if (layer_idx < 0 || layer_idx >= total_layers_) return empty;
    return block_ranges_[layer_idx];
}

void LayerManager::set_block_ranges(int layer_idx, BlockRanges r) {
    if (layer_idx < 0 || layer_idx >= total_layers_) return;
    block_ranges_[layer_idx] = std::move(r);
}

} // namespace gizmo
