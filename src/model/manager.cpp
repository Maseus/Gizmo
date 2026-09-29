#include "model_manager.hpp"
#include "gguf.h"
#include <climits>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>

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
    if (url.empty()) {
        std::cerr << "Error: download URL is empty\n";
        return false;
    }

    // Determine the destination directory.
    std::string dir = models_dir_;
    if (dir.empty()) {
        const char* home = std::getenv("HOME");
        if (home) {
            dir = std::string(home) + "/.local/share/gizmo/models";
        } else {
            dir = "./models";
        }
    }

    // Determine the output filename.
    std::string file;
    if (!output_path.empty()) {
        file = output_path;
    } else {
        auto pos = url.find_last_of('/');
        file = (pos == std::string::npos || pos == url.size() - 1) ? "model.gguf" : url.substr(pos + 1);
    }

    // If only a filename was given, place it in the models directory.
    fs::path out_path;
    if (fs::path(file).is_absolute() || file.find('/') != std::string::npos) {
        out_path = file;
    } else {
        out_path = fs::path(dir) / file;
    }

    try {
        fs::create_directories(out_path.parent_path());
    } catch (const std::exception& e) {
        std::cerr << "Error: could not create directory " << out_path.parent_path()
                  << ": " << e.what() << "\n";
        return false;
    }

    std::cout << "Downloading:\n  URL:  " << url << "\n  To:   " << out_path.string() << "\n";

    // Prefer curl; fall back to wget.
    std::string cmd;
    const char* term = std::getenv("TERM");
    (void)term;
    cmd = "curl -L --fail --progress-bar -o '" + out_path.string() + "' '" + url + "'";

    int rc = std::system(cmd.c_str());
    if (rc == 0 && fs::exists(out_path) && fs::file_size(out_path) > 0) {
        std::cout << "Download complete: " << out_path.string() << "\n";
        return true;
    }

    // Retry with wget if curl failed.
    std::cerr << "curl failed (rc=" << rc << "), trying wget...\n";
    cmd = "wget -q --show-progress -O '" + out_path.string() + "' '" + url + "'";
    rc = std::system(cmd.c_str());
    if (rc == 0 && fs::exists(out_path) && fs::file_size(out_path) > 0) {
        std::cout << "Download complete: " << out_path.string() << "\n";
        return true;
    }

    std::cerr << "Error: download failed (rc=" << rc << ")\n";
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

static int64_t read_int_meta(gguf_context* ctx, int64_t key) {
    if (key < 0) return 0;
    switch (gguf_get_kv_type(ctx, key)) {
        case GGUF_TYPE_INT8:   return gguf_get_val_i8(ctx, key);
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(ctx, key);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(ctx, key);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, key);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, key);
        case GGUF_TYPE_UINT32: return static_cast<int64_t>(gguf_get_val_u32(ctx, key));
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(ctx, key);
        case GGUF_TYPE_UINT64: return static_cast<int64_t>(gguf_get_val_u64(ctx, key));
        default: return 0;
    }
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

    if (fs::exists(path)) {
        info.size_bytes = static_cast<int64_t>(fs::file_size(path));
    }

    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;

    gguf_context* ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) {
        return info;
    }

    // Architecture / name
    const int64_t arch_key = gguf_find_key(ctx, "general.architecture");
    if (arch_key >= 0) {
        info.arch = gguf_get_val_str(ctx, arch_key);
    }

    const int64_t name_key = gguf_find_key(ctx, "general.name");
    if (name_key >= 0) {
        info.name = gguf_get_val_str(ctx, name_key);
    }

    const int64_t param_key = gguf_find_key(ctx, "general.parameter_count");
    if (param_key >= 0) {
        info.parameter_count = read_int_meta(ctx, param_key);
    }

    // Total number of GGUF KV entries; reused by metadata scans below.
    const int64_t n_kv = gguf_get_n_kv(ctx);

    // Context length: prefer the explicit top-level key, then llama.context_length,
    // then <arch>.context_length. Store 0 if none are present.
    const char* ctx_keys[] = {
        "context_length",
        "llama.context_length",
        nullptr
    };
    for (int i = 0; ctx_keys[i]; ++i) {
        const int64_t ctx_key = gguf_find_key(ctx, ctx_keys[i]);
        if (ctx_key >= 0) {
            int64_t v = read_int_meta(ctx, ctx_key);
            if (v > 0) {
                info.context_length = static_cast<int32_t>(
                    v > INT32_MAX ? INT32_MAX : v);
                break;
            }
        }
    }
    if (info.context_length == 0) {
        for (int64_t i = 0; i < n_kv; ++i) {
            const char* key = gguf_get_key(ctx, i);
            if (std::strstr(key, ".context_length") != nullptr) {
                int64_t v = read_int_meta(ctx, i);
                if (v > 0) {
                    info.context_length = static_cast<int32_t>(
                        v > INT32_MAX ? INT32_MAX : v);
                    break;
                }
            }
        }
    }

    // Layer count: look for <arch>.block_count (e.g. qwen3.block_count).
    for (int64_t i = 0; i < n_kv; ++i) {
        const char* key = gguf_get_key(ctx, i);
        if (std::strstr(key, ".block_count") != nullptr) {
            info.total_layers = static_cast<int>(read_int_meta(ctx, i));
            break;
        }
    }

    // Vocabulary size from tokenizer.ggml.tokens array length.
    const int64_t vocab_key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (vocab_key >= 0 && gguf_get_kv_type(ctx, vocab_key) == GGUF_TYPE_ARRAY) {
        info.vocab_size = static_cast<int>(gguf_get_arr_n(ctx, vocab_key));
    }

    // Embedding dimension from the output/embedding tensor (ne[0] == hidden size).
    const char* embd_names[] = {
        "output.weight",
        "token_embd.weight",
        "tok_embeddings.weight",
        "embed_tokens.weight",
        nullptr
    };
    for (int i = 0; embd_names[i]; ++i) {
        const int64_t tid = gguf_find_tensor(ctx, embd_names[i]);
        if (tid >= 0) {
            const int64_t* ne = gguf_get_tensor_ne(ctx, tid);
            if (ne) {
                info.embedding_dim = static_cast<int>(ne[0]);
            }
            break;
        }
    }

    gguf_free(ctx);
    return info;
}

} // namespace gizmo
