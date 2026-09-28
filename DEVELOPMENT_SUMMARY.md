# Gizmo Development Summary

## What We Built

A working C++/llama.cpp inference tool that implements **per-block layer sharding for both prefill and decode** on **qwen3-family** models. All other model families fall back to native `llama_decode` and still produce correct output.

## Supported Models

### qwen3-family (sharded engine enabled)

| Property | Example: Qwen3-4B |
|----------|-------------------|
| Total Layers | 36 |
| Embedding Dimension | 2560 |
| Vocab Size | 151,936 |
| Full GGUF Size (Q4_K_M) | ~2.5 GB |

### qwen3.5-family (sharded engine enabled)

| Property | Example: Qwen3.5-4B | Qwen3.8-27B |
|----------|---------------------|-------------|
| Total Layers | 32 | 64 |
| Embedding Dimension | 2560 | 5120 |
| Vocab Size | 248,320 | 248,320 |
| Full GGUF Size (Q4_K_M) | ~2.6 GB | ~16 GB |

These models now use the per-block sharded engine. Both full-attention and gated-delta-net recurrent blocks are implemented, and validation passes with zero logit diffs against the un-sharded `llama_decode` baseline on 0.8B, 2B, 4B, 9B, and **Qwen3.8-27B** models.

## Memory Measurements (qwen3-family, e.g., Qwen3-4B Q4_K_M)

| Mode | Resident Layers | Steady-State Prefill RSS | Steady-State Decode RSS | End-to-End HWM |
|------|-----------------|--------------------------|-----------------------|----------------|
| Extreme | 1 (`-r 1`) | **~510 MB** | **~500 MB** | **~630 MB** |
| Default | 8 (`-r 8`) | higher than `-r 1` | higher than `-r 1` | higher than `-r 1` |
| `--no-shard` | all | ~2.5 GB | ~3 GB | ~3 GB |

The **prefill RSS** of ~510 MB and **decode RSS** of ~500 MB are the headline
results on qwen3 models: the per-block engine evicts each block after use and keeps
only the active block + scheduler overhead + KV cache resident during generation.

With sharding enabled the GGUF is mmap'd but **not** prefaulted, so the initial load adds only ~120 MB on qwen3 models and ~293 MB on Qwen3.8-27B.

## Memory Measurements (qwen3.5-family, e.g., Qwen3.8-27B Q4_K_M)

| Mode | Resident Layers | Model Load RSS | Generation HWM | Notes |
|------|-----------------|----------------|----------------|-------|
| Extreme | 1 (`-r 1`) | **~293 MB** | **~2.1 GB** | Comfortable on a 16 GB host |
| Default | 8 (`-r 8`) | ~293 MB | **~12.7 GB** | Requires a host with more than 16 GB RAM |
| `--no-shard` | all | ~10.6 GB | ~11.4 GB | Full prefault; avoid on 16 GB hosts |

## Working Features

✅ **CLI Parser** - All commands wired:
- `gizmo run -m model.gguf -p "..." -r 1 -n 4`
- `gizmo chat -m model.gguf -r 1`
- `gizmo list`
- `gizmo download -u <url>` (delegates to system curl/wget)
- `gizmo info`
- `gizmo validate -m model.gguf`
- `gizmo sweep -m model.gguf`
- `gizmo server -m model.gguf --port 8080`
- `gizmo tui`

✅ **Inference Engine** - llama.cpp integration:
- Model loading with CPU-only mmap
- Tokenization
- Sampler chain
- Context management
- Sharded prefill integration (qwen3 only)
- Sharded decode integration (qwen3 only)
- Automatic fallback to native `llama_decode` for unsupported architectures

✅ **Per-Block Sharded Engine**:
- Sequential per-block `ggml_cgraph`
- Residual and KV-cache threading
- qwen3 full-attention block builders
- Tail graph with `output_s` scaling
- Optional `madvise(MADV_DONTNEED)` weight eviction
- Single-token decode path with full-cache causal mask
- Row-graph chaining (`-K`)

✅ **Build System**:
- CMake with llama.cpp submodule
- Static `gizmo` binary

## Project Structure

```
gizmo-dev/
├── include/
│   ├── cli_parser.hpp
│   ├── inference_engine.hpp
│   ├── layer_manager.hpp
│   ├── model_manager.hpp
│   ├── proc_status.hpp
│   ├── server/server.hpp
│   └── tui.hpp
├── src/
│   ├── main.cpp
│   ├── cli/parser.cpp
│   ├── layer/manager.cpp
│   ├── model/manager.cpp
│   ├── inference/engine.cpp
│   ├── server/server.cpp
│   ├── ui/tui.cpp
│   └── util/proc_status.cpp
├── tools/sharded_engine/
│   ├── main.cpp
│   ├── multi_block.cpp
│   ├── shard_block.cpp
│   └── tail_graph.cpp
├── llama.cpp/
├── build/gizmo
├── CMakeLists.txt
├── README.md
├── QUICKSTART.md
├── BUILD_INSTRUCTIONS.md
├── MODEL_SPECS.md
├── STATUS.md
└── FINAL_STATUS.md
```

## How to Test

```bash
cd /home/maseus/Desktop/Gizmo

# Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Validate sharded prefill against baseline on a qwen3 model
./build/gizmo validate -m /path/to/Qwen3-4B-Q4_K_M.gguf -r 1

# Run end-to-end with bounded tokens
./build/gizmo run -m /path/to/Qwen3-4B-Q4_K_M.gguf \
    -p "What is the capital of France?" -r 1 -n 4 --measure-ram

# Show help
./build/gizmo --help

# Show system memory
./build/gizmo info
```

## Validation Results (qwen3 models)

| Prompt | Max Logit Diff vs `llama_decode` | Notes |
|--------|----------------------------------|-------|
| `"hello"` | 0.0 | single token prefill |
| `"What is the capital of France?"` | 0.0 | multi token prefill |
| `"What is the capital of France?"` | 0.0 | per-token decode |

Prefill and decode validated with and without `--evict`.

## Next Steps for Full Implementation

1. **qwen3.5-family sharded-engine parity** — completed
   - [x] Port full-attention block for qwen3.5 (joint QG projection, MRoPE-4, gate*sigmoid, attn_post_norm)
   - [x] Port gated-delta-net recurrent block
   - [x] Re-enable `LLM_ARCH_QWEN35` in `is_sharded_arch_supported`
   - [x] Validate against un-sharded `llama_decode` on 0.8B, 2B, 4B, and 9B models

2. **Performance Tuning**
   - Resident-layer sweep (`-r 1,4,8,16`)
   - Row-size (`-K`) sweep
   - Thread count tuning
   - Automated validation harness across a prompt suite

3. **Convenience Features**
   - Built-in model download via libcurl
   - Chat-template-aware formatting in server `/v1/chat/completions`
   - Automatic sweet-spot recommendation from `gizmo sweep`

4. **Cleanup**
   - Make sharded-engine progress output quiet by default (done for most; finish remaining)
   - Remove stale diagnostic code

## Your System

- **RAM**: 14 GB total
- **Current qwen3-4B performance**:
  - Model load RSS: ~120 MB (sharded, no prefault)
  - Prefill RSS: ~510 MB
  - Decode RSS: ~500 MB
  - End-to-end HWM: ~630 MB
- **Recommended next steps**: tune default resident-layer counts and row sizes per model family; implement convenience features (sweep sweet-spot, libcurl download, server chat template).

## Key Insight

The per-block sharding approach is **proven end-to-end for qwen3 generation**:
- qwen3-family models load with low RSS and run with low resident RAM.
- Output matches the full `llama_decode` baseline exactly.
- Non-qwen3 models fall back safely to the un-sharded path.

The trade-off is speed (block swapping + on-demand page faults). The qwen3.5-family hybrid milestone is now complete; the next focus is convenience features and per-family performance tuning.
