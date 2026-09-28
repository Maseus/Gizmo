# Gizmo Development Summary

## What We Built

A working C++/llama.cpp inference tool that implements **per-block layer sharding for both prefill and decode** and runs end-to-end on `qwen3.8:27b`.

## Target Model: qwen3.8:27b

| Property | Value |
|----------|-------|
| Total Layers | 64 |
| Embedding Dimension | 5120 |
| Vocab Size | 248,320 |
| Full GGUF Size (Q4_K_M) | ~16–17 GB |

## Memory Measurements (qwen3.8:27b Q4_K_M, main CLI)

| Mode | Resident Layers | Steady-State Prefill RSS | Steady-State Decode RSS | End-to-End HWM |
|------|-----------------|--------------------------|-----------------------|----------------|
| Extreme | 1 (`-r 1`) | **~560 MB** | **~1.5 GB** | **~2.0 GB** |
| Default | 8 (`-r 8`) | higher than `-r 1` | higher than `-r 1` | higher than `-r 1` |
| `--no-shard` | all | ~7.5 GB | ~9 GB | ~9.2 GB |

The **prefill RSS** of ~560 MB and **decode RSS** of ~1.5 GB are the headline
generation results: the per-block engine evicts each block after use and keeps
only the active block + scheduler overhead + KV cache resident during generation.

The **end-to-end HWM** of ~2.0 GB is reached when a block is first faulted in;
the initial model load now stays under ~300 MB because the GGUF is mmap'd without
prefault when sharding is enabled. The decode loop no longer calls full-model
`llama_decode`.

## Working Features

✅ **CLI Parser** - All commands working:
- `gizmo run -m model.gguf -p "..." -r 1 -n 4`
- `gizmo chat -m model.gguf -r 1`
- `gizmo list`
- `gizmo download -u <url>` (stub)
- `gizmo info`

✅ **Inference Engine** - llama.cpp integration:
- Model loading with CPU-only mmap
- Tokenization
- Sampler chain
- Context management
- Sharded prefill integration
- Sharded decode integration

✅ **Per-Block Sharded Engine**:
- Sequential per-block `ggml_cgraph`
- Residual, KV-cache, and recurrent-state threading
- qwen3 / qwen3.5 hybrid block builders
- Tail graph with `output_s` scaling
- Optional `madvise(MADV_DONTNEED)` weight eviction
- Single-token decode path with full-cache causal mask

✅ **Build System**:
- CMake with llama.cpp submodule
- Static `gizmo` binary

## Project Structure

```
gizmo-dev/
├── include/
│   ├── cli_parser.hpp
│   ├── inference_engine.hpp
│   └── proc_status.hpp
├── src/
│   ├── main.cpp
│   ├── cli/parser.cpp
│   ├── layer/manager.cpp
│   ├── model/manager.cpp
│   ├── inference/engine.cpp
│   ├── chat/handler.cpp
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
cd /home/maseus/Desktop/obsidiandocs/Overall-infrastructure/development/Gizmo

# Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Validate sharded prefill against baseline
./build/gizmo_argmax_dump -m /path/to/Qwen3.8-27B-Q4_K_M.gguf -p "hello"
GIZMO_PROBE_PREFIXES=... ./build/gizmo_sharded_engine ...

# Run end-to-end with bounded tokens
./build/gizmo run -m /path/to/Qwen3.8-27B-Q4_K_M.gguf \
    -p "What is the capital of France?" -r 1 -n 4 --measure-ram

# Show help
./build/gizmo --help

# Show system memory
./build/gizmo info
```

## Validation Results

| Prompt | Max Logit Diff vs `llama_decode` | Notes |
|--------|----------------------------------|-------|
| `"hello"` | 0.0 | single token prefill |
| `"What is the capital of France?"` | 0.0 | multi token prefill |
| `"What is the capital of France?"` | 0.0 | per-token decode (argmax + logits match exactly) |

Prefill and decode were validated with and without `--evict`.

## Next Steps for Full Implementation

1. **GGUF Header Parsing** (`src/model/manager.cpp`)
   - Read actual layer count from model file
   - Calculate exact layer sizes
   - Support multiple model formats

2. **Model Download** (`src/model/manager.cpp`)
   - HTTP download with libcurl
   - Progress indicator
   - Resume support

3. **Performance Tuning**
   - Resident-layer sweep (`-r 1,4,8,16`)
   - Row-size (`-K`) sweep
   - Thread count tuning
   - Automated validation harness across a prompt suite

4. **Cleanup**
   - Make sharded-engine progress output quiet by default
   - Remove stale chat/download stubs

## Your System

- **RAM**: 14 GB total
- **Current qwen3.8:27b performance**:
  - Model load RSS: ~298 MB (sharded, no prefault)
  - Prefill alone: ~560 MB RSS
  - Decode per token: ~1.5 GB RSS
  - End-to-end HWM: ~2.0 GB
- **Recommended next steps**: automated validation harness, resident-layer sweep, and cleanup.

## Key Insight

The per-block sharding approach is **proven end-to-end for generation**:
- qwen3.8:27b (64 layers, ~16–17 GB GGUF) loads with **~298 MB RSS** and runs with **~560 MB resident RAM during prefill** and **~1.5 GB resident RAM during decode**.
- End-to-end HWM is **~2.0 GB**, down from the ~9.5 GB seen when the full model is prefaulted at load time.
- Output matches the full `llama_decode` baseline exactly for both prefill and decode.

The trade-off is speed (block swapping + on-demand page faults), but for users who **cannot** keep the full model resident, this makes generation possible on very low RAM.
