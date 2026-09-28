# Gizmo Quick Start

## Prerequisites

- C++17 compatible compiler (GCC 8+ or Clang 7+)
- CMake 3.14+
- Git (for llama.cpp submodule)
- Linux (primary target)

## Building

```bash
cd gizmo-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

If the llama.cpp submodule is not initialized:

```bash
git submodule init
git submodule update
```

## Running

```bash
# Show help
./build/gizmo --help

# Show system info
./build/gizmo info

# List models
./build/gizmo list

# Run a model with a prompt
./build/gizmo run -m /path/to/model.gguf -p "Hello!"

# Run with bounded tokens and measure RAM
./build/gizmo run -m /path/to/model.gguf -p "Hello!" -r 1 -n 4 --measure-ram

# Run qwen3.8:27b with the lowest-memory end-to-end path (~2 GB HWM)
./build/gizmo run -m /path/to/Qwen3.8-27B-Q4_K_M.gguf \
    -p "What is the capital of France?" -r 1 -n 4 --measure-ram

# Full-model baseline (prefaults all weights; ~9 GB HWM on 27B)
./build/gizmo run -m /path/to/Qwen3.8-27B-Q4_K_M.gguf \
    -p "What is the capital of France?" -r 1 -n 4 --no-shard

# Keep weights resident for faster decode (higher RSS)
./build/gizmo run -m /path/to/model.gguf -p "Hello!" -r 1 -n 4 --no-evict
```

## Current Status

- ✅ Project structure
- ✅ CLI parser (all commands working)
- ✅ Inference engine with llama.cpp integration
- ✅ Per-block sharded prefill engine (validated against `llama_decode`)
- ✅ Per-block sharded decode engine (validated against `llama_decode`)
- ✅ End-to-end low-memory `gizmo run` on qwen3.8:27b
- ⏳ Model download
- ⏳ Full chat command wiring

## Architecture

```
gizmo-dev/
├── include/           # Header files
│   ├── cli_parser.hpp
│   ├── inference_engine.hpp
│   └── proc_status.hpp
├── src/               # Source files
│   ├── main.cpp
│   ├── cli/parser.cpp
│   ├── layer/manager.cpp
│   ├── model/manager.cpp
│   ├── inference/engine.cpp
│   ├── chat/handler.cpp
│   └── util/proc_status.cpp
├── tools/sharded_engine/  # Per-block prefill/decode engine
│   ├── main.cpp
│   ├── multi_block.cpp
│   ├── shard_block.cpp
│   └── tail_graph.cpp
├── llama.cpp/         # Submodule (after init)
├── build/             # Build output
├── CMakeLists.txt
└── README.md
```
