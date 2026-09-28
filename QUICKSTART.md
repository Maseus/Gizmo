# Gizmo Quick Start

## Prerequisites

- C++17 compatible compiler (GCC 8+ or Clang 7+)
- CMake 3.14+
- Linux (primary target)

## Building

```bash
cd gizmo-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Running

```bash
# Show help
./build/gizmo --help

# Show system info
./build/gizmo info

# List models
./build/gizmo list

# Launch the interactive chat TUI (pick a model, then chat with live speed/RSS footer)
./build/gizmo

# Or launch the server dashboard TUI instead
./build/gizmo tui

# Run a model with a prompt
./build/gizmo run -m /path/to/model.gguf -p "Hello!"

# Run with bounded tokens and measure RAM
./build/gizmo run -m /path/to/model.gguf -p "Hello!" -r 1 -n 4 --measure-ram

# Show an in-place block/token progress indicator during prefill/decode (TTY only)
./build/gizmo run -m /path/to/model.gguf -p "Hello!" -r 1 -n 16 --progress

# Run a qwen3 model with the lowest-memory end-to-end path
./build/gizmo run -m /path/to/Qwen3-4B-Q4_K_M.gguf \
    -p "What is the capital of France?" -r 1 -n 4 --measure-ram

# Full-model baseline (prefaults all weights)
./build/gizmo run -m /path/to/Qwen3-4B-Q4_K_M.gguf \
    -p "What is the capital of France?" -r 1 -n 4 --no-shard

# Keep weights resident for faster decode (higher RSS)
./build/gizmo run -m /path/to/model.gguf -p "Hello!" -r 1 -n 4 --no-evict

# Validate sharded parity on a supported (qwen3) model
./build/gizmo validate -m /path/to/Qwen3-4B-Q4_K_M.gguf -r 1

# Sweep configs
./build/gizmo sweep -m /path/to/Qwen3-4B-Q4_K_M.gguf

# Interactive chat (with model picker when no -m is given)
./build/gizmo chat
./build/gizmo chat -m /path/to/model.gguf -n 256

# Add custom model directories (colon-separated; also use GIZMO_MODEL_PATH)
./build/gizmo chat --model-path ~/models:/data/ggufs

# HTTP server (OpenAI-compatible; `serve` and `server` both work)
./build/gizmo serve                 # interactive model + feature picker
./build/gizmo serve -m /path/to/model.gguf --port 8080 --cors
./build/gizmo server -m /path/to/model.gguf --port 8080

curl http://127.0.0.1:8080/v1/health
curl http://127.0.0.1:8080/v1/models
```

## Current Status

- ✅ Project structure
- ✅ CLI parser (all commands wired)
- ✅ Inference engine with llama.cpp integration
- ✅ Per-block sharded prefill engine (validated against `llama_decode` on qwen3 models)
- ✅ Per-block sharded decode engine (validated against `llama_decode` on qwen3 models)
- ✅ End-to-end low-memory `gizmo run` on qwen3 models
- ✅ GGUF header parsing in `gizmo list`
- ✅ Chat command with chat-template support
- ✅ qwen3.5-family / qwen3.8 sharded-engine support (validated on 0.8B–27B)
- ✅ `--progress` in-place indicator for sharded prefill/decode
- ✅ Interactive default chat TUI with model picker + live speed/RSS footer
- ✅ Shared model discovery used by chat TUI and server TUI
- ✅ `--model-path` / `GIZMO_MODEL_PATH` for custom model search directories
- ✅ `serve` command alias and `/v1/health`, `/v1/` endpoints
- ⏳ Built-in model download via libcurl

## Architecture

```
gizmo-dev/
├── include/           # Header files
│   ├── chat_tui.hpp
│   ├── cli_parser.hpp
│   ├── inference_engine.hpp
│   ├── layer_manager.hpp
│   ├── model_discovery.hpp
│   ├── model_manager.hpp
│   ├── proc_status.hpp
│   ├── server/server.hpp
│   └── tui.hpp
├── src/               # Source files
│   ├── main.cpp
│   ├── cli/parser.cpp
│   ├── layer/manager.cpp
│   ├── model/manager.cpp
│   ├── inference/engine.cpp
│   ├── server/server.cpp
│   ├── ui/chat_tui.cpp
│   ├── ui/model_discovery.cpp
│   ├── ui/tui.cpp
│   └── util/proc_status.cpp
├── tools/sharded_engine/  # Per-block prefill/decode engine
│   ├── main.cpp
│   ├── multi_block.cpp
│   ├── shard_block.cpp
│   └── tail_graph.cpp
├── llama.cpp/         # Submodule
├── build/             # Build output
├── CMakeLists.txt
└── README.md
```
