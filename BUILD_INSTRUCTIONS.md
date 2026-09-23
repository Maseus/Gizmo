# Building Gizmo

Gizmo is a C++ wrapper around [llama.cpp](https://github.com/ggerganov/llama.cpp)
with a small CLI for running quantized LLMs locally and reporting real memory
usage.

## Requirements

- A C++17 compiler (GCC/Clang)
- CMake 3.14+
- `git` (for the llama.cpp submodule)
- Patience — the first build compiles llama.cpp as a static library

## Clone and prepare

```bash
git clone https://github.com/Maseus/Gizmo.git
cd Gizmo

# Pull the llama.cpp submodule
git submodule update --init --recursive

# Apply the Gizmo-specific patches to llama.cpp
cd llama.cpp
git apply ../patches/llama.cpp-gizmo.patch
cd ..
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The `gizmo` binary will be at `build/gizmo`.

## Install

```bash
cmake --install build --prefix ~/.local
```

After installing, `gizmo` should be on your PATH (assuming `~/.local/bin` is
in PATH).

## Run

```bash
# Show help
gizmo --help

# Run a single prompt through a GGUF model
gizmo run -m /path/to/model.gguf -p "Hello, world"

# Run with real-time memory sampling printed to stderr
gizmo run -m /path/to/model.gguf -p "Hello, world" --measure-ram
```

## Serve

```bash
# Start an OpenAI-compatible server on port 11434
gizmo serve -m /path/to/model.gguf

# Or let gizmo scan your Ollama / LM Studio models and show a picker
gizmo serve

# Bind to a specific address / port
gizmo serve -m /path/to/model.gguf --host 0.0.0.0 --port 8080

# From another terminal or machine
curl http://localhost:11434/v1/models
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
```

The server prints its listening address and any local-network IPs so you
can connect from another device on the same LAN.

## Ollama model paths

If you already have the model in Ollama, the GGUF blob is under
`~/.ollama/models/blobs/`:

```bash
ollama list
ls -la ~/.ollama/models/blobs/
```

Use the `sha256-...` file path directly with `-m`.

## Memory reporting

Use `--measure-ram` to print `VmRSS` samples to stderr while generating. This
is the most reliable way to see how much RAM a given model + layer count
uses on your machine. Reported numbers depend on the exact GGUF quantization,
context length, and resident layer count.
