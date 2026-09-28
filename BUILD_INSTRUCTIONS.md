# Building Gizmo with llama.cpp

## Current Status

Gizmo uses **CMake** to build both llama.cpp (as a static submodule library) and the `gizmo` executable. The old Makefile is kept for reference but does not link llama.cpp.

## Prerequisites

- C++17 compatible compiler (GCC 8+ or Clang 7+)
- CMake 3.14+
- Linux (primary target)

Install CMake on Arch-based systems:

```bash
sudo pacman -S cmake
```

## Building

```bash
cd gizmo-dev

# Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build (llama.cpp static libs + gizmo executable)
cmake --build build -j$(nproc)
```

The resulting binary is at `build/gizmo`.

## Installing

```bash
cmake --install build --prefix ~/.local
# Now gizmo is on PATH if ~/.local/bin is in your shell profile
```

## Running

```bash
# Show help
./build/gizmo --help

# Run with a prompt
./build/gizmo run -m /path/to/model.gguf -p "Hello!"

# Low-memory sharded path on a qwen3 model
./build/gizmo run -m /path/to/Qwen3-4B-Q4_K_M.gguf -p "Hello!" -r 1 -n 4 --measure-ram
```

## Finding Your Model Path

`gizmo list` scans common directories for `.gguf` files:

- `~/.local/share/gizmo/models`
- `~/.lmstudio/models`
- `~/.ollama/models/blobs`
- `./models`

You can also pass the full path directly with `-m /path/to/model.gguf`.

## Layer Sharding

With sharding enabled (the default), the GGUF is mmap'd but not prefaulted. Only the active block's weights are faulted into RAM; finished blocks can be evicted with `madvise(MADV_DONTNEED)`. Use `-r` to control the resident-layer sliding window and `-K` to chain blocks into row graphs.

```bash
# Extreme low memory (qwen3 only)
./build/gizmo run -m model.gguf -r 1 -p "Hello"

# Conservative window
./build/gizmo run -m model.gguf -r 4 -p "Hello"

# Default window
./build/gizmo run -m model.gguf -r 8 -p "Hello"

# Disable sharding entirely (full prefault)
./build/gizmo run -m model.gguf --no-shard -p "Hello"
```

## Expected Memory Usage (qwen3 Qwen3-4B Q4_K_M)

| Resident Layers | RAM | Use Case |
|-----------------|-----|----------|
| 1 | ~510 MB | Absolute minimum |
| 4 | ~700 MB | Ultra-low memory |
| 8 | ~1.0 GB | Balanced |
| all (`--no-shard`) | ~2.5 GB | Full model |

## Troubleshooting

### cmake not found
Install with: `sudo pacman -S cmake`

### llama.cpp build errors
Make sure the `llama.cpp/` directory contains the upstream source tree. If it is empty or only partially populated, re-clone or re-initialize the submodule:

```bash
git submodule update --init --recursive
```

### Model not found
Use the full path to the GGUF file, or place models in one of the scanned directories and run `gizmo list`.

### Out of memory on large models
If a model is too large for the sharded path or you want the full-model prefault baseline, pass `--no-shard` to use native `llama_decode`. For very large qwen3.5/qwen3.8 models such as Qwen3.8-27B on a 16 GB host, use `-r 1` so the per-block engine keeps only one layer resident at a time (~2.1 GB peak).
