# Building Gizmo with llama.cpp

## Current Status

The basic Gizmo CLI builds and runs with the simple Makefile:

```bash
make
./build/gizmo --help
```

This gives you:
- ✅ Working CLI with all commands
- ✅ Layer sharding math (memory calculations)
- ✅ Model specs for qwen3.8:27b

## Building with llama.cpp Integration

To get actual inference working, you need cmake:

### Step 1: Install cmake

```bash
sudo pacman -S cmake
```

### Step 2: Build llama.cpp and Gizmo

```bash
cd gizmo-dev

# Initialize llama.cpp submodule (if not already done)
git submodule init
git submodule update

# Build with cmake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Step 3: Run with actual inference

```bash
# The model path for Ollama models
MODEL_PATH="$HOME/.ollama/models/blobs/sha256-<digest>"

# Run with layer sharding
./build/gizmo run -m "$MODEL_PATH" -l 3
```

## Finding Your qwen3.8:27b Model Path

Ollama stores models in `~/.ollama/models/`. To find the actual GGUF file:

```bash
# List ollama models
ollama list

# Find the blob path (digest from the list output)
ls -la ~/.ollama/models/blobs/

# The model file will be something like:
# sha256-22130167c4c20e20c71454612966ca8e8171e9b3cc8ab6ce8aa6cbfec79643
```

## Layer Sharding with llama.cpp

llama.cpp uses `n_gpu_layers` to control how many layers to load. In Gizmo:

```bash
# Ultra-low-memory mode (3 layers at a time)
./build/gizmo run -m model.gguf -l 3

# Extreme memory savings (1 layer at a time)
./build/gizmo run -m model.gguf -l 1

# Balanced mode (16 layers)
./build/gizmo run -m model.gguf -l 16
```

## Expected Memory Usage (qwen3.8:27b)

| Layers | RAM | Use Case |
|--------|-----|----------|
| 1 | ~522 MB | Absolute minimum |
| 3 | ~1.5 GB | Ultra-low-memory default |
| 8 | ~4.7 GB | Low memory systems |
| 16 | ~8.2 GB | Balanced |
| 34 | ~17.3 GB | Full model |

## Troubleshooting

### cmake not found
Install with: `sudo pacman -S cmake`

### llama.cpp submodule not found
Run: `git submodule init && git submodule update`

### Model not found
Make sure to use the full path to the GGUF file, not just the model name.
