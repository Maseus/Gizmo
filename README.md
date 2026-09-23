# Gizmo

A C++ command-line wrapper around llama.cpp for running quantized LLMs
locally, with a focus on **honest memory reporting** and an installable
single-binary distribution.

## Quick Start

```bash
# Clone and prepare the llama.cpp submodule
git clone https://github.com/Maseus/Gizmo.git
cd Gizmo
git submodule update --init --recursive
cd llama.cpp
git apply ../patches/llama.cpp-gizmo.patch
cd ..

# Build (uses CMake; build/ contains the static binary)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Install to ~/.local/bin so `gizmo` is on PATH
cmake --install build --prefix ~/.local

# Use it from anywhere
gizmo --help
gizmo info
gizmo run -m /path/to/model.gguf -p "Hello"
gizmo run -m /path/to/model.gguf -p "Hello" --measure-ram

# Start an OpenAI-compatible server (default port 11434, like Ollama)
gizmo serve -m /path/to/model.gguf
gizmo serve -m /path/to/model.gguf --host 0.0.0.0 --port 8080
```

## What Gizmo does

- Wraps llama.cpp's library API with a small CLI (`run`, `chat`, `serve`,
  `bench`, `list`, `download`, `info`).
- Exposes an **OpenAI-compatible HTTP server** via `gizmo serve`, with
  endpoints for `/v1/models`, `/v1/chat/completions`, and `/v1/completions`,
  plus streaming (`stream: true`) SSE responses.
- Reports **real** memory usage (`/proc/self/status:VmRSS` / `VmHWM`) at
  model load, during generation, and at end-of-run. Use `--measure-ram`
  to print periodic VmRSS samples to stderr while generation is running.
- Builds the llama.cpp dependency as a static library so the `gizmo`
  binary is self-contained — one file, runs from anywhere, no
  `LD_LIBRARY_PATH` gymnastics.

## What Gizmo does **not** do (yet)

The original Gizmo README described a "92% memory savings" feature based
on per-block layer sharding. **That feature does not work**, and the
numbers in the earlier docs were arithmetic on hard-coded constants,
not measurements. The reason:

- llama.cpp builds **one** `ggml_cgraph` per `llama_decode` call
  containing **all** transformer blocks. Every decode step touches
  every layer's weights.
- A `posix_madvise(MADV_DONTNEED)` loop that evicts blocks between
  decodes can't save RAM, because the very next decode re-faults them.

To actually achieve "load only N blocks' worth of weights" you need a
**custom forward pass** that walks one block at a time, builds a
single-layer graph per step, runs `llama_decode`, and threads the
residual stream + KV cache across per-block calls. That's a multi-week
to multi-month project on its own and is tracked as future work; the
current code keeps the `BlockRanges` plumbing in `LayerManager` so that
when the custom forward pass lands, the per-block range map is ready.

## Commands

| Command     | Description |
|-------------|-------------|
| `run`       | Run a model with a single prompt |
| `chat`      | Start interactive chat session |
| `bench`     | Run memory/performance benchmarks |
| `serve`     | Start OpenAI-compatible HTTP server |
| `info`      | Show system memory info |
| `list`      | *(stub)* List downloaded models |
| `download`  | *(stub)* Download a model from URL |

## Options

- `-m, --model <path>` — path to a GGUF file
- `-l, --layers <N>` — alias for `--resident-layers` (currently
  informational; see "What Gizmo does not do" above)
- `-r, --resident-layers <N>` — number of blocks to keep resident
  (default 3; currently informational)
- `--no-shard` — explicit "do nothing special" (currently identical to
  default; placeholder for the future sharded engine)
- `--measure-ram` — print VmRSS to stderr every
  `--measure-interval-ms` ms during `run`/`chat`
- `--host <addr>` — `serve` bind address (default `0.0.0.0`)
- `--port <n>` — `serve` port (default `11434`)
- `-p, --prompt <text>` — prompt for `run`
- `-u, --url <url>` — URL for `download`
- `-i, --interactive` — interactive mode
- `-h, --help` — help

## Measured RAM (Qwen2.5-0.5B-Instruct Q4_K_M, on this machine)

```
$ gizmo run -m qwen2.5-0.5b-instruct-q4_k_m.gguf -p "Hello" --measure-ram
...
VmRSS after model load: 552 MB  (delta: +547 MB)
[measure] VmRSS=552 MB  VmHWM=555 MB
...
VmRSS after generation: 569 MB
VmHWM (peak resident):  569 MB
```

These are the actual numbers, not arithmetic. The 552 MB matches the
462 MiB file size plus tokenizer (~50 MB) plus KV cache allocator
overhead.

## Project Structure

```
gizmo-dev/
├── include/
│   ├── cli_parser.hpp
│   ├── inference_engine.hpp
│   ├── layer_manager.hpp
│   ├── model_manager.hpp
│   ├── chat_handler.hpp
│   ├── proc_status.hpp
│   └── server.hpp
├── src/
│   ├── main.cpp
│   ├── cli/parser.cpp
│   ├── layer/manager.cpp
│   ├── model/manager.cpp
│   ├── inference/engine.cpp
│   ├── chat/handler.cpp
│   ├── util/proc_status.cpp
│   └── server/server.cpp
├── llama.cpp/                # Submodule (apply patches/llama.cpp-gizmo.patch)
├── patches/                  # Patches required on top of the llama.cpp submodule
├── build/gizmo               # Static binary (generated)
├── CMakeLists.txt            # Build system
└── README.md
```

## Build System

- `git submodule update --init --recursive`
- `cd llama.cpp && git apply ../patches/llama.cpp-gizmo.patch && cd ..`
- `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` — configure
- `cmake --build build -j$(nproc)` — build
- `cmake --install build --prefix ~/.local` — install to `~/.local/bin`

Use CMake. The generated `Makefile` is produced by CMake and is not
committed in this repo.

## License

MIT

## Contributing

Real areas where help is welcome:

- The custom per-block forward pass (see "What Gizmo does not do" above).
  This is the actual sharded engine and is a substantial project on its
  own.
- GGUF metadata parsing in `ModelManager::parse_gguf_header` (currently
  returns `total_layers=0`; main path uses `llama_model_n_layer()`).
- Real `download` command via libcurl.
- `chat` command implementation (header exists; main.cpp does not wire
  it up).