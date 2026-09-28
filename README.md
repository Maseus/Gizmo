# Gizmo

A C++ command-line wrapper around llama.cpp for running quantized LLMs locally, with a focus on **honest memory reporting** and a per-block sharded inference engine (prefill + decode).

## Quick Start

```bash
# Build (uses CMake; build/ contains the static binary)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Install to ~/.local/bin so `gizmo` is on PATH
cmake --install build --prefix ~/.local

# Use it from anywhere
gizmo --help
gizmo info
gizmo run -m /path/to/model.gguf -p "Hello"
gizmo run -m /path/to/model.gguf -p "Hello" -r 1 -n 4 --measure-ram
gizmo chat -m /path/to/model.gguf -n 64
```

## What Gizmo does

- Wraps llama.cpp's library API with a small CLI (`run`, `chat`, `bench`, `validate`, `sweep`, `list`, `download`, `info`, `server`).
- Reports **real** memory usage (`/proc/self/status:VmRSS` / `VmHWM`) at model load, during generation, and at end-of-run. Use `--measure-ram` to print periodic VmRSS samples to stderr while generation is running.
- Builds the llama.cpp dependency as a static library so the `gizmo` binary is self-contained.
- Implements a **per-block sharded inference engine** for qwen3/qwen3.5-family models. This engine builds a separate `ggml_cgraph` per transformer block, threads the residual/KV/recurrent state, and can evict each block's weights after use. Output matches the native `llama_decode` baseline exactly for both prefill and decode.
- Automatically falls back to native `llama_decode` for unsupported architectures (e.g., qwen2/qwen2.5 and qwen35moe variants), so non-qwen3 models still work correctly with `--no-shard`-equivalent behavior.

## What Gizmo does **not** do (yet)

- Built-in model downloader (use `curl`/`wget` for now).
- Chat-template aware multi-turn chat (`gizmo chat` uses a plain text context, not the model's official template).

## Commands

| Command     | Description |
|-------------|-------------|
| `run`       | Run a model with a single prompt |
| `chat`      | Interactive chat with streaming output and tok/s stats |
| `bench`     | Memory/performance benchmark across prompt lengths |
| `validate`  | Compare sharded vs un-sharded logits on a built-in prompt suite |
| `sweep`     | Run a full generate() sweep over `-t`, `-r` and `-K` configs |
| `list`      | List downloaded models (placeholder) |
| `download`  | Download a model from URL (placeholder; use `curl`/`wget`) |
| `info`      | Show system memory info |
| `server`    | Start HTTP server (OpenAI-compatible API) |

## Options

- `-m, --model <path>` — path to a GGUF file
- `-l, --layers <N>` — alias for `--resident-layers`
- `-r, --resident-layers <N|list>` — number of blocks to keep resident. Single value for normal commands (default 8); comma-separated list for `sweep` (default `1,4,8,16`)
- `-K, --row-size <K|list>` — Phase 9: chain K blocks per `ggml_cgraph`. Single value for normal commands (default 1); comma-separated list for `sweep` (default `1,2,4`)
- `-t, --threads <N|list>` — CPU threads for `llama_decode` and the sharded backend. Single value for normal commands (default 4); comma-separated list for `sweep` (default `4`)
- `-n, --max-tokens <N>` — maximum tokens to generate for `run`/`chat` (default: 128)
- `--no-shard` — disable the sharded engine and use plain `llama_decode`; the full model is prefaulted at load time
- `--no-evict` — skip `madvise(MADV_DONTNEED)` after each block; keeps weights resident for faster generation at higher RSS
- `--measure-ram` — print VmRSS to stderr every `--measure-interval-ms` ms during `run`
- `--measure-interval-ms N` — sample interval (default 500)
- `-v, --verbose` — print sharded-engine diagnostic details (per-block progress, argmax reports)
- `-p, --prompt <text>` — prompt for `run`
- `--json` — emit validation/sweep results as JSON
- `--argmax-only` — validate only argmax parity, ignore logit diffs (for `validate`)
- `--max-diff <f>` — max abs logit diff tolerance (default: 1e-4, for `validate`)
- `--mean-diff <f>` — mean abs logit diff tolerance (default: 1e-5, for `validate`)
- `--sweep-max-tokens N` — `sweep`: decode tokens per configuration (default 16)
- `--sweep-prefill-tokens N` — `sweep`: prefill tokens per configuration (default 64)
- `-h, --help` — help

## Measured RAM

### qwen3.8:27b Q4_K_M (`-r 1 -n 4`)

```
$ gizmo run -m Qwen3.8-27B-Q4_K_M.gguf \
    -p "What is the capital of France?" -r 1 -n 4 --measure-ram
...
VmRSS after model load: 298 MB  (delta: +292 MB)
[measure] VmRSS=560 MB  VmHWM=2023 MB   # during prefill, after eviction
[measure] VmRSS=1874 MB  VmHWM=2023 MB   # during decode
...
VmRSS after generation: 1538 MB
VmHWM (peak resident):  2023 MB
The capital of
```

Add `--verbose` to see the sharded-engine per-block progress and per-token argmax reports.

### Validation

```
$ gizmo validate -m Qwen3.8-27B-Q4_K_M.gguf -r 1
Gizmo Validate: Qwen3.8-27B-Q4_K_M.gguf
  resident_layers=1 row_size=1

prompt                                 | tokens | argmax_match | max_abs_diff | mean_abs_diff | status
---------------------------------------+--------+--------------+--------------+---------------+-------
Hi                                     |      1 |          yes |     0.00e+00 |      0.00e+00 | PASS
Hello world                            |      2 |          yes |     0.00e+00 |      0.00e+00 | PASS
What is the capital of France?         |      7 |          yes |     0.00e+00 |      0.00e+00 | PASS
The capital of France is Paris. The... |     35 |          yes |     0.00e+00 |      0.00e+00 | PASS

All cases passed.
```

The `validate` command loads the model twice: once with `--no-shard` as the
baseline, and once with sharding enabled, then compares last-token logits for a
built-in prompt suite. Default tolerances are `max_abs_diff < 1e-4` and
`mean_abs_diff < 1e-5`; use `--argmax-only` to ignore logit values and only
check that the top token matches.

These are actual measurements, not arithmetic. With sharding enabled the GGUF is mmap'd but **not** prefaulted, so the initial load adds only ~292 MB. During the sharded prefill the resident set drops to ~560 MB, and the per-token sharded decode keeps RSS at ~1.5 GB. The end-to-end HWM is now ~2.0 GB instead of the ~9.5 GB seen with `--no-shard`.

With `--no-shard` the same model loads ~7.5 GB and peaks at ~9.2 GB, so the sharded path reduces the load RSS by ~96% and the end-to-end HWM by ~78%.

### Chat

```
$ gizmo chat -m Qwen3.8-27B-Q4_K_M.gguf -n 64

Gizmo Chat: Qwen3.8-27B-Q4_K_M.gguf
Type a message and press Enter. Commands: /quit, /reset, /clear
---------------------------------------------------------------

You: Hello
Assistant: Hi there! How can I help you today?
  [8 tokens in 108.04 s => 0.1 tok/s; RSS 3091 MB; HWM 5186 MB]

You: /quit
Chat ended.
```

`gizmo chat` is a simple interactive chat loop. It streams the assistant response as tokens arrive and prints throughput and memory after each turn. The context accumulates across turns; use `/reset` to clear it. It does not use the model's official chat template (that remains future work), so quality may vary on instruction-tuned models.

### Performance sweep

```
$ gizmo sweep -m Qwen3.8-27B-Q4_K_M.gguf

Gizmo Sweep: Qwen3.8-27B-Q4_K_M.gguf
  prefill tokens=64  decode tokens=16  no_evict=false

 config  |  t |  r |  K | wall (s) |  tok/s | VmRSS (MB) | VmHWM (MB) | prefill RSS (MB) | prefill HWM (MB)
---------+----+----+----+----------+--------+------------+------------+------------------+------------------
 unshard |  4 |  - |  - |    45.21 |   1.77 |       9234 |       9456 |                - |                -
 shard   |  4 |  1 |  1 |   128.34 |   0.62 |       1456 |       2134 |             1380 |             2023
 shard   |  4 |  1 |  2 |   104.12 |   0.77 |       1623 |       2456 |             1550 |             2345
 shard   |  4 |  4 |  1 |    76.55 |   1.05 |       1890 |       2678 |             1800 |             2567
 ...
```

The `sweep` command runs a full `generate()` for each `(threads, resident_layers, row_size)` configuration and reports wall time, throughput, and peak RSS/HWM. It begins with an un-sharded baseline per thread count so you can compare the speed/memory trade-off directly. Use `--json` for machine-readable output or override the default matrix with `-t 2,4,8 -r 1,4,8,16 -K 1,2,4`.

### Model support notes

The sharded engine is implemented and validated for **qwen3 / qwen3.5-family full-attention** models. MoE variants (`qwen3moe`, `qwen35moe`, `qwen3vlmoe`) and other architectures, such as **qwen2 / qwen2.5**, are detected at load time and the sharded engine is disabled automatically, falling back to native `llama_decode` so you still get correct output. Use `--no-shard` explicitly if you want to force the un-sharded path.

## Project Structure

```
gizmo-dev/
├── include/
│   ├── cli_parser.hpp
│   ├── inference_engine.hpp
│   ├── layer_manager.hpp
│   ├── model_manager.hpp
│   └── proc_status.hpp
├── src/
│   ├── main.cpp
│   ├── cli/parser.cpp
│   ├── layer/manager.cpp
│   ├── model/manager.cpp
│   ├── inference/engine.cpp
│   ├── server/server.cpp
│   └── util/proc_status.cpp
├── tools/sharded_engine/       # Per-block prefill engine
│   ├── main.cpp
│   ├── multi_block.cpp
│   ├── shard_block.cpp
│   └── tail_graph.cpp
├── llama.cpp/                  # Submodule (built static)
├── build/gizmo                 # Static binary
├── CMakeLists.txt              # Build system
└── README.md
```

## Build System

- `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` — configure
- `cmake --build build -j$(nproc)` — build
- `cmake --install build --prefix ~/.local` — install to `~/.local/bin`
- The `Makefile` is kept for backwards compatibility but does **not** link llama.cpp. Use CMake.

## License

MIT

## Contributing

Real areas where help is welcome:

- ✅ Automated validation harness comparing sharded vs un-sharded logits across a prompt suite.
- ✅ Resident-layer, row-size, and thread-count sweeps to find the RSS/speed sweet spot.
- ✅ Simple interactive `chat` mode with streaming output and tok/s stats.
- Automatic sweet-spot selection from sweep results.
- GGUF metadata parsing in `ModelManager::parse_gguf_header` (currently returns `total_layers=0`; main path uses `llama_model_n_layer()`).
- Built-in `download` command via libcurl.
- Chat-template aware multi-turn chat (current `gizmo chat` uses a plain text context).
