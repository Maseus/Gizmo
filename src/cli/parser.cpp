#include "cli_parser.hpp"
#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <vector>

namespace {

// Parse a comma-separated list of integers (e.g. "1,4,8,16").
// Empty entries and surrounding whitespace are ignored.
std::vector<int32_t> parse_int_list(const char* str) {
    std::vector<int32_t> out;
    if (str == nullptr || str[0] == '\0') {
        return out;
    }
    const char* p = str;
    while (*p != '\0') {
        // Skip leading separators / whitespace
        while (*p == ',' || *p == ' ' || *p == '\t') ++p;
        if (*p == '\0') break;
        char* end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end == p) {
            // Not a number; skip one char to avoid infinite loop on garbage.
            ++p;
            continue;
        }
        out.push_back(static_cast<int32_t>(v));
        p = end;
    }
    return out;
}

} // namespace

namespace gizmo {

CommandType CliParser::parse_command(const std::string& cmd) const {
    if (cmd == "run") return CommandType::Run;
    if (cmd == "chat") return CommandType::Chat;
    if (cmd == "bench") return CommandType::Bench;
    if (cmd == "validate") return CommandType::Validate;
    if (cmd == "sweep") return CommandType::Sweep;
    if (cmd == "list") return CommandType::List;
    if (cmd == "download") return CommandType::Download;
    if (cmd == "info") return CommandType::Info;
    if (cmd == "server") return CommandType::Server;
    if (cmd == "serve") return CommandType::Serve;
    if (cmd == "tui") return CommandType::Tui;
    if (cmd == "launch") return CommandType::Launch;
    if (cmd == "help" || cmd == "--help" || cmd == "-h") return CommandType::Help;
    return CommandType::Unknown;
}

void CliParser::parse_flags(CliOptions& options, int start_index, int argc, char* argv[]) {
    for (int i = start_index; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--model" || arg == "-m") {
            if (i + 1 < argc) {
                options.model = argv[++i];
            }
        } else if (arg == "--layers" || arg == "-l") {
            if (i + 1 < argc) {
                options.layers = std::atoi(argv[++i]);
                // Treat -l/--layers as a single-value alias for -r.
                options.resident_layers = {options.layers};
            }
        } else if (arg == "--prompt" || arg == "-p") {
            if (i + 1 < argc) {
                options.prompt = argv[++i];
            }
        } else if (arg == "--help" || arg == "-h") {
            options.help = true;
        } else if (arg == "--resident-layers" || arg == "-r") {
            if (i + 1 < argc) {
                options.resident_layers = parse_int_list(argv[++i]);
            }
        } else if (arg == "--no-shard") {
            options.no_shard = true;
        } else if (arg == "--measure-ram") {
            options.measure_ram = true;
        } else if (arg == "--measure-interval-ms") {
            if (i + 1 < argc) options.measure_interval_ms = std::atoi(argv[++i]);
        } else if (arg == "--max-tokens" || arg == "-n") {
            if (i + 1 < argc) options.max_tokens = std::atoi(argv[++i]);
        } else if (arg == "--sweep-max-tokens") {
            if (i + 1 < argc) options.sweep_max_tokens = std::atoi(argv[++i]);
        } else if (arg == "--sweep-prefill-tokens") {
            if (i + 1 < argc) options.sweep_prefill_tokens = std::atoi(argv[++i]);
        } else if (arg == "--prefill-only") {
            // bench: run the sharded prefill in isolation, skip
            // the decode loop. Gives a clean wall-time comparison
            // of "sharded prefill" vs "un-sharded prefill" without
            // the current Phase 6 double-prefill cost.
            options.prefill_only = true;
        } else if (arg == "--no-evict") {
            // bench: skip madvise(MADV_DONTNEED); disables the
            // per-block eviction that bounds sharded RSS. Useful
            // as a comparator to attribute the "1.2x un-sharded"
            // wall-time gap to re-fault vs graph-build overhead.
            options.no_evict = true;
        } else if (arg == "--row-size" || arg == "-K") {
            // Phase 9 row-graph amortization: chain K consecutive
            // blocks into one ggml_cgraph. K=1 is the per-block
            // baseline; larger K reduces scheduler overhead.
            if (i + 1 < argc) {
                options.row_size = parse_int_list(argv[++i]);
            }
        } else if (arg == "--threads" || arg == "-t") {
            // CPU thread count for llama_decode and the sharded engine.
            // Single value for normal commands (default: 4);
            // comma-separated list for sweep (default: 4).
            if (i + 1 < argc) {
                options.threads = parse_int_list(argv[++i]);
            }
        } else if (arg == "--verbose" || arg == "-v") {
            options.verbose = true;
        } else if (arg == "--progress") {
            options.progress = true;
        } else if (arg == "--json") {
            options.json_output = true;
        } else if (arg == "--argmax-only") {
            options.argmax_only = true;
        } else if (arg == "--max-diff") {
            if (i + 1 < argc) options.max_diff_threshold = std::atof(argv[++i]);
        } else if (arg == "--mean-diff") {
            if (i + 1 < argc) options.mean_diff_threshold = std::atof(argv[++i]);
        } else if (arg == "--host") {
            if (i + 1 < argc) options.host = argv[++i];
        } else if (arg == "--port") {
            if (i + 1 < argc) options.port = std::atoi(argv[++i]);
        } else if (arg == "--server-threads") {
            if (i + 1 < argc) options.server_threads = std::atoi(argv[++i]);
        } else if (arg == "--cors") {
            options.cors = true;
        } else if (arg == "--request-timeout") {
            if (i + 1 < argc) options.request_timeout_seconds = std::atoi(argv[++i]);
        } else if (arg == "--log-file") {
            if (i + 1 < argc) options.log_file = argv[++i];
        } else if (arg == "--json-logs") {
            options.json_logs = true;
        } else if (arg == "--url") {
            if (i + 1 < argc) options.download_url = argv[++i];
        } else if (arg == "--model-path") {
            if (i + 1 < argc) options.extra_model_dirs = argv[++i];
        } else if (arg == "--context-size" || arg == "-c") {
            if (i + 1 < argc) {
                options.context_size = std::atoi(argv[++i]);
                if (options.context_size < 0) {
                    options.context_size = 0;
                }
            }
        } else if (arg == "--backend") {
            if (i + 1 < argc) {
                options.backend = argv[++i];
            }
        } else if (arg == "--rux-path") {
            if (i + 1 < argc) {
                options.rux_path = argv[++i];
            }
        }
    }

    // Positional argument handling: `gizmo download <url>` is allowed.
    if (options.command == CommandType::Download && options.download_url.empty()) {
        for (int i = start_index; i < argc; ++i) {
            std::string arg = argv[i];
            if (!arg.empty() && arg[0] != '-') {
                options.download_url = arg;
                break;
            }
        }
    }
}

CliOptions CliParser::parse(int argc, char* argv[]) {
    CliOptions options;

    if (argc < 2) {
        options.command = CommandType::Help;  // default: show help
        return options;
    }

    options.command = parse_command(argv[1]);

    // For `gizmo launch <target>`, the second positional argument is the
    // launch target (e.g. "claude"). Remaining flags start at argv[3].
    if (options.command == CommandType::Launch) {
        if (argc >= 3) {
            options.launch_target = argv[2];
        }
        parse_flags(options, 3, argc, argv);
    } else {
        parse_flags(options, 2, argc, argv);
    }

    // Environment fallback for context size only when the CLI did not set it.
    if (options.context_size == 0) {
        const char* env_ctx = std::getenv("GIZMO_CONTEXT_SIZE");
        if (env_ctx && env_ctx[0] != '\0') {
            long v = std::strtol(env_ctx, nullptr, 10);
            if (v > 0 && v <= INT32_MAX) {
                options.context_size = static_cast<int32_t>(v);
            }
        }
    }

    // Apply command-specific defaults for list-valued flags.
    if (options.command == CommandType::Sweep) {
        if (options.resident_layers.empty()) {
            options.resident_layers = {1, 4, 8, 16};
        }
        if (options.row_size.empty()) {
            options.row_size = {1, 2, 4};
        }
        if (options.threads.empty()) {
            options.threads = {4};
        }
    } else {
        if (options.resident_layers.empty()) {
            options.resident_layers = {8};
        }
        if (options.row_size.empty()) {
            options.row_size = {1};
        }
        if (options.threads.empty()) {
            options.threads = {4};
        }
    }

    return options;
}

void CliParser::print_help() const {
    std::cout << R"(
Gizmo - Low-resource LLM inference with layer sharding

Usage: gizmo [command] [options]

Commands:
  (none)    Show this help message
  run       Run a model with a single prompt
  chat      Interactive chat mode with streaming output and speed stats
  bench     Run memory/performance benchmark across prompt lengths
  validate  Compare sharded vs un-sharded logits on a prompt suite
  sweep     Run a full generate() sweep over resident-layer / row-size configs
  list      List downloaded models in common model directories
  download  Download a model from URL using curl/wget
  info      Show system memory info
  serve     Start HTTP server (OpenAI-compatible API)
  server    Alias for serve (backwards compatibility)
  tui       Launch interactive server dashboard (model picker + live dashboard)
  launch    Launch an external tool connected to the local API (e.g. `launch claude`)
  help      Show this help message

Options:
  -m, --model <name>            Model name or path to use
  -l, --layers <count>          Layers to load at once (alias for --resident-layers)
  -r, --resident-layers <list>   Number of blocks kept resident in RAM.
                                Single value for normal commands (default: 8);
                                comma-separated list for sweep (default: 1,4,8,16)
  -n, --max-tokens <N>          Maximum tokens to generate for run/chat (default: 128)
  -c, --context-size <N>        Context window size in tokens. Default derives from
                                model metadata (capped at 262144 / 256k) or 4096.
                                Also read from GIZMO_CONTEXT_SIZE env var.
      --backend <gizmo|rux>     Launch backend: gizmo (local GGUF, default) or rux
                                (disk-KV HuggingFace server)
      --rux-path <path>         Explicit path to the `rux` executable (backend=rux)
      --no-shard                Disable sharding; load full model (uses all RAM)
      --measure-ram             Print VmRSS to stderr during run/chat
      --measure-interval-ms N   VmRSS sample interval (default: 500)
      --prefill-only            bench: sharded prefill in isolation, no decode
      --no-evict                disable per-block madvise(MADV_DONTNEED); keeps
                                weights resident after a block runs
  -K, --row-size <list>        Phase 9: chain K blocks per cgraph.
                                Single value for normal commands (default: 1);
                                comma-separated list for sweep (default: 1,2,4)
  -t, --threads <list>         CPU threads for llama_decode / sharded engine.
                                Single value for normal commands (default: 4);
                                comma-separated list for sweep (default: 4)
  -v, --verbose                 Print sharded-engine diagnostic details
      --progress                 Show an in-place progress indicator during prefill/decode
      --json                   Emit validation/sweep results as JSON
      --argmax-only            Validate only argmax parity, ignore logit diffs
      --max-diff <f>           Max abs logit diff tolerance (default: 1e-4)
      --mean-diff <f>          Mean abs logit diff tolerance (default: 1e-5)
      --sweep-max-tokens N      sweep: decode tokens per config (default: 16)
      --sweep-prefill-tokens N  sweep: prefill tokens per config (default: 64)
  -p, --prompt <text>           Prompt text for run command
      --host <addr>             Server bind address (default: 0.0.0.0)
      --port <N>                Server port (default: 8080)
      --server-threads <N>      Server worker threads (default: 4)
      --cors                    Enable CORS headers (required for browser frontends)
      --request-timeout <N>     Per-request generation timeout in seconds (default: 300)
      --log-file <path>         Append structured JSON request logs to a file
      --json-logs               Also emit structured JSON request logs to stderr
      --url <url>               Download URL (for download command)
      --model-path <path[:path]>  Extra directories to scan for GGUF models
                                    (also read from GIZMO_MODEL_PATH env var)
  -h, --help                    Show this help message

Layer Sharding:
  With sharding enabled (the default), the GGUF is mmap'd but not prefaulted.
  Only the blocks that are executed are faulted into RAM; finished blocks are
  evicted via madvise(MADV_DONTNEED) and re-faulted from disk on next access.
  Use --measure-ram to see real savings.

  -r 1      Extreme (~1.2 GB resident; slowest)
  -r 4      Conservative (~1.6 GB; baseline noise)
  -r 8      Default (~2 GB; ~22% faster than r=1 on Qwen3-8B)
  -r 16     Generous (~2.9 GB; saturates the win)
  -r 36+    All blocks resident (~4.9 GB; same speed as r=8 on Qwen3-8B)
  --no-shard   Full model prefaulted into RAM (baseline for comparison)
  gizmo sweep  Run the default -r 1,4,8,16 x -K 1,2,4 matrix and pick the sweet spot

Examples:
  gizmo run -m model.gguf -p "Hello" --measure-ram
  gizmo run -m model.gguf -r 1 -p "Hello" --measure-ram
  gizmo run -m model.gguf --no-shard -p "Hello" --measure-ram
  gizmo bench -m model.gguf
  gizmo sweep -m model.gguf
  gizmo sweep -m model.gguf -r 1,4,8,16 -K 1,2,4 --json
  gizmo sweep -m model.gguf -r 4 -K 1 -t 2,4,8
  gizmo info
  gizmo serve -m model.gguf --port 8080
  gizmo serve -m model.gguf -r 4 --cors
  gizmo chat -m model.gguf -n 256
  gizmo chat --model-path ~/models
  gizmo launch claude -m model.gguf --port 8080
  gizmo launch claude -m model.gguf -c 32768
  gizmo launch claude --backend rux -m Qwen/Qwen2.5-3B-Instruct --port 8000
  gizmo launch claude --backend rux -m meta-llama/Llama-3.2-3B-Instruct -c 65536
)";
}

void CliParser::print_version() const {
    std::cout << "Gizmo v0.1.0\n";
}

} // namespace gizmo
