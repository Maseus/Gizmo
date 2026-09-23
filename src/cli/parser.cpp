#include "cli_parser.hpp"
#include <iostream>
#include <cstdlib>

namespace gizmo {

CommandType CliParser::parse_command(const std::string& cmd) const {
    if (cmd == "run") return CommandType::Run;
    if (cmd == "chat") return CommandType::Chat;
    if (cmd == "bench") return CommandType::Bench;
    if (cmd == "list") return CommandType::List;
    if (cmd == "download") return CommandType::Download;
    if (cmd == "info") return CommandType::Info;
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
            }
        } else if (arg == "--url" || arg == "-u") {
            if (i + 1 < argc) {
                options.url = argv[++i];
            }
        } else if (arg == "--prompt" || arg == "-p") {
            if (i + 1 < argc) {
                options.prompt = argv[++i];
            }
        } else if (arg == "--interactive" || arg == "-i") {
            options.interactive = true;
        } else if (arg == "--help" || arg == "-h") {
            options.help = true;
        } else if (arg == "--resident-layers" || arg == "-r") {
            if (i + 1 < argc) options.resident_layers = std::atoi(argv[++i]);
        } else if (arg == "--no-shard") {
            options.no_shard = true;
        } else if (arg == "--measure-ram") {
            options.measure_ram = true;
        } else if (arg == "--measure-interval-ms") {
            if (i + 1 < argc) options.measure_interval_ms = std::atoi(argv[++i]);
        } else if (arg == "--prefill-only") {
            // bench: run the sharded prefill in isolation, skip
            // the decode loop. Gives a clean wall-time comparison
            // of "sharded prefill" vs "un-sharded prefill" without
            // the current Phase 6 double-prefill cost.
            options.prefill_only = true;
        } else if (arg == "--no-evict") {
            // bench: skip madvise(MADV_PAGEOUT); disables the
            // per-block page-out that bounds sharded RSS. Useful
            // as a comparator to attribute the "1.2x un-sharded"
            // wall-time gap to re-fault vs graph-build overhead.
            options.no_evict = true;
        } else if (arg == "--row-size" || arg == "-K") {
            // Phase 9 row-graph amortization: chain K consecutive
            // blocks into one ggml_cgraph. K=1 is the per-block
            // baseline; larger K reduces scheduler overhead.
            if (i + 1 < argc) options.row_size = std::atoi(argv[++i]);
        }
    }
}

CliOptions CliParser::parse(int argc, char* argv[]) {
    CliOptions options;

    if (argc < 2) {
        return options;  // Will trigger help
    }

    options.command = parse_command(argv[1]);
    parse_flags(options, 2, argc, argv);

    return options;
}

void CliParser::print_help() const {
    std::cout << R"(
Gizmo - Low-resource LLM inference with layer sharding

Usage: gizmo <command> [options]

Commands:
  run       Run a model with a prompt
  chat      Start interactive chat session
  bench     Run memory/performance benchmark across prompt lengths
  list      List available models
  download  Download a model from URL
  info      Show system and model info
  help      Show this help message

Options:
  -m, --model <name>            Model name or path to use
  -l, --layers <count>          Layers to load at once (alias for --resident-layers)
  -r, --resident-layers <count> Number of blocks kept resident in RAM (default: 8)
      --no-shard                Disable sharding; load full model (uses all RAM)
      --measure-ram             Print VmRSS to stderr during run/chat
      --measure-interval-ms N   VmRSS sample interval (default: 500)
      --prefill-only            bench: sharded prefill in isolation, no decode
      --no-evict                bench: disable per-block MADV_PAGEOUT (full model resident)
  -K, --row-size <K>           Phase 9: chain K blocks per cgraph (default: 1)
  -u, --url <url>               URL for download command
  -p, --prompt <text>           Prompt text for run command
  -i, --interactive             Enable interactive mode
  -h, --help                    Show this help message

Layer Sharding:
  Gizmo keeps only N transformer blocks physically resident in RAM at a time.
  The full GGUF stays mmap'd; evicted pages are freed via posix_madvise(MADV_DONTNEED)
  and re-faulted from disk on next access. Use --measure-ram to see real savings.

  -r 1     Extreme (~1.2 GB resident; slowest)
  -r 4     Conservative (~1.6 GB; baseline noise)
  -r 8     Default (~2 GB; ~22% faster than r=1 on Qwen3-8B)
  -r 16    Generous (~2.9 GB; saturates the win)
  -r 36+   All blocks resident (~4.9 GB; same speed as r=8 on Qwen3-8B)
  --no-shard  Full model in RAM (baseline for comparison)

Examples:
  gizmo run -m model.gguf -p "Hello" --measure-ram
  gizmo run -m model.gguf -r 1 -p "Hello" --measure-ram
  gizmo run -m model.gguf --no-shard -p "Hello" --measure-ram
  gizmo bench -m model.gguf
  gizmo info
)";
}

void CliParser::print_version() const {
    std::cout << "Gizmo v0.1.0\n";
}

} // namespace gizmo
