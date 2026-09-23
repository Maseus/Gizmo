#ifndef GIZMO_CLI_PARSER_HPP
#define GIZMO_CLI_PARSER_HPP

#include <string>

namespace gizmo {

enum class CommandType {
    Run,
    Chat,
    Bench,
    Serve,
    List,
    Download,
    Info,
    Help,
    Unknown
};

struct CliOptions {
    CommandType command = CommandType::Unknown;
    std::string model;
    std::string url;
    int32_t layers = 0;  // 0 means auto/load all
    std::string prompt;
    bool interactive = false;
    bool help = false;
    int32_t resident_layers = 8;  // number of blocks kept resident (Phase 8 sweet spot)
    bool no_shard = false;        // disable layer sharding (full model load)
    bool measure_ram = false;     // print VmRSS to stderr during run/chat
    int32_t measure_interval_ms = 500;
    bool prefill_only = false;    // bench: sharded prefill in isolation, no decode
    bool no_evict = false;        // bench: skip madvise(MADV_PAGEOUT) for comparator
    int32_t row_size = 1;         // Phase 9: blocks per cgraph (default 1 = per-block)
    std::string host = "0.0.0.0"; // gizmo serve: bind address
    int32_t port = 11434;           // gizmo serve: default Ollama-compatible port
};

class CliParser {
public:
    CliParser() = default;

    // Parse command-line arguments
    CliOptions parse(int argc, char* argv[]);

    // Print help message
    void print_help() const;

    // Print version
    void print_version() const;

private:
    CommandType parse_command(const std::string& cmd) const;
    void parse_flags(CliOptions& options, int start_index, int argc, char* argv[]);
};

} // namespace gizmo

#endif // GIZMO_CLI_PARSER_HPP
