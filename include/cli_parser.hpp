#ifndef GIZMO_CLI_PARSER_HPP
#define GIZMO_CLI_PARSER_HPP

#include <string>
#include <vector>

namespace gizmo {

enum class CommandType {
    Run,
    Chat,
    Bench,
    Validate,
    Sweep,
    List,
    Download,
    Info,
    Server,   // alias for Serve; kept for backwards compatibility
    Serve,    // HTTP server (OpenAI-compatible API)
    Tui,      // interactive server launcher / dashboard
    Help,
    Unknown
};

struct CliOptions {
    CommandType command = CommandType::Unknown;
    std::string model;
    int32_t layers = 0;  // legacy alias for --resident-layers single value
    std::string prompt;
    bool help = false;
    std::vector<int32_t> resident_layers;  // number of blocks kept resident; first value used by run/chat/bench/validate; comma list used by sweep
    bool no_shard = false;                 // disable layer sharding (full model load)
    bool measure_ram = false;              // print VmRSS to stderr during run/chat
    int32_t measure_interval_ms = 500;
    int32_t max_tokens = -1;               // -1: use command default (run/chat 128, empty prompt 256)
    bool prefill_only = false;             // bench: sharded prefill in isolation, no decode
    bool no_evict = false;                 // bench: skip madvise(MADV_DONTNEED) for comparator
    std::vector<int32_t> row_size;         // Phase 9: blocks per cgraph; first value used by run/chat/bench/validate; comma list used by sweep
    std::vector<int32_t> threads;          // number of CPU threads; first value used by run/chat/bench/validate; comma list used by sweep
    int32_t sweep_max_tokens = 16;         // sweep: decode tokens per configuration
    int32_t sweep_prefill_tokens = 64;     // sweep: prefill tokens per configuration
    bool verbose = false;         // print sharded-engine diagnostic details
    bool progress = false;        // show in-place prefill/decode progress indicator
    bool json_output = false;     // emit validation results as JSON
    bool argmax_only = false;     // validate only argmax parity, ignore logit diffs
    float max_diff_threshold = 1e-4f;   // validation: max abs logit diff tolerance
    float mean_diff_threshold = 1e-5f;  // validation: mean abs logit diff tolerance

    // Server options
    std::string host = "0.0.0.0";
    int32_t port = 8080;
    int32_t server_threads = 4;
    bool cors = false;            // enable CORS headers

    // Download options
    std::string download_url;     // URL to download from

    // Default model search directories (colon-separated on CLI).
    std::string model_path;       // extra directories to scan for GGUFs
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
