#ifndef GIZMO_TUI_HPP
#define GIZMO_TUI_HPP

#include <string>

namespace gizmo {

// Settings that seed the interactive serve launcher.
// If model_path is non-empty, the model picker is skipped.
struct ServeSettings {
    std::string model_path;       // optional pre-selected model
    std::string model_path_extra; // colon-separated custom search dirs
    std::string host = "0.0.0.0";
    int32_t port = 8080;
    int32_t threads = 4;
    int32_t max_tokens = 128;
    int32_t request_timeout_seconds = 300;
    std::string log_file;
    bool json_logs = false;
    bool cors = false;
    bool no_evict = false;
    bool verbose = false;
    bool no_shard = false;
    int32_t resident_layers = 8;
    int32_t row_size = 1;
};

// Launch the interactive Gizmo server TUI.
// Discovers models from default Ollama/LM Studio directories, lets the user
// pick one (or enter a custom path), loads the model, starts the HTTP server,
// and renders a live activity dashboard until the user quits.
//
// Returns 0 on clean exit, non-zero on error.
int run_server_tui(const ServeSettings& settings);

// Start the HTTP server without the interactive TUI.
// Uses the supplied model and settings directly; blocks until the server is
// stopped via signal (SIGINT/SIGTERM) or an error occurs.
int run_server_headless(const ServeSettings& settings);

// Backwards-compatible overload.
inline int run_server_tui(const std::string& host, int port, bool cors) {
    ServeSettings s;
    s.host = host;
    s.port = port;
    s.cors = cors;
    return run_server_tui(s);
}

} // namespace gizmo

#endif // GIZMO_TUI_HPP
