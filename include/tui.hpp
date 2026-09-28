#ifndef GIZMO_TUI_HPP
#define GIZMO_TUI_HPP

#include <string>

namespace gizmo {

// Launch the interactive Gizmo server TUI.
// Discovers models from default Ollama/LM Studio directories, lets the user
// pick one (or enter a custom path), loads the model, starts the HTTP server,
// and renders a live activity dashboard until the user quits.
//
// Returns 0 on clean exit, non-zero on error.
int run_server_tui(const std::string& host, int port, bool cors);

} // namespace gizmo

#endif // GIZMO_TUI_HPP
