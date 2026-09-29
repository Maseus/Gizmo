#ifndef GIZMO_CHAT_TUI_HPP
#define GIZMO_CHAT_TUI_HPP

#include <cstdint>
#include <string>

namespace gizmo {

// Launch an interactive Claude Code–style chat TUI.
// If `model_path` is empty, the user is prompted to pick a discovered model.
// `model_path_extra` is the colon-separated list of extra directories to scan.
// Returns 0 on clean exit, non-zero on error.
int run_chat_tui(const std::string& model_path,
                 const std::string& model_path_extra,
                 int32_t max_tokens,
                 int32_t threads,
                 int32_t context_size);

} // namespace gizmo

#endif // GIZMO_CHAT_TUI_HPP
