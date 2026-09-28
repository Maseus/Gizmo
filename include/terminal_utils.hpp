#ifndef GIZMO_TERMINAL_UTILS_HPP
#define GIZMO_TERMINAL_UTILS_HPP

#include <termios.h>
#include <unistd.h>

namespace gizmo {

// Terminal control helpers shared by the interactive TUIs.

void clear_screen();
void hide_cursor();
void show_cursor();
void move_cursor(int row, int col);
void clear_line();

// RAII wrapper that enables raw terminal mode for stdin on construction and
// restores the previous settings on destruction.
struct RawMode {
    termios old_tio;
    bool active = false;

    bool enable();
    void disable();
    ~RawMode() { disable(); }
};

// RAII wrapper that hides the terminal cursor on construction and restores it
// on destruction.
struct CursorGuard {
    bool hidden = false;
    void hide();
    ~CursorGuard();
};

} // namespace gizmo

#endif // GIZMO_TERMINAL_UTILS_HPP
