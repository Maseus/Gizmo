#include "terminal_utils.hpp"

#include <iostream>
#include <termios.h>
#include <unistd.h>

namespace gizmo {

void clear_screen() {
    std::cout << "\x1b[2J\x1b[H";
}

void hide_cursor() {
    std::cout << "\x1b[?25l";
}

void show_cursor() {
    std::cout << "\x1b[?25h";
}

void move_cursor(int row, int col) {
    std::cout << "\x1b[" << row << ";" << col << "H";
}

void clear_line() {
    std::cout << "\x1b[2K\r";
}

bool RawMode::enable() {
    if (tcgetattr(STDIN_FILENO, &old_tio) != 0) return false;
    termios new_tio = old_tio;
    new_tio.c_lflag &= ~(ICANON | ECHO);
    new_tio.c_cc[VMIN] = 0;
    new_tio.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &new_tio) != 0) return false;
    active = true;
    return true;
}

void RawMode::disable() {
    if (active) {
        tcsetattr(STDIN_FILENO, TCSANOW, &old_tio);
        active = false;
    }
}

void CursorGuard::hide() {
    hide_cursor();
    hidden = true;
}

CursorGuard::~CursorGuard() {
    if (hidden) show_cursor();
}

} // namespace gizmo
