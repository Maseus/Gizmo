#include "proc_status.hpp"

#include <fstream>
#include <string>

namespace gizmo {

namespace {

size_t parse_kb_line(const std::string& line) {
    // "VmRSS:      12345 kB"
    auto pos = line.find_last_of("0123456789");
    if (pos == std::string::npos) return 0;
    auto start = line.find_first_of("0123456789");
    if (start == std::string::npos) return 0;
    try {
        return std::stoull(line.substr(start, pos - start + 1)) * 1024ULL;
    } catch (...) {
        return 0;
    }
}

} // namespace

size_t read_vm_rss_bytes() {
    std::ifstream f("/proc/self/status");
    if (!f) return 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) return parse_kb_line(line);
    }
    return 0;
}

size_t read_vm_hwm_bytes() {
    std::ifstream f("/proc/self/status");
    if (!f) return 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmHWM:", 0) == 0) return parse_kb_line(line);
    }
    return 0;
}

} // namespace gizmo