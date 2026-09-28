#ifndef GIZMO_UTIL_STRING_HPP
#define GIZMO_UTIL_STRING_HPP

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

namespace gizmo {

// Parse a colon-separated list of directories (e.g. "~/models:/data/ggufs").
// Empty entries are ignored. Also used for GIZMO_MODEL_PATH.
inline std::vector<std::string> parse_colon_dirs(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ':') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// Join a list of directories into a colon-separated string.
inline std::string join_colon_dirs(const std::vector<std::string>& dirs) {
    std::string out;
    for (size_t i = 0; i < dirs.size(); ++i) {
        if (i > 0) out += ':';
        out += dirs[i];
    }
    return out;
}

} // namespace gizmo

#endif // GIZMO_UTIL_STRING_HPP
