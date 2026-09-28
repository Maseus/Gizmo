#ifndef GIZMO_PROC_STATUS_HPP
#define GIZMO_PROC_STATUS_HPP

#include <cstddef>

namespace gizmo {

// Read VmRSS (resident set size) from /proc/self/status for the current
// process. Returns 0 if /proc is unavailable (e.g. Windows, macOS sandbox).
size_t read_vm_rss_bytes();

// Read VmHWM (peak resident set size) from /proc/self/status.
size_t read_vm_hwm_bytes();

} // namespace gizmo

#endif // GIZMO_PROC_STATUS_HPP