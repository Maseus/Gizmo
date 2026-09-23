#ifndef GIZMO_SERVER_HPP
#define GIZMO_SERVER_HPP

#include "inference_engine.hpp"

#include <string>
#include <mutex>
#include <atomic>

namespace gizmo {

// Simple OpenAI-compatible HTTP server around one InferenceEngine.
// The server serializes requests with a mutex because it shares a single
// llama_context. It prints activity and listening addresses to stdout/stderr.
class Server {
public:
    Server(InferenceEngine* engine, const std::string& model_path);

    // Start the blocking HTTP server loop. Returns only on error.
    bool run(const std::string& host, int port);

    // Ask the running server to stop (not used by the CLI today, but
    // useful for future programmatic use).
    void stop();

private:
    InferenceEngine* engine_;
    std::string model_path_;
    std::string model_id_;
    std::mutex engine_mutex_;
    std::atomic<bool> stop_flag_{false};
};

} // namespace gizmo

#endif // GIZMO_SERVER_HPP
