#ifndef GIZMO_SERVER_HPP
#define GIZMO_SERVER_HPP

#include "inference_engine.hpp"
#include "httplib.h"
#include "json.hpp"

#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace gizmo {

// One line of request activity surfaced to the TUI dashboard.
struct RequestLogEntry {
    std::string time;
    std::string method;
    std::string path;
    int         status = 0;
    int         tokens = 0;
    double      seconds = 0.0;
};

struct ServerConfig {
    std::string host = "0.0.0.0";
    int32_t port = 8080;
    int32_t threads = 4;
    bool cors = false;
};

class HttpServer {
public:
    HttpServer(InferenceEngine& engine, const ServerConfig& config);
    ~HttpServer();

    // Start the server (returns quickly; the actual listener runs in a
    // background thread). Call stop() to shut down.
    void start();

    // Stop the server (can be called from another thread)
    void stop();

    // Check if server is running
    bool is_running() const;

    // Activity surfaced to the TUI.
    // `limit` caps the number of most recent entries returned.
    std::vector<RequestLogEntry> recent_requests(size_t limit = 20) const;

    // Optional callback invoked on every completed request.
    void set_request_callback(std::function<void(const RequestLogEntry&)> cb);

private:
    InferenceEngine& engine_;
    ServerConfig config_;
    std::atomic<bool> running_{false};

    // The httplib server is kept alive as a member so the listener thread
    // can safely use it after start() returns.
    std::unique_ptr<httplib::Server> svr_;
    std::thread server_thread_;
    mutable std::mutex svr_mutex_;

    // Activity log consumed by the TUI dashboard.
    mutable std::mutex log_mutex_;
    std::deque<RequestLogEntry> request_log_;
    std::function<void(const RequestLogEntry&)> request_callback_;
    static constexpr size_t kMaxLogEntries = 100;

    void record_request(const httplib::Request& req, const httplib::Response& res,
                        int tokens, double seconds);

    // HTTP handlers
    void handle_health(const httplib::Request& req, httplib::Response& res);
    void handle_models(const httplib::Request& req, httplib::Response& res);
    void handle_completions(const httplib::Request& req, httplib::Response& res);
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);

    // Helper: set CORS headers if enabled
    void set_cors_headers(httplib::Response& res);

    // Helper: parse JSON request body
    bool parse_json_body(const httplib::Request& req, nlohmann::json& out_json);

    // Helper: create error response
    void send_error(httplib::Response& res, int status, const std::string& message);
};

} // namespace gizmo

#endif // GIZMO_SERVER_HPP