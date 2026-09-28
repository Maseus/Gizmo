#ifndef GIZMO_SERVER_HPP
#define GIZMO_SERVER_HPP

#include "inference_engine.hpp"
#include "httplib.h"
#include "json.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace gizmo {

// One line of request activity surfaced to the TUI dashboard and structured logs.
struct RequestLogEntry {
    std::string time;
    std::string method;
    std::string path;
    int         status = 0;
    int         tokens = 0;
    double      seconds = 0.0;
    std::string request_id;
    std::string error_message;
};

struct ServerConfig {
    std::string host = "0.0.0.0";
    int32_t port = 8080;
    int32_t threads = 4;
    bool cors = false;
    int32_t request_timeout_seconds = 300; // 5 minutes default
    std::string log_file;                  // append-mode structured JSON log
    bool json_logs = false;                // emit JSON logs to stderr too
};

class HttpServer {
public:
    HttpServer(InferenceEngine& engine, const ServerConfig& config);
    ~HttpServer();

    // Start the server (returns quickly; the actual listener runs in a
    // background thread). Call stop() to shut down.
    void start();

    // Stop the server (can be called from another thread). Waits for the
    // listener thread to finish and attempts to cancel in-flight generation.
    void stop();

    // Request a stop without blocking. Safe to call from a signal handler.
    void request_stop() { stop_requested_.store(true); }

    // Check if server is running
    bool is_running() const;

    // True once the listener thread reports it is accepting connections.
    bool ready() const;

    // True if a SIGINT/SIGTERM has been received and the server should stop.
    bool stop_requested() const { return stop_requested_.load(); }

    // Activity surfaced to the TUI.
    // `limit` caps the number of most recent entries returned.
    std::vector<RequestLogEntry> recent_requests(size_t limit = 20) const;

    // Optional callback invoked on every completed request.
    void set_request_callback(std::function<void(const RequestLogEntry&)> cb);

    // Register SIGINT/SIGTERM to call stop() on this instance. Safe to call once.
    static void install_signal_handlers(HttpServer* instance);

private:
    InferenceEngine& engine_;
    ServerConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<bool> ready_{false};
    std::atomic<bool> stop_requested_{false};

    // The httplib server is kept alive as a member so the listener thread
    // can safely use it after start() returns.
    std::unique_ptr<httplib::Server> svr_;
    std::thread server_thread_;
    mutable std::mutex svr_mutex_;

    // In-flight streaming generation workers that must be joined before
    // the httplib server is torn down.
    mutable std::mutex stream_mutex_;
    std::vector<std::thread> stream_workers_;

    // Activity log consumed by the TUI dashboard.
    mutable std::mutex log_mutex_;
    std::deque<RequestLogEntry> request_log_;
    std::function<void(const RequestLogEntry&)> request_callback_;
    std::ofstream log_stream_;
    static constexpr size_t kMaxLogEntries = 100;

    // Start time for uptime reporting.
    std::chrono::steady_clock::time_point start_time_;

    // Serialize generation requests: the InferenceEngine/llama_context is not
    // thread-safe, and we intentionally run one generation at a time.  A
    // small queue-depth cap prevents unbounded backlog when subagents or
    // multiple clients call concurrently.
    mutable std::mutex engine_mutex_;
    std::condition_variable engine_cv_;
    std::atomic<int> active_generations_{0};
    std::atomic<int> queued_generations_{0};
    static constexpr int kMaxQueueDepth = 8;

    // RAII helper that joins the generation queue, blocks until the engine is
    // free, and keeps it locked for the duration of one request.  If the queue
    // is full, acquired() is false and the caller should return a 503.
    // Movable so a streaming request can hand the lock to its worker thread.
    class GenerationSlot {
    public:
        explicit GenerationSlot(HttpServer& server);
        GenerationSlot(GenerationSlot&& other) noexcept;
        GenerationSlot& operator=(GenerationSlot&& other) noexcept;
        ~GenerationSlot();
        bool acquired() const { return acquired_; }
    private:
        HttpServer* server_;
        bool acquired_;
        std::unique_lock<std::mutex> lock_;
    };
    friend class GenerationSlot;

    void record_request(const httplib::Request& req, const httplib::Response& res,
                        int tokens, double seconds, const std::string& request_id,
                        const std::string& error_message = "");

    void emit_json_log(const RequestLogEntry& entry);

    // HTTP helpers
    void set_cors_headers(httplib::Response& res);
    bool parse_json_body(const httplib::Request& req, nlohmann::json& out_json);
    void send_error(httplib::Response& res, int status, const std::string& message);

    // Format messages with the loaded model's chat template.
    std::string format_chat_messages(
        const std::vector<std::pair<std::string, std::string>>& messages,
        bool add_assistant);

    // HTTP handlers
    void handle_health(const httplib::Request& req, httplib::Response& res);
    void handle_models(const httplib::Request& req, httplib::Response& res);
    void handle_completions(const httplib::Request& req, httplib::Response& res);
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_options(const httplib::Request& req, httplib::Response& res);
};

} // namespace gizmo

#endif // GIZMO_SERVER_HPP
