#include "server/server.hpp"

#include "chat_template.hpp"
#include "inference_engine.hpp"
#include "llama.h"
#include "ggml-backend.h"

#include "httplib.h"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace gizmo {

namespace {

// Thread-safe chunk queue used by the SSE streaming handler.
// The generation thread pushes SSE payload strings; the HTTP content
// provider thread pops them and writes them to the response sink.
class StreamChunkQueue {
public:
    static constexpr size_t kMaxChunks = 64;

    void push(const std::string& chunk) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) return;
            // Keep memory bounded for slow consumers: drop oldest chunks if
            // the backlog grows too large. The final chunks are tiny and
            // still fit after this.
            if (chunks_.size() >= kMaxChunks) {
                chunks_.pop_front();
            }
            chunks_.push_back(chunk);
        }
        cv_.notify_one();
    }

    // Returns true if a chunk was popped into `out`. Returns false when
    // the stream has been closed and the queue is empty.
    bool pop(std::string& out) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return closed_ || !chunks_.empty(); });
        if (!chunks_.empty()) {
            out = std::move(chunks_.front());
            chunks_.pop_front();
            return true;
        }
        return false;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    bool is_closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::string> chunks_;
    bool closed_ = false;
};

} // namespace

using json = nlohmann::json;

namespace {

// Count tokens in a UTF-8 text string using the loaded model's vocabulary.
// Returns 0 if the engine/model is not ready or tokenization fails.
int count_tokens(const llama_model* model, const std::string& text) {
    if (!model || text.empty()) {
        return 0;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) {
        return 0;
    }
    int32_t n_needed = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                                       nullptr, 0,
                                       /*add_special=*/true, /*parse_special=*/true);
    if (n_needed == 0) {
        return 0;
    }
    if (n_needed < 0) {
        n_needed = -n_needed;
    }
    std::vector<llama_token> tokens(static_cast<size_t>(n_needed));
    const int n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                                 tokens.data(), n_needed,
                                 /*add_special=*/true, /*parse_special=*/true);
    return n < 0 ? 0 : n;
}

// Generate a short random request id. Thread-safe so multiple httplib
// worker threads can create ids concurrently without corrupting the RNG.
std::string make_request_id() {
    static std::mutex rng_mutex;
    static std::mt19937_64 rng(std::random_device{}());
    std::lock_guard<std::mutex> lock(rng_mutex);
    std::ostringstream oss;
    oss << "req_" << std::hex << (rng() & 0xffffffffffff);
    return oss.str();
}

// ISO-8601-ish timestamp for structured logs.
std::string format_log_time_iso() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

// Human-readable time for the TUI dashboard.
std::string format_log_time() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%H:%M:%S");
    return oss.str();
}

// Extract the request_id header if present.
std::string request_id_from_header(const httplib::Request& req) {
    auto it = req.headers.find("X-Request-ID");
    if (it != req.headers.end() && !it->second.empty()) {
        return it->second;
    }
    return make_request_id();
}

} // namespace

HttpServer::HttpServer(InferenceEngine& engine, const ServerConfig& config)
    : engine_(engine)
    , config_(config)
    , running_(false)
    , ready_(false)
    , stop_requested_(false)
    , start_time_(std::chrono::steady_clock::now()) {
    if (!config_.log_file.empty()) {
        log_stream_.open(config_.log_file, std::ios::out | std::ios::app);
    }
}

HttpServer::~HttpServer() {
    stop();
}

namespace {

std::atomic<HttpServer*> g_signal_target{nullptr};
std::atomic<bool> g_signal_handlers_installed{false};

} // namespace

void HttpServer::install_signal_handlers(HttpServer* instance) {
    // Only the currently registered server should react to signals.
    g_signal_target.store(instance, std::memory_order_relaxed);

    if (g_signal_handlers_installed.exchange(true, std::memory_order_relaxed)) {
        return;
    }

    auto handler = [](int /*sig*/) {
        HttpServer* target = g_signal_target.load(std::memory_order_relaxed);
        if (target != nullptr) {
            // Async-signal-safe: only flip an atomic flag. The main loop(s)
            // call stop() from normal execution context.
            target->request_stop();
        }
    };
    std::signal(SIGINT, handler);
    std::signal(SIGTERM, handler);
}

void HttpServer::start() {
    if (running_.load()) {
        std::cerr << "Server already running\n";
        return;
    }

    std::lock_guard<std::mutex> lock(svr_mutex_);
    svr_ = std::make_unique<httplib::Server>();

    // Set up routes
    using httplib::Request;
    using httplib::Response;

    // Root redirects to the OpenAI-compatible model list.
    svr_->Get("/", [this](const Request& /*req*/, Response& res) {
        set_cors_headers(res);
        res.status = 302;
        res.set_header("Location", "/v1/models");
    });

    svr_->Get("/health", [this](const Request& req, Response& res) { handle_health(req, res); });
    svr_->Get("/v1/health", [this](const Request& req, Response& res) { handle_health(req, res); });
    svr_->Get("/v1/", [this](const Request& req, Response& res) { handle_models(req, res); });
    svr_->Get("/v1", [this](const Request& req, Response& res) { handle_models(req, res); });
    svr_->Get("/v1/models", [this](const Request& req, Response& res) { handle_models(req, res); });
    svr_->Post("/v1/completions", [this](const Request& req, Response& res) { handle_completions(req, res); });
    svr_->Post("/v1/chat/completions", [this](const Request& req, Response& res) { handle_chat_completions(req, res); });

    // Unknown routes return a JSON error so OpenAI clients can parse it.
    svr_->set_error_handler([this](const Request& /*req*/, Response& res) {
        set_cors_headers(res);
        res.status = 404;
        json err;
        err["error"]["message"] = "Not found";
        err["error"]["type"] = "invalid_request_error";
        err["error"]["param"] = nullptr;
        err["error"]["code"] = 404;
        res.set_content(err.dump(), "application/json");
    });

    // Preflight for CORS (always registered when CORS is on; some clients probe
    // OPTIONS before POST).
    if (config_.cors) {
        svr_->Options(".*", [this](const Request& req, Response& res) { handle_options(req, res); });
    }

    std::cout << "Starting Gizmo HTTP server on " << config_.host << ":" << config_.port << "\n";
    std::cout << "  Threads: " << config_.threads << "\n";
    std::cout << "  CORS: " << (config_.cors ? "enabled" : "disabled") << "\n";
    std::cout << "  Request timeout: " << config_.request_timeout_seconds << "s\n";
    std::cout << "  Model: " << engine_.get_model_info() << "\n";
    std::cout << "  Sharding: " << (engine_.is_sharded() ? "enabled" : "disabled") << "\n";

    running_.store(true);
    ready_.store(false);

    // Run server in a member thread so the httplib::Server object stays alive.
    server_thread_ = std::thread([this]() {
        bool ok = svr_->bind_to_port(config_.host.c_str(), config_.port);
        if (!ok) {
            std::cerr << "Failed to bind to " << config_.host << ":" << config_.port << "\n";
            running_.store(false);
            return;
        }
        ready_.store(true);
        svr_->listen_after_bind();
        ready_.store(false);
        running_.store(false);
    });
}

void HttpServer::stop() {
    stop_requested_.store(true);

    // Wait for any detached streaming generation workers to finish. They
    // observe stop_requested_ and break out of generation quickly.
    {
        std::lock_guard<std::mutex> lock(stream_mutex_);
        for (auto& t : stream_workers_) {
            if (t.joinable()) {
                t.join();
            }
        }
        stream_workers_.clear();
    }

    {
        std::lock_guard<std::mutex> lock(svr_mutex_);
        if (svr_ != nullptr) {
            svr_->stop();
        }
    }
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    running_.store(false);
}

bool HttpServer::is_running() const {
    return running_.load();
}

bool HttpServer::ready() const {
    return ready_.load();
}

std::vector<RequestLogEntry> HttpServer::recent_requests(size_t limit) const {
    std::lock_guard<std::mutex> lock(log_mutex_);
    std::vector<RequestLogEntry> out;
    out.reserve(std::min(limit, request_log_.size()));
    auto it = request_log_.rbegin();
    while (it != request_log_.rend() && out.size() < limit) {
        out.push_back(*it);
        ++it;
    }
    return out;
}

void HttpServer::set_request_callback(std::function<void(const RequestLogEntry&)> cb) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    request_callback_ = std::move(cb);
}

HttpServer::GenerationSlot::GenerationSlot(HttpServer& server)
    : server_(&server)
    , acquired_(false) {
    // Reserve a place in the queue; reject immediately if it is full.
    int expected = server_->queued_generations_.load();
    while (true) {
        if (expected >= HttpServer::kMaxQueueDepth) {
            return;
        }
        if (server_->queued_generations_.compare_exchange_weak(expected, expected + 1)) {
            break;
        }
    }

    // Wait until the engine is free, then take it.
    lock_ = std::unique_lock<std::mutex>(server_->engine_mutex_);
    server_->queued_generations_--;
    server_->active_generations_++;
    acquired_ = true;
}

HttpServer::GenerationSlot::GenerationSlot(GenerationSlot&& other) noexcept
    : server_(other.server_)
    , acquired_(other.acquired_)
    , lock_(std::move(other.lock_)) {
    other.server_ = nullptr;
    other.acquired_ = false;
}

HttpServer::GenerationSlot& HttpServer::GenerationSlot::operator=(GenerationSlot&& other) noexcept {
    if (this != &other) {
        // Release any currently held slot before taking over the new one.
        if (acquired_ && server_ != nullptr) {
            server_->active_generations_--;
        }
        if (lock_.owns_lock()) {
            lock_.unlock();
        }

        server_ = other.server_;
        acquired_ = other.acquired_;
        lock_ = std::move(other.lock_);
        other.server_ = nullptr;
        other.acquired_ = false;
    }
    return *this;
}

HttpServer::GenerationSlot::~GenerationSlot() {
    if (acquired_ && server_ != nullptr) {
        server_->active_generations_--;
    }
    // lock_ is released here (if it owns the mutex); the next waiter proceeds.
}

void HttpServer::emit_json_log(const RequestLogEntry& entry) {
    json j;
    j["timestamp"] = format_log_time_iso();
    j["request_id"] = entry.request_id;
    j["method"] = entry.method;
    j["path"] = entry.path;
    j["status"] = entry.status;
    j["tokens"] = entry.tokens;
    j["seconds"] = entry.seconds;
    if (!entry.error_message.empty()) {
        j["error"] = entry.error_message;
    }
    const std::string line = j.dump();
    if (config_.json_logs) {
        std::cerr << line << "\n";
    }
    if (log_stream_.is_open()) {
        log_stream_ << line << "\n";
        log_stream_.flush();
    }
}

void HttpServer::record_request(const httplib::Request& req, const httplib::Response& res,
                                int tokens, double seconds, const std::string& request_id,
                                const std::string& error_message) {
    RequestLogEntry entry;
    entry.time   = format_log_time();
    entry.method = req.method;
    entry.path   = req.path;
    entry.status = res.status;
    entry.tokens = tokens;
    entry.seconds = seconds;
    entry.request_id = request_id;
    entry.error_message = error_message;

    {
        std::lock_guard<std::mutex> lock(log_mutex_);
        request_log_.push_back(entry);
        while (request_log_.size() > kMaxLogEntries) {
            request_log_.pop_front();
        }
    }
    // Invoke the user callback outside the lock so a callback that reads
    // the log cannot deadlock.
    std::function<void(const RequestLogEntry&)> cb;
    {
        std::lock_guard<std::mutex> lock(log_mutex_);
        cb = request_callback_;
    }
    if (cb) {
        cb(entry);
    }
    emit_json_log(entry);
}

void HttpServer::set_cors_headers(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Request-ID");
    res.set_header("Access-Control-Max-Age", "86400");
}

void HttpServer::handle_options(const httplib::Request& /*req*/, httplib::Response& res) {
    set_cors_headers(res);
    res.status = 204;
}

void HttpServer::handle_health(const httplib::Request& req, httplib::Response& res) {
    (void)req;
    set_cors_headers(res);

    const auto uptime_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time_).count();

    json response;
    response["status"] = "ok";
    response["ready"] = ready_.load();
    response["uptime_seconds"] = uptime_s;
    response["model"] = engine_.get_model_info();
    response["sharding"] = engine_.is_sharded();
    response["layers"] = engine_.n_layer();
    response["embedding_dim"] = engine_.embedding_dim();
    response["vocab_size"] = engine_.vocab_size();
    response["version"] = "0.1.0";
    response["active_generations"] = active_generations_.load();
    response["queued_generations"] = queued_generations_.load();
    response["max_queue_depth"] = kMaxQueueDepth;

    res.set_content(response.dump(), "application/json");
}

void HttpServer::handle_models(const httplib::Request& req, httplib::Response& res) {
    (void)req;
    set_cors_headers(res);

    json response;
    json model_info;
    model_info["id"] = engine_.get_model_id();
    model_info["object"] = "model";
    model_info["created"] = static_cast<int64_t>(std::time(nullptr));
    model_info["owned_by"] = "gizmo";

    response["object"] = "list";
    response["data"] = {model_info};

    res.set_content(response.dump(), "application/json");
}

std::string HttpServer::format_chat_messages(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_assistant) {
    return gizmo::apply_chat_template(engine_.raw_model(), messages, add_assistant);
}

namespace {

InferenceConfig parse_inference_config(const json& request_json) {
    InferenceConfig config;
    if (request_json.contains("max_tokens")) {
        config.max_tokens = request_json["max_tokens"].get<int32_t>();
    }
    if (request_json.contains("temperature")) {
        config.temperature = request_json["temperature"].get<float>();
    }
    if (request_json.contains("top_p")) {
        config.top_p = request_json["top_p"].get<float>();
    }
    if (request_json.contains("top_k")) {
        config.top_k = request_json["top_k"].get<int32_t>();
    }
    if (request_json.contains("seed")) {
        config.seed = request_json["seed"].get<int32_t>();
    }
    if (request_json.contains("stop")) {
        const auto& stop_field = request_json["stop"];
        if (stop_field.is_string()) {
            config.stop.push_back(stop_field.get<std::string>());
        } else if (stop_field.is_array()) {
            for (const auto& s : stop_field) {
                if (s.is_string()) {
                    config.stop.push_back(s.get<std::string>());
                }
            }
        }
    }
    return config;
}

} // namespace

void HttpServer::handle_completions(const httplib::Request& req, httplib::Response& res) {
    const std::string request_id = request_id_from_header(req);
    set_cors_headers(res);

    json request_json;
    if (!parse_json_body(req, request_json)) {
        send_error(res, 400, "Invalid JSON body");
        record_request(req, res, 0, 0.0, request_id, "Invalid JSON body");
        return;
    }

    // Parse request
    std::string prompt;
    if (request_json.contains("prompt")) {
        if (request_json["prompt"].is_string()) {
            prompt = request_json["prompt"].get<std::string>();
        } else if (request_json["prompt"].is_array()) {
            // Handle array of prompts - just use first for now
            if (!request_json["prompt"].empty()) {
                prompt = request_json["prompt"][0].get<std::string>();
            }
        }
    }

    if (prompt.empty()) {
        send_error(res, 400, "Missing or empty 'prompt' field");
        record_request(req, res, 0, 0.0, request_id, "Missing or empty 'prompt' field");
        return;
    }

    InferenceConfig config = parse_inference_config(request_json);

    GenerationSlot slot(*this);
    if (!slot.acquired()) {
        send_error(res, 503, "Server is busy; too many queued generation requests");
        record_request(req, res, 0, 0.0, request_id, "Server busy");
        return;
    }

    engine_.reset_for_next_run();

    std::string completion;
    int completion_tokens = 0;
    const auto t0 = std::chrono::steady_clock::now();

    auto timeout = std::chrono::steady_clock::now() +
                   std::chrono::seconds(config_.request_timeout_seconds);
    auto should_cancel = [this, timeout]() {
        return stop_requested_.load() || std::chrono::steady_clock::now() > timeout;
    };

    bool ok = engine_.generate_stream(prompt, config,
        [&completion, &completion_tokens](const std::string& token_text, int32_t /*token_id*/) {
            completion += token_text;
            ++completion_tokens;
        }, should_cancel);
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    // Build OpenAI-compatible response
    json response;
    response["id"] = "cmpl-" + request_id;
    response["object"] = "text_completion";
    response["created"] = static_cast<int64_t>(std::time(nullptr));
    response["model"] = engine_.get_model_id();

    json choice;
    choice["index"] = 0;
    choice["text"] = completion;
    choice["logprobs"] = nullptr;
    choice["finish_reason"] = ok ? "stop" : "error";

    response["choices"] = {choice};

    const int prompt_tokens = count_tokens(engine_.raw_model(), prompt);
    json usage;
    usage["prompt_tokens"] = prompt_tokens;
    usage["completion_tokens"] = completion_tokens;
    usage["total_tokens"] = prompt_tokens + completion_tokens;
    response["usage"] = usage;

    res.status = 200;
    res.set_content(response.dump(), "application/json");
    record_request(req, res, prompt_tokens + completion_tokens, elapsed, request_id,
                   ok ? "" : "Generation failed or timed out");
}

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    const std::string request_id = request_id_from_header(req);
    set_cors_headers(res);

    json request_json;
    if (!parse_json_body(req, request_json)) {
        send_error(res, 400, "Invalid JSON body");
        record_request(req, res, 0, 0.0, request_id, "Invalid JSON body");
        return;
    }

    // Parse messages
    std::vector<std::pair<std::string, std::string>> messages;
    if (request_json.contains("messages") && request_json["messages"].is_array()) {
        for (const auto& msg : request_json["messages"]) {
            std::string role = msg.value("role", "");
            std::string content = msg.value("content", "");
            if (role.empty()) continue;
            messages.emplace_back(role, content);
        }
    } else {
        send_error(res, 400, "Missing or invalid 'messages' array");
        record_request(req, res, 0, 0.0, request_id, "Missing or invalid 'messages' array");
        return;
    }

    // Use the model's built-in chat template when available; fall back to the
    // legacy plain-text join so older models and unit tests still work.
    std::string prompt = format_chat_messages(messages, /*add_assistant=*/true);
    if (prompt.empty()) {
        for (const auto& msg : messages) {
            if (msg.first == "system") {
                prompt = "System: " + msg.second + "\n";
            } else if (msg.first == "user") {
                prompt += "User: " + msg.second + "\n";
            } else if (msg.first == "assistant") {
                prompt += "Assistant: " + msg.second + "\n";
            }
        }
        prompt += "Assistant: ";
    }

    InferenceConfig config = parse_inference_config(request_json);

    bool stream = false;
    if (request_json.contains("stream")) {
        stream = request_json["stream"].get<bool>();
    }

    // Serialize all generation requests; reject if the queue is full.
    GenerationSlot slot(*this);
    if (!slot.acquired()) {
        send_error(res, 503, "Server is busy; too many queued generation requests");
        record_request(req, res, 0, 0.0, request_id, "Server busy");
        return;
    }

    engine_.reset_for_next_run();

    if (stream) {
        // Streaming (SSE) response. Run generation in a worker thread and
        // push chunks through a thread-safe queue to httplib's chunked
        // content provider. Count generated tokens for logging; usage is
        // not included in SSE chunks by default.
        auto queue = std::make_shared<StreamChunkQueue>();
        const std::string model_id = engine_.get_model_id();
        const int prompt_tokens = count_tokens(engine_.raw_model(), prompt);

        // Capture request metadata by value so the detached logging thread is
        // safe after handle_chat_completions returns.
        const std::string req_method = req.method;
        const std::string req_path = req.path;
        const int response_status = 200;

        std::thread gen_thread([this, prompt, config, queue, model_id, request_id,
                                req_method, req_path, response_status, prompt_tokens,
                                slot = std::move(slot)]() mutable {
            bool first = true;
            int completion_tokens = 0;
            const auto t0 = std::chrono::steady_clock::now();
            auto timeout = std::chrono::steady_clock::now() +
                           std::chrono::seconds(config_.request_timeout_seconds);
            // The cancel predicate only sets the generation abort flag. The
            // worker thread is responsible for emitting the final SSE chunk
            // and [DONE] so the client always sees a complete stream.
            auto should_cancel = [this, timeout]() {
                return stop_requested_.load() || std::chrono::steady_clock::now() > timeout;
            };

            bool ok = engine_.generate_stream(prompt, config,
                [queue, &first, model_id, request_id, &completion_tokens](const std::string& token_text, int32_t) {
                    ++completion_tokens;
                    json chunk;
                    chunk["id"] = "chatcmpl-" + request_id;
                    chunk["object"] = "chat.completion.chunk";
                    chunk["created"] = static_cast<int64_t>(std::time(nullptr));
                    chunk["model"] = model_id;

                    json choice;
                    choice["index"] = 0;
                    json delta;
                    if (first) {
                        delta["role"] = "assistant";
                        first = false;
                    }
                    delta["content"] = token_text;
                    choice["delta"] = delta;
                    choice["finish_reason"] = nullptr;
                    chunk["choices"] = {choice};

                    std::string sse = "data: " + chunk.dump() + "\n\n";
                    queue->push(sse);
                }, should_cancel);
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();

            // Final chunk with finish_reason="stop" (or "error" on failure).
            json final_chunk;
            final_chunk["id"] = "chatcmpl-" + request_id;
            final_chunk["object"] = "chat.completion.chunk";
            final_chunk["created"] = static_cast<int64_t>(std::time(nullptr));
            final_chunk["model"] = model_id;

            json final_choice;
            final_choice["index"] = 0;
            final_choice["delta"] = json::object();
            final_choice["finish_reason"] = ok ? "stop" : "error";
            final_chunk["choices"] = {final_choice};

            queue->push("data: " + final_chunk.dump() + "\n\n");
            queue->push("data: [DONE]\n\n");
            queue->close();

            httplib::Request fake_req;
            fake_req.method = req_method;
            fake_req.path = req_path;
            httplib::Response fake_res;
            fake_res.status = response_status;
            record_request(fake_req, fake_res, prompt_tokens + completion_tokens, elapsed, request_id,
                           ok ? "" : "Streaming generation failed or timed out");
        });

        {
            std::lock_guard<std::mutex> lock(stream_mutex_);
            stream_workers_.push_back(std::move(gen_thread));
        }

        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider("text/event-stream",
            [queue](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                std::string chunk;
                if (queue->pop(chunk)) {
                    sink.write(chunk.data(), chunk.size());
                    // [DONE] signals the end of the stream.
                    if (chunk.find("data: [DONE]") != std::string::npos) {
                        return false;
                    }
                    return true;
                }
                return false;
            });
        return;
    }

    std::string completion;
    int completion_tokens = 0;
    const auto t0 = std::chrono::steady_clock::now();
    auto timeout = std::chrono::steady_clock::now() +
                   std::chrono::seconds(config_.request_timeout_seconds);
    auto should_cancel = [this, timeout]() {
        return stop_requested_.load() || std::chrono::steady_clock::now() > timeout;
    };
    bool ok = engine_.generate_stream(prompt, config,
        [&completion, &completion_tokens](const std::string& token_text, int32_t /*token_id*/) {
            completion += token_text;
            ++completion_tokens;
        }, should_cancel);
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    // Build OpenAI-compatible response
    json response;
    response["id"] = "chatcmpl-" + request_id;
    response["object"] = "chat.completion";
    response["created"] = static_cast<int64_t>(std::time(nullptr));
    response["model"] = engine_.get_model_id();

    json choice;
    choice["index"] = 0;
    json message;
    message["role"] = "assistant";
    message["content"] = completion;
    choice["message"] = message;
    choice["logprobs"] = nullptr;
    choice["finish_reason"] = ok ? "stop" : "error";

    response["choices"] = {choice};

    const int prompt_tokens = count_tokens(engine_.raw_model(), prompt);
    json usage;
    usage["prompt_tokens"] = prompt_tokens;
    usage["completion_tokens"] = completion_tokens;
    usage["total_tokens"] = prompt_tokens + completion_tokens;
    response["usage"] = usage;

    res.status = 200;
    res.set_content(response.dump(), "application/json");
    record_request(req, res, prompt_tokens + completion_tokens, elapsed, request_id,
                   ok ? "" : "Generation failed or timed out");
}

bool HttpServer::parse_json_body(const httplib::Request& req, json& out_json) {
    try {
        out_json = json::parse(req.body);
        return true;
    } catch (const json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << "\n";
        return false;
    }
}

void HttpServer::send_error(httplib::Response& res, int status, const std::string& message) {
    set_cors_headers(res);
    json error;
    error["error"]["message"] = message;
    error["error"]["type"] = "invalid_request_error";
    error["error"]["param"] = nullptr;
    error["error"]["code"] = status;
    res.status = status;
    res.set_content(error.dump(), "application/json");
}

} // namespace gizmo
