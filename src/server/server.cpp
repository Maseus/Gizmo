#include "server/server.hpp"

#include "inference_engine.hpp"
#include "llama.h"
#include "ggml-backend.h"

#include "httplib.h"
#include "json.hpp"

#include <atomic>
#include <chrono>
#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
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
    void push(const std::string& chunk) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
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

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::string> chunks_;
    bool closed_ = false;
};

} // namespace

using json = nlohmann::json;

namespace {

// Count tokens in a UTF-8 text string using the loaded model's vocabulary.
// Returns 0 if the engine/model is not ready.
int count_tokens(const llama_model* model, const std::string& text) {
    if (!model || text.empty()) {
        return 0;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) {
        return 0;
    }
    std::vector<llama_token> tokens(text.size() + 16, 0);
    const int n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                                 tokens.data(), static_cast<int32_t>(tokens.size()),
                                 /*add_special=*/true, /*parse_special=*/true);
    return n < 0 ? 0 : n;
}

} // namespace

HttpServer::HttpServer(InferenceEngine& engine, const ServerConfig& config)
    : engine_(engine)
    , config_(config)
    , running_(false) {
}

HttpServer::~HttpServer() {
    stop();
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
    svr_->Get("/health", [this](const Request& req, Response& res) { handle_health(req, res); });
    svr_->Get("/v1/health", [this](const Request& req, Response& res) { handle_health(req, res); });
    svr_->Get("/v1/", [this](const Request& req, Response& res) { handle_models(req, res); });
    svr_->Get("/v1/models", [this](const Request& req, Response& res) { handle_models(req, res); });
    svr_->Post("/v1/completions", [this](const Request& req, Response& res) { handle_completions(req, res); });
    svr_->Post("/v1/chat/completions", [this](const Request& req, Response& res) { handle_chat_completions(req, res); });

    // Preflight for CORS
    if (config_.cors) {
        svr_->Options(".*", [this](const Request& /*req*/, Response& res) {
            set_cors_headers(res);
            res.status = 204;
        });
    }

    std::cout << "Starting Gizmo HTTP server on " << config_.host << ":" << config_.port << "\n";
    std::cout << "  Threads: " << config_.threads << "\n";
    std::cout << "  CORS: " << (config_.cors ? "enabled" : "disabled") << "\n";
    std::cout << "  Model: " << engine_.get_model_info() << "\n";
    std::cout << "  Sharding: " << (engine_.is_sharded() ? "enabled" : "disabled") << "\n";

    running_.store(true);

    // Run server in a member thread so the httplib::Server object stays alive.
    server_thread_ = std::thread([this]() {
        svr_->listen(config_.host.c_str(), config_.port);
        running_.store(false);
    });
}

void HttpServer::stop() {
    {
        std::lock_guard<std::mutex> lock(svr_mutex_);
        if (svr_ != nullptr) {
            svr_->stop();
        }
    }
    running_.store(false);
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}

bool HttpServer::is_running() const {
    return running_.load();
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

namespace {

std::string format_log_time() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%H:%M:%S");
    return oss.str();
}

} // namespace

void HttpServer::record_request(const httplib::Request& req, const httplib::Response& res,
                                int tokens, double seconds) {
    RequestLogEntry entry;
    entry.time   = format_log_time();
    entry.method = req.method;
    entry.path   = req.path;
    entry.status = res.status;
    entry.tokens = tokens;
    entry.seconds = seconds;

    {
        std::lock_guard<std::mutex> lock(log_mutex_);
        request_log_.push_back(entry);
        while (request_log_.size() > kMaxLogEntries) {
            request_log_.pop_front();
        }
        if (request_callback_) {
            request_callback_(entry);
        }
    }
}

void HttpServer::set_cors_headers(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
}

void HttpServer::handle_health(const httplib::Request& req, httplib::Response& res) {
    (void)req;
    set_cors_headers(res);

    json response;
    response["status"] = "ok";
    response["model"] = engine_.get_model_info();
    response["sharding"] = engine_.is_sharded();
    response["layers"] = engine_.n_layer();
    response["embedding_dim"] = engine_.embedding_dim();
    response["vocab_size"] = engine_.vocab_size();

    res.set_content(response.dump(), "application/json");
}

void HttpServer::handle_models(const httplib::Request& req, httplib::Response& res) {
    (void)req;
    set_cors_headers(res);

    json response;
    json model_info;
    model_info["id"] = engine_.get_model_info();
    model_info["object"] = "model";
    model_info["created"] = static_cast<int64_t>(std::time(nullptr));
    model_info["owned_by"] = "gizmo";

    response["object"] = "list";
    response["data"] = {model_info};

    res.set_content(response.dump(), "application/json");
}

void HttpServer::handle_completions(const httplib::Request& req, httplib::Response& res) {
    set_cors_headers(res);

    json request_json;
    if (!parse_json_body(req, request_json)) {
        send_error(res, 400, "Invalid JSON body");
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
        return;
    }

    int32_t max_tokens = 256;
    if (request_json.contains("max_tokens")) {
        max_tokens = request_json["max_tokens"].get<int32_t>();
    }

    float temperature = 0.8f;
    if (request_json.contains("temperature")) {
        temperature = request_json["temperature"].get<float>();
    }

    float top_p = 0.95f;
    if (request_json.contains("top_p")) {
        top_p = request_json["top_p"].get<float>();
    }

    int32_t top_k = 40;
    if (request_json.contains("top_k")) {
        top_k = request_json["top_k"].get<int32_t>();
    }

    // Generate completion
    InferenceConfig config;
    config.max_tokens = max_tokens;
    config.temperature = temperature;
    config.top_p = top_p;
    config.top_k = top_k;

    engine_.reset_for_next_run();

    std::string completion;
    int completion_tokens = 0;
    const auto t0 = std::chrono::steady_clock::now();
    bool ok = engine_.generate_stream(prompt, config,
        [&completion, &completion_tokens](const std::string& token_text, int32_t /*token_id*/) {
            completion += token_text;
            ++completion_tokens;
        });
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    // Build OpenAI-compatible response
    json response;
    response["id"] = "cmpl-" + std::to_string(std::time(nullptr));
    response["object"] = "text_completion";
    response["created"] = static_cast<int64_t>(std::time(nullptr));
    response["model"] = engine_.get_model_info();

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

    res.set_content(response.dump(), "application/json");
    record_request(req, res, prompt_tokens + completion_tokens, elapsed);
}

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    set_cors_headers(res);

    json request_json;
    if (!parse_json_body(req, request_json)) {
        send_error(res, 400, "Invalid JSON body");
        return;
    }

    // Parse messages
    std::string prompt;
    if (request_json.contains("messages") && request_json["messages"].is_array()) {
        for (const auto& msg : request_json["messages"]) {
            std::string role = msg.value("role", "");
            std::string content = msg.value("content", "");
            if (role == "system") {
                prompt = "System: " + content + "\n";
            } else if (role == "user") {
                prompt += "User: " + content + "\n";
            } else if (role == "assistant") {
                prompt += "Assistant: " + content + "\n";
            }
        }
        prompt += "Assistant: ";
    } else {
        send_error(res, 400, "Missing or invalid 'messages' array");
        return;
    }

    int32_t max_tokens = 256;
    if (request_json.contains("max_tokens")) {
        max_tokens = request_json["max_tokens"].get<int32_t>();
    }

    float temperature = 0.8f;
    if (request_json.contains("temperature")) {
        temperature = request_json["temperature"].get<float>();
    }

    float top_p = 0.95f;
    if (request_json.contains("top_p")) {
        top_p = request_json["top_p"].get<float>();
    }

    bool stream = false;
    if (request_json.contains("stream")) {
        stream = request_json["stream"].get<bool>();
    }

    // Generate completion
    InferenceConfig config;
    config.max_tokens = max_tokens;
    config.temperature = temperature;
    config.top_p = top_p;

    engine_.reset_for_next_run();

    if (stream) {
        // Streaming (SSE) response. Run generation in a worker thread and
        // push chunks through a thread-safe queue to httplib's chunked
        // content provider. Count generated tokens for logging; usage is
        // not included in SSE chunks by default.
        auto queue = std::make_shared<StreamChunkQueue>();
        const std::string model_info = engine_.get_model_info();
        const int prompt_tokens = count_tokens(engine_.raw_model(), prompt);

        std::thread gen_thread([this, prompt, config, queue, model_info, prompt_tokens]() {
            bool first = true;
            int completion_tokens = 0;
            const auto t0 = std::chrono::steady_clock::now();
            bool ok = engine_.generate_stream(prompt, config,
                [queue, &first, model_info, &completion_tokens](const std::string& token_text, int32_t) {
                    ++completion_tokens;
                    json chunk;
                    chunk["id"] = "chatcmpl-" + std::to_string(std::time(nullptr));
                    chunk["object"] = "chat.completion.chunk";
                    chunk["created"] = static_cast<int64_t>(std::time(nullptr));
                    chunk["model"] = model_info;

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
                });
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();

            // Final chunk with finish_reason="stop" (or "error" on failure).
            json final_chunk;
            final_chunk["id"] = "chatcmpl-" + std::to_string(std::time(nullptr));
            final_chunk["object"] = "chat.completion.chunk";
            final_chunk["created"] = static_cast<int64_t>(std::time(nullptr));
            final_chunk["model"] = model_info;

            json final_choice;
            final_choice["index"] = 0;
            final_choice["delta"] = json::object();
            final_choice["finish_reason"] = ok ? "stop" : "error";
            final_chunk["choices"] = {final_choice};

            queue->push("data: " + final_chunk.dump() + "\n\n");
            queue->push("data: [DONE]\n\n");
            queue->close();
            (void)prompt_tokens;
            (void)elapsed;
        });
        gen_thread.detach();

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
        record_request(req, res, prompt_tokens, 0.0);
        return;
    }

    std::string completion;
    int completion_tokens = 0;
    const auto t0 = std::chrono::steady_clock::now();
    bool ok = engine_.generate_stream(prompt, config,
        [&completion, &completion_tokens](const std::string& token_text, int32_t /*token_id*/) {
            completion += token_text;
            ++completion_tokens;
        });
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    // Build OpenAI-compatible response
    json response;
    response["id"] = "chatcmpl-" + std::to_string(std::time(nullptr));
    response["object"] = "chat.completion";
    response["created"] = static_cast<int64_t>(std::time(nullptr));
    response["model"] = engine_.get_model_info();

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

    res.set_content(response.dump(), "application/json");
    record_request(req, res, prompt_tokens + completion_tokens, elapsed);
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