#include "server.hpp"
#include "proc_status.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstring>
#include <functional>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "httplib.h"
#include "json.hpp"
#include "llama.h"

namespace gizmo {

using json = nlohmann::json;

static std::string now_str() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&time, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// Extract the plain prompt string from an OpenAI chat.messages array.
static std::string chat_messages_to_prompt(const json& messages) {
    std::string prompt;
    if (messages.is_array()) {
        for (const auto& msg : messages) {
            if (msg.contains("role") && msg.contains("content")) {
                prompt += msg["role"].get<std::string>() + ": " + msg["content"].get<std::string>() + "\n";
            }
        }
    }
    return prompt;
}

static std::vector<std::string> local_ipv4_addresses() {
    std::vector<std::string> addrs;
    struct ifaddrs *ifap = nullptr;
    if (getifaddrs(&ifap) == 0) {
        for (struct ifaddrs* ifa = ifap; ifa != nullptr; ifa = ifa->ifa_next) {
            if (ifa->ifa_addr == nullptr) continue;
            if (ifa->ifa_addr->sa_family == AF_INET) {
                struct sockaddr_in* sin = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
                std::string s = inet_ntoa(sin->sin_addr);
                if (s != "127.0.0.1") {
                    addrs.push_back(s);
                }
            }
        }
        freeifaddrs(ifap);
    }
    return addrs;
}

Server::Server(InferenceEngine* engine, const std::string& model_path)
    : engine_(engine)
    , model_path_(model_path) {
    model_id_ = model_path;
    size_t pos = model_id_.find_last_of("/\\");
    if (pos != std::string::npos) {
        model_id_ = model_id_.substr(pos + 1);
    }
}

void Server::stop() {
    stop_flag_.store(true);
}

bool Server::run(const std::string& host, int port) {
    httplib::Server svr;

    // CORS: allow the web UI / browser clients to talk to us.
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.status = 204;
    });

    auto log_request = [](const std::string& method, const std::string& path, int status,
                          int tokens, double seconds) {
        std::cout << "[" << now_str() << "] "
                  << method << " " << path
                  << " " << status
                  << " tokens=" << tokens
                  << " time=" << std::fixed << std::setprecision(2) << seconds << "s"
                  << std::endl;
    };

    // Health / model-list endpoints. Atrari's health probe tries /health then
    // falls back to /v1/models, so both must exist.
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        json j;
        j["status"] = "ok";
        res.set_content(j.dump(), "application/json");
        res.set_header("Access-Control-Allow-Origin", "*");
    });

    svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        json j;
        j["object"] = "list";
        json model;
        model["id"] = model_id_;
        model["object"] = "model";
        model["created"] = 0;
        model["owned_by"] = "gizmo";
        j["data"] = json::array({model});
        res.set_content(j.dump(), "application/json");
        res.set_header("Access-Control-Allow-Origin", "*");
    });

    // Chat completions endpoint. Supports both streaming and non-streaming.
    svr.Post("/v1/chat/completions", [this, &log_request](const httplib::Request& req, httplib::Response& res) {
        auto start = std::chrono::steady_clock::now();
        res.set_header("Access-Control-Allow-Origin", "*");

        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& e) {
            json err;
            err["error"]["message"] = std::string("Invalid JSON: ") + e.what();
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            log_request("POST", "/v1/chat/completions", 400, 0, 0);
            return;
        }

        std::string prompt = chat_messages_to_prompt(body.value("messages", json::array()));
        if (prompt.empty()) {
            json err;
            err["error"]["message"] = "No messages provided";
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            log_request("POST", "/v1/chat/completions", 400, 0, 0);
            return;
        }

        InferenceConfig cfg;
        cfg.max_tokens = body.value("max_tokens", 256);
        if (cfg.max_tokens <= 0 || cfg.max_tokens > 4096) cfg.max_tokens = 256;
        cfg.temperature = body.value("temperature", 0.8f);
        cfg.top_p = body.value("top_p", 0.95f);
        cfg.top_k = body.value("top_k", 40);

        bool stream = body.value("stream", false);
        std::string request_id = "gizmocmpl-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());

        if (stream) {
            // Build the full SSE stream in memory while holding the engine
            // lock. Simpler than a chunked provider and avoids lifetime
            // issues with the callback and httplib worker threads.
            std::lock_guard<std::mutex> lock(engine_mutex_);
            engine_->reset_for_next_run();

            std::string sse;
            int generated = 0;
            std::string model = model_id_;
            auto append_chunk = [&sse, &request_id, &model](const std::string& token,
                                                             const char* finish_reason) {
                json chunk;
                chunk["id"] = request_id;
                chunk["object"] = "chat.completion.chunk";
                chunk["created"] = std::chrono::system_clock::to_time_t(
                    std::chrono::system_clock::now());
                chunk["model"] = model;
                json choice;
                choice["index"] = 0;
                if (finish_reason == nullptr) {
                    choice["delta"]["content"] = token;
                    choice["finish_reason"] = nullptr;
                } else {
                    choice["delta"] = json::object();
                    choice["finish_reason"] = finish_reason;
                }
                chunk["choices"] = json::array({choice});
                sse += "data: " + chunk.dump() + "\n\n";
            };

            generated = engine_->generate_stream(prompt, cfg,
                [&append_chunk](const std::string& token) {
                    append_chunk(token, nullptr);
                    return true;
                });

            if (generated >= 0) {
                append_chunk("", "stop");
            }

            res.set_content(sse, "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            log_request("POST", "/v1/chat/completions", 200, generated, elapsed);
            return;
        }

        // Non-streaming path
        std::lock_guard<std::mutex> lock(engine_mutex_);
        engine_->reset_for_next_run();

        std::string full_response;
        int generated = engine_->generate_stream(prompt, cfg,
            [&full_response](const std::string& token) {
                full_response += token;
                return true;
            });

        json response;
        response["id"] = request_id;
        response["object"] = "chat.completion";
        response["created"] = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        response["model"] = model_id_;
        json choice;
        choice["index"] = 0;
        choice["message"]["role"] = "assistant";
        choice["message"]["content"] = full_response;
        choice["finish_reason"] = generated >= 0 ? "stop" : "error";
        response["choices"] = json::array({choice});
        json usage;
        usage["prompt_tokens"] = 0; // we don't track this exactly today
        usage["completion_tokens"] = generated > 0 ? generated : 0;
        usage["total_tokens"] = usage["completion_tokens"];
        response["usage"] = usage;

        res.set_content(response.dump(), "application/json");
        auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        log_request("POST", "/v1/chat/completions", 200, generated, elapsed);
    });

    // Legacy /v1/completions endpoint for broader OpenAI compatibility.
    svr.Post("/v1/completions", [this, &log_request](const httplib::Request& req, httplib::Response& res) {
        auto start = std::chrono::steady_clock::now();
        res.set_header("Access-Control-Allow-Origin", "*");

        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& e) {
            json err;
            err["error"]["message"] = std::string("Invalid JSON: ") + e.what();
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            log_request("POST", "/v1/completions", 400, 0, 0);
            return;
        }

        std::string prompt = body.value("prompt", "");
        if (prompt.empty()) {
            json err;
            err["error"]["message"] = "No prompt provided";
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            log_request("POST", "/v1/completions", 400, 0, 0);
            return;
        }

        InferenceConfig cfg;
        cfg.max_tokens = body.value("max_tokens", 256);
        if (cfg.max_tokens <= 0 || cfg.max_tokens > 4096) cfg.max_tokens = 256;
        cfg.temperature = body.value("temperature", 0.8f);
        cfg.top_p = body.value("top_p", 0.95f);

        std::lock_guard<std::mutex> lock(engine_mutex_);
        engine_->reset_for_next_run();

        std::string full_response;
        int generated = engine_->generate_stream(prompt, cfg,
            [&full_response](const std::string& token) {
                full_response += token;
                return true;
            });

        json response;
        response["id"] = "gizmocmpl-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
        response["object"] = "text_completion";
        response["created"] = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        response["model"] = model_id_;
        json choice;
        choice["text"] = full_response;
        choice["index"] = 0;
        choice["finish_reason"] = generated >= 0 ? "stop" : "error";
        response["choices"] = json::array({choice});

        res.set_content(response.dump(), "application/json");
        auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        log_request("POST", "/v1/completions", 200, generated, elapsed);
    });

    std::cout << "Gizmo server starting...\n";
    std::cout << "Model: " << model_path_ << "\n";
    std::cout << "Listening on http://" << host << ":" << port << "\n";

    std::vector<std::string> addrs = local_ipv4_addresses();
    if (!addrs.empty()) {
        std::cout << "Available on your network:\n";
        for (const auto& a : addrs) {
            std::cout << "  http://" << a << ":" << port << "\n";
        }
    } else {
        std::cout << "No external IPv4 interfaces found (localhost only)\n";
    }
    std::cout << "Press Ctrl+C to stop.\n\n" << std::flush;

    if (!svr.listen(host.c_str(), port)) {
        std::cerr << "Failed to start server on " << host << ":" << port << "\n";
        return false;
    }
    return true;
}

} // namespace gizmo
