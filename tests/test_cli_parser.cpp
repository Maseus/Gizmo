#include "cli_parser.hpp"
#include "test_harness.hpp"

#include <cstdlib>

using namespace gizmo;

namespace {

struct Argv {
    std::vector<std::string> storage;
    std::vector<char*> argv;

    explicit Argv(std::initializer_list<std::string> args) {
        storage = args;
        argv.reserve(storage.size() + 1);
        for (auto& s : storage) {
            argv.push_back(s.data());
        }
        argv.push_back(nullptr);
    }

    int argc() const { return static_cast<int>(storage.size()); }
    char** data() { return argv.data(); }
};

} // namespace

TEST(cli_parse_context_size_long) {
    Argv a{"gizmo", "run", "--model", "m.gguf", "--context-size", "32768"};
    CliParser p;
    auto opts = p.parse(a.argc(), a.data());
    ASSERT_EQ(opts.command, CommandType::Run);
    ASSERT_EQ(opts.context_size, 32768);
}

TEST(cli_parse_context_size_short) {
    Argv a{"gizmo", "chat", "-m", "m.gguf", "-c", "65536"};
    CliParser p;
    auto opts = p.parse(a.argc(), a.data());
    ASSERT_EQ(opts.context_size, 65536);
}

TEST(cli_parse_context_size_negative_becomes_zero) {
    Argv a{"gizmo", "run", "-m", "m.gguf", "-c", "-100"};
    CliParser p;
    auto opts = p.parse(a.argc(), a.data());
    ASSERT_EQ(opts.context_size, 0);
}

TEST(cli_parse_context_size_env_fallback) {
    // Environment fallback only applies when CLI did not set a value.
    setenv("GIZMO_CONTEXT_SIZE", "131072", 1);
    {
        Argv a{"gizmo", "run", "-m", "m.gguf"};
        CliParser p;
        auto opts = p.parse(a.argc(), a.data());
        ASSERT_EQ(opts.context_size, 131072);
    }
    unsetenv("GIZMO_CONTEXT_SIZE");
}

TEST(cli_parse_context_size_cli_overrides_env) {
    setenv("GIZMO_CONTEXT_SIZE", "131072", 1);
    {
        Argv a{"gizmo", "run", "-m", "m.gguf", "-c", "8192"};
        CliParser p;
        auto opts = p.parse(a.argc(), a.data());
        ASSERT_EQ(opts.context_size, 8192);
    }
    unsetenv("GIZMO_CONTEXT_SIZE");
}

TEST(cli_parse_context_size_env_invalid_ignored) {
    setenv("GIZMO_CONTEXT_SIZE", "not_a_number", 1);
    {
        Argv a{"gizmo", "run", "-m", "m.gguf"};
        CliParser p;
        auto opts = p.parse(a.argc(), a.data());
        ASSERT_EQ(opts.context_size, 0); // not set, kept as default
    }
    unsetenv("GIZMO_CONTEXT_SIZE");
}

TEST(cli_parse_launch_target_and_backend) {
    Argv a{"gizmo", "launch", "claude", "--backend", "rux", "--rux-path", "/usr/local/bin/rux"};
    CliParser p;
    auto opts = p.parse(a.argc(), a.data());
    ASSERT_EQ(opts.command, CommandType::Launch);
    ASSERT_EQ(opts.launch_target, "claude");
    ASSERT_EQ(opts.backend, "rux");
    ASSERT_EQ(opts.rux_path, "/usr/local/bin/rux");
}

TEST(cli_parse_launch_default_backend) {
    Argv a{"gizmo", "launch", "claude", "-m", "m.gguf"};
    CliParser p;
    auto opts = p.parse(a.argc(), a.data());
    ASSERT_EQ(opts.command, CommandType::Launch);
    ASSERT_EQ(opts.launch_target, "claude");
    ASSERT_EQ(opts.backend, "gizmo");
    ASSERT_EQ(opts.model, "m.gguf");
}

TEST(cli_parse_launch_no_target) {
    Argv a{"gizmo", "launch"};
    CliParser p;
    auto opts = p.parse(a.argc(), a.data());
    ASSERT_EQ(opts.command, CommandType::Launch);
    ASSERT_TRUE(opts.launch_target.empty());
}

TEST(cli_parse_sweep_keeps_context_size) {
    Argv a{"gizmo", "sweep", "-m", "m.gguf", "-c", "262144"};
    CliParser p;
    auto opts = p.parse(a.argc(), a.data());
    ASSERT_EQ(opts.command, CommandType::Sweep);
    ASSERT_EQ(opts.context_size, 262144);
    ASSERT_EQ(opts.resident_layers.size(), 4u); // sweep default
    ASSERT_EQ(opts.row_size.size(), 3u);        // sweep default
}
