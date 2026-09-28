cmake_minimum_required(VERSION 3.14)
project(gizmo VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# llama.cpp submodule - disable GGML_CUDA by default for low-resource builds
set(GGML_CUDA OFF CACHE BOOL "Enable CUDA support" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "Enable Vulkan support" FORCE)
set(GGML_METAL OFF CACHE BOOL "Enable Metal support" FORCE)

# Build llama.cpp as static libraries so the gizmo binary is self-contained
# and can be installed to a single prefix without shipping .so files.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries (off = static)" FORCE)

# Add llama.cpp
if(EXISTS "${CMAKE_SOURCE_DIR}/llama.cpp/CMakeLists.txt")
    add_subdirectory(llama.cpp EXCLUDE_FROM_ALL)
else()
    message(WARNING "llama.cpp not fully initialized. Some features will be stubs.")
endif()

# Per-block forward pass spike (breather session deliverable 3)
# Standalone executable, does not link against gizmo sources.
# Per-block sharded inference engine (Phase 6, becomes a reusable
# static library linked into gizmo).
add_subdirectory(tools/one_block_spike)
add_subdirectory(tools/sharded_engine)

# Gizmo executable
add_executable(gizmo
    src/main.cpp
    src/cli/parser.cpp
    src/layer/manager.cpp
    src/model/manager.cpp
    src/inference/engine.cpp
    src/util/proc_status.cpp
    src/server/server.cpp
    src/ui/tui.cpp
    src/ui/model_discovery.cpp
    src/ui/chat_tui.cpp
)

target_include_directories(gizmo PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

# Link llama.cpp libraries if available
if(EXISTS "${CMAKE_SOURCE_DIR}/llama.cpp/CMakeLists.txt")
    target_link_libraries(gizmo PRIVATE
        llama
    )
else()
    # Stub build - no llama.cpp yet
    target_compile_definitions(gizmo PRIVATE GIZMO_STUB_MODE)
endif()

# Wire the sharded inference engine into gizmo. gizmo_sharded_engine_core
# is a static lib whose PUBLIC link interface pulls in llama and the
# llama.cpp/ggml include dirs. We also need to add the sharded engine
# headers to gizmo's include path so the engine can include multi_block.h.
if(TARGET gizmo_sharded_engine_core)
    target_link_libraries(gizmo PRIVATE gizmo_sharded_engine_core)
    target_include_directories(gizmo PRIVATE
        ${CMAKE_SOURCE_DIR}/tools/sharded_engine
    )
endif()

# Install target
install(TARGETS gizmo RUNTIME DESTINATION bin)
