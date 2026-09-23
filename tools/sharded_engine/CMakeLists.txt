# Per-block sharded inference engine (breather session, Phase 2).
#
# Two targets:
#   1. gizmo_sharded_engine_core (STATIC) — the engine itself
#      (multi_block, shard_block, tail_graph, evict). Reusable from
#      the main gizmo binary.
#   2. gizmo_sharded_engine (EXECUTABLE) — the spike binary, links
#      the static library plus its own main.cpp driver.
#
# Uses the model's native Q4_K/Q6_K weights directly (no F32 dequant).

# Core engine as a reusable static library.
add_library(gizmo_sharded_engine_core STATIC
    multi_block.cpp
    shard_block.cpp
    tail_graph.cpp
    evict.cpp
)

target_include_directories(gizmo_sharded_engine_core PUBLIC
    ${CMAKE_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/llama.cpp/src
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/src
)

target_link_libraries(gizmo_sharded_engine_core PUBLIC
    llama
)

# Spike binary: links the core library + its own main.cpp driver.
add_executable(gizmo_sharded_engine
    main.cpp
)

target_link_libraries(gizmo_sharded_engine PRIVATE
    gizmo_sharded_engine_core
)
