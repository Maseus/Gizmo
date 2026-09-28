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

# Reference argmax probe: stand-alone binary that calls llama_decode
# directly and prints the last-token argmax. Used as the baseline for
# validating gizmo_sharded_engine output. Lives under the same
# directory so the include paths line up automatically.
add_executable(gizmo_argmax_probe
    argmax_probe.cpp
)
target_include_directories(gizmo_argmax_probe PRIVATE
    ${CMAKE_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/llama.cpp/src
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/src
)
target_link_libraries(gizmo_argmax_probe PRIVATE
    llama
)

# Last-token logits dumper: writes int32 n_vocab + float[n_vocab] to
# a binary file. Used for offline column-by-column diff between
# sharded engine and llama_decode baselines.
add_executable(gizmo_argmax_dump
    argmax_dump.cpp
)
target_include_directories(gizmo_argmax_dump PRIVATE
    ${CMAKE_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/llama.cpp/src
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/src
)
target_link_libraries(gizmo_argmax_dump PRIVATE
    llama
)

# Per-layer residual dumper: runs llama_decode and dumps the post_ffn
# tensor's last-token column for each layer. Uses a debug hook in
# llama-context.cpp::graph_get_cb (gizmo_probe_post_ffn[]).
add_executable(gizmo_ref_per_layer_dump
    ref_per_layer_dump.cpp
)
target_include_directories(gizmo_ref_per_layer_dump PRIVATE
    ${CMAKE_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/llama.cpp/src
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/src
)
target_link_libraries(gizmo_ref_per_layer_dump PRIVATE
    llama
)

# Per-layer diff tool: compares sharded vs reference per-layer dumps.
add_executable(gizmo_diff_per_layer
    diff_per_layer.cpp
)
target_include_directories(gizmo_diff_per_layer PRIVATE
    ${CMAKE_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/llama.cpp/src
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/src
)
target_link_libraries(gizmo_diff_per_layer PRIVATE
    llama
)
