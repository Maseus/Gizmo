# Per-block forward pass spike (breather session deliverable 3).
#
# Standalone executable that links the same llama.cpp static library
# as the main gizmo binary but does not depend on any Gizmo code.
# Two modes: capture (writes logits baseline) and run (per-block ggml
# graph, future).

add_executable(gizmo_one_block_spike
    main.cpp
    reference.cpp
    spike_graph.cpp
)

# Public llama.h API
target_include_directories(gizmo_one_block_spike PRIVATE
    ${CMAKE_SOURCE_DIR}/llama.cpp/include
    # Private llama.cpp headers (llama-model.h, llama-hparams.h)
    # needed to access struct llama_model::layers[] and per-layer
    # weight tensors. The spike is a private fork tool; the
    # surface area is small and the structs are stable.
    ${CMAKE_SOURCE_DIR}/llama.cpp/src
    # ggml internal headers for Q*_K block dequant (ggml-quants.h,
    # ggml-common.h). Used to dequantize embedding rows from raw
    # bytes when CPU repack is disabled.
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/src
)

target_link_libraries(gizmo_one_block_spike PRIVATE
    llama
)
