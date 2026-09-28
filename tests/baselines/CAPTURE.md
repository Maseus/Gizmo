# Numerical Baselines

These files are the **oracle** the per-block forward pass spike is
validated against. They were captured by running the existing Gizmo
inference engine on a fixed prompt and dumping the post-prefill logits
at position -1 (the most recent token).

## Model

- **Name:** qwen3:4b
- **Quantization:** Q4_K_M
- **Family:** qwen3 (pure full-attention, no hybrid)
- **File:** `/home/maseus/.ollama/models/blobs/sha256-3e4cb14174460404e7a233e531675303b2fbf7749c02f91864fe311ab6344e4f`
- **Size:** 2,497,280,480 bytes (2.33 GiB)
- **Shape:** n_layer=36, n_embd=2560, n_head=32, n_head_kv=8, n_swa=0, vocab=151936

## Why qwen3:4b and not qwen3.5:2b

The breather session plan specified qwen3.5:2b as the spike target. The
two qwen3.5 blobs in the local Ollama store both fail to load with the
current llama.cpp fork:

```
llama_model_load: error loading model: error loading model
hyperparameters: key qwen35.rope.dimension_sections has wrong array
length; expected 4, got 3
```

This is a model-vs-fork incompatibility (the GGUF was quantized with a
slightly different RoPE metadata format than our fork expects). Fixing
this would require either re-quantizing the model or patching the fork
to accept both formats — out of scope for the breather session.

qwen3:4b is a better spike target anyway:
- Pure full-attention (no hybrid recurrent blocks to skip around)
- Same end-goal learning: validating the per-block forward pass approach
- Loads cleanly with the current fork
- 2.5 GB, fits comfortably in RAM

The end-goal model (qwen3.8:27b) is still in the qwen3.5 family. When
we move from the spike to the full engine, the qwen3.5 fork issue must
be resolved first. The spike validates the *architecture*, not a
specific model file.

## Prompts

- **Prompt A:** "The capital of France is"
- **Prompt B:** "Hello world"

Each produces a different logits file. Both are valid oracles.

## File format

Each `.logits` file is binary, little-endian:

```
offset  size    contents
0       4       int32  n_vocab (= 151936)
4       n_vocab*4  float32 logits[n_vocab]
```

`n_vocab * 4 = 607,744` bytes for the per-logit section, plus 4 bytes
for the header = 607,748 bytes total. This matches the file sizes on
disk.

## How to re-capture

```bash
# Build the spike
cd /home/maseus/Gizmo/gizmo-dev
cmake --build build --target gizmo_one_block_spike -j$(nproc)

# Capture
./build/tools/one_block_spike/gizmo_one_block_spike capture \
  /home/maseus/.ollama/models/blobs/sha256-3e4cb14174460404e7a233e531675303b2fbf7749c02f91864fe311ab6344e4f \
  "The capital of France is" \
  tests/baselines/qwen3-4b-prompt-A.logits
```

The expected argmax token id and logit value for prompt A as of
2026-09-13:

```
argmax token id: 12095 (logit=14.8294)
```

The exact argmax is sensitive to model quantization noise and runtime
floating-point ordering, so re-captures may produce a different
argmax in the last few bits. The logits vector itself should be
byte-identical for a given build, model file, and prompt.

## What the spike compares against

The spike's correctness criterion: **the per-block ggml graph's output
for one transformer block should match a direct-C++-math reference
implementation to within 1e-3 absolute tolerance on F32 values**.

The captured logits files are NOT the direct comparison target for the
spike — they're the engine-level sanity check that the model produces
sensible outputs. The spike's true reference will be computed inline
during the `run` mode of the spike executable (next deliverable).

When the per-block forward pass is built into the full engine
(future work), the engine's output on a given prompt should match
these captured logits within 1e-3. That's the end-to-end validation
that the per-block approach produces equivalent results to the
existing full-graph approach.
