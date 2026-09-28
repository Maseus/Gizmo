# Model Specifications

## qwen3-family (e.g., Qwen3-4B)

A supported sharded-engine target.

| Property | Value |
|----------|-------|
| **Architecture** | qwen3 |
| **Total Layers** | 36 |
| **Embedding Dimension** | 2560 |
| **Vocab Size** | 151,936 |
| **Model Size** | ~2.5 GB (Q4_K_M GGUF) |

### Memory Usage (sharded, Qwen3-4B Q4_K_M)

| Resident Layers | RAM Required |
|-----------------|--------------|
| 1 | ~510 MB |
| 4 | ~700 MB |
| 8 | ~1.0 GB |
| all (`--no-shard`) | ~2.5 GB |

## qwen3.5-family

`LLM_ARCH_QWEN35` is now handled by the per-block sharded engine. Both full-attention and gated-delta-net recurrent blocks are implemented and validated against the un-sharded `llama_decode` baseline.

### Validated sizes

| Model | Layers | Embedding | Quant | Status |
|-------|--------|-----------|-------|--------|
| Qwen3.5-0.8B | 24 | 2048 | Q8_0 | ✅ sharded parity |
| Qwen3.5-2B | 24 | 2048 | Q4_K_S | ✅ sharded parity |
| Qwen3.5-4B | 32 | 2560 | Q4_K_M | ✅ sharded parity |
| Qwen3.5-9B | 32 | 4096 | Q4_K_M | ✅ sharded parity |
| **Qwen3.8-27B** | 64 | 5120 | Q4_K_M | ✅ sharded parity |

### Example properties (Qwen3.5-4B Q4_K_M)

| Property | Value |
|----------|-------|
| **Architecture** | qwen35 |
| **Total Layers** | 32 |
| **Embedding Dimension** | 2560 |
| **Vocab Size** | 248,320 |
| **Model Size** | ~2.6 GB |
| **Block Layout** | Hybrid: full-attention + recurrent (gated delta net) blocks |

### How the sharded engine handles qwen3.5

The qwen3.5 architecture differs from qwen3 in several ways; all are now covered by the per-block builders:

- Hybrid block types dispatch to either the full-attention builder or the gated-delta-net SSM builder.
- Full-attention blocks use the joint Q+gate projection, MRoPE-4 (multi-section RoPE), gate×sigmoid, and `attn_post_norm`.
- The recurrent path threads `llama_memory_recurrent` state (R/S/P buffers) through each recurrent block.
- The KV cache for full-attention blocks is threaded through the hybrid memory context alongside the recurrent state.

### Large-model notes

Qwen3.8-27B (64 layers, ~16 GB Q4_K_M) uses the same `LLM_ARCH_QWEN35` gate and has been validated end-to-end. Because the full-model un-sharded footprint exceeds the model file size, use the sharded path with `-r 1` on hosts with 16 GB RAM or less:

| Mode | RAM Required |
|------|--------------|
| Sharded `-r 1`, model load | ~293 MB |
| Sharded `-r 1`, generation HWM | ~2.1 GB |
| Sharded `-r 8`, generation HWM | ~12.7 GB |
| Un-sharded `--no-shard` | ~11.4 GB (full prefault) |

## Other Architectures

Models based on qwen2, qwen2.5, qwen3moe, qwen35moe, and qwen3vlmoe are detected at load time and automatically run through native `llama_decode` (the un-sharded fallback).
