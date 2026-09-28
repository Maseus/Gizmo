# Model Specifications

## qwen3.8:27b

The target model for Gizmo development.

| Property | Value |
|----------|-------|
| **Total Layers** | 34 |
| **Model Size** | ~17.7 GB (GGUF) |
| **Layer Size** | ~522 MB per layer |
| **Layer Breakdown** | 1 embedding + 32 transformer + 1 output norm |
| **Parameter Count** | 27.3B |

### Memory Usage by Layer Count

| Layers Loaded | RAM Required | Savings |
|---------------|--------------|---------|
| 2 | ~1.1 GB | 95% |
| 4 | ~2.3 GB | 89% |
| 8 | ~4.7 GB | 77% |
| 12 | ~6.3 GB | 64% |
| 16 | ~8.2 GB | 53% |
| 24 | ~12.1 GB | 30% |
| 32 | ~16.2 GB | 7% |
| 34 (full) | ~17.3 GB | 0% |

### Recommended Configurations

| Use Case | Layers | RAM | Notes |
|----------|--------|-----|-------|
| Minimal | 4 | ~2.3 GB | Very slow, constant swapping |
| Low Memory | 8 | ~4.7 GB | Usable for short prompts |
| Balanced | 16 | ~8.2 GB | Good balance for most tasks |
| Performance | 24 | ~12.1 GB | Near-full speed |

### Layer Types

```
Layer 0:    token_embedding (embedding)
Layers 1-32: transformer_block_N (transformer)
Layer 33:   output_normalization (normalization)
```

## Testing

```bash
# Test with minimal layers (lowest RAM)
./build/gizmo run -m qwen3.8:27b -l 4

# Test with balanced config
./build/gizmo run -m qwen3.8:27b -l 16

# Full model (requires ~18GB RAM)
./build/gizmo run -m qwen3.8:27b -l 34
```
