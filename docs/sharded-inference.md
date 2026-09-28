# Per-Block Sharded Inference in Gizmo

**Status:** Implemented and validated end-to-end for **qwen3-family** and **qwen3.5-family** prefill and decode. The per-block sharded engine in `tools/sharded_engine/` is integrated into the main `gizmo` CLI and produces output that matches the un-sharded `llama_decode` baseline exactly for both the prompt forward pass and every generated token on supported models. With sharding enabled the model is mmap'd without prefault, so the initial load RSS stays low.

Non-supported architectures (qwen2/qwen2.5, qwen3next, qwen3vl, and MoE variants) are detected at load time and automatically fall back to native `llama_decode` so they still produce correct output, albeit with the full-model memory footprint.

## 1. Problem statement

Gizmo's reason to exist is to run quantized LLMs locally with honest memory accounting. The un-sharded path loads the full model and reports real `VmRSS`. The problem it does **not** solve on its own: models that are bigger than the host's comfortable RAM. The per-block engine addresses this for supported architectures.

`llama_decode` builds one `ggml_cgraph` containing **all N transformer blocks** of the model and runs the whole graph in a single `ggml_backend_sched_graph_compute_async` call. The graph visits every block per call, so its memory accesses span every block's weights every time. Calling `posix_madvise(MADV_DONTNEED)` between calls does not help because the next `llama_decode` immediately page-faults them back in.

To actually reduce peak RAM, the forward pass has to be split so that one graph contains **one** (or a small row of) block's work and only references **those** blocks' weights. Then that graph genuinely touches only those pages, and eviction between graphs is real.

## 2. The per-block approach

The per-block forward pass replaces `llama_decode` with a loop that, for each block (or each row of K consecutive blocks), builds and runs a focused graph. The pieces of state that have to be threaded across the loop:

- **The residual stream.** Each block takes the previous block's output and adds it to the attention/FFN sub-block outputs.
- **The KV cache.** Full-attention blocks write K and V rows for the current token into the cache and read the K and V history for attention. The cache is shared across blocks.
- **The recurrent state (qwen3.5-family).** Gated-delta-net recurrent blocks don't have a KV cache; they hold conv1d + SSM (`R`/`S`) state in `llama_memory_recurrent`. The per-block runner threads this state alongside the attention KV state through the hybrid memory context.

Eviction policy: after a block/row finishes, the per-block loop calls the raw Linux `madvise(MADV_DONTNEED)` syscall on that block's weight mmap region. For file-backed mappings this drops the pages from RSS; the next access faults them back in from the GGUF file.

## 3. Current architecture support

### qwen3 (validated)

Pure full-attention qwen3 models work end-to-end:

- Model load is mmap'd without prefault.
- Prefill runs per-block graphs that write K/V into `llama_kv_cache`.
- Decode runs per-token per-block graphs with a full-cache causal mask.
- Last-token logits are copied into `llama_get_logits` so the sampler can consume them.

### qwen3.5 / qwen3.8 (validated)

`LLM_ARCH_QWEN35` is now enabled in the sharded engine. Gizmo dispatches each block through the appropriate builder:

- Full-attention blocks use the qwen3.5 attention path (joint Q+gate projection, MRoPE-4, gate×sigmoid, `attn_post_norm`).
- Recurrent blocks use the gated-delta-net SSM path, threading `llama_memory_recurrent` state.

Validated sizes (all pass the built-in `gizmo validate` prompt suite with zero logit diffs):

- Qwen3.5-0.8B-Q8_0
- Qwen3.5-2B-Q4_K_S
- Qwen3.5-4B-Q4_K_M
- Qwen3.5-9B-Q4_K_M
- **Qwen3.8-27B-Q4_K_M** (64 layers, 16 GB on disk; use `-r 1` to stay under ~2.1 GB resident)

`LLM_ARCH_QWEN3NEXT` / `QWEN3VL` / `QWEN35MOE` and other MoE variants still fall back to native `llama_decode`.

### qwen2 / qwen2.5 / MoE (fallback only)

Detected at load time and run through native `llama_decode`.

## 4. The four candidate architectures (historical design context)

### A. Per-block ggml driver from Gizmo

Gizmo drives `ggml_*` directly, porting the forward pass one block at a time. This is the approach the current code uses for qwen3.

### B. `llama_decode_block(il, n)` fork patch in llama.cpp

Add a new public function to llama.cpp that builds and runs a single block's graph. Not currently used.

### C. Super-block windowing

Divide the model into super-blocks and run `llama_decode` on one at a time. Broken because KV-cache writes for evicted blocks are lost.

### D. Hybrid: `llama_decode_block` + fine-grained residency

Same as B plus `MADV_DONTNEED` on blocks not about to run. Future optimization.

## 5. Sharded decode details

The decode loop reuses the same per-block machinery as prefill:

1. For each generated token, `llama_kv_cache_init_for_decode(ctx, token)` allocates a single KV cache cell at the position after the existing prompt.
2. `mctx->apply()` commits the cell.
3. `run_multi_block(model, sched, &token, 1, ..., mctx, pos_first=n_past)` runs all blocks with a single-token mask that sees the full cache history.
4. The resulting F32 logits are copied into `llama_get_logits(ctx)`, so `llama_sampler_sample` consumes them on the next loop iteration.

## 6. Row-graph amortization (`-K`)

The driver can chain K consecutive blocks into a single `ggml_cgraph` (a "row"), so per-block overhead amortizes over K. Default K=1 preserves byte-exact behavior. Higher K values are supported on qwen3 models but do not close the un-sharded speed gap because the dominant cost is `mul_mat` compute, not scheduler overhead.

## 7. Known limits

- **Architecture scope:** qwen3 full-attention and qwen3.5-family hybrid (full-attention + recurrent) models use the sharded path.
- **MoE / VL / qwen3next:** not yet ported; fall back to native `llama_decode`.
- **Row-graph K limits:** large K values can expose scheduler split-boundary issues on some models.

## 8. References

- `tools/sharded_engine/shard_block.cpp` — current qwen3 block builder
- `tools/sharded_engine/multi_block.cpp` — per-block runner and eviction logic
- `tools/sharded_engine/tail_graph.cpp` — final norm + LM head
- `docs/qwen35-sharded-engine.md` — qwen3.5-specific design notes (older phase doc)
