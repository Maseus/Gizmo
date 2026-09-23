# Per-Block Sharded Inference in Gizmo

**Status:** Design document for the breather session of 2026-09-13. This is a
written rationale, not a finished implementation. The architecture it describes
is being validated by a per-block forward pass spike (see
`tools/one_block_spike/`); the spike's outcome determines whether the design
is correct and what gets built next.

## 1. Problem statement

Gizmo's reason to exist is to run quantized LLMs locally with honest memory
accounting. The current code in `src/inference/engine.cpp` does that — it
loads a model, generates text, and reports real `VmRSS`. The problem it
solves well: a model that fits in RAM, with no surprise about peak memory
use.

The problem it does **not** solve: models that are bigger than the host's
RAM. The user's machine has 13 GB. The end-goal model,
**qwen3.8:27b Q4_K_M**, is 17.7 GB on disk. That model cannot be loaded
whole — `llama_model_load_from_file` either returns null on the OOM or, if
the kernel overcommits, the system starts swapping and becomes
unresponsive.

**Note on the spike target:** the spike originally planned to use
qwen3.5:2b, but the two qwen3.5 blobs in the local Ollama store fail
to load with the current fork (`qwen35.rope.dimension_sections has
wrong array length; expected 4, got 3`). The spike was retargeted to
**qwen3:4b** (Q4_K_M, 2.5 GB, qwen3 family), which loads cleanly.
qwen3:4b is pure full-attention (no hybrid recurrent blocks), which
is actually a better spike target — simpler forward pass, same
architectural learning goal. The qwen3.5 fork issue must be resolved
before moving from the spike to the full sharded engine.

The naive answer is "swap" or "use a smaller model". The non-naive answer
is **per-block sharding**: keep the full GGUF file mmap'd (so the model is
"loaded" in the sense that all weights are addressable), but at any moment
have only a small sliding window of blocks' weights physically resident
in RAM. Run the model block by block. Evict the block we just left,
prefetch the block we're about to enter. Net RAM at any moment:
`n_resident_blocks * per_block_bytes` plus the always-resident embeddings,
output head, and the KV cache. For qwen3.8:27b that comes out to roughly
1.5 GB peak — well within the 13 GB the user has.

The naive sharding strategy, which this project already tried, is "run
`llama_decode`, evict between calls." It does not work, for a concrete
reason in section 2.

## 2. Why naive `posix_madvise` between `llama_decode` calls fails

`llama_decode` is not a primitive that operates on one block. It is a
full forward pass: it builds one `ggml_cgraph` containing **all N
transformer blocks** of the model and runs the whole graph in a single
`ggml_backend_sched_graph_compute_async` call
(`llama-context.cpp:1829` → `process_ubatch` → `graph_compute`). The graph
visits every block per call, so its memory accesses span every block's
weights every time.

If we call `posix_madvise(MADV_DONTNEED)` on block K's pages after one
`llama_decode` finishes, the next `llama_decode` immediately page-faults
them back in. The eviction never sticks. This was the previous version of
this project's plan, and it was verified-failed in the prior session: see
`README.md` "What Gizmo does not do (yet)" for the writeup of why the
"92% savings" was arithmetic on hard-coded constants, not measurement.

To actually reduce peak RAM, the forward pass has to be split so that one
graph contains **one** block's work and only references **one** block's
weights. Then that graph genuinely touches only that block's pages, and
eviction between graphs is real.

## 3. The per-block approach

The per-block forward pass replaces `llama_decode` with a loop that, for
each block, builds and runs a single-block graph. The pieces of state
that have to be threaded across the loop:

- **The residual stream.** Each block takes the previous block's output
  and adds it to the attention and FFN sub-block outputs (qwen3.5 uses
  post-attention norm + FFN with residual from before the norm, see
  `qwen35.cpp:179-196`). The spike is responsible for carrying the
  residual stream tensor from one block to the next.

- **The KV cache.** Full-attention blocks write K and V rows for the
  current token into the cache, and read the K and V history for the
  attention computation. The cache itself is a single large tensor held
  by `llama_kv_cache` (or `llama_kv_cache_iswa` for qwen3.5's
  sliding-window variant). The per-block loop shares that one tensor;
  each block reads its own slot and writes its own row. The eviction
  strategy must not evict the cache — the cache is a few hundred MB and
  is needed by every full-attention block.

- **The recurrent state (qwen3.5-specific).** Linear-attention blocks
  don't have a KV cache. They have a recurrent state — conv1d state and
  SSM state — that updates in place. This is held by
  `llama_memory_recurrent` and the `llm_graph_input_rs` graph input.
  Same threading story: one shared tensor, each block reads and writes
  its own slot. Different storage than the KV cache, different update
  semantics.

Eviction policy: between blocks, the per-block loop calls
`posix_madvise(MADV_DONTNEED)` on the just-finished block's weight
mmap region (computed from the block's tensor `data` pointers and
`ggml_nbytes`), and `posix_madvise(MADV_WILLNEED)` on the next block's
region. The mmap stays valid; pages re-fault from disk on next access.
The trick Ollama uses for the same purpose.

## 4. Target architecture: qwen3.5

The end-goal model is qwen3.8:27b, family `qwen35`. Reading
`llama.cpp/src/models/qwen35.cpp` and `llama-graph.h` shows that qwen3.5
is **not a vanilla transformer**. It is a hybrid model with two block
types, and the per-block engine has to handle both.

### Block types

**Full-attention blocks** use standard MHA/GQA. The KV cache is the
`llama_kv_cache_iswa` variant — "inference short-long window attention",
a sliding-window + global cache architecture
(`llama-graph.cpp:3379-3409`). Each full-attention block writes one row
to the cache (its K and V at the current token) and reads the full cache
history for the attention computation. The block also does:

- Pre-attention RMSNorm (`attn_norm`)
- Q projection that outputs query + gate interleaved (a qwen3.5 quirk
  — line 266-277 of `qwen35.cpp`)
- Q/K RMSNorm (`attn_q_norm`, `attn_k_norm`)
- MRoPE (multi-modal RoPE) with 4 sections (`rope_sections` from
  `hparams`)
- Attention with `ggml_flash_attn_ext`
- Gate: `sigmoid(gate) * attn_out`
- Output projection (`wo`)
- Residual, post-attention norm (`attn_post_norm`), SwiGLU FFN, final
  residual

See `qwen35.cpp:254-333` (`build_layer_attn`) and
`qwen35.cpp:470-483` (`build_layer_ffn`).

**Recurrent blocks (gated delta net)** use linear attention. There is no
KV cache. There is recurrent state (conv1d + SSM) that updates in place
each token. The block does:

- Conv1D over the input (with state that shifts)
- Q/K/V extraction from the conv output
- SSM recurrence with `ssm_dt`, `ssm_a`, `ssm_beta`, `ssm_alpha` weights
- Gated normalization: `self.norm(core_attn_out, z)`
- Output projection (`ssm_out`)

See `qwen35.cpp:335-468` (`build_layer_attn_linear`) and the
`build_conv_state` / `build_recurrent_attn` / `build_gdn_l2_norm`
helpers in `llm_build_delta_net_base` (`models.h:34-101`).

### Block scheduling

`qwen35.cpp:17-23` shows the scheduling. By default,
`full_attention_interval = 4`, and layer `il` is recurrent iff
`(il + 1) % 4 != 0`. So layers 0, 1, 2, 4, 5, 6, 8, 9, 10, ... are
recurrent, and layers 3, 7, 11, 15, ... are full attention. The
explicit `is_recr_impl` array in the GGUF can override this default
(line 17).

For the 27B target, the model has 64 layers (`qwen35.cpp:28`). Roughly
48 are recurrent and 16 are full attention. The per-block engine
dispatches to one or the other based on `hparams.is_recr(il)`.

### KV cache architecture in detail

`llama_kv_cache_iswa` is two `llama_kv_cache` instances composed: a base
cache for the long-context attention and a sliding-window cache for
short-range attention. `build_attn_inp_kv_iswa` at
`llama-graph.cpp:3379-3409` sets up both with separate k_idxs, v_idxs,
and kq_mask tensors per ubatch. For the per-block spike, we only need
to handle a single token, so both caches just get one row written and
read at position 0. The recurrent state and the SWA cache are
independent allocations; they both stay resident.

## 5. The four candidate architectures

Different ways to do per-block sharding, on a single axis: **how much
of llama.cpp's forward pass do we re-implement?**

### A. Per-block ggml driver from Gizmo

Gizmo drives `ggml_*` directly, porting the qwen3.5 forward pass one
block at a time. Hand-rolled KV cache management, hand-rolled
recurrent state. No llama.cpp fork patch needed.

- **Pros:** No upstream coupling, full control, all sharding logic
  visible in one place.
- **Cons:** ~1500-2000 lines of new code that must track upstream
  llama.cpp changes. Lots of surface area for subtle bugs (RoPE
  sections, K/V cache timing, normalization placement). Hard to
  extend to other model families without re-porting.
- **Estimated scope:** 3-4 weeks for one architecture (qwen3.5).
- **Maintenance burden:** Every llama.cpp refactor of
  `build_attn_mha` or `llama_kv_cache` is a Gizmo rebase task.

### B. `llama_decode_block(il, n)` fork patch in llama.cpp

Add a new public function to llama.cpp that builds and runs a single
block's graph against the existing context. The block build logic is
copied/refactored from the existing `build_graph`, so most of the
per-arch code stays where it is. KV cache and recurrent state are
reused via the existing `llama_memory_hybrid` allocator — the cache
write/read for full-attention blocks goes through `mctx_cur->cpy_k/
cpy_v/get_k/get_v` exactly as in the full-graph path. The scheduler
is reused.

- **Pros:** Less new code than A. KV cache and recurrent state
  handling is free — we use the existing allocators and schedulers.
  Per-architecture work is more localized.
- **Cons:** Still per-architecture (each model has its own
  `build_graph` to refactor). Fork patch that has to track upstream.
- **Estimated scope:** 1.5-2.5 weeks for Qwen2/Qwen3, plus
  ~1-2 days per additional architecture.
- **Maintenance burden:** Lower than A. The shared infrastructure
  (cache allocators, schedulers, attention ops) all stays in
  llama.cpp; only the per-block graph build needs rebasing.

### C. Super-block windowing (the broken approach)

Keep `llama_decode` for the active window, but divide the model into N
"super-blocks" of L/N layers each. During decode, only one super-block's
weights are resident; trick the scheduler into ignoring the others via
`n_gpu_layers` or similar.

- **Why it doesn't work:** the KV cache problem comes back hard.
  `llama_decode` updates all blocks' KV rows. If you've evicted blocks
  0..K-1, their KV writes are lost. You would have to cache their
  K and V writes separately and merge on the next decode, which is
  more complex than just doing the per-block forward pass. This is
  what the prior version of this project was implicitly trying to
  do via `n_gpu_layers = layers_to_load`, and it didn't reduce
  RAM because the per-layer weights still got touched by the
  scheduler.

### D. Hybrid: `llama_decode_block` + fine-grained residency

Same as B, with the additional trick of `MADV_DONTNEED` on blocks
the scheduler isn't about to touch. Block 0's weights are mmap'd;
blocks 1..N-1 are also mmap'd but `MADV_DONTNEED`'d. The
`llama_decode_block` call for block 0 page-faults only block 0's few
hundred KB; blocks 1..N-1 stay evicted.

- **Pros:** Both wins — clean fork patch + genuine page-level
  residency control.
- **Cons:** More complex than pure B, but probably the right
  design.
- **Estimated scope:** 2-3 weeks for Qwen2/Qwen3.5.

### Recommended: B, scoped to qwen3.5

B is the right starting point. D is a refinement we layer in once
B works. A is too much code for the time budget. C is dead on
arrival.

If the spike validates the basic per-block approach, we pursue B
with the qwen3.5 architecture (full-attention and recurrent blocks
both). If the spike fails, we learn what was wrong and revisit
the design before committing further time.

## 6. Test strategy

The spike exists to validate the per-block approach before any
production code is written. The test methodology:

1. **Numerical baseline.** Run the existing engine on a fixed prompt
   with qwen3.5:2b, capture the residual stream tensor at a known
   layer's output, save to `tests/baselines/qwen3.5-2b-layer-N.logits`.
   This is the oracle.

2. **Spike run.** Build the per-block forward pass for the same
   layer, run it on the same input, capture the output.

3. **Compare.** Position-by-position float32 comparison. Tolerance
   for F16/Q4_K_M is 1e-3 absolute; for F32 it's tighter. The spike
   succeeds if max abs diff < 1e-3 and 99% of elements are within
   tolerance.

If the spike fails, the failure mode tells us what was wrong:
- RoPE section misalignment — diff is large at all positions
- Cache write timing wrong — diff is large only at attention output
- Normalization placement wrong — diff is large after the
  normalization step
- Floating-point accumulation order — diff is small but non-zero
  (acceptable for cross-architecture validation)

## 7. Kill criteria

We stop the per-block approach and revisit if:

- **Kill 1:** After 2 sessions of spike work, the per-block output
  for one full-attention block still differs from the baseline by
  more than 1% relative error at any position. This means our
  ported forward pass is wrong, and a from-scratch port (approach A)
  may be needed.

- **Kill 2:** Per-block first-token latency exceeds 5× the
  full-graph first-token latency on the same model on the same
  hardware. The per-block approach adds per-block graph build
  overhead, but it should still be in the same order of magnitude.
  If it's not, the per-block approach is not viable for production
  use, only for memory-constrained scenarios.

- **Kill 3:** The full sharded engine on qwen3.8:27b still OOMs
  even with all blocks evicted except the active one. This would
  mean the embeddings, output head, and KV cache alone exceed
  available RAM, which would force a different design (e.g.
  embedding sharding too).

We do not stop on:
- Numerical drift of < 1% (acceptable cross-architecture noise)
- Performance regressions within 2-3× (expected for v1, optimizable later)

## 8. Phased plan

The breather session itself produces three deliverables. After that:

- **Phase 2 (multi-block):** Run all full-attention blocks sequentially,
  threading the residual stream and KV cache. ~3-5 days.

- **Phase 3 (recurrent blocks):** Add a per-block graph for the
  gated delta net path, dispatching based on `hparams.is_recr(il)`.
  ~3-5 days.

- **Phase 4 (hybrid memory):** Replace the spike's stub K/V tensors
  with calls into the real `llama_kv_cache_iswa`, and the recurrent
  state stub with the real `llama_memory_recurrent`. ~1 week.

- **Phase 5 (eviction):** Wire `posix_madvise` into the loop. Verify
  VmRSS scales with `n_resident_blocks`. ~1 week.

- **Phase 6 (engine integration):** Replace `llama_decode` in
  `src/inference/engine.cpp` with the per-block loop. Add CLI flags
  `--resident-layers`, `--no-shard`. ~1 week.

- **Phase 7 (end-to-end):** Run qwen3.8:27b with `--resident-layers 3`,
  verify it generates coherent text at ~1.5 GB peak RAM. Compare
  generation quality against the un-sharded engine on a smaller
  model. ~3-5 days.

Total post-breather scope: 4-6 weeks of focused work, broken into
~one-week milestones with verification at each step.

## 9. What this document is not

- It is not a tutorial on `ggml_*` or `llama.cpp`. For that, see
  the inline comments in `llama-graph.cpp:2591-2727` (build_attn_mha)
  and `llama-kv-cache.cpp:1266-1351` (KV cache get/cpy impls).

- It is not a finished design. The architecture it describes is
  being validated. If the spike fails, the design changes.

- It is not a commitment to ship. The user explicitly said "this is
  out of my expertise area" and asked for a breather before
  committing to a multi-week project. The breather produces this
  document and the spike; the user's review of both determines
  whether the project continues.

## 9.1 Row-graph amortization (Phase 9)

The Phase 8 driver builds N=36 separate `ggml_cgraph`s per prefill,
each ~37 nodes. With 36 graphs that is ~1332 nodes built, allocated,
and dispatched across 36 calls each. Phase 9 chains K consecutive
blocks into a single cgraph (a "row"), so the per-block overhead
amortizes over K.

### Knob

`--row-size <K>` (alias `-K`) on `gizmo bench` and `gizmo run`.
Default K=1 preserves Phase 8 byte-exact behavior. K=2..8 are
numerically correct against the un-sharded baseline. K>=9 triggers a
scheduler split-boundary bug and is rejected at runtime until
further work lands.

### How it works

- Per-row ctx sized as `(256 * K_this) * ggml_tensor_overhead()` +
  `ggml_graph_overhead_custom(64 * K_this, false)`. For K=36 that's
  9216 tensor slots and 2304 graph nodes, well within the
  scheduler's existing `graph_size=8192` capacity.
- Per-row shared input tensors (`positions`, `kq_mask`, `token_ids`,
  `carrier`) live in a `carrier_ctx` (CPU-buft-backed), pre-filled
  once before the per-row loop, then marked as inputs in each row's
  cgraph via `ggml_set_input` (idempotent).
- `k_idxs` / `v_idxs` are built once per row from `llama_kv_cache_context`;
  the slot offsets written by `set_input_k_idxs` depend only on the
  ubatch, not on `il`, so they are safely reused for all K blocks
  in the row.
- Residual threading within a row passes block j's output
  `ggml_tensor *` directly into block j+1's `inpL`. Host-staging
  copy happens only BETWEEN rows.
- Per-block eviction runs once per row, evicting any block whose
  index fell out of the residency window (`il - resident_layers >= 0`).

### Measured result on Qwen3-8B-Q4_K_M at N=64, r=8

| Mode      | Wall-time | Vs un-sharded |
|-----------|-----------|---------------|
| Un-sharded| 65.84 s   | 1.00x         |
| K=1       | 77.51 s   | 1.18x         |
| K=2..8    | 77.5 s    | 1.18x         |

Identical across K=1..8. **Row-graph amortization does NOT close
the 1.17x un-sharded gap.** The amortized scheduler overhead (a few
hundred ms) is dominated by actual `mul_mat` compute time on
dequantized weights. To close the gap, deeper changes are needed:
drop to `ggml_gallocr` (manual allocator) so buffer allocations
reuse across blocks, or run blocks in parallel.

### Known limits

- **K>=9 scheduler split-boundary bug:** when a row graph spans
  multiple backend buffers (the per-row kv-cache cells push it past
  one buft boundary), `ggml_backend_sched`'s split allocator
  misassigns cross-ctx shared tensors. Symptom: K=9 wrong argmax,
  K=36 RMSNorm assertion. Untouched by row-graph changes — visible
  because row graphs expose what was always there. Fix requires
  manual `ggml_gallocr`.
- **Recurrent blocks:** row-graph chaining is correct for
  full-attention blocks only. Hybrid (recurrent) blocks still
  require per-block graphs and inherit the Phase 6 qwen3-only
  restriction.

## 10. References

- `llama.cpp/src/models/qwen35.cpp:254-333` — `build_layer_attn` (full
  attention block, the spike's primary port target)
- `llama.cpp/src/models/qwen35.cpp:335-468` — `build_layer_attn_linear`
  (gated delta net, future spike)
- `llama.cpp/src/llama-graph.cpp:2591-2727` — `build_attn_mha` (the
  inner attention op, the spike inlines this)
- `llama.cpp/src/llama-graph.cpp:2839-2912` — `llm_graph_context::build_attn`
  (the high-level block builder, the spike mimics this without the
  llm_graph_input_attn_kv wrapper)
- `llama.cpp/src/llama-kv-cache.cpp:1266-1351` — KV cache
  `get_k`/`get_v`/`cpy_k`/`cpy_v` impls
- `llama.cpp/src/llama-model-loader.cpp:1635-1664` — `load_all_data`,
  the mmap-relative tensor data binding
- `README.md` "What Gizmo does not do (yet)" — the writeup of why
  the prior `posix_madvise`-between-`llama_decode` approach failed
