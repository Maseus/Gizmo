# qwen3.5 (qwen3.8) sharded engine design

**Status:** Implemented and validated. Phases 10–12 are complete: the qwen3.5
full-attention block, gated-delta-net recurrent block, and hybrid memory
threading are all wired into the per-block sharded engine. `gizmo validate`
passes with zero logit diffs on Qwen3.5-0.8B, 2B, 4B, 9B, and **Qwen3.8-27B**. This document
is kept as historical design context for the port.

## 1. Target

**Model:** qwen3.8:27b Q4_K_M, 17.7 GB on disk, 64 layers, 32 full-attn +
32 recurrent (default `full_attention_interval = 4`, layers 0/1/2/4/5/6/...
are SSM, 3/7/11/.../59 are full attention).

**Memory budget:** 13 GB host RAM, ~9.7 GB currently free.

**Memory shape at Q4_K_M, ~4.5 bits/weight:**
- 27B params * 4.5 bits = ~15.2 GB on disk
- Per-block ~240 MB at 64 layers, ~480 MB at 32 layers (so the 32
  recurrent blocks are larger than the 16 full-attn blocks at 27B's
  d_inner = 2560 etc.)
- Resident sliding window of N=8 layers: ~2-4 GB resident
- Embeddings + output head + KV cache: ~1-2 GB
- Recurrent state: a few hundred MB (one cell per SSM block)
- Projected peak RSS for r=8: **~3-5 GB**, comfortably under 13 GB.

The 27B is achievable. The current sharded engine's `multi_block.cpp:47-55`
hard-rejects hybrid models, so we need to remove that wall and dispatch
on `hparams.is_recr(il)` per block.

## 2. Architectural differences from qwen3

| Concern | qwen3 (current) | qwen3.5 (target) |
|---------|-----------------|------------------|
| Block types | All full-attention | Hybrid: full-attn + DeltaNet |
| Per-block weight layout | `wq`, `wk`, `wv` separately | Full-attn: same. SSM: `wqkv` (fused QKV), `wqkv_gate` (gate), `ssm_conv1d`, `ssm_dt`, `ssm_a`, `ssm_beta`, `ssm_alpha`, `ssm_norm`, `ssm_out` |
| Pre-attention norm | `attn_norm` then `ffn_norm` | `attn_norm` then **post-attention norm** (`attn_post_norm`) |
| FFN placement | After `ffn_norm`, then `ffn_out + ffn_norm_in` | After `attn_post_norm(attn_out + residual)`, then `ffn_out + ffn_residual` |
| Residual structure | `attn_out + inpL` -> norm -> FFN -> `+ ffn_inp` | `attn_out + inpL` -> `attn_post_norm` -> FFN -> `+ ffn_residual` |
| Q projection | Q only | Q + gate interleaved in one tensor (`Qcur_full`), with stride-2 view |
| RoPE | `ggml_rope_ext` (1D) | `ggml_rope_multi` (MRoPE, 4 sections from `hparams.rope_sections`) |
| KV cache | Single `llama_kv_cache` | `llama_kv_cache_iswa` (sliding-window + global) for full-attn blocks |
| Recurrent state | n/a | `llama_memory_recurrent` (conv1d + SSM per block) |
| Memory context | `llama_kv_cache_context` | Hybrid: both `llama_kv_cache_iswa::context` (full-attn) and `llama_memory_recurrent_context` (SSM) |
| `cparams.fused_gdn_ar/ch` | n/a | Optional fused GDN ops when `num_k_heads != num_v_heads` |

**Key insight:** the qwen3.5 full-attention path is the qwen3 path PLUS:
fused QG projection with stride-2 view, MRoPE-4 instead of standard RoPE,
attn_post_norm instead of ffn_norm, and ISWA cache. The qwen3 SSM path
(DeltaNet) is brand-new.

## 3. qwen3.5 full-attention block (Phase 11 work)

Reference: `llama.cpp/src/models/qwen35.cpp:254-333` (`build_layer_attn`).

```cpp
// (illustrative only — actual implementation must mirror upstream line-by-line)
// Order: joint QG projection, QG split, Q/K norm, KV projection, K norm,
//        RoPE, attention, gate*sigmoid, output projection, residual,
//        attn_post_norm, FFN, FFN residual.

// 1. Pre-attention norm
x_norm = rms_norm(inpL) * attn_norm

// 2. Joint QG projection: wq outputs [query | gate] interleaved per head
//    qwen3.5 line 266-270: n_embd, n_embd_head * 2 (query+gate), n_head
[Qcur_full, Kcur, Vcur] = build_qkv(layer, x_norm,
    n_embd_head * 2, n_head,         // wq -> Q+G interleaved
    n_embd_head,     n_head_kv,      // wk
    n_embd_head,     n_head_kv,      // wv
    il, false)

// 3. Split Q from QG tensor (stride-2 view, head interleaved)
Qcur = view_3d(Qcur_full, n_embd_head, n_head, n_tokens,
               stride = 2 * n_embd_head * elem_size,
               offset = 0)

// 4. Per-head Q/K RMSNorm (qwen3.5 line 281-287)
Qcur = rms_norm(Qcur) * attn_q_norm
Kcur = reshape_3d(Kcur, n_embd_head, n_head_kv, n_tokens)
Kcur = rms_norm(Kcur) * attn_k_norm

// 5. Gate view (the second half of Qcur_full, head-interleaved)
gate = view_3d(Qcur_full, n_embd_head, n_head, n_tokens,
               stride = 2 * n_embd_head * elem_size,
               offset = n_embd_head * elem_size)
gate = cont_2d(gate, n_embd_head * n_head, n_tokens)  // needed for mul_mat layout

// 6. Reshape Vcur for RoPE
Vcur = reshape_3d(Vcur, n_embd_head, n_head_kv, n_tokens)

// 7. MRoPE (line 299-309): sections[4] from hparams.rope_sections
sections[4] = { hparams.rope_sections[0..3] }  // e.g. {11, 11, 10, 0}
Qcur = ggml_rope_multi(ctx, Qcur, inp_pos, nullptr,
        n_rot, sections, rope_type, n_ctx_orig, freq_base, ...)
Kcur = ggml_rope_multi(ctx, Kcur, inp_pos, nullptr, ...)

// 8. Phase 7 KV cache path (unchanged from qwen3) — write K/V, read back
if (mctx_kv != nullptr) {
    cpy_k(mctx_kv, Kcur_f16, k_idxs, il)
    cpy_v(mctx_kv, Vcur_f16, v_idxs, il)
    Kcur = get_k(mctx_kv, il)
    Vcur = get_v(mctx_kv, il)
}

// 9. Flash attn (same as qwen3)
attn_out = ggml_flash_attn_ext(Qcur_permuted, Kcur_permuted, Vcur_permuted, kq_mask, kq_scale)

// 10. Gated: sigmoid(gate) * attn_out (line 323-326)
gate_sig = sigmoid(gate)
attn_gated = attn_out * gate_sig

// 11. Output projection
wo_out = wo * attn_gated

// 12. Residual + attn_post_norm (THE KEY DIFFERENCE FROM qwen3)
//     qwen3:  result = wo_out + inpL
//             ffn_normed = rms_norm(result) * ffn_norm
//             ffn_out = ffn_down(silu(ffn_gate * ffn_normed) * ffn_up * ffn_normed)
//             final  = ffn_out + result
//     qwen3.5: same residual, but uses attn_post_norm (not ffn_norm)
//              line 187-191 of qwen35.cpp: norm is applied AFTER the
//              residual add, named "post-attention norm", and the FFN
//              runs WITHOUT an additional pre-FFN norm.
ffn_residual = wo_out + inpL
post_normed  = rms_norm(ffn_residual) * attn_post_norm
ffn_out      = ffn_down(silu(ffn_gate * post_normed) * ffn_up * post_normed)
final        = ffn_out + ffn_residual
```

**Reuses from existing sharded engine:**
- Phase 7 KV cache wiring (`kv_ctx->cpy_k/cpy_v/get_k/get_v`)
- F16 cast before cpy_k (qwen3.5 KV cache is F16 too)
- Residual threading via tensor pointers across blocks in a row
- Per-row shared `positions`, `kq_mask`, `carrier` (CPU-buft-backed)

**New code:**
- `ggml_rope_multi` instead of `ggml_rope_ext` (4 sections)
- Joint QG projection + stride-2 view
- `attn_post_norm` + post-residual norm (replacing `ffn_norm`)
- `gate_sig * attn_out` before `wo`

**Validation:** argmax + per-position logit diff against un-sharded
`llama_decode`, target `max_abs_diff < 1e-3` for Q4_K_M. Use a 1-token
and a 5-token prompt for sanity.

## 4. qwen3.5 SSM block (Phase 12 work — sketch only)

Reference: `llama.cpp/src/models/qwen35.cpp:335-468` (`build_layer_attn_linear`).

The gated delta net (GDN) is a linear-attention variant. Per block:
1. Project `cur` through three weight matrices:
   - `wqkv`: outputs Q+K+V fused (`[key_dim * 2 + value_dim]`)
   - `wqkv_gate`: outputs gate Z (`[value_dim]`)
   - `ssm_beta`, `ssm_alpha`: per-head gate and time-step
2. Build conv input via `build_conv_state(inp, conv_states_all, qkv_mixed, ...)`.
   This updates the conv1d ring buffer in-place; reads `ssm_d_conv` history.
3. Apply 1D causal conv with `ssm_conv1d` weight, then SiLU.
4. Slice conv output into `q_conv`, `k_conv`, `v_conv` (head-major views).
5. Apply `build_gdn_l2_norm` to q_conv and k_conv (L2-norm with eps).
6. If `num_k_heads != num_v_heads` and `cparams.fused_gdn_ar` is false,
   `ggml_repeat_4d` to expand q_conv and k_conv to v_heads shape.
7. Apply gated delta recurrence via `build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il)`. Updates the SSM state in-place.
8. `build_norm_gated` = `rms_norm(output) * silu(z)` (line 243-252).
9. `ssm_out` projection back to `n_embd`.
10. Add residual (same as full-attn: `cur + inpL`).

**Reuses from llama.cpp:**
- `llm_build_delta_net_base` (in models.h) already exposes
  `build_conv_state`, `build_rs`, `build_recurrent_attn`,
  `build_gdn_l2_norm` as virtual methods.
- `llama_memory_recurrent_context::get_r_l(il)` / `get_s_l(il)` give
  the conv/SSM state tensor for layer il.

**New code:**
- Block dispatcher in `shard_block.cpp` that detects `is_recr(il)` and
  calls the GDN builder path.
- Five new weight tensors wired: `wqkv`, `wqkv_gate`, `ssm_conv1d`,
  `ssm_dt`, `ssm_a`, `ssm_beta`, `ssm_alpha`, `ssm_norm`, `ssm_out`.
- Conv1d state + SSM state must remain resident (these are not on
  disk; they're allocated by `llama_memory_recurrent` and grow per
  ubatch). They are NOT subject to MADV_DONTNEED.

**Validation:** argmax + logit diff vs `llama_decode` with sharded
disabled. Recurrent blocks at a single token are deterministic given
the conv/SSM state, so test on N=1 first.

## 5. Phase 10 scaffold (this session's deliverable)

Make `multi_block.cpp` and `shard_block.cpp` cleanly dispatch on
`is_recr(il)` WITHOUT porting the actual qwen3.5 ops yet.

### 5.1 Add a model-type sniff

In `multi_block.cpp:run_multi_block`, replace the early-reject loop
with a per-block dispatcher stub:

```cpp
// Today: hard-reject if any block is recurrent.
// Phase 10: count block types, log them, but proceed with a stub
// builder for recurrent blocks until Phase 12 lands.

int n_full = 0, n_recr = 0;
for (int il = 0; il < n_layer; ++il) {
    if (model->hparams.is_recr((uint32_t)il)) ++n_recr; else ++n_full;
}
if (n_recr > 0) {
    std::fprintf(stderr,
        "WARNING: sharded engine does not yet implement SSM blocks "
        "(%d full-attn + %d recurrent in this model). The first "
        "recurrent layer (%d) will fail; full-attn layers will "
        "still work correctly.\n",
        n_full, n_recr, /* first recurrent il */);
}
```

### 5.2 Per-block dispatch in `shard_block.cpp`

Replace `build_block_graph_into`'s hard-error with a dispatch:

```cpp
if (model->hparams.is_recr((uint32_t)il)) {
    // Phase 12 work. Stub returns nullptr; driver logs and skips
    // to the tail graph. This means the model will produce
    // nonsense for any layer >= first_recurrent_il but the
    // sharded engine will run end-to-end without crashing.
    return build_block_graph_ssm_stub(model, ctx, gf, il, n_tokens,
                                      inpL_source, token_ids, positions,
                                      kq_mask, k_idxs, v_idxs, mctx);
} else {
    return build_block_graph_attn_qwen35(model, ctx, gf, il, n_tokens,
                                         inpL_source, token_ids, positions,
                                         kq_mask, k_idxs, v_idxs, mctx);
}
```

`build_block_graph_attn_qwen35` is a copy of the current qwen3 builder
plus the qwen3.5 attention changes (joint QG, MRoPE-4, gate*sigmoid).
This is **Phase 11 work**, not Phase 10.

### 5.3 Refactor `build_block_graph_into` signature

Keep the public signature stable (multi_block.cpp doesn't change).
Add internal helpers:

```cpp
namespace {
// Returns true if the layer is qwen3-style full-attention.
// Returns false for qwen3.5 full-attention (different layout).
bool is_qwen3_full_attn(const llama_model * model, int il) {
    if (model->hparams.is_recr((uint32_t)il)) return false;
    // Heuristic: qwen3 has layer.ffn_norm, qwen3.5 does not.
    return model->layers[il].ffn_norm != nullptr;
}
}
```

Phase 10 keeps the current qwen3 builder for qwen3 models and
introduces the dispatcher; qwen3.5 full-attention is Phase 11.

### 5.4 What Phase 10 actually does

- Edit `multi_block.cpp` to print a warning instead of rejecting
  hybrid models.
- Edit `shard_block.cpp` to dispatch on `is_recr(il)` with the
  qwen3 path as the default and a stub for SSM.
- Run the existing qwen3:8b bench to verify no regression on the
  default path.
- Verify `gizmo_sharded_engine validate -m qwen3.8:27b` fails
  gracefully at the first recurrent layer (layer 1 or 4 depending
  on `full_attention_interval`) — i.e. we don't crash, we don't
  produce numerical garbage silently, we report the failure clearly.

### 5.5 What Phase 10 does NOT do

- Does NOT port qwen3.5 attention blocks (that's Phase 11).
- Does NOT port the GDN/recurrent path (that's Phase 12).
- Does NOT modify KV cache handling — full-attn path stays on the
  existing `llama_kv_cache`; ISWA upgrade is a Phase 11+ item.
- Does NOT change the row-graph driver (K, r, eviction).
- Does NOT touch the carrier_ctx, kq_mask, or positions threading.

## 6. Risks

- **Layer count vs scheduler capacity.** 64 layers * 37 nodes = 2368
  nodes per row. Within scheduler capacity (8192). With K=1 (Phase 10's
  default), no issue. Phase 11 may need to revisit K limits.
- **MRoPE-4 vs ggml_rope_ext.** The sharded engine today uses
  `ggml_rope_ext`. qwen3.5 needs `ggml_rope_multi` with 4 sections.
  Different op signature; the dispatch in 5.3 must swap both.
- **conv/SSM state residency.** SSM blocks keep in-RAM state that
  must NOT be MADV_DONTNEED'd. The Phase 8 eviction policy evicts
  blocks; if conv/SSM state is allocated INSIDE the block weight
  range, we'd lose it. Verify state is allocated from
  `llama_context_`'s scratch, not block weight tensors. Likely fine
  but worth checking in Phase 12.
- **n_layer_all vs n_layer.** qwen3.5 has MTP layers
  (`hparams.n_layer_nextn > 0`). The sharded engine today uses
  `model->hparams.n_layer()` only and never touches MTP. If we
  dispatch on `is_recr(il)` correctly, MTP layers (which are
  full-attn-only and live at `il >= n_layer`) are skipped naturally
  by `is_recr(il)` returning true. Verify in Phase 12.
- **fused_gdn_ar/ch.** The qwen3.5 graph code branches on
  `cparams.fused_gdn_ar` and `cparams.fused_gdn_ch`. These are
  context-level params, not model-level. Phase 12 should read
  `cparams` from `llama_context` (we'd need access — see
  `llama_cparams`). For Phase 10/11, set both to false (the
  repeat-4d path), which is correct but slower. Optional
  fused path is a later optimization.

## 7. Phased plan

| Phase | Goal | Wall-time est. |
|-------|------|----------------|
| **10** (this session) | Refactor driver + dispatch stub. qwen3 path unchanged. qwen3.5 SSM stub returns nullptr with a clear error. | 30-60 min |
| **11** (next session) | Port qwen3.5 full-attention block. Validate argmax vs un-sharded `llama_decode`. | 1-2 sessions |
| **12** (after) | Port GDN/recurrent block. Validate argmax on qwen3.5:9b first (smaller, fits comfortably), then 27b. | 1-2 sessions |
| **13** | First end-to-end 27B run in 13 GB. Tune `--resident-layers` sweet spot. Add ISWA if needed. | 1 session |
| **14** (TUI) | New `gizmo chat` TUI. Raw-mode terminal, live token streaming, scrollback, esc-to-quit. | 1-2 sessions |

## 8. Files this design touches (Phase 10 only)

- `tools/sharded_engine/multi_block.cpp` — replace reject-loop with
  warning + log block counts.
- `tools/sharded_engine/shard_block.cpp` — dispatch on `is_recr(il)`
  with stub for SSM, current qwen3 builder for full-attn.
- `tools/sharded_engine/shard_block.h` — declare
  `build_block_graph_ssm_stub` returning nullptr.
- `docs/qwen35-sharded-engine.md` — this document.

## 9. References

- `llama.cpp/src/models/qwen35.cpp` — full qwen3.5 graph builder
  (the spike target for Phase 11/12).
- `llama.cpp/src/models/models.h` — `llm_build_delta_net_base`
  with `build_conv_state`, `build_rs`, `build_recurrent_attn`,
  `build_gdn_l2_norm`.
- `llama.cpp/src/llama-memory-recurrent.cpp:1286-1300` — recurrent
  context `get_r_l` / `get_s_l` / `get_rs_z`.
- `llama.cpp/src/llama-memory-recurrent.cpp:428-464` — `init_batch`
  (the analogue of `llama_kv_cache_init_for_batch`).
- `llama.cpp/src/llama-hparams.cpp:260` — `is_recr(uint32_t il) const`.
- `gizmo-dev/docs/sharded-inference.md` — Phase 8/9 design context.
- `gizmo-dev/tools/sharded_engine/shard_block.cpp` — current qwen3
  builder (the Phase 11 starting point).
- `gizmo-dev/tools/sharded_engine/multi_block.cpp:47-55` — current
  reject-loop (the Phase 10 target).
