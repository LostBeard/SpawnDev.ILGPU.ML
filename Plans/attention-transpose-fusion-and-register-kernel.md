# Next attention levers (after per-query + reshape-CopyFrom) — 2026-06-22

Two universal levers identified + scoped this session but deliberately NOT rushed (intricate / blocked).
Per-query attention (shared-slice) is shipped (WebGPU 65×, all backends ex-WebGL); reshape→CopyFrom shipped
(`fa31974`, ~112 fewer dispatches/step). These are the NEXT wins.

## Lever A — Eliminate the per-layer Transpose[0,2,1,3] dispatches (universal, ~84-112/decode-step)

**Finding:** decode is ~703 nodes/step; **Transpose n=112 (4/layer)** + Reshape n=112 are ~32% of nodes.
`TransposeOperator.Execute` runs a REAL transpose kernel (`reg.Transpose.Transpose`) — necessary data movement,
no `Scale(×1)`-style freebie (unlike Reshape, already fixed).

**Where they come from** (`GGUFGraphBuilder.cs`):
- PRE-attention, per Q/K/V (`EmitAttnHead`, ~line 600-641): `Reshape projOut→[1,seq,heads,hd]` → (QK/V norm) →
  (RoPE, done on seq-major `[1,seq,heads,hd]`, rows_per_position=heads) → **`Transpose[0,2,1,3]`→[1,heads,seq,hd]**
  because `FusedAttention` indexes heads-major (`qBase=(bh*SQ+sq)*D`, K/V `kvHead*kvStride+kv*D`).
- POST-attention (~line 254): attn out `[heads,seq,hd]` → **`Transpose[0,2,1,3]`** → `Reshape` → merged
  `[seq, heads*hd]` for the output projection.

**The win:** make `FusedAttention` read/write the **seq-major** `[1,seq,heads,hd]` layout directly via strides
(it already reads K/V with an explicit `kvRowStride`). Seq-major index for (head bh, seq sq, dim dd) =
`(sq*heads+bh)*hd+dd` vs heads-major `(bh*SQ+sq)*hd+dd`. Add a seq-major-stride mode (Q/output index + a
headStride param) → drop the pre- AND post-attention transposes.

**The catch (why it's a careful multi-file change, not a quick edit):** K/V's transpose output is what the
**KV cache** stores (`[kvHeads, maxSeq, hd]`, heads-major; attention reads it with `kvStride`). Eliminating the
K/V transpose requires the **KV-cache store layout + the attention K/V read to BOTH go seq-major** — touches
`GGUFDecodeKVCache.cs` + `FusedAttentionKernel` + the graph wiring. Q-only + output transpose are more contained
(no KV cache) but partial. PMT-gate every step (GGUFDecodeKVCache 8/8 byte-identical + Attn oracle 92/92 catch
any layout corruption); revert if red. Sequence: (1) post-attention transpose via seq-major OUTPUT write
(contained, 28/step). (2) Q pre-transpose via seq-major Q read (contained, 28/step). (3) K/V pre-transpose +
KV-cache seq-major (the deep one, 56/step). Biggest on WebGPU (each transpose = a dispatch + a full copy).

## Lever B — REGISTER per-query attention (the real attention compute win)

Per-query currently accumulates D outputs in a per-thread SHARED-memory slice (CUDA 5.3×, WebGPU 65×). Tried the
dynamic-D **local-memory** array (Geordi's 4.15.3-local.1 LowerArrays fix): MEASURED ≈ tie (WebGPU 68.7× ~noise,
CUDA 8% SLOWER — shared already had no contention; `.local` is off-chip). **Reverted.** The real win = **true
registers**, which needs a **D-specialized kernel** (compile-time-constant D so the `for dd<D` loops UNROLL and
`acc[dd]` scalar-replaces into registers). Approach: specialize FusedAttentionPerQuery on D (D=128 qwen, D=256
gemma) — e.g. separate D128/D256 kernel methods with `[Unroll]`-friendly constant bounds, or a const-generic
wrapper. BLOCKED on a clean specialize-on-D / unroll pattern — asked Geordi
(`tuvok-to-geordi-localarray-works-but-dynamicD-localmem-ties-sharedslice`). His LowerArrays fix is the
prerequisite (done); the unroll/specialize pattern is pending. Registers >> shared >> local for this pattern.

## Not these (Geordi's lane, in flight): GEMV 128-bit vectorized load (decode), Wasm large-local-array
sequencing. Cache = task #6 (deferred). WebGL attention stays per-element (multi-store vs Transform-Feedback
one-store-per-thread — architectural).
