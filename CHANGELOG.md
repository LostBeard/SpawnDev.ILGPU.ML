# SpawnDev.ILGPU.ML Changelog

Notable changes per release. Pre-stable; API will change between preview drops.

## Unreleased — FusedLinear register-blocked path for native low-precision weights (+ SiLU/ReLU)

**The fused MatMul+bias+activation path now register-blocks native low-p weights and supports SiLU/ReLU.**
`FusedLinear` (the GraphOptimizer fuses MatMul+Add+activation into it — the SD ResNet/FFN + LLM decoder-FFN
lever) had a register-blocked variant only for **f32** weights and only **None/GELU**; a fused fp16/bf16 linear
(SD's UNet/VAE) fell to the per-element `ForwardLowP`, forfeiting tiled throughput (this is the SD-specific
follow-on flagged when the MatMul-operator low-p reg-block shipped). New `FusedRegBlockedLowPActivation<T>` is the
f32 register-blocked FusedLinear with the weight decoded to float ONCE on the shared-mem load (`PrecisionConvert`,
amortized over the 4× register reuse) and bias+activation fused in the write-back; `ForwardLowP` routes `M,N ≥ 64`
there (same gate as the f32 path → M small / CPU / WebGL fall back to per-element). `FusedActivate` extended to
ReLU + SiLU, and the **f32** register-blocked gate extended to ReLU/SiLU too — so f32 SiLU FusedLinears (SD's
ResNet/FFN) also get the tiled kernel, not just None/GELU. Verified: `FusedLinear_LowP_RegBlocked_LargeMatchesReference`
(BFloat16 + Half × None/ReLU/GELU/SiLU, partial-tile dims) + `FusedLinear_Silu_RegBlocked_LargeMatchesReference`
(f32) + the existing FusedLinear regression green on all 6 backends (PMT `FusedLinear` 56/0). SD-Turbo (seed 42)
end-to-end went ~12.1 s → ~10.9 s on CUDA (the fused fp16 linears now tile; the rest is conv/attention-bound),
image still a real PASS (lumStd 114.6).

## Unreleased — SpawnDev.ILGPU 4.14.2 + MXFP4 scale single-source-of-truth decode

**Migrated to SpawnDev.ILGPU 4.14.2-local.1** (Fork / Algorithms.Fork `2.0.41`) — the WebGL fix for the
multi-exit `RawBitsToFloat` GLSL explosion (this consumption surfaced it: a multi-exit decode inlined before a
loop made the WebGL/GLSL structurizer duplicate the loop continuation per branch arm, blowing the MXFP4 kernel
to 76,876 GLSL lines / depth-56 → shader compile-fail). Geordi made `Float8E8M0Extensions.RawBitsToFloat` and
`Float4E2M1Extensions.RawBitsToFloat` single-exit / branchless (value-identical), so the MXFP4 kernel compiles.

**MXFP4 scale decode now composes the verified library E8M0 primitive (element AND scale).** Completing the MXFP4
single-source-of-truth story (the FP4 element was already composed; the scale was still inline): the shared
power-of-two block scale — `FusedDequantMatMul` GEMM + M=1 GEMV `DecodeMXFP4Element`, `FusedDequantGather`, and the
CPU `GGUFModel` full-dequant / per-row paths — now decodes through `ILGPU.Float8E8M0Extensions.RawBitsToFloat`
(OCP `float8_e8m0fnu`: `2^(e-127)`, `e==0xFF→NaN`; pure bit-math, transpiles on all 6 backends, block stays
packed). The inline `E8M0ToFloat` wrapper (`MathF.Pow(2f, e-127f)`) and the two CPU-path `MathF.Pow` sites are
deleted. Value-identical on every real MX scale byte (`e` in `1..254`; differs only at the degenerate `e==0xFF`,
where the library is spec-correct NaN). One verified decode for the MXFP4 element AND its scale. PMT `MXFP4` 20/0
on all 6 backends (incl. WebGL, now that the library decode compiles there).

## Unreleased — Register-blocked GEMM for native low-precision weights

**`MatMulLowPWeight` now uses the register-blocked tiled kernel for large matrices, not the per-element kernel.**
A native low-p weight (fp16 / bf16 / FP8) kept the memory win but forfeited GEMM throughput — every
`MatMulLowPWeight<T>` ran the simple one-result-per-thread kernel. New `RegisterBlockedMatMul.MatMulLowPWeight<T>`
(`RegBlockedLowPImpl<T>`) is the same 64×64-tile / 4×4-register GEMM as the f32 path, with the weight decoded to
float ONCE as it stages into the float shared-memory tile (`PrecisionConvert`) — so the decode is amortized over
the 4× register reuse and the hot register-block math is byte-identical to the f32 kernel (16 results/thread vs 1).
`MatMulKernel.MatMulLowPWeight` routes `M,N ≥ 64` here (the same gate as the f32 `MatMul`, so M==1 GEMV and
CPU/WebGL fall back to the per-element kernel). Verified: `F16_MatMulLowPWeight_RegBlocked_LargeMatchesReference`
(BFloat16 + Half, partial-tile dims, matches the low-p-weight fp32 reference) green on all 6 backends (PMT `F16`
98/0). NOTE: this covers the **MatMul-operator** low-p path; SD's fused MatMul+bias+activation linears go through
`FusedLinear` (per-element low-p), so extending the FusedLinear register-blocked path to low-p weights is the
follow-on for the SD-fused-linear throughput. (SD-Turbo end-to-end timing was not measurable this session — the
hub/WebTorrent model load was failing on a transient network condition.)

## Unreleased — Conv2D / ConvTranspose2D accumulate in f32 (not f64)

**Convolution now accumulates in f32, the ML standard, instead of f64.** Every `Conv2DKernel` variant (NCHW,
NHWC, depthwise, native-low-p-weight) and `ConvTranspose2DKernel` previously accumulated each output element's
`inC*kH*kW` MACs in `double` ("ultimate quality" caution). f64 is far slower on every GPU backend — consumer
cards run f64 at ~1/64 of f32, and WebGPU/WebGL **emulate** it via Dekker (multiple f32 ops per MAC) — so the
conv-heavy SD UNet/VAE paid it on every multiply-accumulate. f32 accumulation's rounding error over the MACs is
~1e-5 relative, far below an 8-bit pixel and within the existing MAC-scaled conv-test tolerance. Verified: Conv2D
+ ConvTranspose2D CPU-reference tests **164/0 on all 6 backends**; SD-Turbo (seed 42) output is **perceptually
identical** to the f64 baseline (`maxAbsDiff=1/255, meanAbsDiff=0.0007, 0.07% of bytes differ`) and the same run
went **~2.4× faster on CUDA (29.4 s → 12.1 s)**. (The numerically-sensitive double *reductions* in
Norm/Softmax/Stats kernels — variance, softmax-sum — are intentionally left in f64; only the conv MAC
accumulation changed.)

## Unreleased — GPU argmax greedy decode (no per-token full-vocab readback)

**Greedy next-token selection now runs on the GPU and reads back one int, not the ~1 MB vocab row.** Both decode
pipelines (`GgufTextGenerationPipeline`, `Gemma4MultimodalPipeline`) previously copied the entire last-position
logits row (~262K floats) to the host every token and ran a CPU argmax (`TextGenerationSampler.Greedy`). The new
`GpuArgMax` kernel does a parallel partial-argmax on the GPU — P threads each scan a strided slice and emit one
interleaved (value, index) pair, combined on the host over P (~1024) entries — so the per-token transfer drops
from `vocab*4` bytes to `~P*8` bytes in a single round-trip and the host scan from `vocab` to `P` (the latter
matters most in WASM, where a 262K single-threaded argmax per token is real work). No shared memory / barriers, so
it runs on all 6 backends including WebGL; tie-break is the LOWEST index, byte-identical to the CPU greedy path, so
greedy tokens are unchanged. The full-vocab readback is kept only for sampling (top-k / top-p) and repetition
penalty, which need the whole distribution on the host. Reused partial buffer (no per-token GPU alloc).
`GpuArgMax_MatchesCpuGreedy_VariousSizes` + `GpuArgMax_LowestIndexOnTie` green on all 6 backends (PMT `GpuArgMax`
14/0).

## Unreleased — SpawnDev.ILGPU 4.14.1 migration + FusedLinear native low-precision weight support

**Migrated to SpawnDev.ILGPU 4.14.1 (nuget.org stable).** Off `4.14.0` onto `4.14.1` (Fork / Algorithms.Fork
`2.0.40` transitive) - hardens the packed 4-bit tier (the per-backend `[NoInlining]` helper-function
generators now decode `Float4E2M1`/`QInt4`/`QUInt4` correctly on all 6 backends) and adds the new
`Float8E8M0` OCP MX-scale type.

**FusedLinear now supports native low-precision (bf16 / fp16 / FP8) weights.** A linear weight kept NATIVE
low-p by the GGUF loader (no f32 upcast - e.g. gpt-oss attention/output projections, after the 2026-06-19
bf16-native loader) has an EMPTY float `Data` view (its data lives in the typed low-p view). `FusedLinearOperator`
read `weights.Data` unconditionally, and its bounds check uses the SHAPE-derived `ElementCount` (still the real
count for a low-p tensor), so the check passed and the fp32 kernel's `Data.SubView(0, K*N)` on the empty view threw
`Index/Extent X out of bounds` - terminating a gpt-oss-20b forward at its first fused attention/output projection.
The op now branches on `Tensor.DType` (via `LowPWeightDispatch.FusedLinear`) and routes a low-p weight to a new
generic `FusedLinearKernel.ForwardLowP<T>` that reads the weight in its native type and converts to float
in-register (`ILGPU.PrecisionConvert`, no f32 weight temp - Rule 4) with the bias + activation
(None / ReLU / GELU / SiLU) fused, mirroring `MatMulKernel.MatMulLowPWeight<T>`. Verified: gpt-oss-20b CPU forward
now runs end-to-end with finite, non-degenerate logits (`finite=201088/201088`); new
`F16_FusedLinearOperator_RoutesBFloat16Weight` (bf16 None + GELU operator routing) + the fp32 FusedLinear
regression green on all 6 backends (PMT `FusedLinear` 44/0).

**Audited the low-p empty-`.Data` trap across every weight-consuming operator.** MatMul / Gemm / Conv already
branch on `IsLowP` and route to the native path (`LowPWeightDispatch`); FusedLinear was the only real gap (fixed
above). `EinsumOperator`'s GPU matmul fast-path now FAILS LOUD on a native low-p operand - a clear
`NotSupportedException` (route the low-p linear as MatMul/Gemm) instead of the cryptic `Index/Extent X out of
bounds` FailFast - since Einsum has no low-p kernel and its operands are normally both fp32 activations (no current
model exercises it; the guard is defensive). PMT `Einsum` 20/0 unchanged.


**BF16/F16 GGUF linear weights stay NATIVE on load (no f32 upcast).** The GGUF loader upcast every
non-block-quantized weight to f32 at load (`ExtractWeight` → `GetTensorFloat32`), doubling the VRAM and
upload bandwidth of BF16/F16 weights (e.g. gpt-oss attention/output projections, any fp16/bf16 GGUF). Now
a 2-D linear-B BF16/F16 weight is kept in its native 2-byte elements end-to-end: `GGUFGraphBuilder` records
it on a new `GGUFLowPWeight` channel (mirroring the quantized channel — raw bytes or stream offset + dtype
+ a transpose flag, with a presence marker in the float `weights` dict) instead of expanding to `float[]`;
both `InferenceSession` loaders upload the packed bytes, reinterpret (`Cast<byte,T>`), transpose in the
element dtype from the GGUF `[N rows][K]` storage to the declared MatMul-B `[K, N]`, and wrap as a
`Tensor.FromLowP<T>` (the MatMul/Gemm operators already decode it in-register via `MatMulLowPWeight<T>`).
The streaming path is zero-copy (stream → GPU byte buffer → reinterpret → drain before freeing). Halves
these weights' device memory (e.g. a `[vocab, n_embd]` bf16 lm-head stays ~1.16 GB instead of ~2.3 GB).
Norms, biases and the embedding/Gather table stay f32 (tiny, or no native kernel yet). PMT green all 6
backends (`F16_GgufLoad_BFloat16LinearWeight_TransposedNative_MatchesFp32` numerical + bf16 regression
74/0; `GGUFGraphBuilder_BF16Linear_RoutesToNativeLowP` routing).

**Migrated to SpawnDev.ILGPU 4.14.0 (nuget.org stable).** Off the `4.14.0-local.6` feed pin onto the
published `4.14.0` (Fork / Algorithms.Fork `2.0.39` transitive) - the 4-bit / low-precision data-type tier
(true packed `Float4E2M1`/`QInt4`/`QUInt4`, FP8/Half/bf16 parity, plus the kernel-safe raw-bits decode
primitives this release exposes).

**MXFP4 dequant now composes ONE verified FP4 decode.** The GGUF MXFP4 (ggml `GGML_TYPE_MXFP4`, gpt-oss
MoE experts) fused-dequant paths - `FusedDequantMatMul` (GEMM + M=1 GEMV), `FusedDequantGather`, and the
CPU full-dequant / per-row-embedding fallback in `GGUFModel` - now decode each FP4 nibble through the
library primitive `ILGPU.Float4E2M1Extensions.RawBitsToFloat` (bit-exact `ml_dtypes.float4_e2m1fn`, pure
bit-math so it transpiles on all 6 backends, buffer stays packed = Rule-4 no-unpack-on-load). The two
hand-rolled `kvalues_mxfp4` tables (`DecodeMXFP4Kvalue`, `MXFP4Kvalue`) are deleted. The decode is value-
identical: ggml's doubled kvalues `{0,1,2,3,4,6,8,12,..}` × halved scale `2^(e-128)` folds exactly into
the canonical MX form (OCP E2M1 element × E8M0 scale `2^(e-127)`); `E8M0HalfToFloat` → `E8M0ToFloat`. The
independent unit-test oracle remains the literal `RefMXFP4` kvalues table (no tautology). PMT `MXFP4`
(MatMul + Gather + GEMV oracle tests) green on all 6 backends.

## Unreleased — Gemma 4 multimodal chat + selectable-precision decode KV cache + SpawnDev.ILGPU 4.13.0 migration

**Browser model delivery: WebTorrent → OPFS → zero-copy GPU.** The browser gemma4 chat (and the WebTorrent
load path generally) now: (1) streams weights **zero-copy** — `AllocateQuantizedBytesFromStreamAsync` uses
`ArrayView.CopyFromStreamAsync`, which on browser backends pipes each torrent `IJSReadStream` chunk's
`Uint8Array` straight to the GPU via `queue.writeBuffer`, never entering the .NET/WASM managed heap (verified
313 MB distilgpt2, 312 MB zero-copy); (2) PARSES the GGUF header + small non-quantized tensors via an async
gather (`GGUFModel.HydrateNonQuantizedAsync` / `GetTensorRowFloat32Async`) so no synchronous `Stream.Read`
hits the async-only browser stream; (3) PERSISTS to OPFS — the demo wires the WebTorrent client to the OPFS
`IAsyncFS` and calls `RestoreFromStorageAsync` on startup, so a reload reuses the cached pieces instead of
re-downloading (verified: fresh client restores + re-reads from OPFS, no re-download). The `/cache` page lists
live WebTorrent transfers (progress, ↓/↑ share speed, peers, cancel/remove/seed).

**Gemma 4 multimodal CHAT.** `Gemma4MultimodalPipeline.StartChat()` returns a `Gemma4Chat` for multi-turn
conversation — the KV cache is REUSED across turns (each turn prefills only its new tokens, O(new) not
O(whole conversation)), so it stays coherent like llama.cpp/ollama chat. Text + image + audio in any turn.
`InferenceSession.DisableGGUFDecode()` detaches the cache (clears the session reference + resets the cursor)
so a caller never holds a dangling reference to freed GPU buffers. Example 05 is now an interactive chat
console (startup → chat until `/exit`; `/image`, `/audio`, `/reset` commands).

**Selectable-precision GGUF decode KV cache — BF16 now the default.** `GGUFDecodeKVCache` gains a
`KVCachePrecision` option (`F32` | `BF16`). **BF16 is now the default** (~half the KV-cache VRAM — VRAM is
the binding constraint running gemma4:12b's 7 GB weights on a 12 GB card), built on SpawnDev.ILGPU's
first-class `BFloat16` type; the earlier BFloat16 CUDA store/load codegen bug is fixed in 4.13.0-local.4, so
bf16 store/load is correct on CUDA/OpenCL/WebGPU/WebGL/Wasm. F32 (exact) remains available; the regression
suite runs BOTH arms (F32 for a tight layout/RoPE/kv_offset gate, BF16 argmax-strict with bf16 tolerance).
The cache write/read is ONE async path on every backend: a contiguous element-wise f32↔bf16 convert kernel
(write-index == thread-index, so WebGL-Transform-Feedback-safe — no scatter) plus `CopyFromAsync` between the
contiguous scratch and the maxSeq-strided store. `CopyFromAsync` (not sync `CopyFrom`) is what orders the
copy against the producing kernel on the Wasm worker pool — a sync `CopyFrom` of a kernel output silently
races there. Scoped PMT: passes on CUDA/OpenCL/WebGPU/WebGL/Wasm (5/6). The CPU desktop lane is held out by a
separate, tracked SpawnDev.ILGPU CPU-backend async-path deadlock (reproduces on a plain console forward,
before any KV-cache code, and on committed HEAD — DevComms
`tuvok-to-geordi-CopyFrom-silent-wasm-race-and-cpu-decode-hang-2026-06-16`), not by this cache. (This BF16
path also replaces an earlier manual `ushort`+bit-shift version that the WebGL GLSL emitter mis-compiled at
~17% error — the native type is the right fix.)

### WebGL multi-store kernel fixes + SpawnDev.ILGPU 4.12.0 migration

**WebGL Transform-Feedback "one store at the thread's own index" contract.** On the WebGL TF
vertex path a kernel thread may write exactly ONE output element, at its own index — no multi-store
(a per-element loop into a shared output) and no scatter (`out[reindexed] = in[thread]`). Violations
land only one store per thread and SILENTLY produce garbage on WebGL while every other backend is
correct. A sweep of every ML GPU kernel found and fixed the violators:

- **`TurboQuantKernels.Normalize`/`Denormalize`** — split the per-vector loop into a per-vector
  reduction + a per-element write. (`TurboQuant_QuantizedAttention` was cosine -0.0088 on WebGL.)
- **`FWHTKernel`** butterfly (batched + single-vector) — the in-place butterfly wrote two non-adjacent
  elements per thread; replaced with a one-store-per-thread out-of-place butterfly that ping-pongs
  between a work buffer and scratch. (`TurboQuant_KVCache_FlashAttention_EndToEnd` was 0.654 on WebGL.)
- **`MissingElementWiseKernels` TopK operator** — one-thread-per-row writing all k slots; now
  `rows*k` threads, each self-contained, writing its own slot. (Top-3 returned `9,0,0` on WebGL.)
  Also worked around an interpreted-Wasm codegen quirk: a `bool`-guard + nested `if/else if`
  selection mis-executed (all -inf) where each piece worked alone — flattened to `if (v > bestVal)`.
- **`TrainingKernels.SoftmaxCrossEntropy` forward** — split per-sample stats + per-element probs.
  (`Training_SoftmaxCE_Backward` saw probs summing to 1.149 on WebGL.)
- **`PostProcessingKernels.L2NormalizeRows` / `SoftmaxRows`** (multi-store) and **`TensorLayoutKernel`
  NCHW<->NHWC** (scatter: `out[reindexed] = in[thread]`, flipped to gather) — found by a follow-up
  audit; all unused in production today but shipped public API. Now covered by
  `MLTestBase.PostProcessingKernelTests` (CPU-reference, distinct values).

In all, **nine** ML kernels violated the WebGL TF "one store at the thread's own index" contract;
every one is now fixed and tested. (Filed an ILGPU request for a compile-time fail-loud guard.)

**Migrated to SpawnDev.ILGPU 4.12.0** (the sync/async contract: `Synchronize()` now throws on browser
backends — submit+wait is desktop-only — while `Flush()` is the sync submit everywhere). ML built
clean, but several browser code paths called sync `Synchronize()` purely to flush a command encoder
(a no-op-flush on 4.10.0, a throw on 4.12.0); a full sweep caught 11 such runtime failures the build
could not. Fixed by `Synchronize()` -> `Flush()` at the flush-intent sites in `GraphExecutor.Run()`
and the GGUF weight-load transpose; the wait-before-sync-read sites are desktop-only by nature.

**Test-harness memory-cascade guard.** A capture-enabled test that times out never runs its `finally`
to null `GraphExecutor.CapturedOutputs` (and the instance-norm GPU-buffer capture), so the leaked
static kept accumulating every later test's per-node tensors — turning a long Wasm lane into a
DistilGPT2 OutOfMemory + a cascade of follow-on timeouts. `RunTest` now evicts the static capture
state at each test's start (deterministic, like the existing zombie-accelerator eviction).

## Unreleased — gemma4:12b GGUF forward is CORRECT end-to-end (CUDA)

gemma4:12b greedy-decodes coherent thinking-model text through the pure-ILGPU engine
("What is the capital of France?" -> a `<|channel>thought` reasoning block then "The capital
of France is Paris.", clean self-stop on `<turn|>`). Verified 12/12 token-for-token against
ollama's bundled llama.cpp (`llama-server.exe`) on the same gguf. Three forward-correctness
bugs in `GGUFGraphBuilder`, each root-caused against the llama.cpp source:

- **Token embedding scale** — gemma multiplies the token embedding by `sqrt(n_embd)` right
  after the lookup (`gemma4.cpp`; no metadata key). Missing it left the token-identity signal
  at <1% of the RMS-normed residual stream -> every position collapsed to one argmax.
- **Norm `(1+weight)` was double-counted** — `conversion/gemma.py` (Gemma3/Gemma4) bakes the
  `+1` into the GGUF norm weights AT CONVERSION, so the graph must use them RAW. Folding a
  second `+1` 9x-inflated the small qk/post norms (k_norm 0.12 -> 1.12) -> QK blowup -> logits
  pinned at the 30 soft-cap. Now uses the stored weights verbatim.
- **Attention scale = 1.0, not 1/sqrt(head_dim)** — `gemma4.cpp`: `f_attention_scale = 1.0f`
  (the QK-norm folds the scale away). Defaulting to 1/sqrt(512) on the global heads made
  softmax ~22x too flat -> attention averaged all positions (cross-position cosine -> 1 by
  layer ~23) -> generation degenerated to whitespace. Now emits scale=1.0 on gemma4 attention.

Tests: `Gemma4_GraphBuilder_NormWeightsRaw_NoDoublePlusOne`,
`Gemma4_GraphBuilder_EmbeddingScaleSqrtNEmbd`, `Gemma4_Attn_AttentionScaleIsOne`
(scoped PMT Gemma4 74/74). `Examples/04.GGUFTextGen.Console` gains greedy decode + the gemma4
chat template, a teacher-forcing per-position comparison vs a reference, and GGUF_PROBE /
cross-position-cosine diagnostics. Decode is currently full-recompute (~7s/token); KV-cache is
planned (`Plans/gemma4-kvcache-decode-plan-2026-06-12.md`).

## Unreleased — gemma4 decode-path kernels (GEMV routing, masked flash attention, RoPE generalization)

The three kernel prerequisites for the gemma4 bring-up (the graph wiring selects and
passes the per-layer values; the kernels honor call parameters):

- **GEMV routing**: M==1 matmuls (every LLM decode matmul) no longer pad through the
  16x16 tiled kernel (15/16 of each group idled); the simple thread-per-output kernel is
  the coalesced GEMV for row-major B. Applies to `MatMul` and `BatchedMatMul` (per-head
  decode attention).
- **`FusedAttentionKernel` masking**: new `Forward(..., causal, window, kvOffset)` -
  index-computed causal + sliding-window masking (gemma4's 5:1 SWA/global interleave
  passes per-layer windows; global layers pass window >= seqKV), KV-cache decode at any
  position via kvOffset. Branch-free body (WebGL emitter rule), exact-bits softmax scale
  (the old param quantized it to 1e-4), params-buffer ring (fixes a dispose-while-
  pending-dispatch hazard). Original API unchanged via delegation.
- **`RoPEKernel` generalization**: per-call `ropeBase` (gemma4 dual 10000 local /
  1000000 global), pairing-style flag (NeoX split-half | GPT-J interleaved - the old doc
  described interleaved while implementing split-half; both are now real), partial
  `rotaryDim` with exact tail pass-through. Original API unchanged via delegation.

All three: CPU-oracle test suites, scoped PMT green on all 6 backends, offline GLSL
size probes (the WebGL emitter constraint) before gating.

## Unreleased — GGUF quantization correctness overhaul (the K-quant landmine)

### The bug class (gemma4 gap #0)

The GGUF GPU path dropped the GGML quantization type: every quantized tensor was decoded
by the fused MatMul kernel as **Q4_0**, so any K-quant model (Q4_K_M = the modern default,
including gemma4) computed with silently garbage weights. Root-caused by Seven 2026-06-11;
fixing it surfaced two more layers of the same disease:

- **All five K-quant CPU dequant routines (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K) were wrong** vs the
  ggml reference (element permutations inside super-blocks); Q8_1 treated ggml's `s`
  (a dot-product term) as a per-element min. All rewritten as direct ports of
  ggml-quants.c `dequantize_row_*` (fetched verbatim). Legacy Q4_0/Q4_1/Q5_0/Q5_1/Q8_0
  were verified correct.
- **The GPU Q4_0 kernel read nibbles in interleaved order**; the GGUF format is split
  order (low nibbles of a 16-byte run = elements 0-15, high = 16-31). Even pure-Q4_0
  files decoded permuted. Its unit test encoded synthetic data in the kernel's OWN wrong
  order - self-consistent, never spec-checked.

### What shipped

- **Typed fused dequant MatMul** (`FusedDequantMatMul`): Q4_0 + Q8_0 + Q4_K + Q6_K
  (gemma4's Q4_K_M mix), `Forward(..., GGMLType)`, single packed-int kernel per type on
  all 6 backends, host-side block-size validation, per-shape params-buffer cache
  (replaces a dispose-while-dispatch-pending hazard). Weights stay COMPRESSED in GPU
  memory; blocks decode in registers.
- **NEW `FusedDequantGather`**: quantized embedding lookup - the table stays compressed
  in VRAM and only gathered rows decode in-register (a gemma-class Q6_K table is ~770MB
  compressed vs ~4GB as F32). No CPU dequant pass, ever (interpreted Blazor WASM cannot
  afford one and the heavy work belongs on the GPU).
- **Type travels with the bytes**: `GGUFGraphBuilder` returns `GGUFQuantizedWeight(bytes,
  type)`; `OperatorRegistry.QuantizedWeightTypes` carries types to operators (executor
  plumbing untouched); MatMul/Gather operators route by type and THROW on an untyped
  quantized tensor instead of guessing a layout. Unsupported quantized types
  (Q2_K/Q3_K/Q5_K/IQ*) throw a clear error at load - no silent fallback.
- **Orientation contracts**: GGUF linear storage is [N rows][K contig]; fused kernels
  read it transposed (that IS the contract, B declared [K,N]); F32/F16 linears get a
  one-time GPU transpose at upload; embeddings declare physical [vocab, n_embd] order.
  **Tied-embed LM head is zero-copy**: the head MatMul reads the SAME compressed buffer
  as the embedding Gather via an alias initializer - the old per-forward Transpose node
  is gone (it also read a wrongly-declared shape).
- **`Tensor.ShapeOnly`**: quantized graph entries carry shape without a dead F32 buffer
  (previously a full-size buffer was rented per quantized weight purely for shape
  tracking - ~4GB for a gemma-class embedding).
- **9-test oracle suite** (`MLTestBase.FusedDequantMatMulTests`): CPU dequant locked to
  ggml-reference ports for ALL TEN types; GPU fused MatMul + Gather vs reference oracle
  per type; operator routing incl. the untyped-throw; graph-builder contract assertions.
  Synthetic blocks are encoded in the REAL GGUF layout.

### Breaking (preview API)

`FusedDequantMatMul.Forward` requires a `GGMLType`; `GGUFGraphBuilder.BuildGraph` returns
4 items (typed quantized dict + transpose set). No silent-default overloads were kept -
an untyped quantized decode is exactly the bug this release removes.

## 4.0.0-preview.4 (2026-05-23) — Transformers.js-style Tensor API + /depth GPU-direct rendering + palette swap on accelerator

### Headline: native C# Tensor types

Three new types under `SpawnDev.ILGPU.ML.Tensors` give the library a Transformers.js / ONNX-Runtime-style API surface while preserving the zero-cost GPU paths C# can actually deliver:

- **`Tensor<T>`** (class, generic over `T : unmanaged`) — host-side shape-tracked view over an accelerator buffer. Reshape / Slice / SubTensor zero-copy. The legacy non-generic `Tensor` stays for backwards compatibility and now exposes a `.View` property returning `TensorView<float>`.
- **`OwnedTensor<T>`** (`IDisposable`) — owns a `MemoryBuffer1D<T>` and disposes it. Composition over inheritance: holds an internal `Tensor<T>`, exposes it through `.AsTensor`, exposes the kernel view through `.View`. Implicit conversions to both types so call sites stay clean. Factories `Allocate(accelerator, shape)` and `FromHost(accelerator, data, shape)`.
- **`TensorView<T>`** (struct, blittable) — kernel-passable. Inline `D0..D3 + Rank` ints (no managed array). Generic over `T : unmanaged`. Indexers `Get1D..Get4D` / `Set1D..Set4D` compute row-major strides inline so kernels stop having to take a 6-parameter shape signature.
- **`OwnedTensorMap<T>`** (`IDisposable`) — named-tensor output bag. Used as the return type of `InferenceSession.RunOwnedAsync`. `using var outputs = await session.RunOwnedAsync(...)` disposes every contained tensor in one go when the map goes out of scope.

`InferenceSession.RunOwnedAsync(IDictionary<string, Tensor<float>>) → Task<OwnedTensorMap<float>>` is the headline new method. Inputs are `Tensor<float>` views (`OwnedTensor<float>` converts implicitly). Outputs are each copied off the executor's pool-managed buffer to a fresh caller-owned buffer (GPU-to-GPU `CopyFrom`, no host readback) so subsequent runs cannot mutate previously-returned tensors. The legacy `RunAsync(Dictionary<string, Tensor>)` is unchanged; migration is opt-in per consumer.

### Phase 2 kernel migrations (proof-of-concept)

`ImagePostprocessKernel.ResizeBilinear` and `ImagePostprocessKernel.DepthToColormapPalette` now both expose `TensorView<T>` overloads alongside their legacy raw-`ArrayView` signatures. Kernels read source/dest dimensions directly from the tensor view; the host-side call-site stops having to keep `(srcW, srcH, dstW, dstH)` straight. `DepthEstimationPipeline` migrated to both new overloads. Other shape-heavy kernels (`SuperResCompositeYCbCr`, `AccumulateYTile`, Conv2D, attention) follow in subsequent previews.

### /depth: GPU-direct rendering via `ICanvasRenderer`

The depth demo no longer produces a base64 PNG data URL for display. `DepthPage` instantiates `CanvasRendererFactory.Create(accelerator)` (backend-optimized: WebGPU texture copy / WebGL blit / CPU `putImageData`), attaches it to the `BeforeAfterSlider`'s After-side `<canvas>`, and calls `ICanvasRenderer.PresentAsync(buffer)` — the GPU buffer renders straight to the page with zero PNG encode, zero base64 string, zero `Blob` shuffling, zero host readback of any rendered pixel data.

### /depth: palette swap is one kernel dispatch

`ImagePostprocessKernel.DepthToColormapPalette` now takes a palette index parameter (plasma / viridis / inferno / grayscale) and branches into the appropriate piecewise-linear interpolation inline. `PaletteFromName(string)` does the UI-string → int translation. `DepthEstimationPipeline.EstimateGpuRawAsync` returns a raw depth GPU buffer + min/max scalars; `DepthEstimationPipeline.ApplyColormapGpuAsync` applies the colormap kernel with any palette against that cached raw depth. Result: the dropdown is one accelerator dispatch + one `PresentAsync`. No re-inference, no host readback of depth values.

### Dependencies (unchanged from preview.2)

- `SpawnDev.ILGPU` 4.9.8
- `SpawnDev.WebTorrent` 2.3.1
- `Microsoft.AspNetCore.Components.Web` 10.0.4

## 4.0.0-preview.3 (2026-05-23) — SuperResolutionPipeline: real tile-based super-resolution + /remove-bg slider rendering fix

### Fixes
- **SuperResolutionPipeline** rewritten for proper tile-based super-resolution. The
  prior pipeline emitted grayscale output (only the Y channel was used,
  Cb/Cr discarded) at the model's fixed square output dimensions (e.g. 672×672 for
  a 224×224 ESPCN @ 3×) regardless of source aspect. preview.3:
  - Tiles the source into overlapping `modelW × modelH` patches (overlap defaults
    to 16 source pixels) and runs each through the model independently.
  - Accumulates super-resolved Y outputs on the accelerator into a single
    `(sourceWidth * scale) × (sourceHeight * scale)` destination plane with a
    per-pixel contribution count. Overlap regions are averaged (mean of all
    contributing tiles) to smooth boundary seams.
  - Composites the assembled Y with source-derived Cb/Cr (BT.601, bilinear) to
    produce color RGBA at the correct aspect ratio.
  - Sequential per-tile kernel dispatches mean no atomics are required — works
    across all 6 backends including WebGL.
  - All per-pixel work on the accelerator; CPU only does orchestration (which
    tile index, what coordinates) and the final result readback in
    `UpscaleAsync`. `UpscaleGpuAsync` keeps the result on the GPU.
- **BeforeAfterSlider component** (demo): mirror the clip-path onto the Before
  image so it only renders on the LEFT of the slider. The prior layout left the
  Before image full-width underneath the After image — for opaque results that
  was fine, but the background-removal demo's transparent RGBA After image let
  the original brick wall show through the alpha-transparent areas, making the
  result visually identical to the source even when the alpha mask was correct.
  Demo only; library unchanged.
- **`AfterHasTransparency` parameter** on `BeforeAfterSlider`: renders a 16×16
  grey checkerboard behind the After image so transparent areas read as
  "transparent" rather than blending with the page theme. `RemoveBgPage` opts in
  when bg-mode is Transparent.

### Dependencies (unchanged from preview.2)

- `SpawnDev.ILGPU` 4.9.8
- `SpawnDev.WebTorrent` 2.3.1
- `Microsoft.AspNetCore.Components.Web` 10.0.4

### Known rough edges

- RMBG-1.4 (`/remove-bg`) inference itself is slow during load + compile on
  WebGPU. The slider rendering bug above is fixed; once load + compile finishes
  the actual segmentation works. Inference perf improvements are a follow-up.

## 4.0.0-preview.2 (2026-05-23) — dep bump + DepthEstimationPipeline aspect-ratio fix + /remove-bg demo cleanup

### Fixes
- **DepthEstimationPipeline.EstimateAsync / EstimateGpuAsync** gained optional
  `outputWidth = 0, outputHeight = 0`. Default `(0, 0)` matches source image dimensions
  (preserves aspect — previously the depth result came back at the model's square
  input size and visibly squished against the source). `(w, 0)` and `(0, h)` fit
  one axis and derive the other from source aspect; `(w, h)` is exact. Resize runs
  on the accelerator via a new `ImagePostprocessKernel.ResizeBilinear` — no CPU
  readback of the raw depth tensor.
- **CoreML/ONNX format detection false positive** closed. `CoreMLParser.IsCoreML`
  now refuses when the next protobuf tag after `specificationVersion` is `0x3A`
  (the ONNX graph field-7 tag). Without this guard, ONNX models with no producer
  string (e.g. SqueezeNet 1.1) were misclassified as CoreML and routed through
  the CoreML placeholder graph — surfaced as `Inference failed: KeyNotFoundWithKey, output`
  in the `/classify` demo.
- **/remove-bg demo backend selector** now honors the dropdown (was always
  WebGPU regardless of pick), and the **Transparent / White / Blur buttons** now
  actually composite the result on mode change.

### Dependency bumps
- `SpawnDev.ILGPU` 4.9.7 → **4.9.8** — WebGL device probe no longer leaks an
  `OffscreenCanvas` + `WebGL2RenderingContext` per registration. This is what
  caused Chrome's "too many WebGL contexts" warning in v4.0.0-preview.1 even
  when the WebGL backend was never selected.

### Known rough edges (unchanged from preview.1)

- RMBG-1.4 (`/remove-bg`) on WebGPU is sluggish during load + compile and the
  output mask is still being investigated. WebGL works for this model. Other
  pipelines (depth, classify, style) are smooth on WebGPU.
- Wasm has a tighter memory ceiling than other backends; large models may exceed
  it on Wasm.

## 4.0.0-preview.1 (2026-05-23) — first nuget.org preview

First public preview. SpawnDev.ILGPU.ML provides native GPU neural-network inference
for .NET — C# compute kernels transpiled to WebGPU, WebGL, Wasm, CUDA, OpenCL, and CPU
via [SpawnDev.ILGPU](https://www.nuget.org/packages/SpawnDev.ILGPU). No ONNX Runtime, no
JavaScript bridge, no native binaries.

This is a **preview**: API is stabilizing but will change. Not yet recommended for
production. Ship feedback as GitHub issues — bugs that surface in your model usually
become regression tests in our PMT suite.

### What works today

- **6-backend coverage** — same kernels run on WebGPU, WebGL, Wasm, CUDA, OpenCL, CPU.
- **Universal model loading** — `InferenceSession.CreateFromFileAsync()` auto-detects
  ONNX / TFLite / GGUF / SafeTensors / TF GraphDef / PyTorch / Core ML from magic bytes.
- **16 inference pipelines** — Classification, StyleTransfer, SuperResolution,
  DepthEstimation, ObjectDetection, PoseEstimation, FaceDetection, TextClassification,
  ZeroShotClassification (CLIP), BackgroundRemoval, SpeechRecognition (Whisper),
  TextGeneration, FeatureExtraction, Diffusion (DDPM), TextToSpeech (SpeechT5),
  Image3D (TripoSR).
- **Verified demo pipelines for this preview**: depth estimation (Depth Anything V2),
  style transfer, image classification (SqueezeNet). Other pipelines build and run
  but may have rough edges. See the [live demo](https://lostbeard.github.io/SpawnDev.ILGPU.ML/).
- **Zero-copy GPU pipeline** — `ImagePreprocessKernel` → inference → `ImagePostprocessKernel`
  (incl. plasma colormap, bilinear resize for depth/masks) → `CanvasRendererFactory`.
  Data enters the GPU at pre-processing and stays until the pixel lands on the canvas.
- **Aspect-aware depth pipeline** — `DepthEstimationPipeline.EstimateAsync` /
  `EstimateGpuAsync` accept optional `outputWidth = 0, outputHeight = 0`. Default
  matches source dimensions (preserves source aspect); explicit values fit one
  axis (derives the other from aspect) or set exact size. Resize runs on the
  accelerator via bilinear interpolation.
- **HuggingFace CDN integration** via `ModelHub` with OPFS caching in the browser.
- **Streaming weight loader** for large models (GPT-2 652MB single-tensor-at-a-time).
- **30 GPU kernel files** — MatMul (tiled 16×16 shared mem, ~92-101 GFLOPS validated),
  Conv2D, FWHT, TurboQuant KV cache compression, RoPE, QKNorm, GroupNorm, SelectiveScan
  (Mamba-3), MarchingCubes, SpatialMemoryUnit, and more.

### Known rough edges

- This is a preview — pipelines other than the three named above are not all verified
  end-to-end across every backend.
- Some operators have backend-specific limitations on WebGL (no shared memory / atomics
  / barriers). `AcceleratorRequirements` in the underlying SpawnDev.ILGPU lets consumers
  declare requirements and have an incapable backend filtered out at selection time.
- Memory pressure on Wasm is higher than other backends due to the 2 GiB heap ceiling.
  Large models (GPT-2 scale) may exceed this on Wasm; prefer WebGPU for those.

### Dependencies

- `SpawnDev.ILGPU` 4.9.7 — the underlying transpiler and runtime.
- `SpawnDev.WebTorrent` 2.3.1 — for the optional P2P model delivery code path.
  (Will track WebTorrent 3.x in a follow-up preview once interop is verified.)
- `Microsoft.AspNetCore.Components.Web` 10.0.4 — for the demo's Blazor surface.

### Credits

The SpawnDev Crew:

- **LostBeard** (Todd Tanner) — captain, library author, vision.
- **Data** — operations officer, ML library lead.
- **Geordi** — chief engineer, ILGPU internals and GPU kernels.
- **Riker** — first officer, WebRTC / WebTorrent / BlazorJS plumbing.
- **Tuvok** — security/research officer, codecs, design review.
