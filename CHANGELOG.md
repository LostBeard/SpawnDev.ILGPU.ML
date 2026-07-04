# SpawnDev.ILGPU.ML Changelog

Notable changes per release. Pre-stable; API will change between preview drops.

## 4.0.0-preview.7 (2026-07-04)

### 🔧 Storage-quota browser-download runaway fixed via dependency bump — WebTorrent 3.2.12 + BlazorJS 3.5.15 (Tuvok)

Bumps `SpawnDev.WebTorrent` 3.2.11 → **3.2.12** and `SpawnDev.BlazorJS` 3.5.14 → **3.5.15** to pull in the OPFS storage-quota runaway fix. Previously, a browser model download whose OPFS piece store failed (the origin out of space) would re-request the same piece forever - 169 identical range GETs observed live on the 1.5B model - because the failing write leaked an OPFS swap file per attempt (BlazorJS) and the picker re-requested the unflagged piece (WebTorrent). Now the write aborts-on-throw (no swap leak) and the torrent classifies the failure and pauses with an `OnError` event instead of hot-looping. The successful download/decode path is byte-identical - this only changes the out-of-space failure behavior. No ML API change.

### 🔧 DA2 content-free depth FIXED: FuseAttention double-scaled Q-pre-scaled ViT attention (Seven)

Since 89180ea (2026-06-16) the MatMul-form attention fusion accepted DA2/onnx-community ViT exports
(Q PRE-scaled, plain-Transpose K, no post-MatMul scale) with scale left at the kernel default -
stacking 1/sqrt(hd) on Q's embedded scale. Scores /8 → near-uniform softmax → plausible-range but
CONTENT-FREE outputs that passed every range-based gate for 17 days (found from Captain's "depth map
isn't good" via bisect over 353 commits + onnxruntime ground truth). Fix: detect the Q-side scalar
Mul → scale=1.0. Plus: `ML_NO_ATTN_FUSION=1` kill-switch; structure-measuring gates (cross-path
identity at 224 AND 518 + calibrated strongEdges content-free guard, colormap oracles, hub-bytes
identity); `DepthEstimationPipeline.EnableGraphCapture` now defaults ON (the always-on switch lives
in the pipeline - /depth went 6252ms → ~300ms per estimate).

### 🖼️ SD-Turbo generates IN THE BROWSER: E2E green on WebGPU + CUDA + OpenCL (Seven)

`SDTurbo_Generate_E2E` passes on WebGPU (the browser lane), CUDA, and OpenCL - hub-streamed 2.5GB
trio, single-step diffusion, verified non-degenerate 512x512 (photorealistic on visual check).
- **GPU Euler step**: `ElementWiseKernels.AddScaledInPlace` (axpy) replaces the per-denoise-step
  two-readbacks-plus-upload host round trip; element-exact gate vs the CPU scheduler reference.
- **`Graph.SessionGraphCapture`** (new): reusable capture-once/replay-many wrapper per
  InferenceSession, owning stable input clones (capture binds input buffers into the graph - caller
  transients being disposed was a native crash). CLIP/UNet/VAE wired; **defaults OFF for this
  pipeline**: SD-class activation volumes guarantee pool misses mid-capture and ILGPU's
  AllocateWithReclaim (flush/alloc/free during stream capture) corrupts the CUDA context -
  bisect-proven per sub-model; tracked in `Plans/sd-capture-pool-priming.md`
  (`SDTURBO_FORCE_CAPTURE=1` opts in for that work).
- Capture-safety fixes (permanent, all captured models): `WhereBroadcastND` stable param slot
  (d0cd5a6 pattern - was missed), `InstanceNorm(/InPlace)` cached stats scratch,
  `BufferPool.DisposeBucketedBuffers` refuses to free under SuppressDrains, `RangeOperator`
  empty-scalar guards + diagnostics.
- E2E hardening: cold OPFS WebTorrent client (restored-shared-client trap), Wasm lane skip with a
  tracked plan (`Plans/wasm-weight-paging.md` - fp16-resident weights / OPFS per-layer paging).
- Known remaining: CPU console lane crashes during VAE weight upload (captures off too - separate
  investigation); WebGL lane timeout (slow-lane class).

### 🏆 WebGPU LLM decode 34.1 tok/s (23x): GEMV un-exclusion + argmax folded into the plan (Seven)

Two more levers on top of the patched-replay decode, both token-identical:
- **`FusedDequantMatMul.EnableWebGPUGemv`**: WebGPU M==1 re-routed onto the cooperative shared-mem
  GEMV. The old exclusion ("~75x slower", 2026-06-13) was measured on the SwiftShader-CPU harness -
  VOID; on real Dawn the GEMV nearly HALVED the decode GPU floor (23.3 -> 12.3ms/tok, the
  FusedDequantQ8_0 term). Opt-in flag (env GGUF_GEMV_WEBGPU=1) pending broader soak; the decode
  gates run with it ON.
- **Greedy argmax FOLDED into the captured plan** (`GpuArgMax.DispatchPartials`/`ReadPartialsAsync` +
  `WebGPUDecodeCapture.PatchAndDecodeGreedyAsync`): the partial-argmax kernel is recorded as the
  plan's final dispatch, and the partials readback's own mapAsync fence orders after the replay -
  ONE GPU round-trip per token (was three: replay sync + argmax sync + readback). Decode
  38.7 -> 29.3ms/tok. The generator's greedy path uses it automatically under
  EnableWebGPUDecodeCapture; sampling/repetition-penalty steps keep the full-logits path.
- Net campaign: **1.5 -> 34.1 tok/s in the browser (23x), token-identical** (CUDA reference 87.9).
  Remaining wall: the single mapAsync fence (~15ms round-trip latency, not bytes) - candidates:
  speculative double-buffering or a JS-side decode loop (the shared-worker library-ization).

### 🏆 WebGPU LLM decode 15x: 1.5 -> 22.3 tok/s TOKEN-IDENTICAL via patched dispatch-plan replay (Seven)

The browser Ollama-clone/ai-chat decode lever, end-to-end. `WebGPUDecodeCapture` captures the decode
step ONCE as a dispatch plan and replays it per token with a single interop crossing; every
KV-cursor-dependent value is DISCOVERED (not hard-coded): two captures at pastLen P0/P0+1 plus
stable-slot observer probes are diffed, and each difference is fitted as value(P)=v0+slope*(P-P0)
(fail-loud on any structural/non-affine mismatch). Discovered surface on qwen2.5-0.5b: 48 scalar
patches (RoPE startPosition x q/k x 24 layers), 48 copy-destination patches (K/V cache appends),
24 attention param slots. Per token: raw queue.WriteBuffer patches (1.9ms) + plan call (0.3ms) +
GPU 23.3ms. Opt-in via `GgufTextGenerationPipeline.EnableWebGPUDecodeCapture` (no-op elsewhere;
prefill/multi-token steps stay direct; patches are affine in pastLen so ONE capture survives across
turns and prefix-cache reuse).
- Gates: `GGUF_WebGPU_DecodeCapture_TokenIdentical` (greedy 48 toks, captured == direct, char-exact)
  + `GGUF_WebGPU_DecodePlanReplay_Probe` (same-state bit-exact, 20.8ms floor) + KV-cache suite +
  attention oracle unchanged.
- Uses SpawnDev.ILGPU `4.17.2-local.7` (dispatch-plan patch surface). Enablers: stable-slot
  observers on `FusedAttentionKernel`/`CaptureParamArena` (null-checked, inert in production).
- FOUND on the way (tracked): ILGPU `ArrayView.CopyFromCPU` on WebGPU costs ~14ms/call (vs 0.02ms
  raw writeBuffer) - it made the first patched replay 456ms/tok; the driver uses raw writes.
- Remaining per-token costs: GPU 23.3ms (72% = the WGSL Q8_0 dequant GEMV - stage 3) + ~19ms
  argmax/detok/loop residual (the argmax readback fence - stage 4 candidate).

### 🎥 WebGPU VIDEO PATH: DepthEstimationPipeline.EnableGraphCapture now works in the browser - 10.4 fps end-to-end (Seven)

`EnableGraphCapture` gains the WebGPU twin of the CUDA wiring: first frame captures the forward into
a `WebGPUGraphCapture` dispatch plan, every subsequent frame replays it (the per-frame preprocess
dispatch writes fresh pixels into the capture's stable input buffer; re-captures on resolution
change). Measured end-to-end (`DA3_WebGPU_Pipeline_GraphCapture_VideoPath`, RTX 4070): capture frame
27.8s one-time, then **96.2ms/frame = 10.4 fps INCLUDING preprocess + bilinear resize + the min/max
readback** - the full SpawnScene webcam-depth cost. Capture output bit-exact vs direct
(capDiff=0.00E+000); a flipped-input frame changes the depth (stale-replay guard).
Found + fixed upstream in SpawnDev.ILGPU `4.17.2-local.6`: the plan replay submitted its own command
buffer WITHOUT flushing the accelerator's pending encoder, so an input refreshed by a KERNEL
DISPATCH (this exact preprocess pattern) landed after the replay - the replay silently read the
previous frame. `ReplayAsync` now flushes first; the ILGPU DispatchPlan guard grew the
dispatch-written-input case. Known headroom: the min/max readback should become a GPU reduction
(existing TODO) - it is a large share of the 96.2-vs-~70ms gap.

### WebGPU dispatch-elide gap FIXED (zero-length values) + elide-ON plan capture (Seven)

The tracked executor gap - "Tensor 'X' not found (needed by Cast)" under fold+elide on WebGPU -
root-caused by a new throw-site invariant diagnostic (`producerOp=Slice elidedThisRun=True
inRuntimeConstants=True constLen=0`): a shape op producing a legitimately EMPTY value (DAv3
`blocks.N/attn` Slice -> Cast, an empty dim-list slice) was elided, but the on-demand materializer
requires len>0, so its GPU consumer threw. Fix: zero-length values are excluded from `elideSafe`
and DISPATCH via the proven path (which produces the real empty GPU tensor). New gate
`DA3_WebGPU_ElideOn_Forward` - full elide-ON forward, range 0.136470 correct. `WebGPUGraphCapture`
now captures under dispatch-elide (CUDA-capture parity): plan 2515 -> 2130 ops. Honest accounting:
the elidable class on this graph is ~385 tiny shape passes (~2-3ms), NOT the ~1200 once hoped - the
tail is mostly feature-tensor elementwise ops. Replay frame: best measured **64.4ms**, run-to-run
band ~64-76ms (GPU clock state; measured by identical back-to-back runs), vs ORT-Web 73ms.
Also: measured-NEGATIVE result recorded - a vec4 F4/AsAligned16 tile-load variant of the
reg-blocked fused linear passed 62/62 but measured 11.0 -> 11.9ms in-frame (shared-tile staging is
already coalesced); reverted, documented in the new aligned-shape test's doc block.

### 🏆 DAv3 WebGPU 66.1ms/frame bit-exact - UNDER ORT-Web's 73ms (Seven)

The beat-ORT milestone. Attribution named `FusedAttentionPerQueryRegister` as 56% of the frame
(49.5ms of 88ms GPU in 12 dispatches) and the kernel read found two load-path defects; fixing them
took the replay frame from ~104ms to **66.1ms, maxAbsDiff=0.00E+000** (253.6x over the 16.8s direct
forward; ORT-Web warm = 73ms):

- **Q-hoist (generic register kernel, all K/V types):** the kv loop re-read the SAME 16 Q floats
  from global memory every iteration (SKV x 16 redundant loads per lane). Q is now hoisted into a
  const-16 register array once per query. Pure load hoist - arithmetic order unchanged, output
  bit-identical; the GGUF bf16/quant KV decode path gets it too.
- **`FusedAttentionPerQueryRegisterF32Impl`:** f32 specialization where K/V (+ the one-time Q hoist)
  stream through 128-bit `F4`/`AsAligned16()` loads (PTX `ld.v4.b32` / WGSL `vec4<f32>`) - 4 loads
  per 16-dim tile instead of 16, same MAC order (bit-identical). Q/K/V are cast to `ArrayView<F4>`
  HOST-side (the WGSL backend has no in-kernel ViewCast lowering - fail-loud NotSupportedException;
  flagged to the ILGPU lane). Host routes here for T=float when all three view offsets are
  16-byte-aligned (AsAligned16 is an alignment assertion); falls back to the scalar generic form
  otherwise. Type specialization, identical on all 6 backends - not a backend variant.
- **Measured:** attention 49.5ms -> **13.4ms (3.7x)**. New attribution top: RegBlockedLinear 11.0ms,
  Conv2DImplicitGemm 7.1ms, LayerNormFused 5.6ms - the next vec4 candidates; ~10ms shape-op tail
  pending dispatch-elide.
- **Gates:** Attn oracle 98/98 all backends (incl. WGSL vec4 trigger on real Dawn); PlanReplay
  bit-exact; GGUF decode byte-identical gate.

### Per-kernel GPU-time attribution of the replay frame (Seven)

`WebGPUGraphCapture.ReplayTimedAsync()` (consuming SpawnDev.ILGPU `4.17.2-local.5`'s timed
dispatch-plan replay): replays the captured forward with a WebGPU 'timestamp-query' timestamp per
compute pass and returns GPU ms aggregated by kernel label, sorted descending - the instrument that
decomposes the ~102ms GPU floor into named kernels on hardware. `DA3_WebGPU_PlanReplay_VsOrtWeb` now
logs the full attribution JSON and appends the top-5 kernels to its report. PMT harness adds
`--enable-webgpu-developer-features` (unquantized GPU timestamps; Chrome otherwise rounds to 100us).

### Replay-frame split measured: the WebGPU replay is ~98.5% GPU execution - pipelining is NOT the lever (Seven)

`WebGPUGraphCapture` replay instrumentation (consuming SpawnDev.ILGPU `4.17.2-local.4`'s
`WebGPUDispatchPlan.GetLastReplayTimings()`): new `CollectTimings` opt-in + per-replay
`LastInputCopyMs` / `LastPlanCallMs` / `LastSyncMs` / `LastJsEncodeMs` / `LastJsSubmitMs`
(.NET splits are Stopwatch-free and always recorded; the JS split costs 2 interop reads only when
`CollectTimings`). Measured on hardware (`DA3_WebGPU_PlanReplay_VsOrtWeb`, RTX 4070): per ~104ms frame,
planCall = **1.5ms** (jsEncode 1.3ms for all 2515 ops + jsSubmit ~0.05ms + interop wrapper ~0.2ms),
gpuWait = **~102ms**. Verdict: the frame is ~98.5% GPU execution - pipelining encode/writeBuffer against
GPU work can hide at most ~1.5%; the sub-73ms path is reducing GPU work itself (dispatch-elide of the
~1200 shape ops in the plan + WGSL kernel throughput). Replay stayed bit-exact under instrumentation.

### BREAKTHROUGH: WebGPU dispatch-plan capture/replay - DAv3 at 99.5ms/frame bit-exact (1.36x from ORT-Web), + the SwiftShader discovery (Seven)

Two coupled findings. (1) The PMT Playwright bundled Chromium NEVER had hardware WebGPU - every prior
WebGPU perf number (77s warm, 4-5 GF GEMM benches, the drains decomposition) was the SwiftShader CPU
software rasterizer (caught by the new `WebGPU_AdapterIdentity_Probe`; correctness results all stand).
Harness fixed: `Channel = "chrome"` + `--disable-software-rasterizer` + removed the Windows-hostile
Vulkan feature flag -> vendor=nvidia arch=lovelace. (2) New `WebGPUGraphCapture` (consuming SpawnDev.ILGPU
4.17.2-local.3's `WebGPUDispatchPlan` capture/replay - the browser twin of `CudaGraphCapture`, same warm
A/B + stable-slots + readback-cache + suppress-drains regime; dispatch-elide off on WebGPU pending its
executor gap): capture the forward once, replay per frame with ONE interop crossing. Measured on real
hardware (`DA3_WebGPU_PlanReplay_VsOrtWeb`): direct 18.9s -> **replay 99.5ms/frame, 190x, bit-exact
(maxAbsDiff=0)** vs ORT-Web's 73ms warm. Also: `MatMulKernel.MatMulTiled16` direct entry + tiled16 bench
column (the register-pressure control), adapter probe as a permanent harness guard.


### Fix: kernel params-buffer reuse/inline-dispose hazards on async backends - 4 kernels converted to fresh-buffer + deferred disposal (Seven)

The WebGPU range-deviation hunt's isolation tests caught a REAL defect family: SliceKernel reused ONE
params buffer (CopyFromCPU overwrite per call, inline Dispose on rank growth) - on the async backends a
pending dispatch in an un-submitted WebGPU encoder / on the Wasm worker pool still references that buffer,
so the overwrite hands it the NEXT call's params and the growth-dispose frees memory under it (Wasm fault:
"RangeError: offset is out of bounds", reproduced by the new `SliceKernel_MixedRankHistory_MatchesCPU`).
SliceKernel, Conv1DKernel, ColorConversionKernel (flip + colormap), and
MissingElementWiseKernels.EnsureParamsBuf (DepthToSpace/Expand) all converted to the proven GatherKernel
pattern: FRESH params buffer per call, previous retired for deferred disposal at Dispose(). Four new
regression tests (`MLTestBase.SliceKernelRaceTests.cs`: single / back-to-back-no-sync / offset-subviews /
mixed-rank-history at production rotate-half geometry) - 26/26 on all 6 backends.

### Diagnosed: the WebGPU depth range deviation (0.1616 vs 0.1365) = executor Slice runtime-SHAPE cascade failure (Seven -> Tuvok)

Root cause CONFIRMED by per-node first-divergence capture + resolved-params probes: at the graph's first
rope rotate-half Slices (blocks.4), the executor's runtime output-shape override fails on WebGPU (its
starts/ends come from runtimeConstants, filled by SYNC readback on desktop but not yet present on the
async path) -> falls back to unreliable compile-time OutputShapes -> the Slices execute with
outShape=[1,6,1370,1] instead of [...,16], reading input[16+32*i]; downstream broadcasting recovers the
shape so the graph completes with plausible-but-wrong depth. Params resolution itself is fine (path-1
compiler-resolved, byte-identical across backends) - the fix belongs in the executor's shape override
(consult the compiler-resolved _resolved_* attrs). Diagnostics added: `SliceOperator.CaptureResolvedParams`
(gated static) + value-pattern/input-forensic reporting in the DA3 divergence tests. Full handoff:
DevComms `seven-ROOT-CAUSE-webgpu-range-executor-slice-shape-cascade-outshape1-2026-07-02`.

### Measured: 128-bit vec4 GEMM loads are NOT a lever - vec4 arc closed with data on all 3 GPU backends (Seven)

Consumed SpawnDev.ILGPU 4.17.1-local.1 (the WGSL AsAligned16 -> `array<vec4<f32>>` trigger + the AsAligned
lowering fix that never shipped in 4.17.0). New `Kernels/Vec4LoadMatMul` + `GemmVec4Tests` = a THREE-way
GEMM A/B at real DAv3 shapes: production scalar-float vs F4-struct-load (packing-only control) vs
F4+AsAligned16 (ld.v4.b32 on PTX, single vec4<f32> load on WGSL). Full CPU-reference correctness + GPU-side
full-output comparison vs RegisterBlockedMatMul, green on CUDA/OpenCL/WebGPU/Wasm (WebGL = tracked GLSL
struct-load bug; CPU = 64-thread group cap). RESULT (4070): the access LAYOUT (one contiguous 16-byte
element per thread instead of 4 strided floats) is worth 1.05-1.24x on CUDA/OpenCL and 1.12-1.14x on
WebGPU; the 128-bit load itself adds ZERO beyond that on every backend - drivers coalesce the contiguous
struct loads already. No production routing (decision + revisit conditions: `Plans/vec4-gemm-webgpu-integration.md`);
the tests stay as end-to-end consumer regression coverage of the ILGPU trigger. Bench numbers persist per
run in the PMT results JSON via test-returned resultText.

### Measured: DAv3-5D WebGPU cold run - the 121s "shader-JIT wall" is dead; executor overhead is the whole fight (Seven)

New `DA3_WebGPU_ColdRun_JitAndRange` (WebGPU-only, HeavyModel): create 4.2s, COLD 84.2s, WARM 69.9s at
2524 nodes with the runtime shape interpreter's readback-skip active. Cold-warm delta = ~14s over 98 unique
WGSL modules (~146ms/module, no specialization explosion) - shader-JIT is a small one-time cost, not a
target. Warm decomposition: 125 surviving shape readbacks x ~345ms = ~43s (62%) + ~10.6ms/node dispatch
orchestration = the ORT-Web-73ms gap lives entirely in the executor lane (readbacks, dispatch-elide,
bind-group caching), not in kernels. OPEN (tracked): WebGPU depth range = 0.161563 deterministic vs desktop
bit-exact 0.1365; the interpreter-off isolation test (`DA3_WebGPU_InterpOff_RangeIsolation`) is in this drop
but currently blocked by in-flight GraphExecutor diagnostics.

### Perf: Conv2D implicit-GEMM tiled kernel - 3-9x on CUDA, the last by-design-naive FLOP carrier (Seven)

The naive one-thread-per-output Conv2D kernel has zero data reuse (every MAC = two global loads; a 3x3 conv
re-reads each input pixel 9x from DRAM) - measured 420-960 GFLOPS on the 4070 while the register-blocked
GEMM does 4.3-5.7 TFLOPS at the same scale. Convolution IS a GEMM
(`C[outC, oH*oW] = W[outC, inC*kH*kW] x im2col(input)`), and the NCHW weight is already row-major
`[outC, K]`, so `Conv2DImplicitGemmImpl` reuses the exact RegisterBlockedMatMul structure (64x64 output
tile, 4x4 register block, 256-thread groups) with the B-tile stage doing the im2col ADDRESSING on the fly -
no materialized im2col buffer. Batch via `Grid.IdxY` (DAv3 multi-view preserved). Measured on the 4070 at
DAv3/DPT shapes: patch-embed 14x14 1.30 -> 0.147 ms (8.9x), DPT 3x3 refines ~3.9 -> 0.72-0.87 ms (~4.7x),
518² head 3x3 5.17 -> 1.68 ms (3.1x), 1x1 projections 4.7x; ~2-5 TFLOPS = GEMM-class. Routing: backends
with a 256-thread group + shared memory and at least one full tile of work; WebGL/CPU and tiny convs keep
the naive kernel. Low-precision-weight conv variants stay on their existing path (tiled low-p follows).

Context (measured this session, correcting the strategy post's "everything ~100 GFLOPS" claim - that was
WebGPU-era data): CUDA kernels were NOT the DAv3 wall - all matmuls/linears ~20 ms + convs ~60-100 ms of
the ~1350 ms clean total; the residual is per-node orchestration (GraphExecutor lane). This change plus the
attention fusion closes the kernel side; conv also matters strategically for SD-Turbo's conv-heavy UNet and
much more on browser backends where the naive kernel was proportionally slower.

### Perf: attention fusion now fires on DAv3 - pre-scaled-QK MatMul-form support (Seven)

DAv3's actual export is MatMul-form attention with DINOv2's split scale: Q and K are each pre-multiplied by
the SAME runtime-computed scalar (`s = sqrt(1/sqrt(head_dim))` via a `Shape -> Slice(-1,MAX) -> Cast -> Sqrt
-> Div -> Sqrt` chain), the K side as `Mul(Transpose(K), s)`, and NO scale node between the QK^T MatMul and
the Softmax. `GraphOptimizer.FuseAttention` required the MatMul's K input to be produced directly by a
Transpose, so the K-side pre-scale Mul blocked the fusion - all 24 attention MatMuls ran raw (the dominant
per-op cost after the readback work). Now:
- The K side accepts `Mul(Transpose(k), s)` when `s` is PROVABLY a 1-element tensor (new sound
  `IsProvablyScalar` walk: scalar constants/initializers, scalar-preserving unary/binary ops,
  `Gather(Shape(x), scalar-idx)`, and `Slice(Shape(x), starts, ends)` with resolvable indices producing
  exactly one element - rank-independent for `[-1, MAX)`). A scalar multiply commutes with the transpose, so
  the Mul is retargeted to the UN-transposed k and the Transpose is dropped:
  `FusedAttention(q·s, k·s, V)` computes `(q·s)·(k·s)^T = s²·qk^T` - exactly the unfused graph's math.
- The fused node then carries `scale=1.0` explicitly (the graph's scaling lives in the two Muls; the
  kernel's 1/sqrt(hd) default would double-scale). An explicit between-scale node, when also present,
  still folds in as before.
- The rewrite (Mul retarget + Transpose removal) is DEFERRED until the whole pattern matches - a
  half-matched pattern never edits the graph.
Verified offline against the real DAv3-Small graph: 12/12 attention blocks fuse (Softmax remaining 0,
2560 -> 2524 nodes). With `FusedAttentionKernel.Forward`'s delegation (below), DAv3's fused attention now
runs the warp-register kernel on CUDA + WebGPU. New regression test
`AttentionFusion_PreScaledQK_DAv3Form_FusedMatchesCpu_AllBackends` reproduces the exact block structure
including the runtime scale chain.

### Perf: batched register-blocked GEMM, Einsum-form attention fusion, register attention for fused graphs, fused LayerNorm (Seven)

Four kernel-lane changes from the DAv3 beat-ORT campaign (all bit-consistent on the DAv3 5-D rig, CUDA/OpenCL/CPU
`range=0.1365`; scoped PMT 392/392 across all 6 backends; clean-timing OpenCL DAv3 3642 -> ~1350 ms):
- **`RegisterBlockedMatMul.BatchedMatMul` (new):** batched GEMM finally has a register-blocked route (4x4 per
  thread, 64x64 tiles, batch = `Grid.IdxY`). `MatMulKernel.BatchedMatMul` routes to it at M,N >= 64 - before
  this every batched matmul (attention scores, probs@V, einsum contractions) ran the 16x16
  one-result-per-thread tiled kernel. Gated by `MatMul_BatchedAttentionScores` (the exact DAv3 attention
  shape, batch=6 M=N=1370 K=64, vs CPU reference).
- **`EinsumOperator` batched fast path = ONE dispatch:** the per-batch C# loop (batchSize sequential MatMul
  dispatches) is gone.
- **`GraphOptimizer.FuseAttentionEinsum` (pass 3c):** Einsum-form decomposed attention
  (`Einsum(Q,K) -> [Mul/Div scale] -> Softmax -> [Cast] -> Einsum(probs,V)`, the DINOv2-lineage export form)
  fuses into the same `FusedAttention` node as the MatMul form. Natural-K (`bhid,bhjd->bhij`) and
  pre-transposed-K (`bhid,bhdj->bhij`, walks back through the Transpose) both handled. Scale semantics: a
  found Mul/Div scalar folds into the `scale` attr; NO scale node emits `scale=1.0` explicitly (the unfused
  graph applied none - the kernel's 1/sqrt(hd) default would diverge). New CPU-referenced tests:
  `MLTestBase.AttentionFusionEinsumTests`.
- **`FusedAttentionKernel.Forward` now delegates to `ForwardStrided<float>`** (`kvRowStride = seqKV*headDim`,
  the documented byte-identical combination). Forward's old inline dispatch predated register attention and
  never reached it, so FUSED (non-KV-cache) attention - vision transformers, SD - ran the shared-slice
  per-query kernel even on CUDA/WebGPU where the measured-2.7x-prefill warp-register kernel was available.
  One dispatch chain, no drift; removed the dead inline-dispatch delegate fields.
- **`LayerNormKernel` fused single-pass path:** one GROUP per row with two barrier-separated cooperative
  reductions (strided f64 partials; mean subtracted before squaring = Welford-grade stability), mirroring the
  proven `RMSNormFusedImpl` gating. The old Pass 1 ran ONE thread per row doing a serial C-length f64 Welford
  - 1/8 occupancy at DAv3 shape and Dekker-emulated f64 on WebGPU. Two-pass path kept for WebGL.
- ORT-Web comparison harness: DAv3-Small case in `ort-comparison.html` (per-EP buttons, load/cold/warm
  timing) + `dav3-ort-baseline.mjs` driver (serves wwwroot with COOP/COEP, drives both EPs). Baseline on the
  4070: WebGPU load 5.1 s / cold 3.45 s / warm ~73-78 ms; Wasm(4 threads) warm ~9.3 s.

### Fix: WebGPU "buffer used in submit while destroyed" in the RoPE dynamic-shape subgraph (GatherKernel)

DAv3 (Depth Anything V3) at its native 5-D input `[1,1,3,518,518]`, run through `DepthEstimationPipeline` on
WebGPU, aborted at `/backbone/blocks.4/attn/rope/Add` (node ~177) with `[Buffer] used in submit while destroyed`
- SpawnScene's depth blocker. Root cause: `GatherKernel.GatherGenericFloat` disposed the PREVIOUS call's params
buffer INLINE on every call. The RoPE dynamic-shape subgraph issues several `GatherGenericFloat` calls that batch
into one un-submitted WebGPU command encoder, so the 2nd Gather destroyed the 1st's params buffer while the 1st's
dispatch was still pending - the next `Queue.Submit` then referenced a destroyed buffer. Synchronous CUDA/OpenCL
submit eagerly so never hit it. Fix: defer the old params buffer to `Dispose()` via an `_oldGenericParams` list
(the same pattern `ElementWiseKernels.BroadcastBinaryOpND` already uses) - each call still gets a fresh buffer (no
write-after-read), the old ones are freed at a safe point. New regression guard `DA3Small_Pipeline_5D_WebGPU_ProducesDepth`
exercises the real 5-D pipeline path. (Depth output verified byte-stable: CUDA/OpenCL `range=0.1365, NaN=0`.)

### Fix: DAv3 5-D multi-view (num_images>1) - three "computes only batch 0" batch>1 bugs

Depth Anything V3 multi-view (`pixel_values=[1,N,3,H,W]`, N>1) was catastrophically wrong (depth mad 1.39 vs
ORT, confidence exploded to ~1e10) while single-view (N=1) was bit-exact. Root: the engine was built and
tested at batch=1, so three ops each computed only batch index 0 and left every later view stale. All fixed;
**batch=1 is byte-identical** (b==0, inBatchBase==0) so single-inference LLM/BERT/SD is unaffected.
- **Conv2D** (`Conv2DKernel.Conv2DImpl`/`ForwardPadded`): the kernel decoded idx -> (oc,oy,ox) with no batch
  offset on the input read. Now decodes the batch index from idx and strides the input; the ConvOperator passes
  `batchN` from the input's leading dim.
- **MatMul** (`MatMulOperator`): batched activations `[N,S,K]` @ a shared 2-D weight `[K,Nn]` (a Linear) were
  routed to `BatchedMatMul`, which strides the 2-D weight by batch and reads off its end -> the qkv Linear blew
  to ~1e19. Now: when B is 2-D, flatten all rows into M (`[N*S,K] @ [K,Nn]`); only a genuinely batched B
  (e.g. attention QK^T) uses BatchedMatMul.
- **ConvTranspose** (`ConvTranspose2DKernel`): same class as Conv2D (the DPT head `resize_layers` ConvTranspose
  left view 1 stale, which corrupted the entire head downstream).

Result: multi-view (N=2) is now bit-exact vs ORT (depth + confidence mad 0.000000, max 2e-6); single-view still
bit-exact. Regression guards added: `Conv2D_Batch2_ComputesBothViews`, `ConvTranspose2D_Batch2_ComputesBothViews`,
`AllOps_MatMul_BatchedActivationsSharedWeight`. Known remaining batch-1-only paths (no current model exercises
them at batch>1): depthwise / grouped / NHWC / native-low-p / precision-aware-fp16 Conv variants.

### Fix: gemma3 forward pass - weightless V-norm + sliding-window rope base (fixes factually-wrong output)

gemma3:270m produced fluent but factually WRONG output ("The capital of France is a place where you can
experience the thrill"; "Paris" ranked 143rd) while Ollama's identical blob answered "Paris". Two distinct
bugs, both in `GGUFGraphBuilder`:
- **Weightless V-norm wrongly applied to gemma3 (dominant).** The weightless RMS-norm of V before attention
  is a gemma4 / gemma3n behavior only; standard Gemma 3 leaves V raw (llama.cpp `src/models/gemma3.cpp` norms
  Q and K, never V). It was gated on "has QK-norm", which is true for gemma3 too, so we normed V and corrupted
  the attended values - factual retrieval collapsed while the FFN kept fluency. New `UsesWeightlessVNorm(arch)`
  gates it to gemma4*/gemma3n. Result: "Paris" rank 143 -> rank 0.
- **Sliding-window rope base.** gemma3 interleaves a 5:1 local:global layer pattern (period 6); local layers
  use rope base 10000, globals 1e6. gemma3:270m's GGUF carries neither `sliding_window_pattern` nor
  `rope.freq_base_swa`, so llama.cpp's hardcoded gemma3 defaults apply. `GetLayerAttnConfig` now defaults
  gemma2/gemma3 to period 6 + local base 10000 when the GGUF omits them.

Verified on CUDA (gemma3 -> "Paris", matches Ollama greedy; "2+2" -> "2 + 2 = 4"); qwen2/smollm2/gemma4
unregressed. Guard: `GemmaArchWiringTests`. PMT Gemma 98/0 + GGUF green, all 6 backends.

### Fix: GGUF RoPE pairing style is per-architecture (NORM vs NeoX) - fixes llama-arch degenerate output

GGUF inference applied **NeoX / split-half** RoPE to every architecture. That is correct for qwen2/gemma
(true NeoX), but the **LLaMA lineage** (llama, mistral, minicpm, granite, ...) uses **NORM /
consecutive-pair** RoPE and ships q/k weights *permuted at conversion* to match it (mirrors llama.cpp
`llama_model_rope_type`). Applying NeoX to a NORM-permuted model scrambles every q/k channel, producing
degenerate logits - smollm2:360m looped (`"TheThe answerThe answer the following question:"`) from correct
tokens while Ollama's identical blob was coherent. Fix: `GGUFGraphBuilder.UsesNormRope(arch)` selects the
NORM lineage and stamps `interleaved` on the RoPE node (the kernel and operator already supported both
styles; the builder simply never set it). One place covers prefill, decode, and the KV-cache on all
backends. Verified on CUDA (smollm2 → "The capital of France is Paris."; qwen2 unregressed); regression
guard `RopeStyleWiringTests` builds the real graph per arch and asserts each RoPE node's pairing flag. PMT
GGUF + Rope suites green on all 6 backends.

## 4.0.0-preview.5 (2026-06-23) - GGUF LLM inference + register attention (CUDA + WebGPU) + ~4.6x decode

Headline: native GPU **GGUF LLM inference** (Qwen / Gemma / Llama, KV-cache decode) with an **Ollama-compatible
server example** (Example 06: OpenAI, Ollama, and Anthropic-Messages APIs; works with the Claude CLI). Decode on
qwen2.5-coder:7b Q4_K_M (RTX 4070) went ~11 -> ~51 tok/s this cycle - **1.75x of Ollama, down from a 7x gap** - via
dp4a int8 GEMV, warp-cooperative **register / flash-class per-query attention** (now on CUDA *and* WebGPU subgroups,
plus a universal barrier-free per-query path for the other backends), kernel fusions (SwiGLU, AddRMSNorm), transpose
elimination (zero transpose nodes), zero-copy reshape views, and a queue-ordered sync-CopyFrom KV-write that drops
per-token browser GPU round-trips. Consumes **ILGPU 4.16.2** (CUDA graph capture API, WGSL subgroup shuffle +
`subgroup_uniformity` directive, Wasm large-local-array fix). The remaining Ollama gap is prefill (tensor-core / WMMA,
not yet emitted). Itemized below.

### KV-cache write: queue-ordered sync CopyFrom on ALL backends (drop the per-copy await round-trip)

The decode KV-cache write (`CaptureSafeCopy`) used a stream-ordered sync `CopyFrom` on CUDA (no host sync — "the
whole forward is one stream, so ordering is free") but fell back to an awaited `CopyFromAsync` on every other backend
(WebGPU, Wasm, WebGL). That fallback existed for a Wasm worker-pool ordering race — **which has since been fixed in
SpawnDev.ILGPU**: `CopyFrom` is now reliably ordered on all 6 backends (native `CopyBufferToBuffer` on WebGPU, TF on
WebGL, serialized work-stream on Wasm), so the consumer kernel always reads after the copy with no race. The awaited
`CopyFromAsync` was therefore a pure GPU round-trip per K/V per layer = 2×nLayers per decode token (56/token on a
28-layer 7B) on WebGPU AND Wasm — the two backends where a round-trip hurts most. All non-CUDA backends now take the
sync-enqueue path; CUDA keeps the explicit `DefaultStream` form for graph-capture safety. Correctness gate: PMT
GGUFDecodeKVCache (incremental == full-recompute, byte-identical) green all 6 backends. Pure round-trip removal — the
KV math is unchanged.

### Register attention on WebGPU (subgroups) + consume ILGPU 4.16.2

**Register per-query attention now runs on WebGPU**, extending the flash-class register accumulator (CUDA default-on,
2.7× prefill / ~20% decode) to the browser — where it wins most (no workgroup memory at all; the per-thread 16-wide
register tile + `Warp.ShuffleXor` butterfly map to WGSL `subgroupShuffleXor`). Two things made it valid on Dawn:
(1) ILGPU **4.16.2** (consumed; was 4.16.0) — `Warp.Shuffle{Xor,Up,Down}` now lower to the matching WGSL builtin
(the ShuffleKind was being dropped → plain `subgroupShuffle`), plus a module-level `diagnostic(off,
subgroup_uniformity)` directive so a subgroup op inside a storage-buffer-bound loop compiles. (2) A **uniform-shuffle
correctness fix** in `FusedAttentionPerQueryRegisterImpl`: dropped the divergent `if (query >= BH*SQ) return;` before
the shuffle — now ALL lanes run `Warp.ShuffleXor` uniformly (out-of-range lanes read a clamped in-bounds query and
skip only the final store), so the shuffle is genuinely uniform (correct on WebGPU AND CUDA, not just
whole-group-active-by-luck). Dispatch gate extended to WebGPU when `WarpSize==32` (adapter exposes subgroups);
CPU/Wasm/WebGL/OpenCL keep the shared-slice. PMT GGUFDecodeKVCache **8/8 all 6 backends incl WebGPU** (register ==
full-recompute, byte-identical on real Dawn); CUDA still byte-identical.

### Fused AddRMSNorm (residual add + RMSNorm in one cooperative pass, universal)

Fused the per-layer residual `Add` into the following `RMSNorm`. New `AddRMSNorm` op/kernel: one cooperative pass
reads `x = a + b` per element, writes BOTH the residual stream (`a+b`, for the next residual add) AND the
normalized output (`rmsnorm(a+b)·weight`) — replacing a separate Add kernel + RMSNorm (2 graph nodes → 1). Same
single-pass f64-partial reduction shape as the existing fused RMSNorm, so it matches the Add→RMSNorm chain to the
RMSNorm tolerance; byte-identical decode. Wired for the within-layer ffn-norm on plain weighted RMSNorm archs
(qwen/llama; gemma's (1+w) fold is baked into the GGUF weights so RMSNorm is uniform) — LayerNorm / no-weight archs
keep the separate Add+norm. **EXCLUDES the gemma 2/3/4 norm-sandwich** (a layer with `post_attention_norm`): there
the residual Add consumes the post-attn-norm output, so fusing it would hide the sandwich wiring (the residual add
must stay a visible `Add` fed by `post_attention_norm`) — those layers stay unfused. WebGL (no workgroup shared mem)
falls back to `ElementWise.Add` + the two-pass norm (same op, two kernels). Decode node count 535→507 (−28/step; Add
56→28, RMSNorm 57→29). PMT full sweep green all 6 backends incl `Gemma4_GraphBuilder_PostNormSandwich`. (The 28
residual-stream "boundary" Adds before each attn-norm are a pattern-2 follow-up.)

### Zero-copy Reshape views at decode (drop 112 CopyFrom dispatches/step, universal)

The executor's zero-copy metadata-only-op path (Reshape/Squeeze/Unsqueeze/Flatten hand off a single-consumer
pooled buffer as a view instead of `CopyFrom`-ing it) was gated to `ElementCount >= 4096` ("large reshapes, the
memory win"). At **decode** (seq=1) the per-layer q/k/v + head-merge reshapes are only ~512-3584 elems → below the
gate → they fell through to a real device→device copy (**112 CopyFrom/step on qwen, 8.1% of decode**). Lowered the
threshold to **256**: those reshapes now become zero-copy views (no copy, no dispatch) — the single-consumer
ref-count gate keeps it safe. `Reshape` disappears entirely from the decode op profile; byte-identical. Universal,
biggest on WebGPU (each was a CopyBufferToBuffer dispatch). PMT GGUFDecodeKVCache green all 6 backends.

### Register attention DEFAULT-ON for CUDA (2.7× prefill attention, ~20% decode)

Promoted the warp-cooperative register per-query attention from opt-in to **default-on** (the dispatch still gates
it to CUDA: warp==32 + Warp.Shuffle, D%16==0; all other backends keep the shared-slice; `GGUF_ATTN_REG=0` forces
OFF for A/B). **RTX 4070: 2.7× prefill attention (shared-slice 1226.8 → register 452.9 ms on a 324-token prompt)**
+ ~20% decode — the prefill win directly attacks the Ollama prefill-throughput gap (prefill is ~94% attention).
Argmax-identical. Oracle `Attn_RegisterPerQuery_MatchesCpu_Cuda` now covers hd 64/128/256 (4/8/16 lanes/query);
GGUFDecodeKVCache green with register as the CUDA default.

### Warp-cooperative REGISTER per-query attention (opt-in, CUDA-first) — ~20% faster decode

**The flash-class register accumulator (Geordi's D-tiling recipe).** The barrier-free per-query attention holds its
D outputs in a per-thread workgroup-shared slice (the shared-RMW is its ~5.3× ceiling). New opt-in
(`GGUF_ATTN_REG=1`) `FusedAttentionPerQueryRegisterImpl`: **T = D/16 lanes cooperate on one query**, each owning
dims `[t·16,(t+1)·16)` and holding its 16 online-softmax accumulators in **REGISTERS** (the const-16 array
scalar-replaces — no shared slice, no barrier). Per kv: each lane computes its partial Q·K dot, the T lanes
butterfly-reduce it via `Warp.ShuffleXor` (aligned power-of-2 group, every lane gets the full dot), then each lane
runs the same scalar online-softmax and updates its 16 register accs. **RTX 4070 clean A/B (qwen2.5-coder:7b
decode): shared-slice 23.9 → register 19.2 ms/tok (~20%)**, argmax-identical (the per-tile+shuffle dot sum reorders
vs sequential, within GEMV float-reduction tolerance). CUDA-first (warp==32 + `Warp.Shuffle`, D%16==0); other
backends keep the shared-slice per-query (default OFF → opt-in, master behavior unchanged). PMT GGUFDecodeKVCache
green (CUDA register-incremental == full-recompute) + default sweep unaffected. WebGPU-subgroup extension is the
next step (the bigger win there — avoids workgroup memory).

### Fused SwiGLU (SiLU-gated MLP activation, dispatch reduction)

**Fused the SiLU MLP activation into ONE kernel.** The SwiGLU path emitted three elementwise nodes per layer —
`Sigmoid(gate)` → `Mul(gate, sig)` → `Mul(silu, up)`. New `SwiGLU` op/kernel computes `(gate · sigmoid(gate)) · up`
in a single pass (3 dispatches/layer → 1 = **56 fewer dispatches/decode-step**; biggest on WebGPU dispatch overhead,
continuing the transpose-fusion theme). **Bit-identical**: the sigmoid clamps (>80→1, <-80→0) match `SigmoidInPlaceImpl`
and the multiply order `(gate·sig)·up` matches the two-Mul chain. Elementwise gather (no scatter) → WebGL-safe.
Decode node count 591→535 (qwen2.5-coder:7b; Sigmoid + 56 Mul folded out). PMT GREEN all 6 backends:
GGUFDecodeKVCache 8/8 (the tiny-llama test model uses SiLU → exercises SwiGLU, incremental == full-recompute
byte-identical) + Attn 92/92; qwen 16-tok decode byte-identical. (gemma's GeGLU `Gelu+Mul` left as a follow-up.)

### Consume ILGPU 4.16.0 (CUDA graph API + Wasm large-local-array fix)

Bumped `SpawnDev.ILGPU` 4.15.1 → **4.16.0** (Geordi's stable, forks 2.1.0): rolls up the CUDA graph capture API
(`CudaStream.BeginCapture/EndCapture`, `CudaGraph`/`CudaGraphExec`, `Accelerator.WithDefaultStream`) + the device-
local dynamically-indexed `new T[N>32]` codegen fix now correct on **all 6 backends incl Wasm**. Verified on the ML
lane: GGUFDecodeKVCache 8/8 + qwen decode byte-identical, no regression. Unblocks Example 04's `GGUF_DECODE_GRAPH_PROBE`
(decode CUDA-graph capture/replay probe) + the new `GGUF_PTX_PROBE` (dumps the dp4a GEMV PTX to verify `ld.v4.b32`).

### Transpose-fusion step 3: seq-major KV-cache — ZERO transpose nodes (universal)

**Eliminated the last 56 K/V PRE-attention transposes — the decode graph now has ZERO `Transpose` nodes (703→591,
all 112 gone across steps 1-3).** The KV-cache store flips from head-major `[kvHeads, maxSeq, hd]` to **seq-major
`[maxSeq, kvHeads, hd]`**, so the per-step K/V (already seq-major from the dropped transpose) write as ONE
contiguous copy (`WriteAsync`, was a per-head strided loop) and the live region is contiguous for the repack
(`PackedAsync`). New `seqMajorKV` mode (kernel param `p[13]`): all 7 attention variants read K/V with
`kBase = kvHead·kvHeadStride + kv·kvTokenStride` — seq-major (headStride=hd, tokenStride=kvHeads·hd) or the
existing head-major (contiguous `SKV·hd` / strided `p[10]`). K/V read is a GATHER (no WebGL scatter issue). The
WebGL+bf16 fallback packs seq-major + reads via the contiguous kernel with `seq_major_kv`. Set by the GGUF builder
(`EmitAttnHead(skipTranspose)` for K+V + `seq_major_kv` attr); decode preserves the attr (GraphExecutor copies
node.Attributes). PMT GREEN all 6 backends (GGUFDecodeKVCache 8/8 incl WebGL+bf16 fallback + Attn 92/92);
qwen2.5-coder:7b 16-tok decode byte-identical. Transpose-fusion lever COMPLETE.

### Transpose-fusion step 2: drop the Q pre-attention transpose (universal)

**Eliminated the per-layer Q PRE-attention `Transpose[0,2,1,3]` (another 28 dispatches+copies/decode-step).**
Symmetric to step 1 but on the READ side: new `seqMajorQ` mode (kernel param `p[12]`) makes FusedAttention read Q
with the seq-major base `(sq*BH+bh)*D`, so `EmitAttnHead(..., skipTranspose: true)` for Q drops its transpose and
feeds the post-RoPE `[1,seq,heads,hd]` tensor straight in. Independent of `seqMajorOut` (step 1, the output side) —
both set together by the GGUF builder now (`seq_major_q` attr). K/V keep their transposes until step 3 (the
KV-cache store must go seq-major). **Decode Transpose nodes 84→56 (56 = the remaining K+V pre-transposes).** PMT
GREEN all 6 backends (GGUFDecodeKVCache 8/8 + Attn 92/92); qwen2.5-coder:7b 16-tok decode byte-identical.

### Transpose-fusion step 1: drop the post-attention transpose (universal)

**Eliminated the per-layer post-attention `Transpose[0,2,1,3]` (28 dispatches+copies/decode-step, universal).**
FusedAttention output was heads-major `[1,heads,seq,hd]`, then a real Transpose kernel → `[1,seq,heads,hd]`, then
the head-merge Reshape. New `seqMajorOut` mode (kernel param `p[11]`) makes FusedAttention write its output
**directly seq-major**, so the graph drops the transpose and the Reshape (native CopyFrom) consumes the attention
output as-is. Universal + WebGL-safe: the 5 group-cooperative variants scatter to the seq-major base; the 2
per-element variants (the WebGL path) instead enumerate `idx` in seq-major order so each thread writes its OWN
slot (one store per thread — no scatter, which WebGL's Transform-Feedback forbids). Set unconditionally by the
GGUF graph builder (`seq_major_out` attr); the SafeTensors builder + the softmax→matmul fusion keep the default
heads-major path. **Decode node count 703→675 (−28/step); biggest on WebGPU (each transpose = a dispatch + a full
copy).** PMT GREEN all 6 backends: GGUFDecodeKVCache 8/8 (incremental == full-recompute, byte-identical incl
WebGL) + Attn oracle 92/92 (default path intact); real qwen2.5-coder:7b 16-tok decode byte-identical. Next
(steps 2-3): the 84 remaining PRE-attention Q/K/V transposes (needs the KV-cache store to go seq-major too).

### Universal barrier-free per-query attention (WebGPU prefill win) + Reshape native-copy

**Barrier-free per-query fused attention — the universal (esp. WebGPU) prefill win.** Prefill is ~94% attention
at long context, and WebGPU/WebGL were stuck on the per-element kernel (O(n²·D²), D× redundant Q·K dot) because
the grouped kernel's workgroup reduction is ~75× slow on Tint/Dawn. New `FusedAttentionPerQueryStridedImpl`: one
thread per (bh, sq), computes the dot ONCE and accumulates all D outputs in its own slice of workgroup shared
memory — no `Group.Barrier`, no reduction — byte-identical to the per-element kernel. It's the non-grouped path on
every backend **except WebGL** (no workgroup shared memory → keeps per-element; also a multi-store-per-thread that
WebGL's Transform-Feedback can't do). Wired for both the GGUF-strided (`ForwardStrided`) and contiguous (`Forward`)
paths. **Measured (8h×256×256×128): WebGPU per-element 18258ms → per-query 265ms = ~69×; Wasm 16×; CUDA prefill
~5×.** PMT GREEN all 6 backends (GGUFDecodeKVCache 8/8, attention oracle 92/92); decode byte-identical. (A
dynamic-D *local-memory* accumulator was tried on Geordi's LowerArrays fix and measured ≈ tie / CUDA-slower vs the
shared slice — reverted; true registers need D-specialized unrolling, a follow-up.)

**Reshape → native `CopyFrom` instead of a `Scale(×1)` kernel dispatch.** `ReshapeOperator` only relabels
dimensions (identical element order) but was copying via a `Scale(×1)` shader dispatch (~112×/decode step on
qwen). Now a native device→device `CopyFrom` (WebGPU `CopyBufferToBuffer`; queue-ordered on every backend) — drops
~112 shader dispatches/step to native copies, biggest on WebGPU's dispatch overhead. Verified CUDA byte-identical
+ GGUFDecodeKVCache 8/8 + Attn 92/92.

**128-bit vectorized weight load in the dp4a Q4_K decode GEMV.** `GemvDp4aQ4_KImpl` read its 8 nibble-words per
t-unit as 8 scalar `ld.b32`. Each t-unit's 8 words are two 16-byte-aligned chunks, so reading each as a `W16`
struct-of-4-ints via `BaseView.Cast<W16>().AsAligned16()` makes ILGPU's PTX backend emit a single 128-bit
`ld.v4.b32` (= SASS `LDG.E.128`) per 4 words — weight load-issue cut 4× (the llama.cpp MMVQ bandwidth level).
PTX-verified (2× `ld.v4.b32`, was 8 scalar); bit-identical (PMT `Gemv_M1_Dp4a` 14/14, qwen 12-tok byte-identical).
RTX 4070 kernel A/B: Q4_K dp4a GEMV 266→292 GB/s (+10-16% MLP shapes), 148→238 (+61% attn-proj), ~53→58% of peak.
(E2E decode unchanged within noise — the GEMV is a fraction of the 703-node step; next GEMV lever = 4-warps-per-row.)

(Also this cycle, shipped earlier: RMSNorm cooperative parallel reduction; dp4a int8-activation GEMV 4-warps/block
(CUDA); Example 06 `/api/show` + `run-pi.bat`.)

### Example 06 server: interactive bounds (fast model + capped context/output)

**Made the Claude-CLI experience usable on a slow local engine.** `run-claude-cli.bat` now defaults to
**qwen2.5-coder:7b** (~90ms/tok — the fast interactive pick; gemma4:12b is 12B/slower and has a separate
large-context decode bug) and sets bounds so no request runs for minutes: `OLLAMA_NUM_CTX=8192` (agentic
clients send ~38K-token prompts; we cap/tail-truncate — smaller = faster first token) and
`OLLAMA_MAX_OUTPUT=1024` (hard output cap, env-configurable; at ~90ms/tok ≈ 90s worst case vs ~6 min at the old
4096). Verified: qwen2.5-coder:7b on a moderate Claude-Code-shaped request (system-array + tools, non-stream)
answers "The capital of France is Paris." in **2.4s with `stop_reason=end_turn`** (stops cleanly). The huge-
prompt edge (38K truncated to ctx can break the chat structure → the model rambles to the cap) is now bounded
by the output cap rather than open-ended; cleaner truncation + engine speed are the follow-ups.

### Example 06 server: stream the tools path (fix Claude CLI "API error")

**Claude CLI got "API error" on the first real message and the GPU stayed pegged for minutes.** Claude CLI
ALWAYS sends its toolset, and the `/v1/messages` handler routed any request with `tools` to **buffered (fully
non-streaming) generation** — so even with `stream:true`, nothing was sent until the entire response finished
(a large agentic prompt's prefill + up to MaxOutputTokens is minutes). Claude got zero data → time-to-first-
token timeout → "API error" → retries piled up; and the buffered generation ran to completion after Claude
disconnected (pegged VRAM). Fix: the tools+stream path now **streams text deltas live** (SSE opened immediately,
`message_start` first → time-to-first-byte ~14ms verified), holding back only a partial `<tool_call>` suffix so
tool markup never leaks, and emits `tool_use` blocks at the end. Streaming also makes a client disconnect ABORT
generation (the SSE write throws → generation stops → GPU frees). Non-stream tool requests still buffer + format
tool blocks (unchanged). Verified: tools+stream returns a correct streamed answer with immediate TTFB.

Also: a **client-canceled request is now handled gracefully** — when the client closes the connection (an
agentic frontend dropping a queued auxiliary request, or timing out waiting for the single generation gate
behind a longer request), the server logs a benign "client-canceled" instead of a scary `EXCEPTION`/500. The
generation gate is released cleanly. (True *concurrent* serving of multiple requests on one GPU needs
continuous batching — tracked as the v-next concurrency feature; until then requests serialize on the gate and
the practical lever is generation speed.)

### Example 06 server: fix Claude CLI cold-start (pre-load the model)

**Claude CLI failed to connect to the Ollama-replacement server** — the request log showed every startup
`/v1/messages` throwing `OperationCanceledException` in `ModelRegistry.AcquireAsync`. Root cause (diagnosed
from the captured traffic + reproduction, not guessed): the first request lazily loads the multi-GB model
**while holding the single generation gate**, and Claude CLI fires several requests at once on startup (title +
main + warmup) — they pile up behind the load and Claude cancels them (the client abort = `RequestAborted`).
Once loaded, everything works (200s, coherent answers, concurrent requests serialize cleanly — all verified).
Fix: the server **pre-loads a model at startup** (`OLLAMA_PRELOAD=<model>`) before it begins listening, so a
client that waits for `/api/version` never races the load; `run-claude-cli.bat` sets `OLLAMA_PRELOAD` and now
polls `/api/version` for readiness instead of a fixed 10s sleep. Verified: pre-load (~4s), readiness gating,
title (`json_schema`/`output_config` shape), and a realistic main request (system-as-array, SSE) all work.

### Decode GEMV scale-cache (Q4_K M=1, the per-token path)

**The M=1 dequant GEMV (run for every decoded token) cached its Q4_K sub-block scales.** A direct comparison
against Ollama (same qwen2.5-coder-7B-Q4_K_M blob, RTX 4070) showed decode is the dominant gap (~120ms/tok vs
Ollama's ~12ms); an isolated bench localized it to the M=1 GEMV running at only ~34 GB/s (~7% of the card's
~504 GB/s). The Q8_0 GEMV (trivial dequant) ran at ~86 GB/s on the same path → the limiter was the per-element
6-bit `get_scale_min_k4` extraction (ALU), not bandwidth. Fix: the 8 sub-block `{d·sc, dmin·mn}` of a Q4_K
block are now decoded ONCE per block into shared (8 cooperating threads), and the per-element work is a nibble
fetch + one multiply + one subtract (`DecodeQ4KNibble`). **Q4_K GEMV ~34 → ~51 GB/s (~1.5×); qwen decode steps
~120 → ~90 ms.** Bit-identical (qwen decode tokens + logits unchanged; `GemvTests` oracle green on all 6
backends). The remaining gap to Ollama is a deeper GEMV-structure / occupancy issue (even trivial-dequant Q8_0
caps at ~17% bandwidth here) — tracked.

### KV-tiled grouped attention (unbounded SKV for long prompts)

**`FusedAttentionKernel` now handles unbounded SKV via a KV-tiled grouped kernel.** The single-pass grouped
kernel holds all scores in shared memory, so it was capped at SKV ≤ 4096 — beyond that (8k–16k agentic prompts)
attention fell back to the per-element kernel, which recomputes the D-length Q·K dot per output dim (D-fold
redundant) and is O(seq²): catastrophically slow at long context. The new tiled kernel processes KV one
512-block at a time (scores for just one block resident), so SKV is unbounded; each thread keeps one online
softmax shared across its ≤4 owned output dims and the per-kv recurrence runs in order — **bit-identical** to
the single-pass kernel and the per-element reference. Dispatch is gated by SKV: ≤ 4096 keeps the faster
single-pass kernel (fewer barriers, no per-kv owned-dim branch — verified no regression, ~1.22 s
FusedAttention @1081 tok), above it uses the tiled kernel. A >4096-token prefill that previously fell back now
runs the grouped path. New PMT config (SKV=5000, multi-block) in `FusedAttention_Grouped_MatchesPerElement`
asserts the tiled kernel matches per-element on all 6 backends. Opt-in via the same
`EnableGroupedAttention` / `GGUF_ATTN_GROUP` flag; non-browser-GPU (browser keeps per-element).

### Last-position-only logits (prefill LM-head as an M=1 GEMV)

**`GGUFGraphBuilder.EnableLastPositionLogits`: at prefill the LM head computes logits for ONLY the last
sequence position instead of all prompt positions.** For autoregressive generation only the last token's
logits are sampled, so computing the rest is pure waste. A `Slice` node (axis=1, start=-1) on the final hidden
state before output_norm turns the vocab projection — qwen's single biggest prefill node — from an M=seq GEMM
into an M=1 GEMV (~0 ms). Resolved against the concrete seq length each shape-recompile, so it's correct at any
prefill length and a no-op at seq=1 decode. The win **scales with prompt length** (at 16k context the logits
node would be seconds → ~0). Token-identical: both generation consumers (`GgufGenerator`, Example 04) already
read only the last position; verified bit-identical generated tokens + logits on qwen2.5-coder (KV-decode and
large-prefill) and gemma4 (incl. the final-logit soft-cap on the sliced output). New PMT test
`GGUFLastPositionLogits_MatchesFullLastPosition` (sliced last row == full-recompute last position, argmax-strict)
green on all 6 backends. Opt-in (env `GGUF_LAST_POS=1`); library-default off (the GGUF graph's all-position
logits are still available for eval/perplexity). Example 06 server opts in.

### Metadata-cached dequant GEMM + two cross-backend dequant-path fixes

**`FusedDequantMatMul` register-blocked Q4_K AND Q6_K GEMMs now cache per-column block metadata once per K-tile.**
An isolated GFLOPS benchmark (Example 04 `GGUF_GEMM_BENCH=1`, RTX 4070, qwen MLP shapes @M=1081) localized the
prefill MatMul ceiling: the f32 register-blocked GEMM hits **~5700 GFLOPS** (≈20% of the card's f32 peak — the
codegen is fine), while the Q4_K dequant version was **~2294 GFLOPS** — the dequant-on-load was 2.5× overhead.
The B-tile load re-decoded each block's fp16 d/dmin + 6-bit sub-block scale **per element**. A K-tile (16 deep,
16-aligned) is provably within one Q4_K sub-block (and one Q6_K scale group of 16), so each column's folded
products (`{d·sc, dmin·mn}` for Q4_K, `d·sc` for Q6_K) are now decoded **once per K-tile** into shared memory;
the per-element load is a quant fetch + a multiply (+ a subtract for Q4_K). **Result: Q4_K ~2294 → ~4268
GFLOPS (~1.86×); Q6_K ~4460 GFLOPS, at parity with Q4_K (the Q6_K logits projection is qwen's single biggest
prefill node).** Bit-identical (A/B logits match the prior register-blocked exactly; tokens identical to the
per-element oracle).

Two pre-existing dequant-path defects surfaced by new register-blocked + multi-row oracle tests (the old oracle
used M=2 with MultiRowGemm off, so neither opt-in prefill path was ever unit-tested) and fixed:
- **CPU backend:** the register-blocked kernel launches 256-thread groups, but ILGPU's CPU accelerator caps a
  group dimension at 64 → it threw `Invalid group dimensions`. Now gated on `MaxGroupSize.X >= 256`; CPU falls
  back to the multi-row kernel (group 64).
- **Wasm backend:** the multi-row Q4_K/Q6_K kernels' device-local `float[]` accumulator **miscompiled on the
  ILGPU Wasm backend** (correct on the other 5; wrong only on Wasm). Root-caused by Geordi to two ILGPU Wasm
  codegen bugs (no `LowerArrays` pass + a Local-alloca aliasing onto shared memory), both **fixed in
  SpawnDev.ILGPU 4.15.1**. This project now references 4.15.1 and the kernels use the clean parametric
  `new float[GemmMTile]` array again (the temporary scalar-unroll workaround is removed) — verified green on all
  6 backends (PMT 54/54).

New PMT tests `FusedDequantMatMul_{MultiRow,RegBlocked}_MatchesOracle_{Q4_K,Q6_K}` (M=40 multi-row + M=80
register-blocked, non-tile-aligned dims) — green on all 6 backends.

### Grouped-per-query fused attention (prefill attention win)

**`FusedAttentionKernel` grouped-per-query path: each Q·K score is computed ONCE instead of once per output
dim.** The per-element kernel launches one thread per `(head, query, dim)` output and recomputes the full
D-length Q·K dot for every output dim — D-fold redundant (D=128 → 128×). After the dequant-GEMM landed this
made FusedAttention ~70% of long-prompt prefill. The grouped kernel runs one thread GROUP per `(head, query)`:
phase 1 computes each score once into shared `scores[]`, phase 2 has each thread own a slice of the D output
dims and replay the IDENTICAL online-softmax recurrence per dim reading the shared score. Because the
per-output-element math is reproduced operation-for-operation (same dd-order dot, same Max/Exp recurrence, same
sink epilogue), the result is **BIT-IDENTICAL** to the per-element kernel.
- **Measured (qwen2.5-coder Q4_K_M, RTX 4070, 1081-token prefill): FusedAttention 17.8s → 1.2s (~14.5×,
  72.3% → 15.2% of prefill); total prefill 24.6s → 8.1s.** MatMul is again the dominant node.
- **Verified:** A/B byte-identical generated tokens AND logits on qwen2.5-coder + gemma4 (head_dim 128 and 256,
  GQA, causal + sliding-window, custom scale, bf16 strided KV-cache path). New PMT test
  `FusedAttention_Grouped_MatchesPerElement` asserts grouped == per-element across all 6 backends (42/42 green);
  grouped runs on CPU/CUDA/OpenCL/Wasm, falls back to per-element on WebGL (no workgroup shared memory) /
  WebGPU (slow workgroup reduction) and for SKV beyond the shared cap.
- **Opt-in** (`FusedAttentionKernel.EnableGroupedAttention` / env `GGUF_ATTN_GROUP=1`); library-default off
  pending the full sweep (mirrors `EnableMultiRowGemm`). Example 06 server opts in. Huge-context prefill
  (SKV > 4096) still uses the per-element kernel — kv-tiled flash attention is the follow-up.

### Ollama-compatible inference server (Example 06) + general generation core

**New library generation core, plus a drop-in Ollama-replacement Example so prebuilt agentic frontends
(Claude CLI, Pi, Codex, OpenCode, …) can use native-GPU GGUF inference.**

Library:
- **`SentencePieceStreamingDecoder`** (+ `SentencePieceTokenizer.TokenToBytes`) — stateful UTF-8-safe
  incremental detokenizer for streaming; holds incomplete multi-byte sequences until complete. Also fixes
  a latent bug where `Decode` reinterpreted multi-byte byte-fallback glyphs as Latin-1 (mojibake). Verified
  all 6 backends (PMT) incl. a `é = <0xC3><0xA9>` case.
- **`GgufGenerator`** — architecture-agnostic generation on `InferenceSession`: incremental KV-cache decode +
  sampling + the streaming detokenizer + stop handling (EOS / stop token ids / **arbitrary stop strings**,
  with held-back tails so partial stop strings never leak). Reproduces the gemma4 pipeline byte-identical;
  stop-string truncation verified.
- **`ChatTemplates`** — multi-turn token-level chat-prompt builders (ChatML, Llama 3, gemma4) emitting
  structural markers as single control-token ids, dispatched by `DetectChatFormat` from the model's own
  `tokenizer.chat_template`. (Text only; tool_calls + full Jinja2-from-GGUF rendering are follow-ups.)

Example `06.OllamaServer.Console`:
- Reads **Ollama's model cache** (`~/.ollama/models`) zero-copy — serves the GGUF blobs Ollama already
  pulled (verified 17/17 models). ASP.NET Core/Kestrel host on :11434.
- Three client protocols, all verified end-to-end against gemma4 from the cache: OpenAI
  (`/v1/chat/completions` SSE, `/v1/models`), Ollama-native (`/api/chat`, `/api/generate` NDJSON,
  `/api/tags`), and **Anthropic Messages** (`/v1/messages` SSE + `/v1/messages/count_tokens`) for Claude CLI.

Known (testbed surfaced these): qwen2 GGUF weight load FailFasts (`Index/Extent out of bounds`) — a loader
bug to fix; general Jinja2-from-GGUF templating + tool-calling are the next increments.

### Single-pass fused RMSNorm + SpawnDev.ILGPU 4.15.0

**RMSNorm is now single-pass (one group per row) on every backend with a group — fusing the two-pass stats +
apply into ONE dispatch (no second dispatch, no invRms global round-trip, no scratch).** Thread 0 of each group
computes the row's invRms with the EXACT same f64 sum-of-squares as the old stats pass (so the result is
byte-identical — zero precision drift), then the whole group applies the normalization in parallel. WebGL (TF,
no shared-mem group reduction) keeps the two-pass (with the reusable invRms ring below). This was previously
blocked by the SpawnDev.ILGPU WGSL `_uf_group_iter` redeclaration bug (two `Grid.IdxX` `LoadStreamKernel`s
colliding on WebGPU) — now confirmed FIXED: MatMul's tiled kernel + this RMSNorm group kernel both run on
WebGPU (Norm 194/0). Verified byte-identical: PMT `Norm` + `GGUFDecodeKVCache` decode-equivalence + gemma4-12b
identical tokens, all 6 backends.

**Bumped SpawnDev.ILGPU 4.14.2-local.1 → 4.15.0** (Wasm SIMD128 auto-vectorization with a byte-identical scalar
path; all in-register quant decoders single-exit/branchless — extends the WebGL/GLSL multi-exit-decode fix to
FP8 E4M3/E5M2).

### RMSNorm invRms scratch from a reusable ring (no per-token alloc / leak)

**RMSNorm's two-pass `invRms` scratch now comes from a fixed reusable ring instead of a fresh per-call
`Allocate1D` that was never freed until `Dispose`.** Each RMSNorm call allocated a rows-sized invRms buffer and
appended it to a list released only at `Dispose` — for a 48-layer gemma4 decode (~6 RMSNorm/layer) that is ~288
tiny GPU buffers per token accumulating for the entire generation (a real, if small-per-item, leak the code
comment acknowledged). The invRms ring (64 reusable buffers, each grown to the max rows it has seen) eliminates
the per-call allocation AND the unbounded growth; a slot is reused only after 64 calls — far past the
two-dispatch (Pass 1 → Pass 2) lifetime of any one call's invRms — so the race the per-call alloc avoided cannot
return. The two-pass math is byte-identical (no precision change). Verified: PMT `Norm` + `GGUFDecodeKVCache`
decode-equivalence + gemma4-12b byte-identical tokens, all 6 backends.

NOTE: the deeper single-pass (fuse stats + apply into one dispatch, dropping the second dispatch + the invRms
round-trip) is now DONE — see the single-pass fused RMSNorm entry above. The `_uf_group_iter` WGSL redeclaration
bug that previously blocked a second `Grid.IdxX` `LoadStreamKernel` on WebGPU is fixed/stale (MatMul's tiled
kernel + the RMSNorm group kernel coexist on WebGPU, Norm 194/0).

### FusedAttention params buffer ring pre-allocated (no per-call alloc)

**`FusedAttentionKernel` reuses a pre-allocated params-buffer ring instead of `Allocate1D` per call.** Each
attention dispatch built its tiny (10-11 int) params buffer with a fresh `Allocate1D` (+ dispose-on-overwrite)
— ~48 attention nodes/token → ~48 tiny GPU allocations per token. The ring slots (still RingSize=64, so a slot
isn't reused until well past any unflushed batch) are now each allocated ONCE at a fixed size and refilled via
`CopyFromCPU`; the kernel reads only the used prefix. Same upload, no per-call allocation. Attention results
unchanged (PMT `Attention` 122/0 all 6 backends).

### Hoist the per-token executor refcount/constant rebuild out of RunAsync

**`GraphExecutor.RunAsync` no longer re-walks the whole graph every token to rebuild the buffer-recycling
refcounts + the runtime-constant map.** Each decode step rebuilt, over all ~1400 nodes: the refcount map (a full
node-input walk), the graph-output HashSet, and the constant-output set (a LINQ `Constant` scan + a node walk to
strip stale compile-time constants) — O(nodes) CPU per token, the super-linear per-node residual that forced
multimodal prefill to go token-by-token. Since the executor's graph is fixed (`readonly`; recompile builds a NEW
executor), these are now precomputed ONCE (`EnsureRunTemplates`): a base refcount template (node inputs + graph
outputs + weights pinned to `int.MaxValue`) and a clean-constants map (compile-time constants with
non-Constant-node outputs pre-stripped). Each run clones the templates and pins only that call's inputs —
byte-identical result, without the per-token node walks / LINQ. Verified: `GGUFDecodeKVCache` decode-equivalence
(incremental == full recompute) + gemma4-12b greedy generation byte-identical (CUDA) + the inference Pipeline
sweep green on all 6 backends. The sync `Run()` path (non-decode) is unchanged.

### FusedAttention reads the bf16 KV store directly (no per-token repack)

**GGUF decode attention now reads the `[kvHeads, maxSeq, hd]` KV store DIRECTLY in its native bf16/f32 type,
maxSeq-strided — eliminating the per-token O(history) repack + bf16→f32 widen.** Previously every decode token
re-converted the ENTIRE K/V history (each layer, each token) from the bf16 store into a fresh contiguous f32 pack
for FusedAttention — O(history) memory-bandwidth per token, growing with context (the contiguous
`[kvHeads, totalLen, hd]` pack layout shifts every token, so it can't be appended incrementally). New
`FusedAttention.ForwardStrided<T>` reads K/V in their native type (`BFloat16` / `float`) with in-register
`PrecisionConvert` (branchless — the same read-native-convert-in-register philosophy as the low-p MatMul/FusedLinear
work) and an explicit per-head stride, so the cache exposes its store directly (`GGUFDecodeKVCache.StoreK/StoreV`)
and the per-token KV work drops to O(1) (just writing the new token). The existing f32-contiguous FusedAttention
kernel/`Forward` (SD / ONNX attention) is UNTOUCHED. Per-token KV cost is now flat in context length, not linear.

Verified: `FusedAttention_Strided_MatchesReference` (f32 anchor byte-identical to the existing kernel + bf16 +
maxSeq-strided store, all with GQA + causal + sliding-window + kvOffset) + `GGUFDecodeKVCache_IncrementalMatchesFullRecompute`
(incremental decode == full recompute, F32 + BF16) green on all 6 backends; gemma4-12b greedy generation
byte-identical to the pre-change tokens (CUDA). NOTE: WebGL's sub-word (bf16) kernel read of the large strided
store mis-addresses (an ILGPU WebGL backend limitation — f32-strided + all 5 other backends incl. WebGPU are
byte-exact), so WebGL+bf16 falls back to the existing repack — correct, just the old O(history); surfaced to
Geordi for the durable WebGL fix.

### GGUF decode pipelines enable CacheShapeReadbacks

**`GgufTextGenerationPipeline` and `Gemma4MultimodalPipeline` now set `CacheShapeReadbacks = true`.** The flag
(designed for exactly this fixed-shape decode loop) was never enabled on the gemma4 decode path, leaving two
decode-loop wins off: (1) per-step output-buffer recycling (the prior step's logits buffer is returned to the
pool before the next rent, so the same-shape output reuses it — without this, long generations accumulate ~13
fresh buffers/step and OOM), and (2) the warm shape-readback cache (skips the GPU→host round-trips for
shape-derived values once they're proven stable — the browser-readback latency win). The cache self-validates
(probe→stable→finalize, falling back to live readback for anything not proven stable), and both loops consume
each step's logits (greedy argmax) before the next step, satisfying the output-recycling contract. Verified
decode-equivalent: gemma4-12b greedy generation is byte-identical with the flag on vs off (same tokens, CUDA).

### FusedLinear register-blocked path for native low-precision weights (+ SiLU/ReLU)

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

### SpawnDev.ILGPU 4.14.2 + MXFP4 scale single-source-of-truth decode

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

### Register-blocked GEMM for native low-precision weights

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

### Conv2D / ConvTranspose2D accumulate in f32 (not f64)

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

### GPU argmax greedy decode (no per-token full-vocab readback)

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

### SpawnDev.ILGPU 4.14.1 migration + FusedLinear native low-precision weight support

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

### Gemma 4 multimodal chat + selectable-precision decode KV cache + SpawnDev.ILGPU 4.13.0 migration

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

### gemma4:12b GGUF forward is CORRECT end-to-end (CUDA)

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

### gemma4 decode-path kernels (GEMV routing, masked flash attention, RoPE generalization)

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

### GGUF quantization correctness overhaul (the K-quant landmine)

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
