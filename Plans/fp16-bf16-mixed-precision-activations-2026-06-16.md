# Mixed-precision activations (fp16 + bf16) via generic `INumber<T>` kernels — design

**Goal (Rule 1/4, "works as well as possible for others"):** cut the GPU working set ~2× AND speed up via
bandwidth by running activations (not just weights) in low precision — `ILGPU.Half` (fp16) for SD-Turbo's
ort-web models, `ILGPU.BFloat16` for gemma4. Today activations are all fp32; SD-Turbo on a 12 GB card sits at
~11.5 GB resident (fits, no shared spill after the byte-bounded-release fix `823b3f8`, but too high for 6–8 GB
cards). Halving activations is the lever.

## The enabling insight (verified 2026-06-16, in the fork)
`ILGPU.Half` AND `ILGPU.BFloat16` BOTH implement `INumber<T>`/`ISignedNumber<T>` with **transpilable**
operators (`ILGPU/Half.GenericMath.cs:28`, `ILGPU/BFloat16.GenericMath.cs:27`) — added so a generic-math
kernel `where T : INumber<T>` binds to ILGPU's operators, NOT `System.Half`'s (whose INumber lowers to an
unsupported BitCast). **So one generic kernel covers `float` + `Half` + `BFloat16` from a single source — no
per-op Half/bf16 rewrites.** bf16 arithmetic/compare already PROVEN cross-backend (Geordi `PMT_FILTER=BFloat16`
93/0 all 6 backends, local.2→.6); Half proven by the 2026-06-05 f16 spike. (Caveat below: instantiating a
WHOLE op kernel generically still must be confirmed to transpile per backend — verify as each op is wired.)

## Current state
- **Weights:** fp16-source weights are kept `Half` on GPU ONLY when consumed exclusively as the weight operand
  of MatMul or group-1 Conv (`InferenceSession.cs:1065-1118` gating); else expanded to fp32. So SD UNet/VAE
  conv+matmul weights are likely ALREADY fp16; norms/biases/etc. are fp32. (MEASURE the exact count via
  `ML_VERBOSE=1` → "f16 weights: X of Y" — Example 03 toggle added, env-gated, uncommitted.)
- **Activations:** ALL fp32. `BufferPool.Rent` returns fp32 `Tensor`; the executor's `tensors` dict carries
  fp32; ops consume/produce fp32 `Tensor`. `HalfTensor` exists for WEIGHTS only. This is the gap.
- Hand-written mixed kernels (`MatMulHalfWeight`, `Conv2D.ForwardHalfWeight`): Half WEIGHT × fp32 activation →
  fp32 accumulate (ORT-style, no accuracy loss). These are the precedent for the "mixed" policy below.

## Design
### 1. Activation precision is a per-tensor property
Extend the executor so an intermediate buffer can be `Half`/`BFloat16`, not only fp32. Options:
- (A) A generic `Tensor<T>` flowing through the graph (cleanest long-term; but the f16-spike note warns a
  fully-generic `Tensor<T>` weight path hit issues — re-verify for ACTIVATIONS).
- (B) Pragmatic: `BufferPool.RentHalf(shape)` / `RentLowP(shape, dtype)` returning a low-precision buffer +
  the executor tracks each tensor's element type (a `Dictionary<string,DType>` alongside `tensors`). Ops that
  support low-p read/write it; boundaries insert convert kernels. **Recommended first** — least invasive.

### 2. Per-op precision POLICY (choose per op, like the weight path already does)
- **Full-low-precision** (T=Half/bf16 throughout): element-wise ops (SiLU/GELU/Add/Mul/Scale), Resize/Upsample,
  Concat, Transpose — no large reductions, so low-p compute is safe + smallest/fastest. Use the GENERIC kernel.
- **Mixed (low-p storage, fp32 accumulate):** MatMul, Conv, GroupNorm/LayerNorm/RMSNorm, Softmax, attention —
  anything with a reduction over many terms. Read low-p inputs, accumulate fp32, write low-p output. (The
  generic kernel can still help: parameterize storage T, keep `float` accumulator.)

### 3. Boundaries / conversions
- Model input latent + final output: convert fp32↔low-p at the edges (cheap element-wise).
- A node whose downstream consumer needs fp32 (or a not-yet-low-p op): insert a convert. Track via the dtype map.
- Pick the activation dtype per pipeline: SD-Turbo → Half (model is fp16); gemma4 → BFloat16.

### 4. Incremental, VERIFIABLE rollout (one op at a time, CPU-oracle gated — never ship unverified)
Order by memory impact on the SD-Turbo VAE decode (the working-set peak):
1. Convert the VAE-decode subgraph's element-wise ops (SiLU, Add) + Resize to generic low-p first (safe,
   full-low-p). Measure working-set drop (BufferPool TrackPeaks) + image stays sharp vs the ORT oracle.
2. Conv2D mixed (low-p activation in/out, fp32 accumulate; weight already Half) — the big activation buffers.
3. GroupNorm mixed. 4. Attention/MatMul mixed.
Each step: generic kernel instantiated for Half → **confirm it transpiles + runs on each backend** (CPU/CUDA/
OpenCL/WebGPU/WebGL/Wasm) via a CPU-oracle PMT test BEFORE moving on. SD-Turbo image must stay sharp (diff vs
`_Models/ort_vae_oracle.cs`). Then repeat the win for gemma4 with bf16.

### 5. Verification harness (already partly built)
- `BufferPool.TrackPeaks`/`PeakTotalBytes`/`PeakLiveBytes` (committed) — measure working-set drop per step.
- `_Models/ort_vae_oracle.cs` + `SDTURBO_DUMP_LATENT` — image-fidelity vs ORT (must stay sharp).
- New per-op CPU-oracle PMT tests (the generic kernel vs a CPU reference, all backends) — the gate.

## Open questions to resolve while building
- Does a WHOLE generic op kernel (e.g. GroupNorm `where T : INumber<T>`) transpile on WGSL/GLSL/Wasm/PTX/CL, or
  only the scalar arithmetic? (Geordi proved scalar bf16 ops do; a full kernel body is the unknown — verify
  per op, fall back to mixed/hand-written if a backend chokes, like the bf16 radix WebGPU path did.)
- Accuracy budget: full-low-p element-wise is fine; confirm mixed-accumulate norms/matmul keep SD sharp +
  gemma4 coherent (the bf16 KV cache already showed a constant ~0.3% delta = acceptable).
- Pool interaction: low-p buffers are half the bytes → the byte-bounded release cap (`MaxPendingReleaseBytes`)
  and bucket reuse must account for element size (currently fp32-bytes assumed).

## EXECUTOR INTEGRATION — precise design (piece 3/n, the core cut). Built dtype-PARAMETERIZED (Geordi is
## wiring more low-p types; a new type = one `switch(dtype)` case + its convert kernel + a RentX pool — TJ).
Foundation DONE + verified all-backends: piece 1/n `PrecisionConvertKernels` fp32↔fp16/bf16 (`74ce795`, 14/0),
piece 2/n `BufferPool.RentHalf`/`ReturnHalf` (`181a5a1`, 8/0). Now wire `GraphExecutor.RunAsync`:

- **Opt-in** `GraphExecutor.ActivationPrecision` (enum `F32` default | `F16` | later BF16/…), set per
  `InferenceSession` (SD-Turbo VAE session → F16). F32 = byte-identical to today (zero change when off).
- **Approach (ii) convert-around-node (FIRST cut — operators stay fp32; lowest risk, real win, dtype-generic):**
  store eligible intermediates low-p; convert at node boundaries. NO operator rewrites.
  - **Eligible intermediate** = node OUTPUT that is a float feature map AND NOT: a graph output, a weight
    (`_weights`), an integer/shape/runtime-constant tensor (`_integerTensorNames`/`runtimeConstants`/Shape/
    ConstantOfShape), or a KV-cache tensor. Conservative — when unsure, keep fp32.
  - Mixed storage: parallel `Dictionary<string,HalfTensor> _halfTensors` ALONGSIDE `tensors` (least invasive;
    don't widen `tensors`). A name resolves to whichever holds it.
  - Per node: gather inputs — any input in `_halfTensors` → RentHalf-free fp32 temp via `HalfToFloat` (return
    right after the op). Run op (fp32, unchanged). Each ELIGIBLE fp32 output → RentHalf + `FloatToHalf` into it,
    register in `_halfTensors`, Return the fp32 temp. Non-eligible outputs stay fp32 in `tensors`.
  - Release: `ReturnHalf` for `_halfTensors`, `Return` for `tensors` (extend pendingReleases; count low-p at
    2 B/elem in `MaxPendingReleaseBytes`). Graph OUTPUTS stay/convert to fp32 (caller reads fp32).
- **dtype seam:** convert+rent `switch(ActivationPrecision)` (F16: RentHalf + Float<->Half). Adding BF16/fp8 =
  a case + `RentBFloat16` + the bf16 convert (already in PrecisionConvertKernels) — THE practice for Geordi's types.
- **VERIFY (measure twice):** (1) CONTROLLED minimal-graph executor unit test (tiny Conv/Relu/Add graph, F32 vs
  F16, assert ≈ within fp16 tol on ALL backends) — de-risk the wiring in isolation BEFORE the VAE. (2) SD-Turbo
  pipeline with the VAE session `ActivationPrecision=F16`: image SHARP vs `_Models/ort_vae_oracle.cs` + working-set
  drop measured (`BufferPool.TrackPeaks`).
- **Risk:** the hot RunAsync loop interleaves shape-inference / runtime-const capture / shape-cache / KV-cache /
  refcount — the convert insertion only touches eligible FLOAT feature-map I/O and must not perturb those.
  Approach (ii) keeps every operator fp32 = zero per-op correctness risk; risk is purely executor wiring →
  hence the controlled-graph test first. Do this cut DELIBERATELY (core path, every model), not rushed.
- **Later (perf, not first cut):** approach (i) precision-aware hot ops (Conv/GroupNorm low-p I/O, no fp32 temps)
  — generic `INumber<T>` where INumber-expressible; mixed read-(float)-compute-write-(low-p) for SiLU/GELU/Softmax
  (INumber has no exp).

### ⚠ MEASURED RESULT (2026-06-16): approach (ii) is CORRECT but does NOT reduce peak — needs (i) or immediate-free
Executor mixed-precision SHIPPED + verified (`cc085e5`, `MixedPrecision` 8/0 all backends; SD-Turbo VAE image
stays SHARP at F16 = mechanism correct). BUT wiring the VAE session to F16 **raised** peak GPU memory
**3507 → 4030 MiB** (live unchanged 2194). Root cause (verified-by-design, not guessed): the fp32 op-output is
**deferred-released** (added to `pendingReleases`, freed only at the byte-cap/N-node drain — the 823b3f8 fix), so
between drains the accumulated fp32 outputs coexist with their new fp16 copies + the fp32 convert-temps → MORE,
not less. The fp16 storage saving is negated by the transient/deferred fp32.
**So VAE-F16 via (ii) was NOT shipped (reverted the pipeline wiring).** The executor mechanism is KEPT (correct,
F32-guarded off-by-default, the dtype template Geordi's types plug into, the foundation for (i)).
**To actually win memory, ONE of:** (a) approach (i) precision-aware ops — Conv/GroupNorm read+write fp16
directly (NO fp32 temp, no convert-around) → true half-size I/O; the real fix, bigger (per-op). (b) free the
converted fp32 op-output IMMEDIATELY (not deferred) after its convert — safe on in-order backends (the convert
is the buffer's last reader, queued before any reuse), needs care on the Wasm pool. (c) note the WEIGHTS
(~5GB, fp32 in the ONNX for text_enc/unet — measured) + sequential model residency are the BIGGER 11.5GB
levers anyway; activations matter more at higher resolution / batch. Re-measure each.

## NOT this (kept separate, also on the roadmap)
Sequential model residency, tiled VAE (exact GroupNorm-sync). Those cut the OTHER parts of the 11.5 GB (the
co-resident weights, the VAE peak); this doc is the activation-precision lever. All three compound.
