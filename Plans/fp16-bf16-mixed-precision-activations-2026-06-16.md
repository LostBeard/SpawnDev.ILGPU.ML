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

## NOT this (kept separate, also on the roadmap)
Sequential model residency, tiled VAE (exact GroupNorm-sync). Those cut the OTHER parts of the 11.5 GB (the
co-resident weights, the VAE peak); this doc is the activation-precision lever. All three compound.
