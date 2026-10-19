# WebGPU: the Q4_K GEMV does not compile (barrier in non-uniform control flow)

Found 2026-10-19 on the Image Generation page: Qwen3-4B-Q4_K_M with `FusedDequantMatMul.EnableWebGPUGemv = true` (what the AI Chat page sets) fails in Chrome WebGPU at the first decode:

```
[WebGPU] Shader _main_ failed validation: Error while parsing WGSL: error: 'storageBarrier' must only be called from uniform control flow
  note: control flow depends on possibly non-uniform value
  note: reading from 'v_827' may result in a non-uniform value
```

then every later dispatch fails with `Invalid BindGroupLayout`. The AI Chat page only lists Q8_0 models (verified), which is why it never showed.

**Cause.** `GemvDequantQ4_KImpl` caches the 8 sub-block scales of a super-block in shared memory and calls `Group.Barrier()` twice per super-block, inside `for (blk < K / 256)` and inside `if (n < N)`. `K` and `N` come from the parameter buffer (`p[1]`, `p[2]`): a storage-buffer read, which WGSL's uniformity analysis treats as non-uniform, so a barrier under a loop or branch that depends on it is rejected. The Q8_0, Q4_0 and Q6_K GEMVs have their barriers only in the final reduction (outside any such loop) and compile.

**Ways out (not done).** (1) In SpawnDev.ILGPU's WGSL emitter, wrap a barrier-containing loop bound / branch condition in `workgroupUniformLoad` (the real fix: every kernel with a barrier in a data-dependent loop benefits, including the M>1 multi-row GEMMs that are currently gated off for browser GPUs). (2) Restructure the Q4_K GEMV to decode the per-thread scales without shared memory (costs the ALU saving that the cache bought on desktop; would need an A/B). 

**Meanwhile.** The demo's caption writer uses `unsloth/Qwen3-4B-GGUF` `Qwen3-4B-Q4_0.gguf` (Q4_0 GEMV compiles).
