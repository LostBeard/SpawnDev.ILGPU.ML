# fp32 GEMM accumulation order: the port's per-call error vs ComfyUI's CPU fp32 (2026-11-28, Tuvok)

## Finding
Anima's DiT matches ComfyUI's fp32 dumps to ~1-2e-5 a call, ComfyUI itself is ~2.5e-6 from float64: ours is ~5x further from exact. Through the turbo
LoRA's 8 steps that becomes 1.0e-3 (ComfyUI 3.8e-4 from float64; tools/anima-ref/replay_lora_steps.py). Bisected op by op (bisect_block0.py):
the attention kernels add nothing; every Linear carries the error, the first one already (the adaLN vector: 1.3e-6 where ComfyUI's is 2.6e-7).

Root cause (gemm_accumulation_error.py, Anima block 0's cross-attention v projection, exact inputs, K = 1024):

| fp32 summation of the K products | relRMS from float64 |
|---|---|
| sequential, one register per output (our TiledGemm / LinearImpl order) | 4.92e-7 (the GPU measured 4.9e-7 on OpenCL: the same) |
| numpy fp32 matmul (blocked BLAS = ComfyUI's CPU) | 2.04e-7 |
| two-level: blocks of 8 / 16 / 32 / 64, block sums added | 1.80e-7 / 1.30e-7 / 9.84e-8 / 8.71e-8 |

CUDA's precise path (MmaGemm.KernelBf16x2Dense: activations as bf16 hi + lo = ~16 bits) is worse on the same op: 8.6e-7. Its accumulation is already
two-level (each k step's MMA from zero, then an fp32 add); the activation split is the limit there (the exact attention fixed the same with fp16 hi + lo
at power-of-two scales, 22 bits: 3e-6 -> 6e-8).

## MEASURED (same day): TiledGemm two-level (gen_tiled_gemm.py GEMM_TWO_LEVEL=1, GEMM_FOLD=k steps per partial; default output unchanged)

| RTX 4070, Trellis2_DiTKernels_Benchmark (4096 tokens, qkv / mlp-up / mlp-down) | OpenCL TFLOPS | WebGPU TFLOPS |
|---|---|---|
| sequential (shipped) | 11.5 / 11.9 / 11.1 | 10.3 / 10.6 / 10.0 |
| two-level, a partial per k step of 8 | 9.3 / 9.4 / 9.1 (-20%) | 8.0 / 8.2 / 8.2 (-21%) |
| two-level, a partial per 4 k steps (32) | 8.8 / 9.5 / 9.2 | 7.6 / 7.8 / 7.7 |

The cost is the second set of 64 accumulators (register pressure), not the adds: folding 4x less often is no faster. Accuracy (OpenCL, blocks of 8):
Anima DiT call_0 vs ComfyUI's dump 1.69e-5 -> 2.46e-6 (7x; the float64 reference is 2.4e-6 from the same dump), call_1 1.35e-6; adaLN 1.3e-6 -> 3.2e-7;
the cross v projection 4.9e-7 -> 1.8e-7 (numpy's prediction for blocks of 8). CUDA's GEMMs at M >= 16 are MmaGemm (not affected); its small-M ones are.

Conclusion proposed (needs TJ: a speed / parity trade on a core kernel): a second TiledGemm variant chosen by the existing DiTKernels.PreciseActivations
(the ComfyUI-parity models already set it and already pay for x2 MMA passes on CUDA), plus fp16 hi + lo for MmaGemm's precise path. Speed models keep
the sequential kernel. Not shipped.

## The fix to measure before deciding (a core change: every model, every backend, bitwise)
1. TiledGemm (and LinearImpl): a fresh partial per BK tile (or every 32 k), added into the output's running sum: 5x on the order alone. Cost: a second
   accumulator per register-tile output (register pressure; measure the GEMM speed on WebGPU, OpenCL, CUDA before / after).
2. MmaGemm precise: fp16 hi + lo activations at a power-of-two scale (MmaAttention.KernelExact's trick) instead of bf16 hi + lo.
3. Gate: the Anima / LTX-2 / Z-Image / Qwen-Image / FLUX parity tests before / after, plus the speed benchmarks; full sweep.
