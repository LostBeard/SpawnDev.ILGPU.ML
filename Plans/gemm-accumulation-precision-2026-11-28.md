# GEMM accuracy and speed against cuBLAS on the same card (2026-11-28, Tuvok)

TJ (2026-11-28): a fix that makes us slower than ComfyUI on the same hardware at the same quality is a bad fix; it means the problem is not solved
correctly. So the yardstick is PyTorch CUDA = what ComfyUI runs on this GPU, measured on this machine (D:\Projects\venv-torchcuda, torch 2.14.1+cu126).

## Measured, RTX 4070 (tools/anima-ref/torch_gemm_compare.py, Benchmark_Gemm_VsCublas)

Speed at the DiT shapes, 4096 tokens: qkv 1536 -> 4608 / mlp-up 1536 -> 6144 / mlp-down 6144 -> 1536. Accuracy: random [M, 2048] x [2048, 2048]^T,
bf16-exact weights, relRMS from float64 (cuBLAS: M = 1024; ours: M = 256; the per-element error is the same statistic).

| path | relRMS | TFLOPS |
|---|---|---|
| cuBLAS bf16 (ComfyUI's normal GPU dtype) | 2.6e-3 | 50.1 / 53.6 / 51.9 |
| OURS tensor cores fast (MmaGemm, bf16 activations) | 1.7e-3 | 33.2 / 33.6 / 34.6 |
| cuBLAS TF32 | 2.1e-4 | 26.6 / 25.6 / 23.6 |
| cuBLAS fp32 (ComfyUI --force-fp32) | 3.1e-7 | 16.9 / 17.1 / 16.1 |
| OURS tensor cores precise (MmaGemm x2, bf16 hi + lo activations) | 2.5e-6 | 19.6 / 19.9 / 21.1 |
| OURS tiled fp32 (TiledGemm): CUDA (tensor cores off) / OpenCL / WebGPU | 8.1e-7 | 11.0-10.0 / 11.4-12.0 / 7.7-9.9 |

On Anima block 0's real cross v projection: cuBLAS fp32 1.3e-7, numpy CPU fp32 2.7e-7, ours sequential 4.9e-7 (OpenCL) / 8.6e-7 (CUDA precise).

## The three gaps (each must close WITHOUT trading speed for accuracy)
1. Fast path: 1.5x slower than cuBLAS bf16 at the same quality -> MmaGemm tuning (tile shape, cp.async multi-stage pipeline, ldmatrix, epilogue).
2. Precise path: faster than cuBLAS fp32 but 8x less accurate. The bf16 hi + lo split keeps ~16 activation bits. Fix: fp16 hi + lo at power-of-two
   scales (MmaAttention.KernelExact's trick: 3e-6 -> 6e-8 there): 22 bits, still two MMA passes (same speed), weights bf16 -> fp16 exact under a
   per-tensor power-of-two scale. Target: <= 3e-7 at >= 19 TFLOPS (better than cuBLAS fp32 on both).
3. Tiled fp32 (WebGPU, OpenCL, CPU): 1.5x slower and 2.6x less accurate than cuBLAS fp32. The two-level sum (below) fixed the accuracy (7x on Anima)
   but cost 20% (a second register set): rejected. Needs a redesign: split K across warps within the CTA (each warp's sum is K / S long, one
   shared-memory reduction at the end: no extra registers), plus the speed work cuBLAS has (double-buffered shared memory, wider loads).

## Rejected: two-level accumulation in TiledGemm (gen_tiled_gemm.py GEMM_TWO_LEVEL=1, kept as an A/B switch, default output unchanged)
Anima DiT vs ComfyUI's dump 1.69e-5 -> 2.46e-6 on OpenCL, but TiledGemm 11.5 -> 9.3 TFLOPS (OpenCL), 10.3 -> 8.0 (WebGPU); folding every 4 k steps
was no faster (the cost is the registers, not the adds).

## Root cause of our per-call parity error (how this started)
Anima's DiT matched ComfyUI's CPU fp32 dumps to ~1-2e-5 a call where ComfyUI itself is ~2.5e-6 from float64; bisected op by op (bisect_block0.py):
every Linear carries it (attention adds nothing). gemm_accumulation_error.py: sequential fp32 over K = 4.92e-7 on the v projection (= the GPU's 4.9e-7).
