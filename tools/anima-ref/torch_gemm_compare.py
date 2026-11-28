"""PyTorch CUDA (what ComfyUI runs on this GPU) on the same GEMMs as our kernels: accuracy against float64 and speed.

Run with a CUDA build of torch (D:\\Projects\\venv-torchcuda\\Scripts\\python.exe). Accuracy: Anima block 0's cross-attention v projection on its
real context rows (SpawnGen's tap) and a [1024, 2048] x [2048, 2048] product on random data, in fp32 (TF32 off: torch's default for matmul), TF32,
and bf16. Speed: the Trellis2_DiTKernels_Benchmark shapes (4096 tokens: qkv 1536 -> 4608, mlp-up 1536 -> 6144, mlp-down 6144 -> 1536), fp32 / TF32 / bf16.
Our numbers (same card): sequential tiled GEMM 4.9e-7 on the v projection; 11.5 / 11.9 / 11.1 TFLOPS OpenCL, 10.3 / 10.6 / 10.0 WebGPU (fp32 math).
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st_reader import SafeTensors  # noqa: E402
from taps import tap  # noqa: E402

dev = torch.device('cuda')
print(torch.__version__, torch.cuda.get_device_name(0))


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def gpu_mm(x, w, mode):
    torch.backends.cuda.matmul.allow_tf32 = mode == 'tf32'
    if mode == 'bf16':
        xt = torch.tensor(x, dtype=torch.bfloat16, device=dev)
        wt = torch.tensor(w, dtype=torch.bfloat16, device=dev)
        return (xt @ wt.T).float().cpu().numpy().astype(np.float64)
    xt = torch.tensor(x, dtype=torch.float32, device=dev)
    wt = torch.tensor(w, dtype=torch.float32, device=dev)
    return (xt @ wt.T).cpu().numpy().astype(np.float64)


st = SafeTensors(r'D:\Projects\anima\anima-preview3-base.safetensors')
W = st.get('net.blocks.0.cross_attn.v_proj.weight')
ctx = tap('blocks_0_0_pre-in2_0')[0]
x = ctx[np.flatnonzero(np.abs(ctx).sum(1))]
ref = x @ W.T
print(f'\nAnima block 0 cross v_proj ({x.shape[0]} rows, K = {x.shape[1]}), relRMS from float64:')
for mode in ('fp32', 'tf32', 'bf16'):
    print(f'  torch CUDA {mode:5s} {rel(gpu_mm(x.astype(np.float32), W.astype(np.float32), mode), ref):.3e}')
print(f'  numpy CPU fp32  {rel((x.astype(np.float32) @ W.astype(np.float32).T).astype(np.float64), ref):.3e}')

rng = np.random.default_rng(3)
xr = rng.standard_normal((1024, 2048)).astype(np.float32)
wr = (rng.standard_normal((2048, 2048)) * 0.02).astype(np.float32)
wr = (wr.view(np.uint32) & 0xFFFF0000).view(np.float32)                # bf16-exact weights, as the models'
refr = xr.astype(np.float64) @ wr.astype(np.float64).T
print('\nrandom [1024, 2048] x [2048, 2048]^T, relRMS from float64:')
for mode in ('fp32', 'tf32', 'bf16'):
    print(f'  torch CUDA {mode:5s} {rel(gpu_mm(xr, wr, mode), refr):.3e}')
print(f'  numpy CPU fp32  {rel((xr @ wr.T).astype(np.float64), refr):.3e}')

print('\nspeed, 4096 tokens (best of 10 after warm-up):')
T = 4096
for name, K, N in (('qkv', 1536, 4608), ('mlp-up', 1536, 6144), ('mlp-down', 6144, 1536)):
    parts = []
    for mode in ('fp32', 'tf32', 'bf16'):
        torch.backends.cuda.matmul.allow_tf32 = mode == 'tf32'
        dt = torch.bfloat16 if mode == 'bf16' else torch.float32
        a = torch.randn(T, K, device=dev, dtype=dt)
        b = torch.randn(N, K, device=dev, dtype=dt)
        for _ in range(3):
            a @ b.T
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(10):
            t0 = time.perf_counter(); a @ b.T; torch.cuda.synchronize(); best = min(best, time.perf_counter() - t0)
        parts.append(f'{mode} {best * 1000:.2f} ms {2 * T * K * N / best / 1e12:.1f} TFLOPS')
    print(f'  {name:8s} ' + ' | '.join(parts))
