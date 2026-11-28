"""How much of the GPU port's per-Linear error is the fp32 accumulation ORDER (not the arithmetic): Anima block 0's cross-attention v projection
(context [512, 1024] from SpawnGen's tap, weight [2048, 1024] bf16, exact inputs) in float64 vs:
  numpy fp32 matmul (blocked BLAS, what ComfyUI's CPU run does), a sequential fp32 sum over K (one register per output, the GPU kernels' order),
  and a two-level fp32 sum (blocks of B products summed, then the block sums added: the cheap fix).
The GPU measured 4.9e-7 (OpenCL tiled fp32) and 8.6e-7 (CUDA, bf16 hi + lo activations) on this op (bisect_block0.py).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st_reader import SafeTensors  # noqa: E402
from taps import tap  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


st = SafeTensors(r'D:\Projects\anima\anima-preview3-base.safetensors')
W = st.get('net.blocks.0.cross_attn.v_proj.weight')                 # [2048, 1024] (bf16 values, exact in fp32)
ctx = tap('blocks_0_0_pre-in2_0')[0]                                 # [512, 1024]
rows = np.flatnonzero(np.abs(ctx).sum(1))                            # the real rows (the zero padding gives exact zeros)
x = ctx[rows]
ref = x @ W.T                                                        # float64
x32, W32 = x.astype(np.float32), W.astype(np.float32)
print(f'cross v_proj on {len(rows)} real context rows, K = {x.shape[1]}')
print(f'  numpy fp32 matmul (blocked BLAS)      {rel(x32 @ W32.T, ref):.3e}')
K = x.shape[1]
seq = np.zeros((x.shape[0], W.shape[0]), dtype=np.float32)
for k in range(K):
    seq = (seq + np.outer(x32[:, k], W32[:, k])).astype(np.float32)
print(f'  sequential fp32 over K                {rel(seq, ref):.3e}')
for B in (8, 16, 32, 64):
    tot = np.zeros_like(seq)
    for k0 in range(0, K, B):
        blk = np.zeros_like(seq)
        for k in range(k0, min(K, k0 + B)):
            blk = (blk + np.outer(x32[:, k], W32[:, k])).astype(np.float32)
        tot = (tot + blk).astype(np.float32)
    print(f'  two-level fp32 (blocks of {B:2d})          {rel(tot, ref):.3e}')
