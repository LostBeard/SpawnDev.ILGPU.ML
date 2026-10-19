"""Quantise the bf16 FLUX.2 exports to 4-bit blocks a browser can hold (the C# Flux2Weights dequantises a matrix at a time into a bf16 scratch, then the normal GEMM runs).

  python -I quantize_flux2.py <weights_dir> <name>           name = text_encoder | transformer
      reads  <weights_dir>/<name>.safetensors   (export_flux2.py)
      writes <weights_dir>/<name>_q4.safetensors

Format ("Q4_1"-style): every weight matrix [N, K] with K % 32 == 0 and at least 1M elements is cut along K into blocks of 32. A block is 5 little-endian uint32 words:
  word 0       low 16 bits  d  (fp16 scale) = (max - min) / 15
               high 16 bits m  (fp16 offset) = min
  words 1..4   32 nibbles, element 8*i + j of the block is (word[1 + i] >> (4 * j)) & 15
  value = d * nibble + m       (5 bits per weight: 2.5 GB for the 3.9B transformer)
The quantised tensor is stored under the matrix's own name with dtype I32 and shape [N, K / 32 * 5]. Everything else (norms, small projections, vectors) stays bf16.
"""
import os, sys
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

folder, name = sys.argv[1], sys.argv[2]
src = os.path.join(folder, name + '.safetensors')
dst = os.path.join(folder, name + '_q4.safetensors')
MIN_ELEMENTS = 1_000_000


def quantise(w):
    n, k = w.shape
    x = w.float().reshape(n, k // 32, 32)
    mn = x.amin(dim=2, keepdim=True)
    mx = x.amax(dim=2, keepdim=True)
    d = ((mx - mn) / 15).to(torch.float16)
    m = mn.to(torch.float16)
    df = d.float(); mf = m.float()
    q = torch.where(df > 0, torch.round((x - mf) / df.clamp_min(1e-30)), torch.zeros_like(x)).clamp(0, 15).to(torch.int64)
    words = []
    head = (d.view(torch.int16).to(torch.int64) & 0xFFFF) | ((m.view(torch.int16).to(torch.int64) & 0xFFFF) << 16)
    words.append(head.reshape(n, k // 32, 1))
    for i in range(4):
        wd = torch.zeros(n, k // 32, dtype=torch.int64)
        for j in range(8):
            wd |= q[:, :, 8 * i + j] << (4 * j)
        words.append(wd.reshape(n, k // 32, 1))
    allw = torch.cat(words, dim=2) & 0xFFFFFFFF
    allw = torch.where(allw >= 2 ** 31, allw - 2 ** 32, allw).to(torch.int32)
    return allw.reshape(n, k // 32 * 5).contiguous()


def dequantise(words, k):
    n = words.shape[0]
    u = words.reshape(n, k // 32, 5).to(torch.int64) & 0xFFFFFFFF
    d = (u[:, :, 0] & 0xFFFF).to(torch.int16).view(torch.float16).float()
    m = ((u[:, :, 0] >> 16) & 0xFFFF).to(torch.int16).view(torch.float16).float()
    out = torch.empty(n, k // 32, 32)
    for i in range(4):
        for j in range(8):
            out[:, :, 8 * i + j] = ((u[:, :, 1 + i] >> (4 * j)) & 15).float() * d + m
    return out.reshape(n, k)


out = {}
worst = 0.0
with safe_open(src, framework='pt') as f:
    meta = f.metadata() or {}
    for key in f.keys():
        t = f.get_tensor(key)
        if t.ndim == 2 and t.shape[1] % 32 == 0 and t.numel() >= MIN_ELEMENTS:
            qw = quantise(t)
            err = (dequantise(qw, t.shape[1]) - t.float()).pow(2).sum().sqrt() / t.float().pow(2).sum().sqrt()
            worst = max(worst, float(err))
            out[key] = qw
            print(f'{key} {tuple(t.shape)} relRMS {float(err):.4f}', flush=True)
        else:
            out[key] = t
meta = dict(meta); meta['quant'] = 'q4_1 blocks of 32 along K: [d f16 | m f16], 4 words of 8 nibbles; stored I32 [N, K/32*5]'
save_file(out, dst, metadata=meta)
print(dst, round(os.path.getsize(dst) / 1e9, 2), 'GB; worst per-matrix relRMS', round(worst, 4))
