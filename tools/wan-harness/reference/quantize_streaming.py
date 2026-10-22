"""Quantise a bf16 SafeTensors export (the Wan 2.2 DiT, 10 GB) to 8- or 4-bit blocks WITHOUT holding the output in memory (WSL has 7.8 GB of RAM).

The block format is exactly that of tools/unirig-harness/reference/quantize_flux2.py (the C# `Flux2Weights` / `Flux2Kernels.DequantQ8/Q4` read it): every weight matrix [N, K] with K % 32 == 0
and at least 1M elements is cut along K into blocks of 32; a block is one header word (fp16 scale d = (max - min) / levels in the low half, fp16 offset m = min in the high half) followed
by the quantised elements (4-bit: 4 words of 8 nibbles; 8-bit: 8 words of 4 bytes); value = d * q + m. The tensor is stored under the matrix's own name with dtype I32 and shape
[N, K / 32 * (5 | 9)]. Everything else stays bf16. The file's metadata "quant" starts with "q8" or "q4".

  python quantize_streaming.py IN.safetensors OUT.safetensors [bits=8]
"""
import json
import os
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open

src, dst = sys.argv[1], sys.argv[2]
BITS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
assert BITS in (4, 8)
LEVELS = 15 if BITS == 4 else 255
WORDS = 5 if BITS == 4 else 9
MIN_ELEMENTS = 1_000_000


def quantise(w):
    n, k = w.shape
    x = w.float().reshape(n, k // 32, 32)
    mn = x.amin(dim=2, keepdim=True)
    mx = x.amax(dim=2, keepdim=True)
    d = ((mx - mn) / LEVELS).to(torch.float16)
    m = mn.to(torch.float16)
    df = d.float(); mf = m.float()
    q = torch.where(df > 0, torch.round((x - mf) / df.clamp_min(1e-30)), torch.zeros_like(x)).clamp(0, LEVELS).to(torch.int64)
    head = (d.view(torch.int16).to(torch.int64) & 0xFFFF) | ((m.view(torch.int16).to(torch.int64) & 0xFFFF) << 16)
    words = [head.reshape(n, k // 32, 1)]
    per, bits = (8, 4) if BITS == 4 else (4, 8)
    for i in range(32 // per):
        wd = torch.zeros(n, k // 32, dtype=torch.int64)
        for j in range(per):
            wd |= q[:, :, per * i + j] << (bits * j)
        words.append(wd.reshape(n, k // 32, 1))
    allw = torch.cat(words, dim=2) & 0xFFFFFFFF
    allw = torch.where(allw >= 2 ** 31, allw - 2 ** 32, allw).to(torch.int32)
    return allw.reshape(n, k // 32 * WORDS).contiguous()


def dequantise(words, k):
    n = words.shape[0]
    u = words.reshape(n, k // 32, WORDS).to(torch.int64) & 0xFFFFFFFF
    d = (u[:, :, 0] & 0xFFFF).to(torch.int16).view(torch.float16).float()
    m = ((u[:, :, 0] >> 16) & 0xFFFF).to(torch.int16).view(torch.float16).float()
    out = torch.empty(n, k // 32, 32)
    per, bits = (8, 4) if BITS == 4 else (4, 8)
    for i in range(32 // per):
        for j in range(per):
            out[:, :, per * i + j] = ((u[:, :, 1 + i] >> (bits * j)) & LEVELS).float() * d + m
    return out.reshape(n, k)


def main():
    with safe_open(src, framework="pt") as f:
        meta = dict(f.metadata() or {})
        keys = list(f.keys())
        plan = []        # (key, dtype string, shape, quantised?)
        for key in keys:
            shape = list(f.get_slice(key).get_shape())
            if len(shape) == 2 and shape[1] % 32 == 0 and shape[0] * shape[1] >= MIN_ELEMENTS:
                plan.append((key, "I32", [shape[0], shape[1] // 32 * WORDS], True))
            else:
                plan.append((key, "BF16", shape, False))
        header, off = {}, 0
        for key, dt, shape, _ in plan:
            n = int(np.prod(shape)) * (4 if dt == "I32" else 2)
            header[key] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}
            off += n
        meta["quant"] = f"q{BITS}_1 blocks of 32 along K: [d f16 | m f16] then " + ("4 words of 8 nibbles" if BITS == 4 else "8 words of 4 bytes") + f"; stored I32 [N, K/32*{WORDS}]"
        header["__metadata__"] = meta
        hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
        hb += b" " * ((8 - len(hb) % 8) % 8)
        worst, count = 0.0, 0
        with open(dst, "wb") as out:
            out.write(struct.pack("<Q", len(hb)))
            out.write(hb)
            for key, dt, shape, quant in plan:
                t = f.get_tensor(key)
                if quant:
                    qw = quantise(t)
                    err = float((dequantise(qw, t.shape[1]) - t.float()).pow(2).sum().sqrt() / t.float().pow(2).sum().sqrt())
                    worst = max(worst, err); count += 1
                    out.write(qw.numpy().astype("<i4").tobytes())
                    if count % 20 == 1:
                        print(f"{key} {tuple(t.shape)} relRMS {err:.4f}", flush=True)
                else:
                    out.write(t.contiguous().view(torch.int16).numpy().tobytes())
    print(dst, round(os.path.getsize(dst) / 1e9, 2), "GB;", count, "matrices quantised; worst per-matrix relRMS", round(worst, 4))


if __name__ == "__main__":
    main()
