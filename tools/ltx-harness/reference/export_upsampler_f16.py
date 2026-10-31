"""Export the LTX-Video 0.9.8 spatial latent upscaler (ltxv-spatial-upscaler-0.9.8.safetensors: LatentUpsampler, 128 -> 512 channels, 4 + 4 resblocks, x2 spatial pixel shuffle)
to one SafeTensors file in the layout the C# `LtxLatentUpsampler` reads, plus its licence text (the source file carries it in its metadata).

The source is float32. Convolution weights [cout, cin, kt, kh, kw] (3D) or [cout, cin, kh, kw] (the per-frame 2D upsampling conv) become fp16 [cout, taps, cin] (taps 27 or 9,
row-major over the kernel, channel last: the gather-GEMM layout of `SparseKernels.ConvTaps`); biases and GroupNorm weight / bias fp16. Names unchanged.
latents_mean / latents_std (fp32 [128], the VAE's per-channel statistics from the main checkpoint: the upsampler works on denormalised latents) are added.

  python export_upsampler_f16.py SRC.safetensors CKPT.safetensors OUT.safetensors [LICENSE_OUT.txt]
"""
import json
import os
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open


def main():
    src, ckpt, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    f = safe_open(src, framework="pt", device="cpu")
    meta = f.metadata() or {}
    cfg = json.loads(meta.get("config", "{}"))
    print("config", cfg)
    if len(sys.argv) > 4 and "license" in meta:
        with open(sys.argv[4], "w", encoding="utf-8") as fo:
            fo.write(meta["license"])
        print("wrote the licence text to", sys.argv[4])
    entries = []
    for name in sorted(f.keys()):
        t = f.get_tensor(name)
        if name.endswith(".weight") and t.dim() in (4, 5):
            co, ci = t.shape[0], t.shape[1]
            taps = int(np.prod(t.shape[2:]))
            entries.append((name, [co, taps, ci], (lambda t=t, co=co, ci=ci, taps=taps: t.reshape(co, ci, taps).permute(0, 2, 1).contiguous().to(torch.float16))))
        elif t.dim() == 1:
            entries.append((name, list(t.shape), (lambda t=t: t.to(torch.float16))))
        else:
            raise SystemExit(f"unexpected tensor {name} {tuple(t.shape)}")
    c = safe_open(ckpt, framework="pt", device="cpu")
    mean = c.get_tensor("vae.per_channel_statistics.mean-of-means").float()
    std = c.get_tensor("vae.per_channel_statistics.std-of-means").float()
    entries.append(("latents_mean", [128], lambda: mean.contiguous(), "F32"))
    entries.append(("latents_std", [128], lambda: std.contiguous(), "F32"))
    entries = [e if len(e) == 4 else (*e, "F16") for e in entries]
    header, off = {}, 0
    for name, shape, _, dt in entries:
        n = int(np.prod(shape)) * (2 if dt == "F16" else 4)
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    header["__metadata__"] = {"format": "pt", "source": "Lightricks/LTX-Video ltxv-spatial-upscaler-0.9.8 (conv weights fp16 [cout, taps, cin])",
                              "in_channels": str(cfg.get("in_channels", 128)), "mid_channels": str(cfg.get("mid_channels", 512)),
                              "num_blocks_per_stage": str(cfg.get("num_blocks_per_stage", 4)), "dims": str(cfg.get("dims", 3))}
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(out_path, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)))
        fo.write(hb)
        for name, shape, fn, dt in entries:
            t = fn()
            assert list(t.shape) == shape, (name, list(t.shape), shape)
            fo.write(t.view(torch.int16).numpy().tobytes() if dt == "F16" else t.numpy().astype("<f4").tobytes())
    print("wrote", out_path, f"{os.path.getsize(out_path) / 1e6:.0f} MB,", len(entries), "tensors")


if __name__ == "__main__":
    main()
