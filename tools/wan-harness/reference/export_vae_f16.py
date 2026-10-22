"""Export the Wan 2.2 VAE (Wan2.2_VAE.pth, 704.7M parameters in fp32) to one SafeTensors file in the layout the C# `WanVae` reads.

Convolution weights [cout, cin, *k] become fp16 [cout, taps, cin] (taps = product of the kernel sizes, row-major over the kernel, channel last): the gather-GEMM layout of the
TRELLIS.2 sparse convolutions (`SparseKernels.Conv3`, weight [cout][taps][cin]); Conv2d weights get taps = kh*kw, 1x1 convolutions taps = 1. Biases are fp16 [cout], the RMS-norm gammas
fp32 [C] (their [C,1,1,1] / [C,1,1] shape squeezed), and `latents_mean` / `latents_std` (fp32 [48]) are the constants of `Wan2_2_VAE` (decode: z * std + mean, encode: (mu - mean) / std).
Written tensor by tensor with the header computed first.

  python export_vae_f16.py OUT.safetensors
"""
import json
import os
import struct
import sys

import numpy as np
import torch

WAN = os.path.expanduser("~/wan/Wan2.2")
CKPT = os.path.expanduser("~/wan/ckpt")
sys.path.insert(0, WAN)


def main():
    out_path = sys.argv[1]
    sd = torch.load(os.path.join(CKPT, "Wan2.2_VAE.pth"), map_location="cpu", mmap=True, weights_only=True)
    entries = []
    for name, t in sd.items():
        if name.endswith(".weight") and t.dim() >= 4:
            co, ci = t.shape[0], t.shape[1]
            k = list(t.shape[2:])
            taps = int(np.prod(k))
            entries.append((name, "F16", [co, taps, ci], (lambda t=t, co=co, ci=ci, taps=taps: t.reshape(co, ci, taps).permute(0, 2, 1).contiguous().to(torch.float16))))
        elif name.endswith(".bias"):
            entries.append((name, "F16", list(t.shape), (lambda t=t: t.to(torch.float16))))
        elif name.endswith(".gamma"):
            entries.append((name, "F32", [t.numel()], (lambda t=t: t.reshape(-1).contiguous().float())))
        else:
            raise SystemExit(f"unexpected tensor {name} {tuple(t.shape)}")
    # the latent normalisation constants of Wan2_2_VAE (decode: z * std + mean)
    from wan.modules.vae2_2 import Wan2_2_VAE
    vae = Wan2_2_VAE.__new__(Wan2_2_VAE)
    import inspect, re
    src = inspect.getsource(Wan2_2_VAE.__init__)
    lists = re.findall(r"torch\.tensor\(\s*\[([^\]]+)\]", src)
    mean = np.array([float(x) for x in lists[0].replace("\n", " ").split(",") if x.strip()], dtype=np.float32)
    std = np.array([float(x) for x in lists[1].replace("\n", " ").split(",") if x.strip()], dtype=np.float32)
    assert mean.shape == (48,) and std.shape == (48,), (mean.shape, std.shape)
    entries.append(("latents_mean", "F32", [48], lambda: torch.from_numpy(mean)))
    entries.append(("latents_std", "F32", [48], lambda: torch.from_numpy(std)))

    header, off = {}, 0
    for name, dt, shape, _ in entries:
        n = int(np.prod(shape)) * (2 if dt == "F16" else 4)
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    header["__metadata__"] = {"format": "pt", "source": "Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth (conv weights fp16 [cout, taps, cin])", "z_dim": "48"}
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for i, (name, dt, shape, fn) in enumerate(entries):
            t = fn()
            assert list(t.shape) == shape, (name, list(t.shape), shape)
            f.write(t.view(torch.int16).numpy().tobytes() if dt == "F16" else t.numpy().astype("<f4").tobytes())
            if i % 40 == 0:
                print(f"{i}/{len(entries)} {name} {shape}", flush=True)
    print("wrote", out_path, f"{os.path.getsize(out_path) / 1e9:.2f} GB,", len(entries), "tensors; mean[0..2]", mean[:3], "std[0..2]", std[:3])


if __name__ == "__main__":
    main()
