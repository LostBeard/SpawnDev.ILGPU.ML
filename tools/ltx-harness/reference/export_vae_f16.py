"""Export the LTX-Video 0.9.8 VAE DECODER (the `vae.decoder.*` and `vae.per_channel_statistics.*` tensors of the single-file checkpoint) to one SafeTensors file in the layout the C# `LtxVae` reads.

Convolution weights [cout, cin, kt, kh, kw] become fp16 [cout, taps, cin] (taps = 27, row-major over the kernel, channel last): the gather-GEMM layout of `SparseKernels.ConvTaps` (as the Wan VAE
export). Linear weights stay fp16 [cout, cin]; biases fp16; the per-resnet `scale_shift_table` [4, C] and the final one [2, C] are fp32. The redundant ".conv" level of the source names is dropped and the
timestep embedder is renamed:
  conv_in.{weight,bias}, conv_out.{weight,bias}
  up_blocks.N.res_blocks.J.{conv1,conv2}.{weight,bias}, up_blocks.N.res_blocks.J.scale_shift_table
  up_blocks.N.time_embedder.{0,2}.{weight,bias}     (timestep_embedder.linear_1 / linear_2; N = 0, 2, 4, 6)
  up_blocks.N.conv.{weight,bias}                    (the residual upsamplers, N = 1, 3, 5)
  last_scale_shift_table, last_time_embedder.{0,2}.{weight,bias}, timestep_scale_multiplier [1] fp32
  latents_mean, latents_std                         <- vae.per_channel_statistics.mean-of-means / std-of-means  (fp32 [128]; decode: z * std + mean)

  python export_vae_f16.py SRC.safetensors OUT.safetensors
"""
import json
import os
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open


def main():
    src, out_path = sys.argv[1], sys.argv[2]
    f = safe_open(src, framework="pt", device="cpu")
    entries = []
    for key in sorted(k for k in f.keys() if k.startswith("vae.decoder.")):
        name = key[len("vae.decoder."):]
        t = f.get_tensor(key)
        if name == "timestep_scale_multiplier":
            entries.append((name, "F32", [1], (lambda t=t: t.float().reshape(1))))
            continue
        if name.endswith("scale_shift_table"):
            entries.append((name, "F32", list(t.shape), (lambda t=t: t.float().contiguous())))
            continue
        name = name.replace(".conv.weight", ".weight").replace(".conv.bias", ".bias")
        name = name.replace("timestep_embedder.linear_1", "0").replace("timestep_embedder.linear_2", "2")
        if name.endswith(".weight") and t.dim() == 5:
            co, ci = t.shape[0], t.shape[1]
            entries.append((name, "F16", [co, 27, ci], (lambda t=t, co=co, ci=ci: t.reshape(co, ci, 27).permute(0, 2, 1).contiguous().to(torch.float16))))
        elif name.endswith(".weight") or name.endswith(".bias"):
            entries.append((name, "F16", list(t.shape), (lambda t=t: t.to(torch.float16))))
        else:
            raise SystemExit(f"unexpected tensor {name} {tuple(t.shape)}")
    mean = f.get_tensor("vae.per_channel_statistics.mean-of-means").float()
    std = f.get_tensor("vae.per_channel_statistics.std-of-means").float()
    print("latents_mean[:4]", mean[:4].tolist(), "latents_std[:4]", std[:4].tolist())
    entries.append(("latents_mean", "F32", [128], lambda: mean.contiguous()))
    entries.append(("latents_std", "F32", [128], lambda: std.contiguous()))

    header, off = {}, 0
    for name, dt, shape, _ in entries:
        n = int(np.prod(shape)) * (2 if dt == "F16" else 4)
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    header["__metadata__"] = {"format": "pt", "source": "Lightricks/LTX-Video ltxv-2b-0.9.8-distilled vae.decoder (conv weights fp16 [cout, taps, cin])", "latent_channels": "128", "patch_size": "4"}
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(out_path, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)))
        fo.write(hb)
        for i, (name, dt, shape, fn) in enumerate(entries):
            t = fn()
            assert list(t.shape) == shape, (name, list(t.shape), shape)
            fo.write(t.view(torch.int16).numpy().tobytes() if dt == "F16" else t.numpy().astype("<f4").tobytes())
            if i % 40 == 0:
                print(f"{i}/{len(entries)} {name} {shape}", flush=True)
    print("wrote", out_path, f"{os.path.getsize(out_path) / 1e9:.2f} GB,", len(entries), "tensors")


if __name__ == "__main__":
    main()
