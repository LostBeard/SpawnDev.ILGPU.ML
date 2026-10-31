"""Export the LTX-Video 0.9.8 VAE ENCODER (the `vae.encoder.*` and `vae.per_channel_statistics.*` tensors of the single-file checkpoint) to one SafeTensors file in the layout the C#
`LtxVae.EncodeAsync` reads (image-to-video and video conditioning: the condition frames are encoded to latents).

The same conventions as export_vae_f16.py (the decoder): convolution weights [cout, cin, kt, kh, kw] become fp16 [cout, taps, cin] (taps = 27, row-major over the kernel, channel last),
biases fp16, the redundant ".conv" level of the source names dropped. The source names are kept otherwise (the original LTX layout, not diffusers'):
  conv_in.{weight,bias}                          48 -> 128 (the 4x4 pixel patch of 3 channels)
  down_blocks.{0,2,4,6}.res_blocks.J.{conv1,conv2}.{weight,bias}   resnets at 128, 256, 512, 1024 channels (4, 6, 6, 2 of them)
  down_blocks.{1,3,5,7}.conv.{weight,bias}       the residual space-to-depth downsamplers: spatial 2x2, temporal 2, then 2x2x2 twice
  down_blocks.8.res_blocks.J.{conv1,conv2}.{weight,bias}           the mid block: 2 resnets at 2048 channels
  conv_out.{weight,bias}                         2048 -> 129 (128 means + one log-variance channel)
  latents_mean, latents_std                      <- vae.per_channel_statistics.mean-of-means / std-of-means (fp32 [128]; encode: (z - mean) / std)

  python export_vae_encoder_f16.py SRC.safetensors OUT.safetensors
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
    for key in sorted(k for k in f.keys() if k.startswith("vae.encoder.")):
        name = key[len("vae.encoder."):].replace(".conv.weight", ".weight").replace(".conv.bias", ".bias")
        t = f.get_tensor(key)
        if name.endswith(".weight") and t.dim() == 5:
            co, ci = t.shape[0], t.shape[1]
            entries.append((name, "F16", [co, 27, ci], (lambda t=t, co=co, ci=ci: t.reshape(co, ci, 27).permute(0, 2, 1).contiguous().to(torch.float16))))
        elif name.endswith(".bias"):
            entries.append((name, "F16", list(t.shape), (lambda t=t: t.to(torch.float16))))
        else:
            raise SystemExit(f"unexpected tensor {name} {tuple(t.shape)}")
    mean = f.get_tensor("vae.per_channel_statistics.mean-of-means").float()
    std = f.get_tensor("vae.per_channel_statistics.std-of-means").float()
    entries.append(("latents_mean", "F32", [128], lambda: mean.contiguous()))
    entries.append(("latents_std", "F32", [128], lambda: std.contiguous()))

    header, off = {}, 0
    for name, dt, shape, _ in entries:
        n = int(np.prod(shape)) * (2 if dt == "F16" else 4)
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    header["__metadata__"] = {"format": "pt", "source": "Lightricks/LTX-Video ltxv-2b-0.9.8-distilled vae.encoder (conv weights fp16 [cout, taps, cin])", "latent_channels": "128", "patch_size": "4"}
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(out_path, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)))
        fo.write(hb)
        for i, (name, dt, shape, fn) in enumerate(entries):
            t = fn()
            assert list(t.shape) == shape, (name, list(t.shape), shape)
            fo.write(t.view(torch.int16).numpy().tobytes() if dt == "F16" else t.numpy().astype("<f4").tobytes())
            if i % 20 == 0:
                print(f"{i}/{len(entries)} {name} {shape}", flush=True)
    print("wrote", out_path, f"{os.path.getsize(out_path) / 1e9:.2f} GB,", len(entries), "tensors")


if __name__ == "__main__":
    main()
