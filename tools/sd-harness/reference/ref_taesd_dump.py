"""TAESD / TAESDXL reference (diffusers AutoencoderTiny, float32 on the CPU) for the C# SdTaesd port.
  python ref_taesd_dump.py DIFFUSERS_DIR OUT_DIR --scale 0.18215 [--latent 64x64] [--tag taesd] [--seed 5]
Writes OUT_DIR/taesd_<tag>.npz: dec_in [4, h, w] = the UNSCALED latent the C# API takes (the tiny model sees dec_in * scale), image [3, 8h, 8w] (decoder output, [-1, 1]);
enc_image [3, H, W] in [-1, 1], enc_out [4, H/8, W/8] = the encoder output divided by scale (unscaled, as the C# API returns it).
"""
import argparse
import os

import numpy as np
import torch
from diffusers import AutoencoderTiny


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--latent", default="64x64")
    ap.add_argument("--tag", default="taesd")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--original", nargs=2, metavar=("DECODER", "ENCODER"),
                    help="load these original taesd*_decoder / _encoder.safetensors into the diffusers architecture (the published TAESDXL decoder differs from the diffusers one)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    h, w = (int(v) for v in a.latent.split("x"))
    m = AutoencoderTiny.from_pretrained(a.model_dir, torch_dtype=torch.float32).eval().requires_grad_(False)
    if a.original:
        from safetensors.torch import load_file
        sd = {}
        for k, v in load_file(a.original[0]).items():         # DecoderTiny has no Clamp layer: original layer i = diffusers decoder.layers.(i - 1)
            i, rest = k.split(".", 1)
            sd[f"decoder.layers.{int(i) - 1}.{rest}"] = v
        for k, v in load_file(a.original[1]).items():
            sd[f"encoder.layers.{k}"] = v
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise SystemExit(f"original weights do not fit: missing {missing[:5]}, unexpected {unexpected[:5]}")
        print("loaded the original weights", a.original)
    print("config:", dict(m.config))
    g = torch.Generator().manual_seed(a.seed)
    with torch.no_grad():
        z_scaled = torch.randn(1, 4, h, w, generator=g)
        img = m.decode(z_scaled).sample
        H, W = 8 * h, 8 * w
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
        im = torch.stack([torch.sin(3 * xx + yy), torch.cos(2 * yy - xx), xx * yy]) * 0.8 + torch.rand(3, H, W, generator=g) * 0.2 - 0.1
        enc = m.encode(im[None]).latents
    out = {"dec_in": (z_scaled[0] / a.scale).numpy(), "image": img[0].numpy(), "enc_image": im.numpy(), "enc_out": (enc[0] / a.scale).numpy()}
    path = os.path.join(a.out_dir, f"taesd_{a.tag}.npz")
    np.savez(path, **out)
    print("wrote", path, {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
