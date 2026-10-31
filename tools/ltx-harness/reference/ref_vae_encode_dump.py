"""LTX-Video 0.9.8 VAE ENCODER reference (image-to-video / video conditioning), dumped as fixtures for the C# `LtxVae.EncodeAsync`.

Runs on the CPU in float32 (exact math; small inputs keep it fast) so the GPU stays free for the other sessions.
  python ref_vae_encode_dump.py CKPT.safetensors OUT_DIR [--size 3x192x256] [--seed 11] [--tag img]
--size is FRAMESxHEIGHTxWIDTH (frames 8n + 1, sides multiples of 32). Writes OUT_DIR/vae_enc_<tag>.npz:
  video [3, F, H, W] the input in [-1, 1] (smooth random content),
  taps conv_in, down_0..down_3 (after each diffusers down block: resnets + residual downsampler), mid_block, conv_out [129, ...] (channel-first, batch dropped),
  mean [128, Fz, h, w] and logvar [Fz, h, w] (the posterior: logvar clamped to [-30, 20], one channel shared by all 128),
  latent [128, Fz, h, w] = (mean - latents_mean) / latents_std (what LTXConditionPipeline puts in the first latent frames, before its sampling noise).
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLLTXVideo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("--size", default="1x192x256")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tag", default="img")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    nf, H, W = (int(v) for v in a.size.split("x"))

    vae = AutoencoderKLLTXVideo.from_single_file(a.ckpt, torch_dtype=torch.float32).eval().requires_grad_(False)
    g = torch.Generator().manual_seed(a.seed)
    coarse = torch.randn(1, 3, nf, H // 16, W // 16, generator=g)
    video = torch.tanh(1.5 * F.interpolate(coarse, size=(nf, H, W), mode="trilinear", align_corners=False))

    keep = {}

    def hook(name):
        def fn(_m, _i, o):
            keep[name] = o.detach().float().numpy()[0]
        return fn
    enc = vae.encoder
    enc.conv_in.register_forward_hook(hook("conv_in"))
    for i, b in enumerate(enc.down_blocks):
        b.register_forward_hook(hook(f"down_{i}"))
    enc.mid_block.register_forward_hook(hook("mid_block"))
    enc.conv_out.register_forward_hook(hook("conv_out"))

    with torch.no_grad():
        dist = vae.encode(video).latent_dist
    mean, logvar = dist.mean, dist.logvar
    latent = (mean - vae.latents_mean.view(1, -1, 1, 1, 1)) * vae.config.scaling_factor / vae.latents_std.view(1, -1, 1, 1, 1)
    out = {"video": video[0].numpy(), "mean": mean[0].numpy(), "logvar": logvar[0, 0].numpy(), "latent": latent[0].numpy()}
    out.update(keep)
    path = os.path.join(a.out_dir, f"vae_enc_{a.tag}.npz")
    np.savez(path, **out)
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.1f} MB:", ", ".join(f"{k} {v.shape}" for k, v in out.items()))
    print("latent std", float(latent.std()), "logvar range", float(logvar.min()), float(logvar.max()))


if __name__ == "__main__":
    main()
