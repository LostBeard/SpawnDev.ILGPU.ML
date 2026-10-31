"""Wan 2.2 reference: the VAE decoder with diffusers' TILED decoding (AutoencoderKLWan.enable_tiling: spatial tiles of 256 px with a 192 px stride, each decoded through the
whole causal decoder with fresh caches, blended over 64 px; the low-memory path for 720p) dumped as a fixture for the C# `WanVae.DecodeTiledAsync`.

diffusers' AutoencoderKLWan (Wan-AI/Wan2.2-TI2V-5B-Diffusers, vae) is first checked against the official Wan2_2_VAE's untiled decode of the same latent (they must agree), then
decoded tiled. Runs on the GPU in float32 with TF32 off (cuDNN's default TF32 convolutions are not float32).
  python ref_vae_tiled_dump.py OUT_DIR [--frames-z 3] [--h 20] [--w 28] [--seed 9]
Writes OUT_DIR/vae_tiled.npz: z [48, Fz, h, w] (normalised latent), video_tiled [3, 4 Fz - 3, 16 h, 16 w] (diffusers tiled, clamped), video_whole (diffusers untiled).
"""
import argparse
import os
import sys

import numpy as np
import torch

WAN = os.path.expanduser("~/wan/Wan2.2")
CKPT = os.path.expanduser("~/wan/ckpt")
sys.path.insert(0, WAN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--frames-z", type=int, default=3)
    ap.add_argument("--h", type=int, default=20)
    ap.add_argument("--w", type=int, default=28)
    ap.add_argument("--seed", type=int, default=9)
    a = ap.parse_args()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(a.out_dir, exist_ok=True)
    from diffusers import AutoencoderKLWan
    from wan.modules.vae2_2 import Wan2_2_VAE

    g = torch.Generator(device="cuda"); g.manual_seed(a.seed)
    z = torch.randn(48, a.frames_z, a.h, a.w, generator=g, device="cuda", dtype=torch.float32)

    official = Wan2_2_VAE(vae_pth=os.path.join(CKPT, "Wan2.2_VAE.pth"), device="cuda")
    with torch.no_grad():
        v_off = official.decode([z])[0].float()
    mean, std = official.scale[0].float(), 1.0 / official.scale[1].float()   # the official normalisation: scale = [mean, 1 / std], decode = z / (1 / std) + mean
    del official
    torch.cuda.empty_cache()

    # diffusers' own Wan 2.2 VAE (its single-file converter only knows the 2.1 layout); checked against the official decode below
    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae", torch_dtype=torch.float32).to("cuda").eval()
    lm = torch.tensor(vae.config.latents_mean, device="cuda").view(1, -1, 1, 1, 1)
    ls = torch.tensor(vae.config.latents_std, device="cuda").view(1, -1, 1, 1, 1)
    zz = z[None] * ls + lm                                                     # diffusers' pipelines denormalise before decode()
    print("latent statistics: official vs diffusers config, max diff", float((mean.view(-1) - lm.view(-1)).abs().max()), float((std.view(-1) - ls.view(-1)).abs().max()))
    with torch.no_grad():
        v_whole = vae.decode(zz, return_dict=False)[0][0].float()
        vae.enable_tiling()
        v_tiled = vae.decode(zz, return_dict=False)[0][0].float()
    rel = lambda p, q: float(((p - q) ** 2).mean().sqrt() / (q ** 2).mean().sqrt())
    print(f"diffusers untiled vs the official decode: {rel(v_whole, v_off):.3e}; tiled vs untiled: {rel(v_tiled, v_whole):.3e}")
    out = {"z": z.cpu().numpy(), "video_tiled": v_tiled.cpu().numpy(), "video_whole": v_whole.cpu().numpy(), "video_official": v_off.cpu().numpy()}
    path = os.path.join(a.out_dir, "vae_tiled.npz")
    np.savez(path, **{k: np.ascontiguousarray(v) for k, v in out.items()})
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.1f} MB; video {tuple(v_tiled.shape)}")


if __name__ == "__main__":
    main()
