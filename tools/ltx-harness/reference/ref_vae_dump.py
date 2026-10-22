"""LTX-Video 0.9.8 VAE reference: the DECODER (AutoencoderKLLTXVideo with timestep conditioning), dumped as fixtures for the C# port (stage 5 of Plans/ltx-video-2b-2026-10-22.md).

Runs in WSL (venv ~/wan/venv) on the CPU in float32 (exact math; a small latent keeps it fast) so the GPU stays free for the other sessions.
  python ref_vae_dump.py CKPT.safetensors OUT_DIR [--latent 3x4x6] [--timestep 0.05] [--seed 7] [--tag small]
Writes OUT_DIR/vae_<tag>.npz: latent [128,Fz,h,w] (raw model-space latent, what the DiT predicts), timestep, the denormalised latent the decoder receives, and taps
  conv_in, mid_block, up_0..up_3 (post-block outputs), norm_out, conv_out, video [3, F, H, W] (the decoded frames, nominally in [-1, 1]).
The decoder config is printed: it is what the C# `LtxVae` must implement (block widths, layers per block, residual upsamplers).
"""
import argparse
import json
import os

import numpy as np
import torch
from diffusers import AutoencoderKLLTXVideo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("--latent", default="3x4x6")
    ap.add_argument("--timestep", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="small")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    Fz, h, w = (int(v) for v in a.latent.split("x"))

    vae = AutoencoderKLLTXVideo.from_single_file(a.ckpt, torch_dtype=torch.float32).eval().requires_grad_(False)
    cfg = {k: v for k, v in vae.config.items() if not k.startswith("_")}
    print("vae config:", json.dumps(cfg))
    print("latents_mean[:4]", vae.latents_mean[:4].tolist(), "latents_std[:4]", vae.latents_std[:4].tolist(), "scaling", vae.config.scaling_factor)
    print("decoder:", [type(m).__name__ for m in vae.decoder.up_blocks], "mid", len(vae.decoder.mid_block.resnets), "resnets;", "up resnets", [len(b.resnets) for b in vae.decoder.up_blocks])

    g = torch.Generator().manual_seed(a.seed)
    latent = torch.randn(1, 128, Fz, h, w, generator=g)
    # what LTXPipeline does before the decode: denormalise (x * std / scaling + mean); the pipeline then adds noise by decode_noise_scale (left out here)
    den = latent * vae.latents_std.view(1, -1, 1, 1, 1) / vae.config.scaling_factor + vae.latents_mean.view(1, -1, 1, 1, 1)
    out = {"latent": latent[0].numpy(), "denorm": den[0].numpy(), "timestep": np.array([a.timestep], dtype=np.float32)}

    keep = {}

    def hook(name):
        def fn(_m, _i, o):
            keep[name] = o.detach().float().numpy()[0]
        return fn
    dec = vae.decoder
    dec.conv_in.register_forward_hook(hook("conv_in"))
    dec.mid_block.register_forward_hook(hook("mid_block"))
    for i, b in enumerate(dec.up_blocks):
        b.register_forward_hook(hook(f"up_{i}"))
    dec.norm_out.register_forward_hook(hook("norm_out"))
    dec.conv_out.register_forward_hook(hook("conv_out"))

    with torch.no_grad():
        video = vae.decode(den, torch.tensor([a.timestep]), return_dict=False)[0]
    out["video"] = video[0].numpy()
    for k, v in keep.items():
        out[k] = v
    path = os.path.join(a.out_dir, f"vae_{a.tag}.npz")
    np.savez(path, **out)
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.1f} MB; video {tuple(video.shape)}; taps:", ", ".join(f"{k} {v.shape}" for k, v in keep.items()))


if __name__ == "__main__":
    main()
