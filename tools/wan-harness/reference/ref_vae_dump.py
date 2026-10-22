"""Wan 2.2 reference: the VAE decoder (and encoder) on small inputs, dumped as fixtures for the C# port (stage 5 of Plans/wan22-ti2v-5b-2026-10-22.md).

  python ref_vae_dump.py OUT_DIR [--frames-z 3] [--h 8] [--w 12] [--seed 7]
Decode: a random normalised latent z [48, Fz, h, w] -> video [3, 4*Fz-3, 16h, 16w] (float32, clamped to [-1, 1]) through the reference's chunked decode (one latent frame per
chunk, causal convolution caches). Taps (forward hooks, concatenated over the chunks along time): conv2, decoder.conv1, decoder.middle, decoder.upsamples.0..3, decoder.head.
Encode: a random image [3, 1, 16h, 16w] in [-1, 1] -> mu [48, 1, h, w] (normalised) with the encoder taps.
The VAE runs in float32 (the reference's default), on the GPU (2.8 GB).
"""
import argparse
import os
import sys

import numpy as np
import torch

WAN = os.path.expanduser("~/wan/Wan2.2")
CKPT = os.path.expanduser("~/wan/ckpt")
sys.path.insert(0, WAN)

from wan.modules.vae2_2 import Wan2_2_VAE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--frames-z", type=int, default=3)
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--w", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    vae = Wan2_2_VAE(vae_pth=os.path.join(CKPT, "Wan2.2_VAE.pth"), device="cuda")
    model = vae.model

    taps = {}
    def hook(name):
        def fn(_m, _i, out):
            taps.setdefault(name, []).append(out.detach().float().cpu())
        return fn
    model.conv2.register_forward_hook(hook("dec_conv2"))
    model.decoder.conv1.register_forward_hook(hook("dec_conv1"))
    model.decoder.middle[2].register_forward_hook(hook("dec_middle"))       # the Sequential is iterated layer by layer, its own hook never fires
    for i, m in enumerate(model.decoder.upsamples):
        m.register_forward_hook(hook(f"dec_up{i}"))
    model.decoder.head[2].register_forward_hook(hook("dec_head"))

    g = torch.Generator(device="cuda"); g.manual_seed(a.seed)
    z = torch.randn(48, a.frames_z, a.h, a.w, generator=g, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        video = vae.decode([z])[0]
    out = {"z": z.cpu().numpy(), "video": video.float().cpu().numpy()}
    for k, v in taps.items():
        out[k] = torch.cat(v, dim=2).numpy()                    # chunks along time: [1, C, T, H, W]
    print("decode: z", tuple(z.shape), "-> video", tuple(video.shape), f"range {video.min():.3f}..{video.max():.3f}")
    taps.clear()

    # encode a random image (one frame, the image-to-video conditioning path)
    model.encoder.conv1.register_forward_hook(hook("enc_conv1"))
    for i, m in enumerate(model.encoder.downsamples):
        m.register_forward_hook(hook(f"enc_down{i}"))
    model.encoder.middle[2].register_forward_hook(hook("enc_middle"))
    model.encoder.head[2].register_forward_hook(hook("enc_head"))
    img = (torch.rand(3, 1, 16 * a.h, 16 * a.w, generator=g, device="cuda") * 2 - 1).float()
    with torch.no_grad():
        mu = vae.encode([img])[0]
    out["img"] = img.cpu().numpy(); out["mu"] = mu.float().cpu().numpy()
    for k, v in taps.items():
        out[k] = torch.cat(v, dim=2).numpy()
    print("encode: img", tuple(img.shape), "-> mu", tuple(mu.shape), f"mean {mu.mean():.4f} std {mu.std():.4f}")
    path = os.path.join(a.out_dir, "vae.npz")
    np.savez(path, **out)
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.1f} MB:", ", ".join(f"{k} {tuple(v.shape)}" for k, v in out.items() if k not in ("z", "video", "img", "mu")))


if __name__ == "__main__":
    main()
