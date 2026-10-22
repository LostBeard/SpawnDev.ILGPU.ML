"""Wan 2.2 reference: how sensitive is the 4-step sampling trajectory to a small perturbation of the starting latent?

The C# sampler loop (Wan_SamplerLoop_MatchesReference) deviates from the reference by 1e-3 after the first step and ~1e-1 after the fourth. Is that a bug or the sampler's own
sensitivity? This reruns the REFERENCE trajectory from the same noise perturbed by a relative eps (Gaussian, scaled to the latent's RMS) and prints the divergence per step,
for the text-to-video and the clean-first-frame modes. Needs the t5_*.npz and dit_*.npz fixtures (z0 for the second mode) of ref_t5_dump.py / ref_dit_dump.py.

  python ref_sensitivity.py DUMP_DIR [--eps 1.3e-3]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_dit_dump import build_dit  # noqa: E402
sys.path.insert(0, os.path.expanduser("~/wan/Wan2.2"))
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402


@torch.no_grad()
def trajectory(model, latent0, z0, ctx, ctx_null, steps, first, shift=5.0, guide=5.0):
    Fz, h, w = latent0.shape[1:]
    seq = Fz * (h // 2) * (w // 2)
    mask2 = torch.ones_like(latent0)
    if first:
        mask2[:, 0] = 0
    sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    sched.set_timesteps(steps, device="cuda", shift=shift)
    latent = (1.0 - mask2) * z0 + mask2 * latent0 if first else latent0
    out = []
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for t in sched.timesteps:
            ts = (mask2[0][:, ::2, ::2] * t).flatten().unsqueeze(0)
            nc = model([latent], t=ts, context=[ctx], seq_len=seq)[0]
            nu = model([latent], t=ts, context=[ctx_null], seq_len=seq)[0]
            nxt = sched.step((nu + guide * (nc - nu)).unsqueeze(0), t, latent.unsqueeze(0), return_dict=False)[0].squeeze(0)
            if first:
                nxt = (1.0 - mask2) * z0 + mask2 * nxt
            out.append(nxt.float().cpu().numpy())
            latent = nxt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--eps", type=float, default=1.3e-3)
    a = ap.parse_args()
    model = build_dit()
    ctx = torch.from_numpy(np.load(os.path.join(a.dump, "t5_0.npz"))["context"]).to("cuda", torch.bfloat16)
    ctx_null = torch.from_numpy(np.load(os.path.join(a.dump, "t5_neg.npz"))["context"]).to("cuda", torch.bfloat16)
    g = torch.Generator(device="cuda"); g.manual_seed(99)
    for tag, first in (("t2v", False), ("i2v", True)):
        fx = np.load(os.path.join(a.dump, f"dit_{tag}_640x352_f33.npz"))
        noise = torch.from_numpy(fx["noise"]).to("cuda")
        z0 = torch.from_numpy(fx["z0"]).to("cuda") if first else None
        base = trajectory(model, noise, z0, ctx, ctx_null, 4, first)
        pert = noise + a.eps * noise.pow(2).mean().sqrt() * torch.randn(noise.shape, device="cuda", generator=g)
        alt = trajectory(model, pert, z0, ctx, ctx_null, 4, first)
        rel = [float(np.sqrt(((b - p) ** 2).sum() / (b ** 2).sum())) for b, p in zip(base, alt)]
        print(f"{tag}: start perturbed by relative {a.eps:.1e} -> per-step relRMS divergence of the reference from itself: " + ", ".join(f"{r:.2e}" for r in rel))


if __name__ == "__main__":
    main()
