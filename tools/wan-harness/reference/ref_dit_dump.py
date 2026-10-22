"""Wan 2.2 TI2V-5B reference: the DiT (WanModel) and the UniPC sampler loop, dumped as fixtures for the C# port (stages 3 and 4 of Plans/wan22-ti2v-5b-2026-10-22.md).

Runs in WSL against ~/wan/Wan2.2 and ~/wan/ckpt (see ref_t5_dump.py for the memory constraints). The DiT is built on the meta device and filled
from the fp32 safetensors shards tensor by tensor into bf16 on the GPU (10 GB of the 12 GB card), exactly the dtype the reference runs in
(`convert_model_dtype`: every parameter bf16, autocast bf16, time embedding in fp32).

  python ref_dit_dump.py OUT_DIR T5_DIR [--size WxH] [--frames F] [--steps S] [--seed N] [--cond-first-frame] [--tag name] [--prompt 0]
Reads T5_DIR/t5_<prompt>.npz (the positive context) and t5_neg.npz (the default negative prompt), writes OUT_DIR/dit_<tag>.npz with
  noise [48,Fz,h,w], timesteps, sigmas, z0 (the conditioning first frame, with --cond-first-frame), per step: latent_in, t, noise_cond, noise_uncond, latent_out;
  and for step 0 the intermediates: patch_embedding, text_embedding, time_embedding, time_projection, block_<i>, head.
All arrays float32. Latents are [48, Fz, H/16, W/16], tokens N = Fz * H/32 * W/32.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

WAN = os.path.expanduser("~/wan/Wan2.2")
CKPT = os.path.expanduser("~/wan/ckpt")
sys.path.insert(0, WAN)

from wan.modules.model import WanModel, rope_params  # noqa: E402
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402


def build_dit():
    cfg = json.load(open(os.path.join(CKPT, "config.json")))
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = WanModel(model_type=cfg["model_type"], patch_size=(1, 2, 2), text_len=cfg["text_len"], in_dim=cfg["in_dim"], dim=cfg["dim"],
                         ffn_dim=cfg["ffn_dim"], freq_dim=cfg["freq_dim"], text_dim=4096, out_dim=cfg["out_dim"], num_heads=cfg["num_heads"],
                         num_layers=cfg["num_layers"], window_size=(-1, -1), qk_norm=True, cross_attn_norm=True, eps=cfg["eps"])
    torch.set_default_dtype(torch.float32)
    model.to_empty(device="cuda")
    d = cfg["dim"] // cfg["num_heads"]
    model.freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1).to("cuda")
    index = json.load(open(os.path.join(CKPT, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
    params = dict(model.named_parameters())
    loaded = set()
    for shard in sorted(set(index.values())):
        with safe_open(os.path.join(CKPT, shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                params[name].data.copy_(f.get_tensor(name))     # fp32 on the CPU -> bf16 on the GPU, one tensor at a time
                loaded.add(name)
    missing = [n for n in params if n not in loaded]
    print("dit loaded", len(loaded), "tensors; missing", missing[:5], "; params", sum(p.numel() for p in params.values()) / 1e9, "B")
    return model.eval().requires_grad_(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("t5_dir")
    ap.add_argument("--size", default="640x352")
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cond-first-frame", action="store_true")
    ap.add_argument("--prompt", default="0")
    ap.add_argument("--tag", default="t2v")
    ap.add_argument("--guide", type=float, default=5.0)
    ap.add_argument("--shift", type=float, default=5.0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    W, H = (int(v) for v in a.size.split("x"))
    Fz, h, w = (a.frames - 1) // 4 + 1, H // 16, W // 16
    seq_len = Fz * (h // 2) * (w // 2)
    print(f"size {W}x{H}, {a.frames} frames -> latent [48, {Fz}, {h}, {w}], {seq_len} tokens")

    ctx = torch.from_numpy(np.load(os.path.join(a.t5_dir, f"t5_{a.prompt}.npz"))["context"]).to("cuda", torch.bfloat16)
    ctx_null = torch.from_numpy(np.load(os.path.join(a.t5_dir, "t5_neg.npz"))["context"]).to("cuda", torch.bfloat16)

    model = build_dit()
    keep = {}
    capture = {"on": False}

    def hook(name):
        def fn(_m, _i, out):
            if capture["on"]:
                keep[name] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu().numpy()
        return fn
    model.patch_embedding.register_forward_hook(hook("patch_embedding"))
    model.text_embedding.register_forward_hook(hook("text_embedding"))
    model.time_embedding.register_forward_hook(hook("time_embedding"))
    model.time_projection.register_forward_hook(hook("time_projection"))
    for i in (0, 1, 14, 29):
        model.blocks[i].register_forward_hook(hook(f"block_{i}"))
    model.head.register_forward_hook(hook("head"))

    g = torch.Generator(device="cuda")
    g.manual_seed(a.seed)
    noise = torch.randn(48, Fz, h, w, dtype=torch.float32, device="cuda", generator=g)
    out = {"noise": noise.cpu().numpy()}

    sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    sched.set_timesteps(a.steps, device="cuda", shift=a.shift)
    timesteps = sched.timesteps
    out["timesteps"] = timesteps.float().cpu().numpy()
    out["sigmas"] = sched.sigmas.float().cpu().numpy()

    # the conditioning first frame (TI2V): mask2 is 0 on latent frame 0 -> that frame is the (here random) image latent z0 and its tokens get timestep 0
    mask2 = torch.ones_like(noise)
    z0 = None
    latent = noise
    if a.cond_first_frame:
        mask2[:, 0] = 0
        z0 = torch.randn(48, Fz, h, w, dtype=torch.float32, device="cuda", generator=g)
        out["z0"] = z0.cpu().numpy()
        latent = (1.0 - mask2) * z0 + mask2 * latent

    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        for step, t in enumerate(timesteps):
            ts = (mask2[0][:, ::2, ::2] * t).flatten()
            timestep = ts.unsqueeze(0)
            capture["on"] = step == 0
            nc = model([latent], t=timestep, context=[ctx], seq_len=seq_len)[0]
            capture["on"] = False
            nu = model([latent], t=timestep, context=[ctx_null], seq_len=seq_len)[0]
            noise_pred = nu + a.guide * (nc - nu)
            nxt = sched.step(noise_pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False, generator=g)[0].squeeze(0)
            if a.cond_first_frame:
                nxt = (1.0 - mask2) * z0 + mask2 * nxt
            out[f"step{step}_latent_in"] = latent.float().cpu().numpy()
            out[f"step{step}_t"] = np.array([float(t)], dtype=np.float32)
            out[f"step{step}_noise_cond"] = nc.float().cpu().numpy()
            out[f"step{step}_noise_uncond"] = nu.float().cpu().numpy()
            out[f"step{step}_latent_out"] = nxt.float().cpu().numpy()
            print(f"step {step}: t {float(t):.3f}, cond absmax {nc.abs().max():.3f}, uncond absmax {nu.abs().max():.3f}, out absmax {nxt.abs().max():.3f}")
            latent = nxt
    for k, v in keep.items():
        out[f"step0_{k}"] = v
    path = os.path.join(a.out_dir, f"dit_{a.tag}.npz")
    np.savez(path, **out)
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.1f} MB; intermediates:", ", ".join(f"{k} {v.shape}" for k, v in keep.items()))


if __name__ == "__main__":
    main()
