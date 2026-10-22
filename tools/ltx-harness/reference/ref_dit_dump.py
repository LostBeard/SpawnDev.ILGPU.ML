"""LTX-Video 2B 0.9.8 distilled reference: the DiT (LTXVideoTransformer3DModel) and the distilled Euler sampler loop, dumped as fixtures for the C# port (stages 3 and 4 of Plans/ltx-video-2b-2026-10-22.md).

Runs in WSL (venv ~/wan/venv: diffusers 0.41, torch 2.6) against the single-file checkpoint, in bf16 on the GPU (3.9 GB), exactly the dtype the reference pipeline runs in.
The schedule is the official first pass of configs/ltxv-2b-0.9.8-distilled.yaml: sigmas [1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725] then 0 (guidance 1: ONE forward per step), the model
timestep is sigma * 1000, the step is the flow-matching Euler update x += (sigma_next - sigma) * v (FlowMatchEulerDiscreteScheduler with dynamic shifting and the terminal shift OFF, so the sigmas are used as given).

  python ref_dit_dump.py CKPT.safetensors OUT_DIR T5_DIR [--size WxH] [--frames F] [--seed N] [--frame-rate 25] [--tag name] [--prompt 0]
Reads T5_DIR/t5_<prompt>.npz (ids, mask, context from ref_t5_dump.py) and writes OUT_DIR/dit_<tag>.npz with
  noise [128,Fz,h,w], sigmas, per step: latent_in [128,Fz,h,w], t, pred [128,Fz,h,w] (the velocity), latent_out;
  and for step 0 the intermediates: patch_embedding, text_embedding, time_embedding (the embedded timestep), time_projection (the 6-way adaLN vector), block_<i>, head.
All arrays float32. Latents are [128, Fz, H/32, W/32] with Fz = (frames - 1) / 8 + 1; tokens N = Fz * h * w, in (frame, row, column) order.
"""
import argparse
import os

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, LTXVideoTransformer3DModel

SIGMAS = [1.0, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("t5_dir")
    ap.add_argument("--size", default="480x320")
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--frame-rate", type=float, default=25.0)
    ap.add_argument("--prompt", default="0")
    ap.add_argument("--tag", default="t2v")
    ap.add_argument("--fp32", action="store_true", help="run the DiT in float32 (7.7 GB): the exact math on the same bf16 weights, to separate the reference's own bf16 rounding from a port error")
    a = ap.parse_args()
    dtype = torch.float32 if a.fp32 else torch.bfloat16
    os.makedirs(a.out_dir, exist_ok=True)
    W, H = (int(v) for v in a.size.split("x"))
    Fz, h, w = (a.frames - 1) // 8 + 1, H // 32, W // 32
    N = Fz * h * w
    print(f"size {W}x{H}, {a.frames} frames -> latent [128, {Fz}, {h}, {w}], {N} tokens")

    z = np.load(os.path.join(a.t5_dir, f"t5_{a.prompt}.npz"))
    ctx = torch.from_numpy(z["context"]).to("cuda", dtype)[None]                     # [1, 128, 4096], padding rows included
    mask = torch.from_numpy(z["mask"]).to("cuda")[None]                              # [1, 128]

    model = LTXVideoTransformer3DModel.from_single_file(a.ckpt, torch_dtype=dtype).to("cuda").eval().requires_grad_(False)
    print("dit loaded", sum(p.numel() for p in model.parameters()) / 1e9, "B params; heads", model.config.num_attention_heads, "head dim", model.config.attention_head_dim, "layers", model.config.num_layers)

    keep = {}
    capture = {"on": False}

    def hook(name, pick=None):
        def fn(_m, _i, out):
            if capture["on"]:
                o = out if pick is None else out[pick]
                keep[name] = o.detach().float().cpu().numpy()
        return fn
    model.proj_in.register_forward_hook(hook("patch_embedding"))
    model.caption_projection.register_forward_hook(hook("text_embedding"))
    model.time_embed.register_forward_hook(hook("time_projection", 0))
    model.time_embed.register_forward_hook(hook("time_embedding", 1))
    for i in (0, 1, 14, len(model.transformer_blocks) - 1):
        model.transformer_blocks[i].register_forward_hook(hook(f"block_{i}"))
    model.proj_out.register_forward_hook(hook("head"))

    g = torch.Generator(device="cuda")
    g.manual_seed(a.seed)
    noise = torch.randn(1, 128, Fz, h, w, dtype=torch.float32, device="cuda", generator=g)
    out = {"noise": noise[0].cpu().numpy()}

    def pack(x):        # [1,128,Fz,h,w] -> [1,N,128]
        return x.permute(0, 2, 3, 4, 1).reshape(1, N, 128)

    def unpack(x):      # [1,N,128] -> [1,128,Fz,h,w]
        return x.reshape(1, Fz, h, w, 128).permute(0, 4, 1, 2, 3)

    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False, shift_terminal=None)
    sched.set_timesteps(sigmas=SIGMAS, device="cuda")
    out["sigmas"] = sched.sigmas.float().cpu().numpy()
    out["timesteps"] = sched.timesteps.float().cpu().numpy()
    print("sigmas", out["sigmas"], "timesteps", out["timesteps"])

    rope_scale = (8 / a.frame_rate, 32, 32)
    latent = pack(noise)
    cos, sin = model.rope(latent.to(dtype), Fz, h, w, rope_scale)                      # [1, N, 2048] float32 each, the table the blocks use
    out["rope_cos"] = cos[0].float().cpu().numpy()
    out["rope_sin"] = sin[0].float().cpu().numpy()
    with torch.no_grad():
        for step, t in enumerate(sched.timesteps):
            capture["on"] = step == 0
            v = model(hidden_states=latent.to(dtype), encoder_hidden_states=ctx, timestep=t.expand(1), encoder_attention_mask=mask,
                      num_frames=Fz, height=h, width=w, rope_interpolation_scale=rope_scale, return_dict=False)[0].float()
            capture["on"] = False
            nxt = sched.step(v, t, latent, return_dict=False)[0]
            out[f"step{step}_latent_in"] = unpack(latent)[0].cpu().numpy()
            out[f"step{step}_t"] = np.array([float(t)], dtype=np.float32)
            out[f"step{step}_pred"] = unpack(v)[0].cpu().numpy()
            out[f"step{step}_latent_out"] = unpack(nxt)[0].cpu().numpy()
            print(f"step {step}: t {float(t):.2f}, v absmax {v.abs().max():.3f}, out absmax {nxt.abs().max():.3f}")
            latent = nxt
    for k, v in keep.items():
        out[f"step0_{k}"] = v
    path = os.path.join(a.out_dir, f"dit_{a.tag}.npz")
    np.savez(path, **out)
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.1f} MB; intermediates:", ", ".join(f"{k} {v.shape}" for k, v in keep.items()))


if __name__ == "__main__":
    main()
