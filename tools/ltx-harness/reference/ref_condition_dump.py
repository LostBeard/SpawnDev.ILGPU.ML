"""LTX-Video 2B 0.9.8 distilled reference: CONDITIONED generation (diffusers' LTXConditionPipeline: image to video, keyframes, clips), dumped as fixtures for the C# port.

Runs the REAL pipeline (not a re-implementation) on the GPU and records every DiT call with forward hooks. The prompt is the T5 output of ref_t5_dump.py (no text encoder is loaded);
guidance 1 (the distilled model: one forward per step); the first-pass sigmas of configs/ltxv-2b-0.9.8-distilled.yaml as timesteps; denoise strength 1.
--conditions lists the conditions, comma separated, each KIND:FRAME_INDEX:STRENGTH with KIND "img" (one picture) or "vidN" (a clip of N = 8n + 1 frames); the content is smooth
random pictures (seeded per condition). Default "img:0:1.0" = image to video.
The random draws come from a CPU generator seeded --seed, in the pipeline's order, and are re-drawn here identically so the C# side can use the SAME noise:
  noise [128, Fz, h, w] (the starting latent); per condition i: posterior<i> [128, its latent frames, h, w] (the VAE posterior sample), and for a condition at a later frame
  condnoise<i> [128, its kept latent frames (at most 2), h, w] (the noise it is blended with); then per step cond_noise<s> [tokens, 128] (the image-conditioning noise).

  python ref_condition_dump.py CKPT.safetensors OUT_DIR T5_DIR [--size 480x320] [--frames 25] [--seed 99] [--conditions img:0:1.0] [--cond-noise 0.15] [--frame-rate 25] [--fp32] [--tag i2v]
Writes OUT_DIR/cond_<tag>.npz: cond<i>_pixels [3, F, H, W] (what the VAE encodes, in [-1, 1]), the draws above, sigmas, and per step s: latent_in<s> [tokens, 128] (the DiT input:
the extra tokens of later-frame conditions first, then the video's in (frame, row, column) order), t<s> [tokens] (per-token timestep), pred<s> [tokens, 128]; coords [3, tokens]
(the video_coords the transformer gets, time already divided by the frame rate); final [128, Fz, h, w] (the denoised latent, output_type "latent", before the decode);
cond<i>_latent_mean (the normalised posterior mean). For the default single picture also image / posterior / cond_latent_mean (the names the image-to-video gate reads).
float32 on the GPU needs TF32 OFF (set below): cuDNN's default TF32 convolutions put 8e-4 of error into the VAE encoding.
"""
import argparse
import os

import numpy as np
import torch
from diffusers import AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXConditionPipeline, LTXVideoTransformer3DModel
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from PIL import Image

SIGMAS = [1.0, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250]


def smooth_frames(n, H, W, seed):
    g = torch.Generator().manual_seed(seed)
    coarse = torch.rand(1, 3, max(2, (n + 7) // 8), H // 32, W // 32, generator=g)
    up = torch.nn.functional.interpolate(coarse, size=(n, H, W), mode="trilinear", align_corners=False)[0].clamp(0, 1)      # [3, n, H, W]
    return [Image.fromarray((up[:, f].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)) for f in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("t5_dir")
    ap.add_argument("--size", default="480x320")
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--conditions", default="img:0:1.0")
    ap.add_argument("--cond-noise", type=float, default=0.15)
    ap.add_argument("--frame-rate", type=float, default=25.0)
    ap.add_argument("--prompt", default="0")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--encoder-f16", action="store_true", help="round the VAE encoder's weights to fp16 (the hosted export's precision), math still float32: the reference's own sensitivity to it")
    ap.add_argument("--tag", default="i2v")
    a = ap.parse_args()
    dtype = torch.float32 if a.fp32 else torch.bfloat16
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(a.out_dir, exist_ok=True)
    W, H = (int(v) for v in a.size.split("x"))
    Fz, h, w = (a.frames - 1) // 8 + 1, H // 32, W // 32
    hw = h * w

    specs = []
    for i, c in enumerate(a.conditions.split(",")):
        kind, fi, s = c.split(":")
        n = 1 if kind == "img" else int(kind[3:])
        frames = smooth_frames(n, H, W, a.seed + 1 + i)
        specs.append((kind, int(fi), float(s), frames))
    conditions = [LTXVideoCondition(image=fr[0], frame_index=fi, strength=s) if kind == "img" else LTXVideoCondition(video=fr, frame_index=fi, strength=s)
                  for kind, fi, s, fr in specs]

    z = np.load(os.path.join(a.t5_dir, f"t5_{a.prompt}.npz"))
    ctx = torch.from_numpy(z["context"])[None].to("cuda", dtype)
    mask = torch.from_numpy(z["mask"])[None].to("cuda")

    # loaded in bf16 (the checkpoint's dtype: the float32 copy is exact) and widened ON the GPU: the WSL host has 7 GB of RAM, a float32 DiT on the CPU does not fit.
    # The decoder is dropped (output_type "latent" never decodes), so the float32 DiT (7.7 GB) and encoder (2.8 GB) fit a 12 GB card together without offload.
    transformer = LTXVideoTransformer3DModel.from_single_file(a.ckpt, torch_dtype=torch.bfloat16).to("cuda").to(dtype).eval().requires_grad_(False)
    vae = AutoencoderKLLTXVideo.from_single_file(a.ckpt, torch_dtype=torch.bfloat16)
    vae.decoder = None
    vae = vae.to("cuda").float().eval().requires_grad_(False)
    if a.encoder_f16:
        with torch.no_grad():
            for prm in vae.encoder.parameters():
                prm.copy_(prm.half().float())
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False, shift_terminal=None)
    pipe = LTXConditionPipeline(scheduler=sched, vae=vae, text_encoder=None, tokenizer=None, transformer=transformer)

    calls = []

    def pre(_m, args, kwargs):
        calls.append({"latent_in": kwargs["hidden_states"][0].float().cpu().numpy(), "t": kwargs["timestep"][0].float().cpu().numpy().reshape(-1),
                      "coords": kwargs["video_coords"][0].float().cpu().numpy()})

    def post(_m, args, kwargs, out):
        calls[-1]["pred"] = out[0][0].float().cpu().numpy()
    transformer.register_forward_pre_hook(pre, with_kwargs=True)
    transformer.register_forward_hook(post, with_kwargs=True)

    encs = []
    orig_encode = vae.encode

    def encode(x, *args, **kwargs):
        r = orig_encode(x, *args, **kwargs)
        encs.append((x[0].float().cpu().numpy(), r.latent_dist.mean[0].float().cpu().numpy()))
        return r
    vae.encode = encode

    gen = torch.Generator("cpu").manual_seed(a.seed)
    with torch.no_grad():
        final = pipe(conditions=conditions, prompt_embeds=ctx, prompt_attention_mask=mask, width=W, height=H, num_frames=a.frames, frame_rate=a.frame_rate,
                     timesteps=[s * 1000.0 for s in SIGMAS], guidance_scale=1.0, image_cond_noise_scale=a.cond_noise,
                     generator=gen, output_type="latent", return_dict=False)[0]

    # the same draws again, in the pipeline's order (prepare_latents: the starting noise, then per condition its posterior sample and, at a later frame, its blend noise; then one per step)
    Nt = calls[0]["latent_in"].shape[0]
    g2 = torch.Generator("cpu").manual_seed(a.seed)
    out = {"noise": torch.randn(1, 128, Fz, h, w, generator=g2)[0].numpy()}
    lm, ls = vae.latents_mean.view(-1, 1, 1, 1).cpu(), vae.latents_std.view(-1, 1, 1, 1).cpu()
    for i, ((kind, fi, s, fr), (pix, mean)) in enumerate(zip(specs, encs)):
        nc = mean.shape[1]
        out[f"cond{i}_pixels"] = pix
        out[f"cond{i}_latent_mean"] = ((torch.from_numpy(mean) - lm) / ls).numpy()
        out[f"posterior{i}"] = torch.randn(1, 128, nc, h, w, generator=g2)[0].numpy()
        if fi != 0:
            kept = 2 if (len(fr) > 1 and nc > 2) else nc
            out[f"condnoise{i}"] = torch.randn(1, 128, kept, h, w, generator=g2)[0].numpy()
    for s in range(len(SIGMAS)):
        out[f"cond_noise{s}"] = torch.randn(1, Nt, 128, generator=g2)[0].numpy()
    if len(specs) == 1 and specs[0][0] == "img" and specs[0][1] == 0:
        out["image"], out["posterior"], out["cond_latent_mean"] = out["cond0_pixels"], out["posterior0"], out["cond0_latent_mean"]
    out["sigmas"] = np.array(SIGMAS + [0.0], dtype=np.float32)
    for s, c in enumerate(calls):
        out[f"latent_in{s}"], out[f"t{s}"], out[f"pred{s}"] = c["latent_in"], c["t"], c["pred"]
    out["coords"] = calls[0]["coords"]
    out["final"] = final[0].float().cpu().numpy()
    path = os.path.join(a.out_dir, f"cond_{a.tag}.npz")
    np.savez(path, **{k: np.ascontiguousarray(v) for k, v in out.items()})       # torch views can be Fortran-ordered: the C# reader wants C order
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.1f} MB;", len(calls), f"DiT calls, {Nt} tokens ({Nt - Fz * hw} extra); t per step:", [sorted(set(np.round(c['t'], 3).tolist())) for c in calls])
    print("coords frame values:", sorted(set(np.round(calls[0]["coords"][0], 4).tolist())), "final absmax", float(np.abs(out["final"]).max()))
    # check the re-drawn noise is the pipeline's: the step-0 input equals the starting noise on the video tokens whose timestep is the full t (unconditioned)
    t0 = calls[0]["t"]
    free = np.where(np.abs(t0 - t0.max()) < 1e-3)[0]
    free = free[free >= Nt - Fz * hw]
    noise_tok = out["noise"].reshape(128, -1).T
    print("re-drawn noise matches the pipeline's step-0 input on", len(free), "unconditioned tokens:", float(np.abs(calls[0]["latent_in"][free] - noise_tok[free - (Nt - Fz * hw)]).max()))


if __name__ == "__main__":
    main()
