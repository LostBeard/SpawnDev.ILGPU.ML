"""LTX-Video 2B 0.9.8 distilled reference: IMAGE-TO-VIDEO (diffusers' LTXConditionPipeline, a picture as the first frame), dumped as fixtures for the C# port.

Runs the REAL pipeline (not a re-implementation) with model CPU offload, so the transformer (float32 with --fp32: the exact math on the bf16 weights) and the VAE take the GPU in turn,
and records every DiT call with forward hooks. The prompt is the T5 output of ref_t5_dump.py (no text encoder is loaded); guidance 1 (the distilled model: one forward per step); the
first-pass sigmas of configs/ltxv-2b-0.9.8-distilled.yaml as timesteps; denoise strength 1; the picture at frame 0 with conditioning strength --strength.
The random draws come from a CPU generator seeded --seed, in the pipeline's order, and are re-drawn here identically so the C# side can use the SAME noise:
  noise [128, Fz, h, w] (the starting latent), posterior [128, 1, h, w] (the VAE posterior sample of the picture), then per step cond_noise<i> [N, 128] (image_cond_noise_scale noise).

  python ref_condition_dump.py CKPT.safetensors OUT_DIR T5_DIR [--size 480x320] [--frames 25] [--seed 99] [--strength 1.0] [--cond-noise 0.15] [--frame-rate 25] [--fp32] [--tag i2v]
Writes OUT_DIR/cond_<tag>.npz: image [3, 1, H, W] (the pixels the VAE encodes, in [-1, 1]), the draws above, sigmas, and per step i: latent_in<i> [N, 128] (the DiT input, packed tokens
in (frame, row, column) order), t<i> [N] (the per-token timestep), pred<i> [N, 128]; coords [3, N] (the video_coords the transformer gets, time already divided by the frame rate);
mask [N] (the conditioning mask); final [128, Fz, h, w] (the denoised latent, output_type "latent", before the decode); cond_latent [128, 1, h, w] (the normalised condition latent).
"""
import argparse
import os

import numpy as np
import torch
from diffusers import AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXConditionPipeline, LTXVideoTransformer3DModel
from diffusers.utils import load_image
from PIL import Image

SIGMAS = [1.0, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("t5_dir")
    ap.add_argument("--size", default="480x320")
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--cond-noise", type=float, default=0.15)
    ap.add_argument("--frame-rate", type=float, default=25.0)
    ap.add_argument("--prompt", default="0")
    ap.add_argument("--image", default=None, help="a picture file (resized to the size); default: smooth random content")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--encoder-f16", action="store_true", help="round the VAE encoder's weights to fp16 (the hosted export's precision), math still float32: the reference's own sensitivity to it")
    ap.add_argument("--tag", default="i2v")
    a = ap.parse_args()
    dtype = torch.float32 if a.fp32 else torch.bfloat16
    # float32 must mean float32 on the GPU: cuDNN convolutions default to TF32 (a 10-bit mantissa), which put ~8e-4 of error into the VAE encoding of the picture
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(a.out_dir, exist_ok=True)
    W, H = (int(v) for v in a.size.split("x"))
    Fz, h, w = (a.frames - 1) // 8 + 1, H // 32, W // 32
    N = Fz * h * w

    if a.image:
        img = load_image(a.image).convert("RGB").resize((W, H), Image.BICUBIC)
    else:
        g = torch.Generator().manual_seed(a.seed + 1)
        coarse = torch.rand(3, H // 32, W // 32, generator=g)
        up = torch.nn.functional.interpolate(coarse[None], size=(H, W), mode="bicubic", align_corners=False)[0].clamp(0, 1)
        img = Image.fromarray((up.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))

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

    enc = {}
    orig_encode = vae.encode

    def encode(x, *args, **kwargs):
        enc["image"] = x[0].float().cpu().numpy()
        r = orig_encode(x, *args, **kwargs)
        enc["mean"] = r.latent_dist.mean[0].float().cpu().numpy()
        return r
    vae.encode = encode

    gen = torch.Generator("cpu").manual_seed(a.seed)
    with torch.no_grad():
        final = pipe(image=img, prompt_embeds=ctx, prompt_attention_mask=mask, width=W, height=H, num_frames=a.frames, frame_rate=a.frame_rate,
                     timesteps=[s * 1000.0 for s in SIGMAS], guidance_scale=1.0, strength=a.strength, image_cond_noise_scale=a.cond_noise,
                     generator=gen, output_type="latent", return_dict=False)[0]

    # the same draws again, in the pipeline's order (prepare_latents: the starting noise, then the posterior sample; then one per step)
    g2 = torch.Generator("cpu").manual_seed(a.seed)
    out = {"noise": torch.randn(1, 128, Fz, h, w, generator=g2)[0].numpy(), "posterior": torch.randn(1, 128, 1, h, w, generator=g2)[0].numpy()}
    for i in range(len(SIGMAS)):
        out[f"cond_noise{i}"] = torch.randn(1, N, 128, generator=g2)[0].numpy()
    out["image"] = enc["image"]
    lm, ls = vae.latents_mean.view(-1, 1, 1, 1).cpu(), vae.latents_std.view(-1, 1, 1, 1).cpu()
    mean = torch.from_numpy(enc["mean"])
    out["cond_latent_mean"] = ((mean - lm) / ls).numpy()
    out["sigmas"] = np.array(SIGMAS + [0.0], dtype=np.float32)
    for i, c in enumerate(calls):
        out[f"latent_in{i}"], out[f"t{i}"], out[f"pred{i}"] = c["latent_in"], c["t"], c["pred"]
    out["coords"] = calls[0]["coords"]
    out["final"] = final[0].float().cpu().numpy()
    path = os.path.join(a.out_dir, f"cond_{a.tag}.npz")
    np.savez(path, **{k: np.ascontiguousarray(v) for k, v in out.items()})       # torch views can be Fortran-ordered: the C# reader wants C order
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.1f} MB;", len(calls), "DiT calls; t per step:", [sorted(set(np.round(c['t'], 3).tolist())) for c in calls])
    print("coords frame values:", sorted(set(np.round(calls[0]["coords"][0], 4).tolist())), "final absmax", float(np.abs(out["final"]).max()))
    # check the re-drawn noise is the pipeline's: step 0 input on the unconditioned tokens is the starting noise
    hw = h * w
    noise_tok = out["noise"].reshape(128, -1).T
    print("re-drawn noise matches the pipeline's step-0 input on unconditioned tokens:", float(np.abs(calls[0]["latent_in"][hw:] - noise_tok[hw:]).max()))


if __name__ == "__main__":
    main()
