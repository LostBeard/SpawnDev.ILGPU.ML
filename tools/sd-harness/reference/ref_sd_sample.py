"""A whole Stable Diffusion picture from the single-file checkpoint as the reference makes it, for the end-to-end gate of the C# port (Plans/sd-checkpoint-2026-10-25.md).

DDIM (eta 0: deterministic, no noise after the first latent), classifier-free guidance with the empty negative prompt, float32 math (weights rounded to fp16 with
--f16-weights, or kept fp16 with fp32 math with --f16-storage). The test reruns the same loop on the host around our UNet and compares every step's latent.
  python ref_sd_sample.py CKPT.safetensors OUT_DIR [--size 512x512] [--steps 20] [--cfg 7.5] [--seed 3] [--tag t] [--f16-weights | --f16-storage] [--prompt "..."]
Writes OUT_DIR/sample_<tag>.npz: ids, uncond_ids (and ids2 / uncond_ids2 for SDXL), timesteps [steps], alphas_cumprod [1000], noise [4, h, w] (the initial latent, already
x init_noise_sigma = 1 for DDIM), latents [steps + 1, 4, h, w] (the latent before every step and the final one), image [3, H, W] (decoded, in [-1, 1]).
"""
import argparse
import os

import numpy as np
import torch

from ref_sd_dump import upcast_math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("--size", default="512x512")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--tag", default="t")
    ap.add_argument("--f16-weights", action="store_true")
    ap.add_argument("--config", default=None, help="diffusers config repo (SD 2.x: Stability deleted its repos; sd2-community/stable-diffusion-2-1 mirrors them)")
    ap.add_argument("--f16-storage", action="store_true")
    ap.add_argument("--prompt", default="a red fox sitting in a snowy forest, golden hour, photograph")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    H, W = (int(v) for v in a.size.split("x"))
    torch.set_num_threads(os.cpu_count())

    from safetensors import safe_open
    with safe_open(a.ckpt, "pt") as f:
        sdxl = any(k.startswith("conditioner.embedders.1.") for k in f.keys())
    from diffusers import DDIMScheduler
    if sdxl:
        from diffusers import StableDiffusionXLPipeline as P
    else:
        from diffusers import StableDiffusionPipeline as P
    dt = torch.float16 if a.f16_storage else torch.float32
    kw = {} if sdxl else {"safety_checker": None}
    if a.config:
        kw["config"] = a.config
    pipe = P.from_single_file(a.ckpt, torch_dtype=dt, **kw)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    mods = [pipe.unet, pipe.vae, pipe.text_encoder] + ([pipe.text_encoder_2] if sdxl else [])
    if a.f16_storage:
        upcast_math(mods)
    for m in mods:
        m.eval().requires_grad_(False)
        if a.f16_weights and not a.f16_storage:
            with torch.no_grad():
                for p in m.parameters():
                    p.copy_(p.half().float())
    print("scheduler:", dict(pipe.scheduler.config))

    out = {}
    tok = pipe.tokenizer
    out["ids"] = tok(a.prompt, padding="max_length", max_length=77, truncation=True, return_tensors="np").input_ids[0].astype(np.int32)
    out["uncond_ids"] = tok("", padding="max_length", max_length=77, truncation=True, return_tensors="np").input_ids[0].astype(np.int32)
    if sdxl:
        t2 = pipe.tokenizer_2
        out["ids2"] = t2(a.prompt, padding="max_length", max_length=77, truncation=True, return_tensors="np").input_ids[0].astype(np.int32)
        out["uncond_ids2"] = t2("", padding="max_length", max_length=77, truncation=True, return_tensors="np").input_ids[0].astype(np.int32)

    g = torch.Generator().manual_seed(a.seed)
    noise = torch.randn(1, 4, H // 8, W // 8, generator=g)
    out["noise"] = noise[0].numpy()
    lat = []

    def on_step(p, i, t, kwargs):
        lat.append(kwargs["latents"][0].detach().float().numpy().copy())
        return kwargs

    call = dict(prompt=a.prompt, negative_prompt="", height=H, width=W, num_inference_steps=a.steps, guidance_scale=a.cfg, latents=noise.to(torch.float32),
                output_type="latent", callback_on_step_end=on_step, callback_on_step_end_tensor_inputs=["latents"])
    if sdxl:
        # diffusers zeroes the negative embeddings of an empty SDXL negative prompt; ComfyUI (and SpawnGen) encode "" instead
        pipe.register_to_config(force_zeros_for_empty_prompt=False)
    with torch.no_grad():
        res = pipe(**call).images
        final = res[0].float()
        img = pipe.vae.decode(final[None] / pipe.vae.config.scaling_factor).sample[0]
    out["timesteps"] = pipe.scheduler.timesteps.numpy().astype(np.float32)
    out["alphas_cumprod"] = pipe.scheduler.alphas_cumprod.numpy().astype(np.float32)
    out["latents"] = np.stack([noise[0].numpy()] + lat)
    out["image"] = img.numpy()
    path = os.path.join(a.out_dir, f"sample_{a.tag}.npz")
    np.savez(path, **out)
    try:
        from PIL import Image
        Image.fromarray(((img.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()).save(os.path.join(a.out_dir, f"sample_{a.tag}.png"))
    except ImportError:
        pass
    print("wrote", path, "timesteps", out["timesteps"].tolist())


if __name__ == "__main__":
    main()
