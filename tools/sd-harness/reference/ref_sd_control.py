"""ControlNet reference (SD 1.x or SDXL): diffusers ControlNetModel (from_single_file on a ComfyUI single file, or from_pretrained on a diffusers folder) + the checkpoint's UNet,
fp16 weights with fp32 math, CPU.
  python ref_sd_control.py CKPT.safetensors CONTROLNET(.safetensors | diffusers dir) OUT_DIR [--config lllyasviel/control_v11p_sd15_canny] [--latent 32x40] [--tag t] [--scale 0.8]
Writes OUT_DIR/control_<tag>.npz: latent [4, h, w], timestep, ctx [77, C], hint [3, 8h, 8w] in [0, 1], res_0 .. res_N (the residuals at scale 1, channel-first: the skips
then the middle block), eps_plain (UNet alone), eps_control (UNet + residuals x scale), scale; SDXL adds pooled [1280] and time_ids [6].
"""
import argparse
import os

import numpy as np
import torch

from ref_sd_dump import upcast_math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("controlnet")
    ap.add_argument("out_dir")
    ap.add_argument("--config", default="lllyasviel/control_v11p_sd15_canny")
    ap.add_argument("--latent", default="32x40")
    ap.add_argument("--tag", default="t")
    ap.add_argument("--scale", type=float, default=0.8)
    ap.add_argument("--timestep", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=9)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    h, w = (int(v) for v in a.latent.split("x"))
    torch.set_num_threads(os.cpu_count())
    from safetensors import safe_open
    with safe_open(a.ckpt, "pt") as f:
        sdxl = any(k.startswith("conditioner.embedders.1.") for k in f.keys())
    from diffusers import ControlNetModel
    if sdxl:
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_single_file(a.ckpt, torch_dtype=torch.float16)
    else:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_single_file(a.ckpt, torch_dtype=torch.float16, safety_checker=None)
    if os.path.isdir(a.controlnet):
        fp16 = os.path.exists(os.path.join(a.controlnet, "diffusion_pytorch_model.fp16.safetensors"))
        cn = ControlNetModel.from_pretrained(a.controlnet, torch_dtype=torch.float16, variant="fp16" if fp16 else None)
    else:
        cn = ControlNetModel.from_single_file(a.controlnet, torch_dtype=torch.float16, config=a.config)
    mods = [pipe.unet, pipe.text_encoder, cn] + ([pipe.text_encoder_2] if sdxl else [])
    upcast_math(mods)
    for m in mods:
        m.eval().requires_grad_(False)
    g = torch.Generator().manual_seed(a.seed)
    out = {}
    prompt = "a house by a lake, mountains behind it"
    with torch.no_grad():
        ids = pipe.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt").input_ids
        kw = {}
        if sdxl:
            ids2 = pipe.tokenizer_2(prompt, padding="max_length", max_length=77, return_tensors="pt").input_ids
            c1 = pipe.text_encoder(ids, output_hidden_states=True).hidden_states[-2]
            e2 = pipe.text_encoder_2(ids2, output_hidden_states=True)
            ctx = torch.cat([c1, e2.hidden_states[-2]], dim=-1).float()
            pooled = e2.text_embeds.float()
            time_ids = torch.tensor([[8 * h, 8 * w, 0, 0, 8 * h, 8 * w]], dtype=torch.float32)
            kw["added_cond_kwargs"] = {"text_embeds": pooled, "time_ids": time_ids}
            out["pooled"] = pooled[0].numpy(); out["time_ids"] = time_ids[0].numpy()
        else:
            ctx = pipe.text_encoder(ids).last_hidden_state.float()
        latent = torch.randn(1, 4, h, w, generator=g)
        H, W = 8 * h, 8 * w
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        edges = (((xx // 16 + yy // 24) % 3 == 0) | ((xx - yy).abs() % 37 < 2)).float()     # a deterministic line drawing, values 0 / 1
        hint = edges[None].repeat(3, 1, 1)
        t = torch.tensor([a.timestep])
        down, mid = cn(latent, t, encoder_hidden_states=ctx, controlnet_cond=hint[None], conditioning_scale=1.0, return_dict=False, **kw)
        eps_plain = pipe.unet(latent, t, encoder_hidden_states=ctx, **kw).sample
        eps_control = pipe.unet(latent, t, encoder_hidden_states=ctx, down_block_additional_residuals=[d * a.scale for d in down],
                                mid_block_additional_residual=mid * a.scale, **kw).sample
    out.update({"latent": latent[0].numpy(), "timestep": np.array([a.timestep], np.float32), "ctx": ctx[0].numpy(), "hint": hint.numpy(),
                "eps_plain": eps_plain[0].numpy(), "eps_control": eps_control[0].numpy(), "scale": np.array([a.scale], np.float32)})
    for i, d in enumerate(list(down) + [mid]):
        out[f"res_{i}"] = d[0].float().numpy()
    path = os.path.join(a.out_dir, f"control_{a.tag}.npz")
    np.savez(path, **out)
    print("wrote", path, "residuals", len(down) + 1, [tuple(d.shape[1:]) for d in list(down) + [mid]])


if __name__ == "__main__":
    main()
