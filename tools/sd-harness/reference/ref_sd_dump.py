"""Stable Diffusion single-file checkpoint reference (SD 1.x / 2.x / SDXL), dumped as fixtures for the C# SdCheckpoint port (Plans/sd-checkpoint-2026-10-25.md).

Loads the ComfyUI / CivitAI single .safetensors file through diffusers' from_single_file (the same file the C# loader reads) and runs every stage in float32 on the CPU
(exact math; the GPU stays free for the other sessions). With --f16-weights every parameter is first rounded to fp16 (and kept fp32 for the math): the C# side holds fp16
weights, so that dump isolates our arithmetic from the weight rounding; the plain dump measures both.
  python ref_sd_dump.py CKPT.safetensors OUT_DIR [--latent 64x64] [--timestep 500] [--seed 7] [--tag t] [--f16-weights] [--prompt "..."] [--image-size 256x320]
Writes OUT_DIR/sd_<tag>.npz:
  ids [77] (+ ids2 for SDXL), the CLIP outputs ctx [77, C] (last layer) and ctx_skip2 [77, C] (penultimate layer + final layer norm, ComfyUI clip skip 2), pooled [1280] for SDXL;
  unet: latent [4, h, w], timestep, (SDXL: time_ids [6]), taps conv_in, down_0..3, mid, up_0..3, eps [4, h, w];
  vae: dec_in [4, h, w] (the UNSCALED latent the decoder takes), taps post_quant_conv, dec_mid, dec_up_0..3, image [3, 8h, 8w];
       enc_image [3, H, W] in [-1, 1], enc_mean [4, H/8, W/8] (the posterior mean, unscaled).
The configs are printed: they are what the C# side must infer from the file's tensor shapes.
"""
import argparse
import json
import os

import numpy as np
import torch


def upcast_math(mods):
    """fp16 parameters, fp32 math: every layer that reads parameters casts them to fp32 for the call (activations stay fp32 throughout)."""
    import torch.nn as nn
    import torch.nn.functional as F

    def f32(t):
        return None if t is None else t.float()
    for root in mods:
        for m in root.modules():
            if isinstance(m, nn.Linear):
                m.forward = (lambda mm: lambda x: F.linear(x.float(), f32(mm.weight), f32(mm.bias)))(m)
            elif isinstance(m, nn.Conv2d):
                m.forward = (lambda mm: lambda x: mm._conv_forward(x.float(), f32(mm.weight), f32(mm.bias)))(m)
            elif isinstance(m, nn.GroupNorm):
                m.forward = (lambda mm: lambda x: F.group_norm(x.float(), mm.num_groups, f32(mm.weight), f32(mm.bias), mm.eps))(m)
            elif isinstance(m, nn.LayerNorm):
                m.forward = (lambda mm: lambda x: F.layer_norm(x.float(), mm.normalized_shape, f32(mm.weight), f32(mm.bias), mm.eps))(m)
            elif isinstance(m, nn.Embedding):
                m.forward = (lambda mm: lambda x: F.embedding(x, mm.weight).float())(m)
        for name, buf in list(root.named_buffers()):
            if buf.dtype == torch.float16:
                buf.data = buf.data.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("--latent", default="64x64")
    ap.add_argument("--timestep", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="t")
    ap.add_argument("--f16-weights", action="store_true")
    ap.add_argument("--f16-storage", action="store_true", help="keep the weights fp16 in memory and do every layer's math in fp32 (the same numbers as --f16-weights at half the memory: SDXL on a 16 GB machine)")
    ap.add_argument("--prompt", default="a photograph of an astronaut riding a horse on the moon, highly detailed")
    ap.add_argument("--image-size", default="256x320")
    ap.add_argument("--skip", default="", help="comma list of stages to skip: text,unet,dec,enc")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    skip = set(s for s in a.skip.split(",") if s)
    h, w = (int(v) for v in a.latent.split("x"))
    torch.set_num_threads(os.cpu_count())

    from safetensors import safe_open
    with safe_open(a.ckpt, "pt") as f:
        keys = set(f.keys())
    sdxl = any(k.startswith("conditioner.embedders.1.") for k in keys)
    if sdxl:
        from diffusers import StableDiffusionXLPipeline as P
    else:
        from diffusers import StableDiffusionPipeline as P
    dt = torch.float16 if a.f16_storage else torch.float32
    pipe = P.from_single_file(a.ckpt, torch_dtype=dt, safety_checker=None) if not sdxl else P.from_single_file(a.ckpt, torch_dtype=dt)
    mods = [pipe.unet, pipe.vae, pipe.text_encoder] + ([pipe.text_encoder_2] if sdxl else [])
    if a.f16_storage:
        upcast_math(mods)
    for m in mods:
        m.eval().requires_grad_(False)
        if a.f16_weights and not a.f16_storage:
            with torch.no_grad():
                for p in m.parameters():
                    p.copy_(p.half().float())
    print("sdxl", sdxl)
    print("unet config:", json.dumps({k: v for k, v in pipe.unet.config.items() if not k.startswith("_")}, default=str))
    print("vae config:", json.dumps({k: v for k, v in pipe.vae.config.items() if not k.startswith("_")}, default=str))
    print("scheduler:", json.dumps({k: v for k, v in pipe.scheduler.config.items() if not k.startswith("_")}, default=str))

    out = {}
    g = torch.Generator().manual_seed(a.seed)
    with torch.no_grad():
        # ---- text
        tok = pipe.tokenizer(a.prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        ids = tok.input_ids
        out["ids"] = ids[0].numpy().astype(np.int32)
        enc = pipe.text_encoder(ids, output_hidden_states=True)
        if sdxl:
            # SDXL: tower 1 (CLIP-L) penultimate layer WITHOUT the final norm, tower 2 (OpenCLIP bigG) penultimate layer without norm + the projected pooled output
            tok2 = pipe.tokenizer_2(a.prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
            out["ids2"] = tok2.input_ids[0].numpy().astype(np.int32)
            enc2 = pipe.text_encoder_2(tok2.input_ids, output_hidden_states=True)
            c1 = enc.hidden_states[-2][0]
            c2 = enc2.hidden_states[-2][0]
            out["ctx1"] = c1.numpy(); out["ctx2"] = c2.numpy()
            out["ctx"] = torch.cat([c1, c2], dim=-1).numpy()
            out["pooled"] = enc2.text_embeds[0].numpy()
        else:
            out["ctx"] = enc.last_hidden_state[0].numpy()
            tm = getattr(pipe.text_encoder, "text_model", pipe.text_encoder)     # transformers 5 flattened CLIPTextModel
            out["ctx_skip2"] = tm.final_layer_norm(enc.hidden_states[-2])[0].numpy()
        ctx = torch.from_numpy(out["ctx"])[None]

        # ---- unet
        if "unet" not in skip:
            unet = pipe.unet
            latent = torch.randn(1, unet.config.in_channels, h, w, generator=g)
            out["latent"] = latent[0].numpy()
            out["timestep"] = np.array([a.timestep], dtype=np.float32)
            keep = {}

            def hook(name):
                def fn(_m, _i, o):
                    t = o[0] if isinstance(o, tuple) else o
                    keep[name] = t.detach().float().numpy()[0]
                return fn
            unet.conv_in.register_forward_hook(hook("conv_in"))
            for i, b in enumerate(unet.down_blocks):
                b.register_forward_hook(hook(f"down_{i}"))
            unet.mid_block.register_forward_hook(hook("mid"))
            for i, b in enumerate(unet.up_blocks):
                b.register_forward_hook(hook(f"up_{i}"))
            kw = {}
            if sdxl:
                H8, W8 = h * 8, w * 8
                time_ids = torch.tensor([[H8, W8, 0, 0, H8, W8]], dtype=torch.float32)
                out["time_ids"] = time_ids[0].numpy()
                kw["added_cond_kwargs"] = {"text_embeds": torch.from_numpy(out["pooled"])[None], "time_ids": time_ids}
            eps = unet(latent, torch.tensor([a.timestep]), encoder_hidden_states=ctx, **kw).sample
            out["eps"] = eps[0].numpy()
            for k, v in keep.items():
                out[k] = v
            print("unet taps:", ", ".join(f"{k} {v.shape}" for k, v in keep.items()))

        # ---- vae decode (unscaled latent in, image in [-1, 1] out, not clamped)
        vae = pipe.vae
        if "dec" not in skip:
            z = torch.randn(1, 4, h, w, generator=g) / vae.config.scaling_factor
            out["dec_in"] = z[0].numpy()
            keep = {}

            def hook2(name):
                def fn(_m, _i, o):
                    t = o[0] if isinstance(o, tuple) else o
                    keep[name] = t.detach().float().numpy()[0]
                return fn
            vae.post_quant_conv.register_forward_hook(hook2("post_quant_conv"))
            vae.decoder.mid_block.register_forward_hook(hook2("dec_mid"))
            for i, b in enumerate(vae.decoder.up_blocks):
                b.register_forward_hook(hook2(f"dec_up_{i}"))
            img = vae.decode(z).sample
            out["image"] = img[0].numpy()
            for k, v in keep.items():
                out[k] = v

        # ---- vae encode (posterior mean, unscaled)
        if "enc" not in skip:
            ih, iw = (int(v) for v in a.image_size.split("x"))
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, ih), torch.linspace(-1, 1, iw), indexing="ij")
            noise = torch.rand(3, ih, iw, generator=g) * 0.2 - 0.1
            im = torch.stack([torch.sin(3 * xx + yy), torch.cos(2 * yy - xx), xx * yy]) * 0.8 + noise
            out["enc_image"] = im.numpy()
            out["enc_mean"] = vae.encode(im[None]).latent_dist.mean[0].numpy()

    path = os.path.join(a.out_dir, f"sd_{a.tag}.npz")
    np.savez(path, **out)
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.1f} MB:", ", ".join(f"{k} {v.shape}" for k, v in out.items()))


if __name__ == "__main__":
    main()
