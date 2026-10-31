"""LTX-Video 0.9.8 spatial latent upscaler reference: diffusers' LTXLatentUpsamplePipeline (the real pipeline: denormalise, LTXLatentUpsamplerModel, AdaIN to the input,
normalise) on a latent, float32 on the CPU, with the model's stage taps, dumped as a fixture for the C# `LtxLatentUpsampler`.

  python ref_upsampler_dump.py CKPT.safetensors UPSAMPLER.safetensors LATENT.npz KEY OUT.npz [--adain 1.0] [--round-f16]
LATENT.npz[KEY] is a model-space latent [128, F, h, w] (e.g. the final latent of cond_i2v.npz). Writes latent (the input), output [128, F, 2h, 2w] (normalised, output_type "latent"),
output_noadain (adain_factor 0), and the model's taps on the denormalised input: initial (after the first conv, norm, SiLU), res (after the 4 resblocks), upsampled (after the
2D conv + pixel shuffle), post (after the 4 post resblocks), final (the model output, denormalised space), all [C, F, H, W] channel-first.
"""
import argparse
import json

import numpy as np
import torch
from diffusers import AutoencoderKLLTXVideo
from diffusers.pipelines.ltx.modeling_latent_upsampler import LTXLatentUpsamplerModel
from diffusers.pipelines.ltx.pipeline_ltx_latent_upsample import LTXLatentUpsamplePipeline
from safetensors import safe_open
from safetensors.torch import load_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("upsampler")
    ap.add_argument("latent_npz")
    ap.add_argument("key")
    ap.add_argument("out")
    ap.add_argument("--adain", type=float, default=1.0)
    ap.add_argument("--round-f16", action="store_true", help="round the upsampler's weights to fp16 first (the export's precision): the reference for the port's exact math")
    a = ap.parse_args()
    cfg = json.loads(safe_open(a.upsampler, "pt").metadata()["config"])
    cfg.pop("_class_name", None)
    model = LTXLatentUpsamplerModel(**cfg)
    sd = load_file(a.upsampler)
    if a.round_f16:
        sd = {k: v.half().float() for k, v in sd.items()}
    model.load_state_dict(sd)
    model = model.float().eval().requires_grad_(False)
    vae = AutoencoderKLLTXVideo.from_single_file(a.ckpt, torch_dtype=torch.float32).eval()
    pipe = LTXLatentUpsamplePipeline(vae=vae, latent_upsampler=model)

    latent = torch.from_numpy(np.load(a.latent_npz)[a.key]).float()[None]          # [1, 128, F, h, w]
    keep = {}

    def hook(name):
        def fn(_m, _i, o):
            keep.setdefault(name, o.detach().float().numpy()[0])
        return fn
    model.initial_activation.register_forward_hook(hook("initial"))
    model.res_blocks[-1].register_forward_hook(hook("res"))
    model.post_upsample_res_blocks[-1].register_forward_hook(hook("post"))
    model.final_conv.register_forward_hook(hook("final"))
    up = {}
    model.upsampler.register_forward_hook(lambda _m, _i, o: up.setdefault("o", o.detach().float()))
    with torch.no_grad():
        out = pipe(latents=latent, adain_factor=a.adain, output_type="latent", return_dict=False)[0]
        keep2 = dict(keep)
        keep.clear(); up2 = dict(up); up.clear()
        out0 = pipe(latents=latent, adain_factor=0.0, output_type="latent", return_dict=False)[0]
    F = latent.shape[2]
    u = up2["o"]                                                                     # [F, 512, 2h, 2w] (the 2D stage runs per frame)
    res = {"latent": latent[0].numpy(), "output": out[0].numpy(), "output_noadain": out0[0].numpy(), "upsampled": u.permute(1, 0, 2, 3).numpy()}
    res.update(keep2)
    np.savez(a.out, **{k: np.ascontiguousarray(v) for k, v in res.items()})
    print("wrote", a.out, ", ".join(f"{k} {v.shape}" for k, v in res.items()))
    print("adain moves the output", float(((out - out0) ** 2).mean().sqrt() / (out0 ** 2).mean().sqrt()))


if __name__ == "__main__":
    main()
