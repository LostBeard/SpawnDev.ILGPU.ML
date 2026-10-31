"""LTX-Video 2B 0.9.8 distilled reference: the MULTI-SCALE recipe (configs/ltxv-2b-0.9.8-distilled.yaml) with diffusers' real pipelines chained, dumped as a fixture for the C# port.

  pass 1  LTXConditionPipeline at 2/3 of the size (rounded down to multiples of 32), the first-pass sigmas, output_type "latent"
  x2      LTXLatentUpsamplePipeline (ltxv-spatial-upscaler-0.9.8, AdaIN factor 1), output_type "latent"
  pass 2  LTXConditionPipeline at twice the pass-1 size from the upsampled latent: timesteps [1000, 909.4, 725, 421.9] with denoise_strength 0.75 = the second-pass sigmas
          [0.9094, 0.7250, 0.4219] (then 0), the latent re-noised to sigma 0.9094 first (latents = sigma * noise + (1 - sigma) * latents)
One CPU generator seeded --seed runs through all three calls; its draws are re-drawn here in order (pass 1: noise, the picture's posterior sample, one image-conditioning noise per
step; the upsampler draws nothing for latent output; pass 2 the same at its size). The picture (smooth random content, as ref_condition_dump.py) is frame 0 of both passes; the
pixels each pass encoded are dumped (the pipeline resizes the picture per pass). guidance 1, the prompt of ref/t5_<prompt>.npz, float32 with --fp32, TF32 off.
The upsampler's float32 weights are rounded to fp16 (the export's precision; --upsampler-fp32 keeps them), so the fixture checks the port's math alone: the fp16 cost of the
upsampler is its own gate (Ltx_LatentUpsampler_MatchesReference, 7e-4).

  python ref_multiscale_dump.py CKPT.safetensors UPSAMPLER.safetensors OUT_DIR T5_DIR [--size 480x320] [--frames 25] [--seed 77] [--fp32] [--tag ms]
Writes OUT_DIR/ms_<tag>.npz: p1_*, p2_* (pixels [3, 1, H, W], noise, posterior, cond_noise<s>, final), up (the upsampled latent), sizes [W1, H1, W2, H2].
"""
import argparse
import json
import os

import numpy as np
import torch
from diffusers import AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXConditionPipeline, LTXVideoTransformer3DModel
from diffusers.pipelines.ltx.modeling_latent_upsampler import LTXLatentUpsamplerModel
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.pipelines.ltx.pipeline_ltx_latent_upsample import LTXLatentUpsamplePipeline
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

FIRST = [1.0, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250]
SECOND = [0.9094, 0.7250, 0.4219]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("upsampler")
    ap.add_argument("out_dir")
    ap.add_argument("t5_dir")
    ap.add_argument("--size", default="480x320")
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--frame-rate", type=float, default=25.0)
    ap.add_argument("--prompt", default="0")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--upsampler-fp32", action="store_true")
    ap.add_argument("--tag", default="ms")
    a = ap.parse_args()
    dtype = torch.float32 if a.fp32 else torch.bfloat16
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    W, H = (int(v) for v in a.size.split("x"))
    W1, H1 = int(W * 0.6666666) // 32 * 32, int(H * 0.6666666) // 32 * 32
    W2, H2 = 2 * W1, 2 * H1
    print(f"pass 1 {W1}x{H1}, pass 2 {W2}x{H2} (requested {W}x{H})")

    g = torch.Generator().manual_seed(a.seed + 1)
    coarse = torch.rand(3, 10, 15, generator=g)
    up_img = torch.nn.functional.interpolate(coarse[None], size=(H2, W2), mode="bicubic", align_corners=False)[0].clamp(0, 1)
    img = Image.fromarray((up_img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))

    z = np.load(os.path.join(a.t5_dir, f"t5_{a.prompt}.npz"))
    ctx = torch.from_numpy(z["context"])[None].to("cuda", dtype)
    mask = torch.from_numpy(z["mask"])[None].to("cuda")

    transformer = LTXVideoTransformer3DModel.from_single_file(a.ckpt, torch_dtype=torch.bfloat16).to("cuda").to(dtype).eval().requires_grad_(False)
    vae = AutoencoderKLLTXVideo.from_single_file(a.ckpt, torch_dtype=torch.bfloat16)
    vae.decoder = None
    vae = vae.to("cuda").float().eval().requires_grad_(False)
    cfg = json.loads(safe_open(a.upsampler, "pt").metadata()["config"])
    cfg.pop("_class_name", None)
    upm = LTXLatentUpsamplerModel(**cfg)
    usd = load_file(a.upsampler)
    if not a.upsampler_fp32:
        usd = {k: v.half().float() for k, v in usd.items()}
    upm.load_state_dict(usd)
    upm = upm.to("cuda").float().eval().requires_grad_(False)
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False, shift_terminal=None)
    pipe = LTXConditionPipeline(scheduler=sched, vae=vae, text_encoder=None, tokenizer=None, transformer=transformer)
    pipe_up = LTXLatentUpsamplePipeline(vae=vae, latent_upsampler=upm)

    encs = []
    orig_encode = vae.encode

    def encode(x, *args, **kwargs):
        encs.append(x[0].float().cpu().numpy())
        return orig_encode(x, *args, **kwargs)
    vae.encode = encode
    calls = []
    inputs = []
    def pre(_m, _a, kw):                                    # returns None: a returned tuple would replace the call's (args, kwargs)
        calls.append(kw["timestep"][0].float().cpu().numpy().reshape(-1))
        inputs.append(kw["hidden_states"][0].float().cpu().numpy())
    transformer.register_forward_pre_hook(pre, with_kwargs=True)

    gen = torch.Generator("cpu").manual_seed(a.seed)
    common = dict(prompt_embeds=ctx, prompt_attention_mask=mask, num_frames=a.frames, frame_rate=a.frame_rate, guidance_scale=1.0, generator=gen, output_type="latent", return_dict=False)
    cond = [LTXVideoCondition(image=img, frame_index=0, strength=1.0)]
    with torch.no_grad():
        p1 = pipe(conditions=cond, width=W1, height=H1, timesteps=[s * 1000.0 for s in FIRST], **common)[0]
        n1 = len(calls)
        up = pipe_up(latents=p1, adain_factor=1.0, output_type="latent", return_dict=False, generator=gen)[0]
        p2 = pipe(conditions=cond, width=W2, height=H2, latents=up, timesteps=[1000.0] + [s * 1000.0 for s in SECOND], denoise_strength=0.75, **common)[0]
    print("pass 1 timesteps", [sorted(set(np.round(t, 1).tolist())) for t in calls[:n1]])
    print("pass 2 timesteps", [sorted(set(np.round(t, 1).tolist())) for t in calls[n1:]])

    Fz = (a.frames - 1) // 8 + 1
    g2 = torch.Generator("cpu").manual_seed(a.seed)
    out = {"sizes": np.array([W1, H1, W2, H2], dtype=np.int32)}
    for tag, (w_, h_), steps, pix in (("p1", (W1, H1), len(FIRST), encs[0]), ("p2", (W2, H2), len(SECOND), encs[1])):
        h, w = h_ // 32, w_ // 32
        out[f"{tag}_noise"] = torch.randn(1, 128, Fz, h, w, generator=g2)[0].numpy()
        out[f"{tag}_posterior"] = torch.randn(1, 128, 1, h, w, generator=g2)[0].numpy()
        for s in range(steps):
            out[f"{tag}_cond_noise{s}"] = torch.randn(1, Fz * h * w, 128, generator=g2)[0].numpy()
        out[f"{tag}_pixels"] = pix
    out["p1_final"], out["up"], out["p2_final"] = p1[0].float().cpu().numpy(), up[0].float().cpu().numpy(), p2[0].float().cpu().numpy()
    # the re-drawn noise must be the pipeline's: pass 2's first DiT input on the unconditioned tokens is sigma * noise + (1 - sigma) * the upsampled latent
    h2, w2 = H2 // 32, W2 // 32
    exp = (0.9094 * out["p2_noise"] + (1 - 0.9094) * out["up"]).reshape(128, -1).T
    free = slice(h2 * w2, None)
    print("pass 2 start latent vs the re-drawn noise:", float(np.abs(inputs[n1][free] - exp[free]).max()))
    path = os.path.join(a.out_dir, f"ms_{a.tag}.npz")
    np.savez(path, **{k: np.ascontiguousarray(v) for k, v in out.items()})
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.1f} MB:", ", ".join(f"{k} {v.shape}" for k, v in out.items() if "cond_noise" not in k))


if __name__ == "__main__":
    main()
