"""Reference for the SD inpainting UNet (9 input channels): diffusers' UNet2DConditionModel on the CPU, fp32 math on the fp16 weights.

  python ref_sd_inpaint.py UNET_DIR CTX_NPZ OUT.npz [--latent 32x40] [--timestep 500] [--seed 7]

UNET_DIR holds config.json + diffusion_pytorch_model.fp16.safetensors (stable-diffusion-v1-5/stable-diffusion-inpainting, unet/).
CTX_NPZ supplies a real text context ("ctx" [77, 768], e.g. ref/sd_sd15_small_f16w.npz). Writes latent [4, h, w], concat [5, h, w] (the mask, then the masked
image's latent: diffusers' StableDiffusionInpaintPipeline order), ctx, timestep, eps [4, h, w].
"""
import argparse
import numpy as np
import torch
from diffusers import UNet2DConditionModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("unet_dir")
    ap.add_argument("ctx_npz")
    ap.add_argument("out")
    ap.add_argument("--latent", default="32x40")
    ap.add_argument("--timestep", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    h, w = (int(v) for v in a.latent.split("x"))
    torch.set_grad_enabled(False)
    unet = UNet2DConditionModel.from_pretrained(a.unet_dir, variant="fp16", torch_dtype=torch.float32).eval()
    assert unet.config.in_channels == 9, unet.config.in_channels
    ctx = torch.from_numpy(np.load(a.ctx_npz)["ctx"].astype(np.float32)).reshape(1, 77, -1)
    g = torch.Generator().manual_seed(a.seed)
    latent = torch.randn(1, 4, h, w, generator=g)
    masked = torch.randn(1, 4, h, w, generator=g) * 0.8
    mask = torch.zeros(1, 1, h, w)
    mask[:, :, h // 4: 3 * h // 4, w // 3: 2 * w // 3] = 1.0          # the region to repaint
    masked = masked * (1 - mask)                                      # what the pipeline's masked image latent looks like: content outside the hole
    concat = torch.cat([mask, masked], dim=1)
    eps = unet(torch.cat([latent, concat], dim=1), torch.tensor([a.timestep]), encoder_hidden_states=ctx).sample
    np.savez(a.out, latent=latent[0].numpy(), concat=concat[0].numpy(), ctx=ctx[0].numpy(), timestep=np.array([a.timestep], dtype=np.float32), eps=eps[0].numpy())
    print("eps", tuple(eps.shape), "absmax", float(eps.abs().max()), "->", a.out)


if __name__ == "__main__":
    main()
