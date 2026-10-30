"""Reference for ComfyUI's PatchModelAddDownscale (Kohya deep shrink) and for latents whose sides are not multiples of 8.

  python ref_sd_deepshrink.py CKPT.safetensors CTX_NPZ OUT.npz [--latent 36x44] [--timestep 900] [--seed 11]

diffusers' UNet (from_single_file, weights rounded to fp16, fp32 math on the CPU). With sides that are not multiples of 8, diffusers upsamples each up block to the
next skip connection's size, as ComfyUI's Upsample(output_shape = hs[-1].shape) does. Writes latent, ctx, timestep, eps_plain and eps_shrink (block 3, factor 2,
downscale after skip, bicubic both ways: ComfyUI's defaults). Block 3 is the output of down_blocks[0] (its downsampler); the skip it stores stays full size.
ComfyUI's output_block_patch resizes h to the skip's size when the heights differ: done here before the concatenation of every up-block resnet.
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import diffusers.models.unets.unet_2d_blocks as blocks
from diffusers import UNet2DConditionModel

STATE = {"on": False}


def comfy_output_patch(resolution_idx, h, hsp, **_):          # PatchModelAddDownscale.output_block_patch
    if STATE["on"] and h.shape[2] != hsp.shape[2]:
        h = F.interpolate(h, size=(hsp.shape[-2], hsp.shape[-1]), mode="bicubic")
    return h, hsp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("ctx_npz")
    ap.add_argument("out")
    ap.add_argument("--latent", default="36x44")
    ap.add_argument("--timestep", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    h, w = (int(v) for v in a.latent.split("x"))
    torch.set_grad_enabled(False)
    unet = UNet2DConditionModel.from_single_file(a.ckpt, torch_dtype=torch.float32).eval()
    for p in unet.parameters():
        p.data = p.data.half().float()
    ctx = torch.from_numpy(np.load(a.ctx_npz)["ctx"].astype(np.float32)).reshape(1, 77, -1)
    latent = torch.randn(1, 4, h, w, generator=torch.Generator().manual_seed(a.seed))
    t = torch.tensor([a.timestep])
    run = lambda: unet(latent, t, encoder_hidden_states=ctx).sample[0].numpy()
    out = {"latent": latent[0].numpy(), "ctx": ctx[0].numpy(), "timestep": np.array([a.timestep], dtype=np.float32)}
    out["eps_plain"] = run()

    # input_block_patch_after_skip at block 3 (= down_blocks[0]'s output; its res samples, the skips, are untouched)
    def shrink(_m, _inp, res):
        hs, skips = res
        nh, nw = round(hs.shape[-2] * (1.0 / 2.0)), round(hs.shape[-1] * (1.0 / 2.0))
        return F.interpolate(hs, size=(nh, nw), mode="bicubic"), skips
    blocks.apply_freeu = comfy_output_patch
    unet.enable_freeu(s1=1.0, s2=1.0, b1=1.0, b2=1.0)          # only to route every up-block concatenation through comfy_output_patch
    hook = unet.down_blocks[0].register_forward_hook(shrink)
    STATE["on"] = True
    out["eps_shrink"] = run()
    hook.remove(); STATE["on"] = False; unet.disable_freeu()
    np.savez(a.out, **out)
    for k in ("plain", "shrink"):
        e = out["eps_" + k]
        print(k, e.shape, "absmax", float(np.abs(e).max()), "vs plain", float(np.sqrt(((e - out["eps_plain"]) ** 2).mean() / (out["eps_plain"] ** 2).mean())))


if __name__ == "__main__":
    main()
