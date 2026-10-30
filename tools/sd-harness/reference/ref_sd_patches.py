"""Reference for the UNet patches SpawnGen uses from ComfyUI: FreeU, FreeU_V2 and Perturbed-Attention Guidance's perturbed pass.

  python ref_sd_patches.py CKPT.safetensors FIXTURE.npz OUT.npz

diffusers' UNet (from_single_file, every weight rounded to fp16, fp32 math on the CPU) on the fixture's latent / ctx / timestep (ref/sd_sd15_small_f16w.npz):
  eps_plain, eps_freeu (1.1, 1.2, 0.9, 0.2), eps_freeu2 (V2: 1.3, 1.4, 0.9, 0.2), eps_pag (middle block self-attention = identity).
FreeU is ComfyUI's, not diffusers': diffusers' apply_freeu picks the scale by the up block's index, ComfyUI (nodes_freelunch.py) by the channel count of h
(model_channels * 4 -> b1 / s1, * 2 -> b2 / s2), so diffusers' function is replaced by ComfyUI's code. PAG: ComfyUI's perturbed_attention returns v.
"""
import argparse
import math
import numpy as np
import torch
import diffusers.models.unets.unet_2d_blocks as blocks
from diffusers import UNet2DConditionModel

MC = 320  # model_channels


def fourier_filter(x, threshold, scale):          # ComfyUI comfy_extras/nodes_freelunch.py
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)
    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    return torch.fft.ifftn(x_freq, dim=(-2, -1)).real.to(x.dtype)


STATE = {"v2": False, "scales": None}


def comfy_freeu(resolution_idx, h, hsp, **_):
    scale = STATE["scales"].get(int(h.shape[1]))
    if scale is None:
        return h, hsp
    h = h.clone()
    half = h.shape[1] // 2
    if STATE["v2"]:                                   # FreeU_V2.patch.output_block_patch
        hidden_mean = h.mean(1).unsqueeze(1)
        B = hidden_mean.shape[0]
        hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
        hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
        hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
        h[:, :half] = h[:, :half] * ((scale[0] - 1) * hidden_mean + 1)
    else:
        h[:, :half] = h[:, :half] * scale[0]
    return h, fourier_filter(hsp, threshold=1, scale=scale[1])


class HyperTileAttn:                                 # ComfyUI nodes_hypertile.py (hypertile_in / hypertile_out) around diffusers' processor
    def __init__(self, base, lat_hw, tile_size=128, swap_size=1, max_depth=1, scale_depth=False):
        self.base, (self.H, self.W) = base, lat_hw
        self.tile, self.swap, self.max_depth, self.scale_depth = max(32, tile_size) // 8, swap_size, max_depth, scale_depth

    @staticmethod
    def random_divisor(value, min_value, max_options=1):
        min_value = min(min_value, value)
        divisors = [i for i in range(min_value, value + 1) if value % i == 0]
        ns = [value // i for i in divisors[:max_options]]
        return ns[0] if len(ns) - 1 <= 0 else ns[int(torch.randint(low=0, high=len(ns) - 1, size=(1,)).item())]

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        hw = hidden_states.shape[1]
        apply_to = [(self.H / (2 ** i)) * (self.W / (2 ** i)) for i in range(self.max_depth + 1)]
        if hw in apply_to:
            aspect = self.W / self.H
            h, w = round(math.sqrt(hw * aspect)), round(math.sqrt(hw / aspect))
            factor = (2 ** apply_to.index(hw)) if self.scale_depth else 1
            nh, nw = self.random_divisor(h, self.tile * factor, self.swap), self.random_divisor(w, self.tile * factor, self.swap)
            if nh * nw > 1:
                b, _, c = hidden_states.shape
                th, tw = h // nh, w // nw
                x = hidden_states.view(b, nh, th, nw, tw, c).permute(0, 1, 3, 2, 4, 5).reshape(b * nh * nw, th * tw, c)   # b (nh h nw w) c -> (b nh nw) (h w) c
                out = self.base(attn, x, None, attention_mask, temb)
                return out.view(b, nh, nw, th, tw, c).permute(0, 1, 3, 2, 4, 5).reshape(b, nh * th * nw * tw, c)
        return self.base(attn, hidden_states, encoder_hidden_states, attention_mask, temb)


class IdentityAttn:                                  # PAG: softmax(q k^T) -> identity, the attention returns v (then to_out)
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        return attn.to_out[1](attn.to_out[0](attn.to_v(hidden_states)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("fixture")
    ap.add_argument("out")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    unet = UNet2DConditionModel.from_single_file(a.ckpt, torch_dtype=torch.float32).eval()
    for p in unet.parameters():
        p.data = p.data.half().float()
    fx = np.load(a.fixture)
    latent = torch.from_numpy(fx["latent"].astype(np.float32))[None]
    ctx = torch.from_numpy(fx["ctx"].astype(np.float32)).reshape(1, 77, -1)
    t = torch.tensor([float(fx["timestep"][0])])
    run = lambda: unet(latent, t, encoder_hidden_states=ctx).sample[0].numpy()
    out = {"latent": fx["latent"], "ctx": fx["ctx"], "timestep": fx["timestep"]}
    out["eps_plain"] = run()

    blocks.apply_freeu = comfy_freeu
    for name, v2, (b1, b2, s1, s2) in (("freeu", False, (1.1, 1.2, 0.9, 0.2)), ("freeu2", True, (1.3, 1.4, 0.9, 0.2))):
        STATE["v2"], STATE["scales"] = v2, {MC * 4: (b1, s1), MC * 2: (b2, s2)}
        unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)
        out["eps_" + name] = run()
        unet.disable_freeu()

    saved = {}
    for n, m in unet.mid_block.named_modules():
        if n.endswith("attn1"):
            saved[n] = m.processor
            m.set_processor(IdentityAttn())
    assert saved, "no middle-block self-attention found"
    out["eps_pag"] = run()
    for n, m in unet.mid_block.named_modules():
        if n in saved:
            m.set_processor(saved[n])
    # HyperTile (tile 128 -> latent 16, swap 1 = deterministic, depth 0 and 1) on every self-attention
    procs = {}
    for n, m in unet.named_modules():
        if n.endswith("attn1"):
            procs[n] = m.processor
            m.set_processor(HyperTileAttn(m.processor, tuple(latent.shape[-2:])))
    out["eps_hypertile"] = run()
    for n, m in unet.named_modules():
        if n in procs:
            m.set_processor(procs[n])
    np.savez(a.out, **out)
    for k in ("plain", "freeu", "freeu2", "pag", "hypertile"):
        e = out["eps_" + k]
        print(k, e.shape, "absmax", float(np.abs(e).max()), "vs plain", float(np.sqrt(((e - out["eps_plain"]) ** 2).mean() / (out["eps_plain"] ** 2).mean())))


if __name__ == "__main__":
    main()
