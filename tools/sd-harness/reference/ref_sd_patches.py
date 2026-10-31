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
    """ComfyUI's attn1_patch rearranges ONLY q; k and v are projected from the untiled tokens and attention_pytorch reshapes them with q's batch
    (t.view(b, -1, heads, dim_head) with b = q.shape[0]), so query tile t attends to the flat tokens [t * L, (t + 1) * L) of its own picture.
    tile_kv=True is true tiled attention (q, k and v of the tile)."""
    def __init__(self, base, lat_hw, tile_size=128, swap_size=1, max_depth=1, scale_depth=False, tile_kv=False):
        self.base, (self.H, self.W), self.tile_kv = base, lat_hw, tile_kv
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
                if self.tile_kv:
                    out = self.base(attn, x, None, attention_mask, temb)
                else:
                    heads = attn.heads
                    bq, d = x.shape[0], c // heads
                    q = attn.to_q(x).view(bq, -1, heads, d).transpose(1, 2)
                    k = attn.to_k(hidden_states).reshape(bq, -1, heads, d).transpose(1, 2)        # the untiled tokens, split by q's batch: the band of tile t
                    v = attn.to_v(hidden_states).reshape(bq, -1, heads, d).transpose(1, 2)
                    o = torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bq, -1, c)
                    out = attn.to_out[1](attn.to_out[0](o))
                return out.view(b, nh, nw, th, tw, c).permute(0, 1, 3, 2, 4, 5).reshape(b, nh * th * nw * tw, c)
        return self.base(attn, hidden_states, encoder_hidden_states, attention_mask, temb)


class SimAttn:                                       # ComfyUI SAG's attention_basic_with_sim: the same output, the softmax kept
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        b, n, c = hidden_states.shape
        heads = attn.heads
        d = c // heads
        q, k, v = (t.view(b, n, heads, d).permute(0, 2, 1, 3) for t in (attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)))
        sim = ((q @ k.transpose(-1, -2)) * d ** -0.5).softmax(-1)
        STATE["sim"] = sim.reshape(b * heads, n, n)
        return attn.to_out[1](attn.to_out[0]((sim @ v).permute(0, 2, 1, 3).reshape(b, n, c)))


def gaussian_blur_2d(img, kernel_size, sigma):        # ComfyUI comfy_extras/nodes_sag.py
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    x_kernel = pdf / pdf.sum()
    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])
    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]
    img = torch.nn.functional.pad(img, padding, mode="reflect")
    return torch.nn.functional.conv2d(img, kernel2d, groups=img.shape[-3])


def create_blur_map(x0, attn, sigma=3.0, threshold=1.0):   # ComfyUI comfy_extras/nodes_sag.py
    _, hw1, hw2 = attn.shape
    b, _, lh, lw = x0.shape
    attn = attn.reshape(b, -1, hw1, hw2)
    mask = attn.mean(1, keepdim=False).sum(1, keepdim=False) > threshold
    total = mask.shape[-1]
    x = round(math.sqrt((lh / lw) * total))
    xx = None
    for i in range(0, math.floor(math.sqrt(total) / 2)):
        for j in [(x + i), max(1, x - i)]:
            if total % j == 0:
                xx = j
                break
        if xx is not None:
            break
    x = xx
    y = total // x
    mask = (mask.reshape(b, x, y).unsqueeze(1).type(attn.dtype))
    mask = torch.nn.functional.interpolate(mask, (lh, lw))
    blurred = gaussian_blur_2d(x0, kernel_size=9, sigma=sigma)
    return blurred * mask + x0 * (1 - mask)


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
            m.set_processor(HyperTileAttn(procs[n], tuple(latent.shape[-2:]), tile_kv=True))
    out["eps_hypertile_kv"] = run()
    for n, m in unet.named_modules():
        if n in procs:
            m.set_processor(procs[n])
    # SAG: the middle block's first self-attention map (transformer_blocks[0].attn1), its per-key mass, and ComfyUI's blur map of a test latent
    blk = unet.mid_block.attentions[0].transformer_blocks[0].attn1
    saved_p = blk.processor
    blk.set_processor(SimAttn())
    out["eps_sag_record"] = run()                     # must equal eps_plain: recording does not change the prediction
    blk.set_processor(saved_p)
    sim = STATE["sim"]
    heads = sim.shape[0]
    out["sag_mass"] = sim.reshape(1, heads, sim.shape[1], sim.shape[2]).mean(1).sum(1)[0].numpy()
    x0 = latent                                         # any latent: the blur map is a function of x0 and the mass
    out["sag_degraded"] = create_blur_map(x0, sim, sigma=2.0, threshold=1.0)[0].numpy()
    print("sag mass", out["sag_mass"].shape, "min", float(out["sag_mass"].min()), "max", float(out["sag_mass"].max()), "above 1:", int((out["sag_mass"] > 1).sum()),
          "record vs plain", float(np.abs(out["eps_sag_record"] - out["eps_plain"]).max()))
    np.savez(a.out, **out)
    for k in ("plain", "freeu", "freeu2", "pag", "hypertile", "hypertile_kv"):
        e = out["eps_" + k]
        print(k, e.shape, "absmax", float(np.abs(e).max()), "vs plain", float(np.sqrt(((e - out["eps_plain"]) ** 2).mean() / (out["eps_plain"] ** 2).mean())))


if __name__ == "__main__":
    main()
