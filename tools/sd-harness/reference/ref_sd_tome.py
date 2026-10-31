"""Reference for the UNet's token reduction: ToMe for SD (tomesd's own bipartite_soft_matching_random2d, deterministic destinations) and ToDo.

  python ref_sd_tome.py CKPT.safetensors FIXTURE.npz OUT.npz      (pip install tomesd==0.1.3)

diffusers' UNet (from_single_file, every weight rounded to fp16, fp32 math on the CPU) on the fixture's latent / ctx / timestep (ref/sd_sd15_small_f16w.npz),
with every self-attention (attn1) replaced:
  eps_tome_q       ComfyUI's TomePatchModel placement: the metric is the normalised hidden states, only the queries merged (k / v from every token), output unmerged;
                   ratio 0.5, max_downsample 1
  eps_tome_q2      the same, ratio 0.3, max_downsample 2 (the second resolution too)
  eps_tome_tokens  tomesd's placement: the metric is the block's input (before norm1), q / k / v all from the merged states; ratio 0.5, max_downsample 2
  eps_todo_*       ToDo: every query, k / v from the hidden states downsampled with F.interpolate (nearest; nearest-exact; area), factor 2 at the full resolution,
                   factor 2 at the second for exact / area
Destinations are tomesd's no_rand ones (the top-left token of each 2 x 2 window): the originals' random ones come from torch's generator.
"""
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from tomesd.merge import bipartite_soft_matching_random2d
from diffusers import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock


def sdpa(attn, q, k, v):
    b, heads = q.shape[0], attn.heads
    d = q.shape[-1] // heads
    q, k, v = (t.view(b, -1, heads, d).transpose(1, 2) for t in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, -1, heads * d)
    return attn.to_out[1](attn.to_out[0](o))


def grid(n, H, W, max_ds):
    """tomesd's compute_merge: downsample = ceil(sqrt(original_tokens // tokens)); sides ceil(original / downsample)."""
    ds = int(math.ceil(math.sqrt((H * W) // n)))
    if ds > max_ds:
        return None
    return int(math.ceil(H / ds)), int(math.ceil(W / ds)), ds


class TomeAttn:
    def __init__(self, base, block, lat_hw, ratio, max_ds, tokens):
        self.base, self.block, (self.H, self.W), self.ratio, self.max_ds, self.tokens = base, block, lat_hw, ratio, max_ds, tokens

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        n = hidden_states.shape[1]
        g = grid(n, self.H, self.W, self.max_ds)
        r = int(n * self.ratio)
        if g is None or r <= 0:
            return self.base(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        h, w, _ = g
        if self.tokens:                                      # tomesd: metric = the block's input, q / k / v from the merged normalised states
            m, u = bipartite_soft_matching_random2d(self.block._tome_x, w, h, 2, 2, r, no_rand=True)
            x = m(hidden_states)
            return u(sdpa(attn, attn.to_q(x), attn.to_k(x), attn.to_v(x)))
        m, u = bipartite_soft_matching_random2d(hidden_states, w, h, 2, 2, r, no_rand=True)    # ComfyUI: metric = q (the normalised states)
        return u(sdpa(attn, attn.to_q(m(hidden_states)), attn.to_k(hidden_states), attn.to_v(hidden_states)))


class ToDoAttn:
    def __init__(self, base, lat_hw, factor, factor2, mode):
        self.base, (self.H, self.W), self.factor, self.factor2, self.mode = base, lat_hw, factor, factor2, mode

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        n = hidden_states.shape[1]
        g = grid(n, self.H, self.W, 2)
        f = None if g is None else (self.factor if g[2] == 1 else self.factor2)
        if f is None or f <= 1:
            return self.base(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        h, w, _ = g
        c = hidden_states.shape[-1]
        x = hidden_states.view(1, h, w, c).permute(0, 3, 1, 2)
        kvx = F.interpolate(x, size=(max(1, h // f), max(1, w // f)), mode=self.mode).permute(0, 2, 3, 1).reshape(1, -1, c)
        return sdpa(attn, attn.to_q(hidden_states), attn.to_k(kvx), attn.to_v(kvx))


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
    lat_hw = tuple(latent.shape[-2:])
    run = lambda: unet(latent, t, encoder_hidden_states=ctx).sample[0].numpy()
    out = {"latent": fx["latent"], "ctx": fx["ctx"], "timestep": fx["timestep"]}
    out["eps_plain"] = run()

    blocks = [m for m in unet.modules() if isinstance(m, BasicTransformerBlock)]
    for blk in blocks:                                       # tomesd's metric: the block's input
        def keep(mod, args, kwargs):
            mod._tome_x = args[0] if args else kwargs["hidden_states"]
        blk.register_forward_pre_hook(keep, with_kwargs=True)
    base = {blk: blk.attn1.processor for blk in blocks}

    def use(make):
        for blk in blocks:
            blk.attn1.set_processor(make(base[blk], blk))
        e = run()
        for blk in blocks:
            blk.attn1.set_processor(base[blk])
        return e

    out["eps_tome_q"] = use(lambda b, blk: TomeAttn(b, blk, lat_hw, 0.5, 1, False))
    out["eps_tome_q2"] = use(lambda b, blk: TomeAttn(b, blk, lat_hw, 0.3, 2, False))
    out["eps_tome_tokens"] = use(lambda b, blk: TomeAttn(b, blk, lat_hw, 0.5, 2, True))
    out["eps_todo_nearest"] = use(lambda b, blk: ToDoAttn(b, lat_hw, 2, 1, "nearest"))
    out["eps_todo_exact"] = use(lambda b, blk: ToDoAttn(b, lat_hw, 2, 2, "nearest-exact"))
    out["eps_todo_area"] = use(lambda b, blk: ToDoAttn(b, lat_hw, 2, 2, "area"))
    out["eps_plain_again"] = run()
    np.savez(a.out, **out)
    pl = out["eps_plain"]
    for k in sorted(out):
        if k.startswith("eps_"):
            e = out[k]
            print(k, e.shape, "vs plain", float(np.sqrt(((e - pl) ** 2).mean() / (pl ** 2).mean())))


if __name__ == "__main__":
    main()
