"""Anima ControlNet-LLLite (v2) float64 numpy reference.

Architecture from kohya-ss sd-scripts networks/control_net_lllite_anima.py (Apache-2.0; read at main 88ff6b53) and its
anima_minimal_inference_control_net_lllite.py (the conditioning image). ComfyUI's own AnimaLLLiteApply (GPL-3) was NOT read:
check_lllite.py checks this against Lore's black-box ComfyUI run (anima-run4-lllite / run5-control).

Weights: one .safetensors (anima-lllite-inpainting-v2: bf16; metadata lllite.* gives the dims and targets):
  lllite_conditioning1.*                      the shared conditioning trunk (picture -> one cx [S, cond_emb] for every module)
  lllite_dit_blocks_{i}_{target}.{down|mid|cond_to_film|up}.{weight|bias}, .depth_embed
     target = self_attn_q_proj / self_attn_k_proj / self_attn_v_proj / mlp_layer1 (the INPUT of that DiT Linear is perturbed)

Conditioning trunk (_Conditioning1), picture [C_in, 16 Hp, 16 Wp] -> [Hp * Wp, cond_emb]:
  conv1 4x4 s4 (C_in -> cond_dim/2) GN SiLU; conv2 3x3 p1 GN SiLU; conv3 4x4 s4 (-> cond_dim) GN SiLU;
  resblocks x N (GN SiLU conv3x3 GN SiLU conv3x3, + skip; pre-activation); proj 1x1 (-> cond_emb); flatten (row, col); LayerNorm.
  GroupNorm groups = the largest of 8, 4, 2, 1 dividing the channels; eps 1e-5 (torch defaults), affine. LayerNorm eps 1e-5, affine.
Per module (x = the target Linear's input, [L, D]):
  c = cx + depth_embed;  h = SiLU(down(x));  (gamma, beta) = chunk2(cond_to_film(c));  m = SiLU(mid(cat[c, h]) * (1 + gamma) + beta)
  x' = x + up(m) * strength;  the DiT Linear then runs on x'.
Inpainting picture (cond_in_channels 4, inpaint_masked_input): rgb in [-1, 1] (x / 127.5 - 1), the masked part zeroed
(rgb * (mask < 0.5)), mask channel mask * 2 - 1 (1 = the part to repaint). See check_lllite.py for the variants the taps rule out.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import anima_ref as A

EPS_GN = 1e-5
EPS_LN = 1e-5
TARGETS = {'q': 'self_attn_q_proj', 'k': 'self_attn_k_proj', 'v': 'self_attn_v_proj', 'fc1': 'mlp_layer1'}


def conv2d(x, W, b, stride=1, pad=0, dilation=1):
    """x [C, H, W], W [O, C, kh, kw] (torch layout), b [O] -> [O, Ho, Wo] (cross-correlation, zero padding)."""
    O, C, kh, kw = W.shape
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    win = sliding_window_view(x, ((kh - 1) * dilation + 1, (kw - 1) * dilation + 1), axis=(1, 2))
    win = win[:, ::stride, ::stride, ::dilation, ::dilation]           # [C, Ho, Wo, kh, kw]
    y = np.tensordot(win, W, axes=([0, 3, 4], [1, 2, 3]))              # [Ho, Wo, O]
    y = y.transpose(2, 0, 1)
    return y + b[:, None, None] if b is not None else y


def gn_groups(ch):
    g = 8
    while g > 1 and ch % g != 0:
        g //= 2
    return g


def group_norm(x, weight, bias, eps=EPS_GN):
    """x [C, H, W]; groups per gn_groups (torch nn.GroupNorm: biased variance over each group's channels and pixels)."""
    C = x.shape[0]
    g = gn_groups(C)
    xg = x.reshape(g, -1)
    mu = xg.mean(axis=1, keepdims=True)
    xc = xg - mu
    var = (xc * xc).mean(axis=1, keepdims=True)
    y = (xc / np.sqrt(var + eps)).reshape(x.shape)
    return y * weight[:, None, None] + bias[:, None, None]


class LLLite:
    """An LLLite file: weights + metadata. `cond(picture)` -> cx; `pre_for_block(i, cx, strength)` -> anima_ref.dit_block's pre."""

    def __init__(self, st):
        self.st = st
        self.meta = st.meta or {}
        self.names = set(st.header.keys())
        self.cache = {}

    def w(self, name):
        v = self.cache.get(name)
        if v is None:
            v = self.st.get(name)
            self.cache[name] = v
        return v

    def has(self, name):
        return name in self.names

    def cond(self, pic, eps_gn=EPS_GN, eps_ln=EPS_LN):
        """pic [C_in, H, W] (H, W multiples of 16) -> cx [(H/16) * (W/16), cond_emb]."""
        p = 'lllite_conditioning1.'
        w = self.w
        h = conv2d(pic, w(p + 'conv1.weight'), w(p + 'conv1.bias'), stride=4)
        h = A.silu(group_norm(h, w(p + 'norm1.weight'), w(p + 'norm1.bias'), eps_gn))
        h = conv2d(h, w(p + 'conv2.weight'), w(p + 'conv2.bias'), pad=1)
        h = A.silu(group_norm(h, w(p + 'norm2.weight'), w(p + 'norm2.bias'), eps_gn))
        h = conv2d(h, w(p + 'conv3.weight'), w(p + 'conv3.bias'), stride=4)
        h = A.silu(group_norm(h, w(p + 'norm3.weight'), w(p + 'norm3.bias'), eps_gn))
        r = 0
        while self.has(f'{p}resblocks.{r}.conv1.weight'):
            q = f'{p}resblocks.{r}.'
            t = conv2d(A.silu(group_norm(h, w(q + 'norm1.weight'), w(q + 'norm1.bias'), eps_gn)), w(q + 'conv1.weight'), w(q + 'conv1.bias'), pad=1)
            t = conv2d(A.silu(group_norm(t, w(q + 'norm2.weight'), w(q + 'norm2.bias'), eps_gn)), w(q + 'conv2.weight'), w(q + 'conv2.bias'), pad=1)
            h = h + t
            r += 1
        if self.has(p + 'aspp.proj.0.weight'):
            raise NotImplementedError('ASPP conditioning (use_aspp) is not in this reference')
        h = conv2d(h, w(p + 'proj.weight'), w(p + 'proj.bias'))
        C = h.shape[0]
        h = h.reshape(C, -1).T                                             # [S, C], S in (row, col) order = the DiT's token order
        return A.layer_norm(h, eps_ln, w(p + 'out_norm.weight'), w(p + 'out_norm.bias'))

    def module(self, i, target, x, cx, strength=1.0):
        name = f'lllite_dit_blocks_{i}_{TARGETS[target]}'
        if not self.has(name + '.down.weight'):
            return x
        w = self.w
        c = cx + w(name + '.depth_embed')
        h = A.silu(A.linear(x, w(name + '.down.weight'), w(name + '.down.bias')))
        gb = A.linear(c, w(name + '.cond_to_film.weight'), w(name + '.cond_to_film.bias'))
        gamma, beta = np.split(gb, 2, axis=-1)
        cb = np.broadcast_to(c, h.shape[:-1] + (c.shape[-1],))
        m = A.linear(np.concatenate([cb, h], axis=-1), w(name + '.mid.weight'), w(name + '.mid.bias'))
        m = A.silu(m * (1.0 + gamma) + beta)
        return x + A.linear(m, w(name + '.up.weight'), w(name + '.up.bias')) * strength

    def pre_for_block(self, i, cx, strength=1.0):
        return lambda target, x: self.module(i, target, x, cx, strength)

    def drop_block(self, i):
        pre = f'lllite_dit_blocks_{i}_'
        for k in [k for k in self.cache if k.startswith(pre)]:
            del self.cache[k]


def inpaint_picture(rgb_u8, mask01, variant='kohya'):
    """rgb_u8 [H, W, 3] uint8, mask01 [H, W] in {0, 1} (1 = repaint) -> [4, H, W].
    variants (check_lllite.py's red checks): 'kohya' (masked rgb = 0 in [-1, 1], mask channel in {-1, 1}),
    'black' (masked rgb = -1), 'nomask_rgb' (rgb not masked), 'mask01' (mask channel in {0, 1}), 'invert' (mask channel inverted)."""
    rgb = rgb_u8.astype(np.float64).transpose(2, 0, 1) / 127.5 - 1.0
    m = mask01.astype(np.float64)[None]
    keep = (m < 0.5).astype(np.float64)
    mch = m * 2.0 - 1.0
    if variant == 'kohya':
        rgb = rgb * keep
    elif variant == 'black':
        rgb = rgb * keep - (1.0 - keep)
    elif variant == 'nomask_rgb':
        pass
    elif variant == 'mask01':
        rgb = rgb * keep
        mch = m
    elif variant == 'invert':
        rgb = rgb * keep
        mch = -mch
    else:
        raise ValueError(variant)
    return np.concatenate([rgb, mch], axis=0)
