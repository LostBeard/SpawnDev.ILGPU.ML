"""Anima (Cosmos-Predict2-2B MiniTrainDIT + LLM adapter) float64 numpy reference.

Ground truth for the SpawnDev.ILGPU.ML GPU port. Every convention here was checked against Lore's
ComfyUI black-box taps (see check_anima.py and ANIMA-SPEC.md for the numbers).

Sources: the DiT semantics come from Diffusers' transformer_cosmos.py (Apache-2.0); the LLM adapter has
no readable reference and was derived ONLY from the weight header (net.llm_adapter.*) and the tap tensors.

All functions take float64 arrays and a weight getter `w(name) -> float64 ndarray` (header names, net.*).
Linear layers follow torch: y = x @ W.T (+ b), W stored [out, in].
"""
import numpy as np
from scipy.special import erf

# ----------------------------------------------------------------------------------------------
# DiT constants (Anima = Cosmos-Predict2-2B)
# ----------------------------------------------------------------------------------------------
D_MODEL = 2048
N_HEADS = 16
HEAD_DIM = 128
N_BLOCKS = 28
CTX_DIM = 1024
CTX_LEN = 512           # the DiT's context is padded with zero rows to 512
ADALN_LORA = 256
PATCH = (1, 2, 2)       # (p_t, p_h, p_w)
IN_CH = 16              # latent channels (+1 zero padding-mask channel = 17)
OUT_CH = 16
EPS_LN = 1e-6           # adaLN LayerNorm (no affine)
EPS_QK = 1e-6           # q/k RMSNorm (per head) - determined from the taps, see check_anima.py
EPS_TNORM = 1e-6        # t_embedding_norm RMSNorm

# RoPE (3D, per head of 128): 64 rotation pairs = 22 temporal + 21 row (h) + 21 column (w)
ROPE_DIM_H = HEAD_DIM // 6 * 2      # 42
ROPE_DIM_W = HEAD_DIM // 6 * 2      # 42
ROPE_DIM_T = HEAD_DIM - ROPE_DIM_H - ROPE_DIM_W  # 44
ROPE_SCALE_HW = 4.0                 # NTK scale for h and w (from the taps: theta_hw = 10000 * 4^(42/40))
ROPE_SCALE_T = 1.0                  # temporal scale: NOT observable for images (t = 0); see the spec

# Adapter constants
A_DIM = 1024
A_HEADS = 16
A_HEAD_DIM = 64
A_BLOCKS = 6
A_ROPE_BASE = 10000.0
EPS_A = 1e-6            # adapter RMSNorm eps (block norms, q/k norms, final norm) - from the taps


# ----------------------------------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------------------------------
def linear(x, W, b=None):
    y = x @ W.T
    if b is not None:
        y = y + b
    return y


def silu(x):
    return x / (1.0 + np.exp(-x))


def gelu_exact(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def rms_norm(x, weight, eps):
    ms = np.mean(x * x, axis=-1, keepdims=True)
    y = x / np.sqrt(ms + eps)
    return y * weight if weight is not None else y


def layer_norm(x, eps, weight=None, bias=None):
    mu = np.mean(x, axis=-1, keepdims=True)
    xc = x - mu
    var = np.mean(xc * xc, axis=-1, keepdims=True)
    y = xc / np.sqrt(var + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def softmax(s, axis=-1):
    s = s - np.max(s, axis=axis, keepdims=True)
    e = np.exp(s)
    return e / np.sum(e, axis=axis, keepdims=True)


def sdpa(q, k, v):
    """q [B,H,Lq,d], k/v [B,H,Lk,d] -> [B,H,Lq,d]; scale 1/sqrt(d), no mask."""
    d = q.shape[-1]
    s = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(d)
    return softmax(s) @ v


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def apply_rope_half(x, cos, sin):
    """Rotate-half RoPE: dim i pairs with dim i + d/2 (cos/sin already duplicated to d)."""
    return x * cos + rotate_half(x) * sin


def apply_rope_interleaved(x, cos, sin):
    """Interleaved RoPE: dim 2j pairs with 2j+1 (cos/sin given per pair, d/2 wide)."""
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    out = np.empty_like(x)
    out[..., 0::2] = y0
    out[..., 1::2] = y1
    return out


# ----------------------------------------------------------------------------------------------
# DiT stages
# ----------------------------------------------------------------------------------------------
def pad_mask_channel(x):
    """x [B,16,T,H,W] -> [B,17,T,H,W]; channel 16 = padding mask = all zeros for a full image."""
    B, C, T, H, W = x.shape
    return np.concatenate([x, np.zeros((B, 1, T, H, W))], axis=1)


def x_embedder(x17, w):
    """[B,17,T,H,W] -> [B,T,H/2,W/2,2048]. Patch features ordered (c, p_t, p_h, p_w), c slowest.
    Weight net.x_embedder.proj.1.weight [2048, 68], no bias."""
    B, C, T, H, W = x17.shape
    pt, ph, pw = PATCH
    x = x17.reshape(B, C, T // pt, pt, H // ph, ph, W // pw, pw)
    x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, T // pt, H // ph, W // pw, C * pt * ph * pw)
    return linear(x, w('net.x_embedder.proj.1.weight'))


def timestep_proj(t, dim=D_MODEL):
    """t [B] (the timestep the DiT is called with = sigma, NOT sigma*1000) -> [B, 2048].
    freq_k = exp(-ln(10000) * k / 1024), k = 0..1023; out = [cos(t*freq), sin(t*freq)] (cos FIRST)."""
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float64) / half)
    arg = np.asarray(t, dtype=np.float64)[:, None] * freqs[None, :]
    return np.concatenate([np.cos(arg), np.sin(arg)], axis=-1)


def t_embedder(t, w):
    """t [B] -> (t_proj [B,2048], adaln_lora [B,6144], emb_norm [B,2048])."""
    tp = timestep_proj(t)
    h = silu(linear(tp, w('net.t_embedder.1.linear_1.weight')))
    adaln = linear(h, w('net.t_embedder.1.linear_2.weight'))
    emb = rms_norm(tp, w('net.t_embedding_norm.weight'), EPS_TNORM)
    return tp, adaln, emb


def rope_angles(T, Hp, Wp, scale_t=ROPE_SCALE_T, scale_hw=ROPE_SCALE_HW):
    """Per-token angles [T*Hp*Wp, 64]: pairs 0..21 temporal, 22..42 row, 43..63 column.
    Token order = (t, row, col) row-major, the same order as the patch tokens."""
    dh, dw, dt = ROPE_DIM_H, ROPE_DIM_W, ROPE_DIM_T
    th_h = 10000.0 * scale_hw ** (dh / (dh - 2))
    th_w = 10000.0 * scale_hw ** (dw / (dw - 2))
    th_t = 10000.0 * scale_t ** (dt / (dt - 2))
    fh = 1.0 / th_h ** (np.arange(0, dh, 2, dtype=np.float64) / dh)
    fw = 1.0 / th_w ** (np.arange(0, dw, 2, dtype=np.float64) / dw)
    ft = 1.0 / th_t ** (np.arange(0, dt, 2, dtype=np.float64) / dt)
    tt, rr, cc = np.meshgrid(np.arange(T), np.arange(Hp), np.arange(Wp), indexing='ij')
    tt, rr, cc = tt.ravel(), rr.ravel(), cc.ravel()
    return np.concatenate([np.outer(tt, ft), np.outer(rr, fh), np.outer(cc, fw)], axis=1)


def rope_matrices(ang):
    """[L,64] angles -> [L,64,2,2] rotation matrices [[cos,-sin],[sin,cos]] (ComfyUI's pos_embedder form)."""
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def dit_rope_cos_sin(ang):
    """[L,64] -> cos/sin [L,128] duplicated as cat([a, a]) for rotate-half pairing (dim j with j+64)."""
    a = np.concatenate([ang, ang], axis=-1)
    return np.cos(a), np.sin(a)


def _heads(x, H):
    B, L, D = x.shape
    return x.reshape(B, L, H, D // H).transpose(0, 2, 1, 3)


def _merge(x):
    B, H, L, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, L, H * d)


def adaln_mod(emb, adaln, w, prefix):
    """shift, scale, gate each [B,1,2048]: Linear(256->6144)(Linear(2048->256)(SiLU(emb))) + adaln_lora."""
    m = linear(linear(silu(emb), w(prefix + '.1.weight')), w(prefix + '.2.weight')) + adaln
    sh, sc, g = np.split(m, 3, axis=-1)
    return sh[:, None, :], sc[:, None, :], g[:, None, :]


def dit_attention(x, ctx, w, p, cos=None, sin=None, rope_mode='half', eps_qk=EPS_QK, pre=None):
    """Self (ctx None) or cross attention. q/k/v Linear no bias; per-head RMSNorm(128, eps) on q and k with
    weights q_norm/k_norm; RoPE (self only) on q and k AFTER the norm; SDPA scale 1/sqrt(128); output_proj.
    pre(target, x) -> x' (optional, self-attention only): perturbs the INPUT of q_proj / k_proj / v_proj (targets 'q', 'k', 'v'),
    as ControlNet-LLLite does (anima_lllite_ref.py)."""
    src = x if ctx is None else ctx
    xq, xk, xv = (x, src, src) if pre is None else (pre('q', x), pre('k', src), pre('v', src))
    q = _heads(linear(xq, w(p + '.q_proj.weight')), N_HEADS)
    k = _heads(linear(xk, w(p + '.k_proj.weight')), N_HEADS)
    v = _heads(linear(xv, w(p + '.v_proj.weight')), N_HEADS)
    q = rms_norm(q, w(p + '.q_norm.weight'), eps_qk)
    k = rms_norm(k, w(p + '.k_norm.weight'), eps_qk)
    if cos is not None:
        if rope_mode == 'half':
            q = apply_rope_half(q, cos, sin)
            k = apply_rope_half(k, cos, sin)
        else:
            q = apply_rope_interleaved(q, cos, sin)
            k = apply_rope_interleaved(k, cos, sin)
    o = _merge(sdpa(q, k, v))
    return linear(o, w(p + '.output_proj.weight'))


def dit_block(x, emb, ctx, adaln, cos, sin, w, i, rope_mode='half', eps_qk=EPS_QK, gelu=gelu_exact, eps_ln=EPS_LN, pre=None):
    """x [B,L,2048], emb [B,2048] (t_embedding_norm output), ctx [B,512,1024], adaln [B,6144].
    pre(target, x) -> x' (optional): LLLite-style input perturbations of self_attn q/k/v ('q', 'k', 'v') and mlp.layer1 ('fc1')."""
    p = f'net.blocks.{i}'
    sh, sc, g = adaln_mod(emb, adaln, w, p + '.adaln_modulation_self_attn')
    n = layer_norm(x, eps_ln) * (1.0 + sc) + sh
    x = x + g * dit_attention(n, None, w, p + '.self_attn', cos, sin, rope_mode, eps_qk, pre=pre)
    sh, sc, g = adaln_mod(emb, adaln, w, p + '.adaln_modulation_cross_attn')
    n = layer_norm(x, eps_ln) * (1.0 + sc) + sh
    x = x + g * dit_attention(n, ctx, w, p + '.cross_attn', eps_qk=eps_qk)
    sh, sc, g = adaln_mod(emb, adaln, w, p + '.adaln_modulation_mlp')
    n = layer_norm(x, eps_ln) * (1.0 + sc) + sh
    h = gelu(linear(n if pre is None else pre('fc1', n), w(p + '.mlp.layer1.weight')))
    x = x + g * linear(h, w(p + '.mlp.layer2.weight'))
    return x


def final_layer(x, emb, adaln, w):
    """x [B,L,2048] -> [B,L,64]. shift, scale = chunk2(Linear(256->4096)(Linear(2048->256)(SiLU(emb))) + adaln[:, :4096])."""
    m = linear(linear(silu(emb), w('net.final_layer.adaln_modulation.1.weight')),
               w('net.final_layer.adaln_modulation.2.weight')) + adaln[:, :2 * D_MODEL]
    sh, sc = np.split(m, 2, axis=-1)
    n = layer_norm(x, EPS_LN) * (1.0 + sc[:, None, :]) + sh[:, None, :]
    return linear(n, w('net.final_layer.linear.weight'))


def unpatchify(y, T, Hp, Wp):
    """y [B, T*Hp*Wp, 64] -> [B, 16, T, 2*Hp, 2*Wp]. Channel index = (p_h*2 + p_w)*1*16 + p_t*16 + c,
    i.e. unflatten 64 as (p_h, p_w, p_t, C) - NOT the patchify order (C, p_t, p_h, p_w)."""
    B = y.shape[0]
    pt, ph, pw = PATCH
    y = y.reshape(B, T, Hp, Wp, ph, pw, pt, OUT_CH)
    y = y.transpose(0, 7, 1, 6, 2, 4, 3, 5)  # B, C, T, pt, Hp, ph, Wp, pw
    return y.reshape(B, OUT_CH, T * pt, Hp * ph, Wp * pw)


def pad_context(ctx, n=CTX_LEN):
    """[B,M,1024] -> [B,512,1024] with zero rows after the M real rows."""
    B, M, D = ctx.shape
    out = np.zeros((B, n, D))
    out[:, :M] = ctx
    return out


def denoise_flow(x, t, v):
    """ComfyUI's model output for this model (CONST / rectified flow): denoised = x - sigma * v,
    where v is dit_forward's raw output and sigma = the timestep the DiT was called with."""
    return x - np.asarray(t, dtype=np.float64)[:, None, None, None, None] * v


def dit_forward(x, t, ctx, w, block_hook=None, pre_for_block=None):
    """Full DiT: x [B,16,T,H,W], t [B] (sigma), ctx [B,512,1024] -> raw velocity v [B,16,T,H,W]
    (NOT the denoised latent: see denoise_flow). pre_for_block(i) -> dit_block's pre (or None) for an LLLite."""
    B, C, T, H, W = x.shape
    Hp, Wp = H // PATCH[1], W // PATCH[2]
    h = x_embedder(pad_mask_channel(x), w).reshape(B, T * Hp * Wp, D_MODEL)
    _, adaln, emb = t_embedder(t, w)
    cos, sin = dit_rope_cos_sin(rope_angles(T, Hp, Wp))
    for i in range(N_BLOCKS):
        h = dit_block(h, emb, ctx, adaln, cos, sin, w, i, pre=None if pre_for_block is None else pre_for_block(i))
        if block_hook is not None:
            block_hook(i, h)
    y = final_layer(h, emb, adaln, w)
    return unpatchify(y, T, Hp, Wp)


# ----------------------------------------------------------------------------------------------
# LLM adapter (net.llm_adapter.*) - derived from the header + taps only
# ----------------------------------------------------------------------------------------------
def adapter_rope(n, base=A_ROPE_BASE, d=A_HEAD_DIM):
    """positions 0..n-1 -> cos/sin [n, 64]: inv_freq_k = base^(-2k/64), k = 0..31, duplicated cat([f, f])."""
    inv = 1.0 / base ** (np.arange(0, d, 2, dtype=np.float64) / d)
    f = np.outer(np.arange(n, dtype=np.float64), inv)
    e = np.concatenate([f, f], axis=-1)
    return np.cos(e), np.sin(e)


def adapter_attention(x, src, w, p, cos_q, sin_q, cos_k, sin_k, eps=EPS_A):
    q = _heads(linear(x, w(p + '.q_proj.weight')), A_HEADS)
    k = _heads(linear(src, w(p + '.k_proj.weight')), A_HEADS)
    v = _heads(linear(src, w(p + '.v_proj.weight')), A_HEADS)
    q = rms_norm(q, w(p + '.q_norm.weight'), eps)
    k = rms_norm(k, w(p + '.k_norm.weight'), eps)
    if cos_q is not None:
        q = apply_rope_half(q, cos_q, sin_q)
    if cos_k is not None:
        k = apply_rope_half(k, cos_k, sin_k)
    o = _merge(sdpa(q, k, v))
    return linear(o, w(p + '.o_proj.weight'))


def adapter_block(x, ctx, w, i, cos_x, sin_x, cos_c, sin_c, eps=EPS_A, cross_rope=True, gelu=gelu_exact):
    """x [1,M,1024] (T5 side), ctx [1,N,1024] (Qwen3 states, raw, no norm)."""
    p = f'net.llm_adapter.blocks.{i}'
    n = rms_norm(x, w(p + '.norm_self_attn.weight'), eps)
    x = x + adapter_attention(n, n, w, p + '.self_attn', cos_x, sin_x, cos_x, sin_x, eps)
    n = rms_norm(x, w(p + '.norm_cross_attn.weight'), eps)
    if cross_rope:
        x = x + adapter_attention(n, ctx, w, p + '.cross_attn', cos_x, sin_x, cos_c, sin_c, eps)
    else:
        x = x + adapter_attention(n, ctx, w, p + '.cross_attn', None, None, None, None, eps)
    n = rms_norm(x, w(p + '.norm_mlp.weight'), eps)
    h = gelu(linear(n, w(p + '.mlp.0.weight'), w(p + '.mlp.0.bias')))
    x = x + linear(h, w(p + '.mlp.2.weight'), w(p + '.mlp.2.bias'))
    return x


def adapter_forward(qwen, t5_ids, w, block_hook=None):
    """qwen [1,N,1024] (Qwen3-0.6B last hidden state after its final norm), t5_ids [1,M] int ->
    [1,M,1024] (the unpadded context; the DiT pads it to 512 with zero rows)."""
    x = w('net.llm_adapter.embed.weight')[np.asarray(t5_ids)]   # [1,M,1024]; in_proj = Identity
    M, N = x.shape[1], qwen.shape[1]
    cos_x, sin_x = adapter_rope(M)
    cos_c, sin_c = adapter_rope(N)
    for i in range(A_BLOCKS):
        x = adapter_block(x, qwen, w, i, cos_x, sin_x, cos_c, sin_c)
        if block_hook is not None:
            block_hook(i, x)
    x = linear(x, w('net.llm_adapter.out_proj.weight'), w('net.llm_adapter.out_proj.bias'))
    return rms_norm(x, w('net.llm_adapter.norm.weight'), EPS_A)
