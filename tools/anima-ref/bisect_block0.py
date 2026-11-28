"""Where the GPU port's block-0 error comes from: its block-0 intermediates (Anima_DiT_StagesMatchComfyUiTaps with ANIMA_DUMP_B0=<dir>, row 0 of run3's
call_0) against float64 from run3's tapped block-0 inputs, op by op. Usage: python bisect_block0.py <dump dir>

The GPU's taps (AnimaDiT.Forward, block 0): b0_mod_sa = the self-attention modulation [shift, scale, gate] (6144); b0_q / b0_k = q / k after the per-head
RMSNorm and RoPE, b0_v = v ([N, 2048], head-major); b0_attn_sa = the attention output before output_proj; b0_after_sa = x after the gated self-attention
residual; b0_kc = the cross-attention k after its RMSNorm, b0_vc its v ([512, 2048]); b0_attn_ca = the cross-attention output; b0_mlp_hidden = GELU(layer1).
The GPU's own input to block 0 is its x_embedder output (2.4e-8 from the tap), so every row below is that op's error plus what came before it.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anima_ref as A  # noqa: E402
from check_lllite import report  # noqa: E402
from st_reader import SafeTensors  # noqa: E402
from taps import tap  # noqa: E402

dump = sys.argv[1]
st = SafeTensors(r'D:\Projects\anima\anima-preview3-base.safetensors')
w = lambda n: st.get(n)  # noqa: E731


def gpu(name):
    hit = glob.glob(os.path.join(dump, f'{name}_f32_*.raw'))
    assert len(hit) == 1, (name, hit)
    return np.fromfile(hit[0], dtype=np.float32).astype(np.float64)


x = tap('blocks_0_0_pre-in0_0')[0].reshape(1, -1, A.D_MODEL)
emb = tap('blocks_0_0_pre-in1_0')[0:1, 0]
ctx = tap('blocks_0_0_pre-in2_0')[0:1]
adaln = tap('blocks_0_0_pre-kw-adaln_lora_B_T_3D0')[0:1, 0]
cos, sin = A.dit_rope_cos_sin(A.rope_angles(1, 32, 32))
p = 'net.blocks.0'
H = A.N_HEADS


def heads_rms_rope(y, norm, rope):
    h = A._heads(y, H)
    h = A.rms_norm(h, w(norm), A.EPS_QK)
    if rope:
        h = A.apply_rope_half(h, cos, sin)
    return A._merge(h)


# self attention
m = A.linear(A.linear(A.silu(emb), w(p + '.adaln_modulation_self_attn.1.weight')), w(p + '.adaln_modulation_self_attn.2.weight')) + adaln
report('b0 self-attn modulation [shift, scale, gate]', gpu('b0_mod_sa'), m.ravel())
sh, sc, g = np.split(m, 3, axis=-1)
n = A.layer_norm(x, A.EPS_LN) * (1 + sc[:, None]) + sh[:, None]
q = heads_rms_rope(A.linear(n, w(p + '.self_attn.q_proj.weight')), p + '.self_attn.q_norm.weight', True)
k = heads_rms_rope(A.linear(n, w(p + '.self_attn.k_proj.weight')), p + '.self_attn.k_norm.weight', True)
v = A.linear(n, w(p + '.self_attn.v_proj.weight'))
report('b0 q (RMSNorm + RoPE)', gpu('b0_q'), q.ravel())
report('b0 k (RMSNorm + RoPE)', gpu('b0_k'), k.ravel())
report('b0 v', gpu('b0_v'), v.ravel())
att = A._merge(A.sdpa(A._heads(q, H), A._heads(k, H), A._heads(v, H)))
report('b0 self-attention output', gpu('b0_attn_sa'), att.ravel())
x1 = x + g[:, None] * A.linear(att, w(p + '.self_attn.output_proj.weight'))
report('b0 x after the self-attention residual', gpu('b0_after_sa'), x1.ravel())
report('b0 self-attention UPDATE (after - before)', gpu('b0_after_sa') - x.ravel(), (x1 - x).ravel(), 'the residual hides small errors')

# cross attention
m2 = A.linear(A.linear(A.silu(emb), w(p + '.adaln_modulation_cross_attn.1.weight')), w(p + '.adaln_modulation_cross_attn.2.weight')) + adaln
sh2, sc2, g2 = np.split(m2, 3, axis=-1)
n2 = A.layer_norm(x1, A.EPS_LN) * (1 + sc2[:, None]) + sh2[:, None]
q2 = heads_rms_rope(A.linear(n2, w(p + '.cross_attn.q_proj.weight')), p + '.cross_attn.q_norm.weight', False)
kc = heads_rms_rope(A.linear(ctx, w(p + '.cross_attn.k_proj.weight')), p + '.cross_attn.k_norm.weight', False)
vc = A.linear(ctx, w(p + '.cross_attn.v_proj.weight'))
report('b0 cross k (RMSNorm)', gpu('b0_kc'), kc.ravel())
report('b0 cross v', gpu('b0_vc'), vc.ravel())
att2 = A._merge(A.sdpa(A._heads(q2, H), A._heads(kc, H), A._heads(vc, H)))
report('b0 cross-attention output', gpu('b0_attn_ca'), att2.ravel())
x2 = x1 + g2[:, None] * A.linear(att2, w(p + '.cross_attn.output_proj.weight'))

# MLP
m3 = A.linear(A.linear(A.silu(emb), w(p + '.adaln_modulation_mlp.1.weight')), w(p + '.adaln_modulation_mlp.2.weight')) + adaln
sh3, sc3, g3 = np.split(m3, 3, axis=-1)
n3 = A.layer_norm(x2, A.EPS_LN) * (1 + sc3[:, None]) + sh3[:, None]
hmid = A.gelu_exact(A.linear(n3, w(p + '.mlp.layer1.weight')))
report('b0 MLP hidden GELU(layer1)', gpu('b0_mlp_hidden'), hmid.ravel())
y = x2 + g3[:, None] * A.linear(hmid, w(p + '.mlp.layer2.weight'))
report('b0 block output (float64) vs the ComfyUI tap', y.reshape(1, 1, 32, 32, -1), tap('blocks_0_0_out0')[0:1], 'reference check')
