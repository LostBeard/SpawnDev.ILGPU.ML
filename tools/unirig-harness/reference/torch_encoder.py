"""My own PyTorch implementation of the UniRig shape encoder, written from the checkpoint's tensor names/shapes and the 3DShape2VecSet / Michelangelo papers,
checked against the reference's latents (golden_skeleton.py). This is the readable spec the C# port follows; it deliberately shares no code with the reference.

  python -I torch_encoder.py          compares against UNIRIG_DIR/golden/skeleton/latents.npy for the golden's points

Architecture (all widths from the checkpoint): input_proj Linear(54 -> 512) over [fourier(xyz) (51) | normal (3)]; ONE cross-attention block whose queries are the
1024 farthest-point-sampled tokens (embedded the same way) and whose keys/values are the point cloud; 16 residual self-attention blocks (8 heads); ln_post; then
UniRigAR.output_proj Linear(512 -> 1024). Residual blocks are pre-LN with a 4x GELU MLP.
"""
import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F

UNIRIG_DIR = os.environ.get('UNIRIG_DIR', 'E:/spawndev-unirig')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_ckpt import load_checkpoint


def fourier(x, num_freqs=8):
    """[..., 3] -> [..., 3 + 3 * 8 * 2]: the input, then sin and cos of x * 2^k (k = 0..7), coordinate-major."""
    freqs = 2.0 ** torch.arange(num_freqs, dtype=x.dtype)
    emb = (x[..., None] * freqs).reshape(*x.shape[:-1], -1)
    return torch.cat([x, emb.sin(), emb.cos()], dim=-1)


def layer_norm(x, p, name):
    return F.layer_norm(x, x.shape[-1:], p[name + '.weight'], p[name + '.bias'], 1e-5)


def mlp(x, p, name):
    return F.linear(F.gelu(F.linear(x, p[name + '.c_fc.weight'], p[name + '.c_fc.bias'])), p[name + '.c_proj.weight'], p[name + '.c_proj.bias'])


def attend(q, k, v, heads):
    """q [B,Nq,C], k/v [B,Nk,C]: standard multi-head attention, scale 1/sqrt(head dim)."""
    B, Nq, C = q.shape; D = C // heads
    q = q.view(B, Nq, heads, D).transpose(1, 2); k = k.view(B, -1, heads, D).transpose(1, 2); v = v.view(B, -1, heads, D).transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Nq, C)


def split_heads_interleaved(qkv, heads, parts):
    """The fused projection is laid out per head: [H, parts * D] -> parts tensors of [B, N, H * D] (c_qkv: 3 parts, c_kv: 2 parts)."""
    B, N, _ = qkv.shape
    t = qkv.view(B, N, heads, -1)
    D = t.shape[-1] // parts
    return [t[..., i * D:(i + 1) * D].reshape(B, N, heads * D) for i in range(parts)]


class Encoder:
    def __init__(self, state_dict, heads=8):
        pre = 'model.mesh_encoder.encoder.'
        self.p = {k[len(pre):]: v for k, v in state_dict.items() if k.startswith(pre)}
        self.out_w, self.out_b = state_dict['model.output_proj.weight'], state_dict['model.output_proj.bias']
        self.heads = heads

    def embed(self, xyz, nrm):
        return F.linear(torch.cat([fourier(xyz), nrm], -1), self.p['input_proj.weight'], self.p['input_proj.bias'])

    def __call__(self, points, normals, presample_idx, fps_idx, full_keys=True):
        p, H = self.p, self.heads
        pts, nrm = torch.from_numpy(points)[None], torch.from_numpy(normals)[None]
        pre_p, pre_n = pts[:, presample_idx], nrm[:, presample_idx]
        q_in = self.embed(pre_p[:, fps_idx], pre_n[:, fps_idx])                      # 1024 query tokens
        kv_in = self.embed(pts, nrm) if full_keys else self.embed(pre_p, pre_n)      # keys / values
        ca = 'cross_attn.'
        q = F.linear(layer_norm(q_in, p, ca + 'ln_1'), p[ca + 'attn.c_q.weight'])
        k, v = split_heads_interleaved(F.linear(layer_norm(kv_in, p, ca + 'ln_2'), p[ca + 'attn.c_kv.weight']), H, 2)
        x = q_in + F.linear(attend(q, k, v, H), p[ca + 'attn.c_proj.weight'], p[ca + 'attn.c_proj.bias'])
        x = x + mlp(layer_norm(x, p, ca + 'ln_3'), p, ca + 'mlp')
        for i in range(16):
            b = f'self_attn.resblocks.{i}.'
            q, k, v = split_heads_interleaved(F.linear(layer_norm(x, p, b + 'ln_1'), p[b + 'attn.c_qkv.weight']), H, 3)
            x = x + F.linear(attend(q, k, v, H), p[b + 'attn.c_proj.weight'], p[b + 'attn.c_proj.bias'])
            x = x + mlp(layer_norm(x, p, b + 'ln_2'), p, b + 'mlp')
        x = layer_norm(x, p, 'ln_post')
        return F.linear(x, self.out_w, self.out_b)                                   # [1, 1024, 1024]


if __name__ == '__main__':
    g = os.path.join(UNIRIG_DIR, 'golden', 'skeleton')
    sd = load_checkpoint(os.path.join(UNIRIG_DIR, 'ckpt', 'skeleton', 'model.ckpt'))['state_dict']
    enc = Encoder(sd)
    pts, nrm = np.load(g + '/points.npy'), np.load(g + '/normals.npy')
    ind, fps = torch.from_numpy(np.load(g + '/presample_idx.npy')), torch.from_numpy(np.load(g + '/fps_idx.npy'))
    gold = torch.from_numpy(np.load(g + '/latents.npy'))
    with torch.no_grad():
        for full in (True, False):
            lat = enc(pts, nrm, ind, fps, full_keys=full)
            rel = float((lat - gold).norm() / gold.norm())
            print(f'keys from the {"full 65536-point cloud" if full else "4096 pre-sampled points"}: relRMS vs reference {rel:.3e}   max |diff| {float((lat - gold).abs().max()):.3e}')
