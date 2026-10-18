"""Stand-ins for the two CUDA-only packages the UniRig SKIN model imports, so the REAL reference classes (PTv3Object, UniRigSkin, real spconv) run on this PC.

  torch_scatter.segment_csr     min / max / mean / sum over contiguous segments (used for the per-mesh minimum coordinate and the point-cloud origin)
  flash_attn.flash_attn_varlen_qkvpacked_func   softmax attention inside each cu_seqlens segment, in fp16 like the real kernel (used by the serialized patch attention)
  flash_attn.modules.mha.MHA    the cross-attention block (Wq, Wkv, out_proj; head split of kv = [two, h, d]; scale 1/sqrt(d); no mask)

Each one is a few lines of ordinary torch with the documented semantics; nothing here decides what the model computes. spconv (the sparse convolution) is NOT shimmed:
the golden runs the genuine package on the GPU. Import this module BEFORE anything from the reference.
"""
import sys, types, math
import torch
import torch.nn as nn
import torch.nn.functional as F


def segment_csr(src, indptr, reduce='sum'):
    out = []
    for i in range(len(indptr) - 1):
        seg = src[int(indptr[i]):int(indptr[i + 1])]
        if reduce == 'min': out.append(seg.min(0).values)
        elif reduce == 'max': out.append(seg.max(0).values)
        elif reduce == 'mean': out.append(seg.mean(0))
        elif reduce == 'sum': out.append(seg.sum(0))
        else: raise ValueError(reduce)
    return torch.stack(out)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, **kw):
    """qkv [total, 3, H, D] (half) -> [total, H, D] (half). Sequences are cu_seqlens[i]:cu_seqlens[i+1]."""
    assert dropout_p == 0.0
    out = torch.empty(qkv.shape[0], qkv.shape[2], qkv.shape[3], dtype=qkv.dtype, device=qkv.device)
    cu = cu_seqlens.tolist()
    for a, b in zip(cu[:-1], cu[1:]):
        q, k, v = (qkv[a:b, i].transpose(0, 1).unsqueeze(0) for i in range(3))      # [1, H, K, D]
        o = F.scaled_dot_product_attention(q, k, v, scale=softmax_scale)
        out[a:b] = o[0].transpose(0, 1)
    return out


class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, cross_attn=False, **kw):
        super().__init__()
        assert cross_attn
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wkv = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, x_kv=None, **kw):
        B, Sq, _ = x.shape; Sk = x_kv.shape[1]; H, D = self.num_heads, self.head_dim
        q = self.Wq(x).view(B, Sq, H, D).transpose(1, 2)
        kv = self.Wkv(x_kv).view(B, Sk, 2, H, D)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(D))
        return self.out_proj(o.transpose(1, 2).reshape(B, Sq, H * D))


def install():
    ts = types.ModuleType('torch_scatter'); ts.segment_csr = segment_csr
    fa = types.ModuleType('flash_attn'); fa.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
    fm = types.ModuleType('flash_attn.modules'); fmm = types.ModuleType('flash_attn.modules.mha'); fmm.MHA = MHA
    fa.modules = fm; fm.mha = fmm
    sys.modules.update({'torch_scatter': ts, 'flash_attn': fa, 'flash_attn.modules': fm, 'flash_attn.modules.mha': fmm})
