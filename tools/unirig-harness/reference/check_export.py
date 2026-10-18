"""The exported bf16 weights (parts-major fused projections) through my PyTorch encoder, against the reference latents.
  python -I check_export.py       prints relRMS for: bf16 weights / fp32 activations, and the fp32 original (the floor)"""
import os, sys
import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch_encoder as te

UNIRIG_DIR = te.UNIRIG_DIR


class ExportedEncoder(te.Encoder):
    """Same math as torch_encoder.Encoder, reading fused projections as [q | k | v] blocks (the layout export_weights.py writes)."""
    def __init__(self, tensors, heads=8):
        pre = 'mesh_encoder.encoder.'
        self.p = {k[len(pre):]: v.float() for k, v in tensors.items() if k.startswith(pre)}
        self.out_w, self.out_b = tensors['output_proj.weight'].float(), tensors['output_proj.bias'].float()
        self.heads = heads


# parts-major split: [B, N, parts * C] -> parts tensors of [B, N, C]
te.split_heads_interleaved = lambda qkv, heads, parts: list(qkv.chunk(parts, dim=-1))

if __name__ == '__main__':
    g = os.path.join(UNIRIG_DIR, 'golden', 'skeleton')
    pts, nrm = np.load(g + '/points.npy'), np.load(g + '/normals.npy')
    ind, fps = torch.from_numpy(np.load(g + '/presample_idx.npy')), torch.from_numpy(np.load(g + '/fps_idx.npy'))
    gold = torch.from_numpy(np.load(g + '/latents.npy'))
    enc = ExportedEncoder(load_file(os.path.join(UNIRIG_DIR, 'weights', 'skeleton.safetensors')))
    with torch.no_grad():
        lat = enc(pts, nrm, ind, fps)
    print(f'exported bf16 weights, fp32 math: relRMS vs reference {float((lat - gold).norm() / gold.norm()):.3e}   max |diff| {float((lat - gold).abs().max()):.3e}')
