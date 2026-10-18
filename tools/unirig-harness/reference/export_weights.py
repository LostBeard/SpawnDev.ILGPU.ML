"""Export a UniRig checkpoint to bf16 safetensors for the C# port.

  python -I export_weights.py skeleton        ->  UNIRIG_DIR/weights/skeleton.safetensors

- names lose the Lightning `model.` prefix; every float tensor is rounded to bf16 (round-to-nearest-even, what torch does), like the reference's own bf16-mixed inference;
- the perceiver's fused projections are stored per head ([H][parts][D] rows). The C# attention reads `[q | k | v]` blocks, each head-major, so the rows of
  `c_qkv.weight` (parts = 3) and `c_kv.weight` (parts = 2) are permuted to [parts][H][D] HERE, once. metadata records it so a loader can refuse a file without it;
- learned token / position tables and the OPT final norm stay as they are.
"""
import os, sys, json
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_ckpt import load_checkpoint

UNIRIG_DIR = os.environ.get('UNIRIG_DIR', 'E:/spawndev-unirig')
HEADS = 8    # the perceiver's heads (width 512, head dim 64)


def regroup_rows(w, heads, parts):
    """rows [H][parts][D] -> [parts][H][D] (w is [heads * parts * D, in])."""
    rows, cols = w.shape
    d = rows // (heads * parts)
    return w.view(heads, parts, d, cols).permute(1, 0, 2, 3).reshape(rows, cols).contiguous()


def export_skeleton():
    sd = load_checkpoint(os.path.join(UNIRIG_DIR, 'ckpt', 'skeleton', 'model.ckpt'))['state_dict']
    out = {}
    sd = {(k[len('model.'):] if k.startswith('model.') else k): v for k, v in sd.items()}
    # OPT: one fused qkv projection per layer, rows [q | k | v] (each head-major, which is how the HF layout already is), one matmul instead of three
    for layer in range(24):
        pre = f'transformer.model.decoder.layers.{layer}.self_attn.'
        for kind in ('weight', 'bias'):
            sd[pre + f'qkv_proj.{kind}'] = torch.cat([sd.pop(pre + f'{p}_proj.{kind}') for p in ('q', 'k', 'v')], dim=0)
    for name, t in sd.items():
        if name.endswith('attn.c_qkv.weight'): t = regroup_rows(t, HEADS, 3)
        elif name.endswith('attn.c_kv.weight'): t = regroup_rows(t, HEADS, 2)
        out[name] = t.to(torch.bfloat16).contiguous()
    os.makedirs(os.path.join(UNIRIG_DIR, 'weights'), exist_ok=True)
    path = os.path.join(UNIRIG_DIR, 'weights', 'skeleton.safetensors')
    save_file(out, path, metadata={'source': 'VAST-AI/UniRig skeleton/articulation-xl_quantization_256', 'dtype': 'bf16',
                                   'fused_qkv_layout': 'parts-major (rows [3][H][D]); c_kv rows [2][H][D]', 'perceiver_heads': str(HEADS)})
    print(f'{len(out)} tensors, {sum(t.numel() for t in out.values()) / 1e6:.1f}M params -> {path} ({os.path.getsize(path) / 1e6:.0f} MB)')


if __name__ == '__main__':
    {'skeleton': export_skeleton}[sys.argv[1]]()
