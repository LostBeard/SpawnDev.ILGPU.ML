"""Export a UniRig checkpoint to bf16 safetensors for the C# port.

  python -I export_weights.py skeleton        ->  UNIRIG_DIR/weights/skeleton.safetensors
  python -I export_weights.py skin            ->  UNIRIG_DIR/weights/skin.safetensors          (UniRig's PTv3 skin network: reference tooling only, not ported)
  python -I export_weights.py skintokens      ->  UNIRIG_DIR/weights/skintokens.safetensors    (SkinTokens / TokenRig: skeleton + skin in one sequence)

- names lose the Lightning `model.` prefix; every float tensor is rounded to bf16 (round-to-nearest-even, what torch does), like the reference's own bf16-mixed inference;
- the perceiver's fused projections are stored per head ([H][parts][D] rows). The C# attention reads `[q | k | v]` blocks, each head-major, so the rows of
  `c_qkv.weight` (parts = 3) and `c_kv.weight` (parts = 2) are permuted to [parts][H][D] HERE, once. metadata records it so a loader can refuse a file without it;
- learned token / position tables and the OPT final norm stay as they are.

skin (364.5M parameters of the 4.4 GB Lightning checkpoint, which also carries optimizer state):
- the stem's BatchNorm1d (eval: running statistics, eps 1e-3) is folded into the 5x5x5 convolution that precedes it: w' = w * g / sqrt(var + eps) per output
  channel, b' = beta - mean * g / sqrt(var + eps). The convolution has no bias of its own. num_batches_tracked is dropped;
- the sparse convolutions (SubMConv3d, [Co][kx][ky][kz][Ci] = the C# [Co][taps][Ci] with tap = (kx*K + ky)*K + kz) stay fp32 - the reference runs them in float32 on
  purpose ("must cast to float32 to avoid sparse-conv precision bugs"); everything else is bf16;
- the global (Michelangelo) encoder's fused projections are regrouped exactly as the skeleton's. The PTv3 qkv rows are already [3][H][D] and Wq / Wkv rows [H*D] /
  [2][H][D], which is what the C# attention reads.
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


def export_skintokens():
    """SkinTokens (VAST-AI, MIT) grpo_1400 checkpoint -> bf16 safetensors, inference tensors only (the VAE's encoder / quant side, which only training uses, is dropped).
    - Qwen3 layers: q/k/v fused to one [q 2048 | k 1024 | v 1024] projection, gate/up fused to [gate | up];
    - the Michelangelo encoder's fused projections regrouped exactly as the skeleton's (perceiver heads 8);
    - the VAE's attention (diffusers Attention in the Tripo2 layout): the processor concatenates q|k|v and re-splits per head as [head][q|k|v][D], so the projection rows
      cat(to_q, to_k, to_v) are in head-interleaved order. Regrouped here to [3][H][D] (the C# attention reads q, k, v as head-major blocks): self attention
      `attn1.qkv.weight` (heads 12), cross attention `attn2.kv.weight` (cat(to_k, to_v), heads 12); `attn2.q.weight` is plain."""
    sd = load_checkpoint(os.path.join(UNIRIG_DIR, 'skintokens', 'ckpt', 'experiments', 'articulation_xl_quantization_256_token_4', 'grpo_1400.ckpt'))['state_dict']
    out = {}
    for name, t in sd.items():
        if name.startswith('vae.model.encoder.') or name.startswith('vae.model.quant.') or name.startswith('vae.model.FSQ.project_in'): continue
        if '.self_attn.' in name and name.endswith(('q_proj.weight', 'k_proj.weight', 'v_proj.weight')): continue
        if '.mlp.' in name and name.endswith(('gate_proj.weight', 'up_proj.weight')): continue
        if '.attn1.' in name and name.endswith(('to_q.weight', 'to_k.weight', 'to_v.weight')): continue
        if '.attn2.' in name and name.endswith(('to_k.weight', 'to_v.weight')): continue
        if name.endswith('attn.c_qkv.weight'): t = regroup_rows(t, HEADS, 3)
        elif name.endswith('attn.c_kv.weight'): t = regroup_rows(t, HEADS, 2)
        out[name] = t.to(torch.bfloat16).clone().contiguous()
    for i in range(28):
        p = f'transformer.model.layers.{i}.'
        out[p + 'self_attn.qkv_proj.weight'] = torch.cat([sd[p + f'self_attn.{n}_proj.weight'] for n in 'qkv'], 0).to(torch.bfloat16).clone().contiguous()
        out[p + 'mlp.gate_up_proj.weight'] = torch.cat([sd[p + 'mlp.gate_proj.weight'], sd[p + 'mlp.up_proj.weight']], 0).to(torch.bfloat16).clone().contiguous()
    for part, n_self in (('cond_encoder', 2), ('decoder', 10)):
        for b in range(1 + n_self if part == 'cond_encoder' else n_self + 1):
            p = f'vae.model.{part}.blocks.{b}.'
            if p + 'attn1.to_q.weight' in sd:
                w = torch.cat([sd[p + f'attn1.to_{n}.weight'] for n in 'qkv'], 0)
                out[p + 'attn1.qkv.weight'] = regroup_rows(w, 12, 3).to(torch.bfloat16).clone().contiguous()
            if p + 'attn2.to_k.weight' in sd:
                w = torch.cat([sd[p + f'attn2.to_{n}.weight'] for n in 'kv'], 0)
                out[p + 'attn2.kv.weight'] = regroup_rows(w, 12, 2).to(torch.bfloat16).clone().contiguous()
    path = os.path.join(UNIRIG_DIR, 'weights', 'skintokens.safetensors')
    save_file(out, path, metadata={'source': 'VAST-AI/SkinTokens experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt', 'dtype': 'bf16',
                                   'qwen': 'qkv_proj rows [q 2048|k 1024|v 1024]; gate_up_proj rows [gate 3072|up 3072]',
                                   'vae_attention': 'attn1.qkv rows [3][12][64] (regrouped from the Tripo2 head-interleaved layout); attn2.kv rows [2][12][64]',
                                   'perceiver_heads': str(HEADS)})
    print(f'{len(out)} tensors, {sum(t.numel() for t in out.values()) / 1e6:.1f}M params -> {path} ({os.path.getsize(path) / 1e6:.0f} MB)')


def export_skin():
    sd = load_checkpoint(os.path.join(UNIRIG_DIR, 'ckpt', 'skin', 'model.ckpt'))['state_dict']
    sd = {(k[len('model.'):] if k.startswith('model.') else k): v for k, v in sd.items()}
    out = {}
    pre = 'mesh_encoder.embedding.stem.'
    g, b, mean, var = (sd.pop(pre + 'norm.' + n).double() for n in ('weight', 'bias', 'running_mean', 'running_var'))
    sd.pop(pre + 'norm.num_batches_tracked')
    scale = g / torch.sqrt(var + 1e-3)
    sd[pre + 'conv.weight'] = (sd[pre + 'conv.weight'].double() * scale.view(-1, 1, 1, 1, 1)).float()
    sd[pre + 'conv.bias'] = (b - mean * scale).float()
    for name, t in sd.items():
        if name.endswith('attn.c_qkv.weight'): t = regroup_rows(t, HEADS, 3)
        elif name.endswith('attn.c_kv.weight'): t = regroup_rows(t, HEADS, 2)
        out[name] = t.float().contiguous() if name.endswith('conv.weight') or name.endswith('conv.bias') or '.cpe.0.' in name else t.to(torch.bfloat16).contiguous()
    path = os.path.join(UNIRIG_DIR, 'weights', 'skin.safetensors')
    save_file(out, path, metadata={'source': 'VAST-AI/UniRig skin/articulation-xl', 'dtype': 'bf16; sparse convolutions fp32',
                                   'stem_batchnorm': 'folded into mesh_encoder.embedding.stem.conv (weight + bias)',
                                   'fused_qkv_layout': 'global_encoder: parts-major (rows [3][H][D]); c_kv rows [2][H][D]', 'perceiver_heads': str(HEADS)})
    print(f'{len(out)} tensors, {sum(t.numel() for t in out.values()) / 1e6:.1f}M params -> {path} ({os.path.getsize(path) / 1e6:.0f} MB)')


if __name__ == '__main__':
    {'skeleton': export_skeleton, 'skin': export_skin, 'skintokens': export_skintokens}[sys.argv[1]]()
