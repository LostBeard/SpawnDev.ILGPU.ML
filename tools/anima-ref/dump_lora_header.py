"""Summarise a PEFT-style LoRA file for Anima: metadata, ranks, alpha tensors, and which modules it covers (block 0 / adapter block 0 in full)."""
import json
import re
import struct
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else r'D:\Projects\anima\anima-turbo-lora-v0.2.safetensors'
with open(path, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(n))
meta = header.pop('__metadata__', {}) or {}
print('metadata keys:', sorted(meta)[:40])
for k in ('ss_network_alpha', 'ss_network_dim', 'lora_alpha', 'alpha', 'r', 'modelspec.title', 'modelspec.architecture'):
    if k in meta:
        print(f'  {k} = {meta[k][:200]}')
print('tensors:', len(header), 'dtypes:', dict(Counter(v['dtype'] for v in header.values())))
suffixes = Counter(re.sub(r'^.*?\.(lora_[AB]\.weight|alpha|lora_up\.weight|lora_down\.weight|dora_scale)$', r'\1', k) for k in header)
print('suffixes:', dict(suffixes))
mods = sorted({re.sub(r'\.(lora_[AB]\.weight|alpha|lora_up\.weight|lora_down\.weight|dora_scale)$', '', k) for k in header})
groups = Counter()
for m in mods:
    g = re.sub(r'\.\d+\.', '.N.', m)
    groups[g] += 1
print('module groups (N = block index):')
for g, c in sorted(groups.items()):
    print(f'  {c:4d}  {g}')
ranks = Counter(tuple(v['shape']) for k, v in header.items() if k.endswith('lora_A.weight'))
print('lora_A shapes:', dict(ranks))
for k, v in sorted(header.items()):
    if re.search(r'blocks\.0\.', k) and not k.startswith('diffusion_model.llm_adapter.blocks.1'):
        if 'llm_adapter' in k and 'blocks.0.' not in k:
            continue
        print(f'  {k} {v["dtype"]} {tuple(v["shape"])}')
