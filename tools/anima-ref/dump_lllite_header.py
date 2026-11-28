"""Print an Anima LLLite file's metadata and tensor names / shapes (block 0 and the conditioning trunk only, plus counts)."""
import json
import re
import struct
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else r'D:\Projects\anima\anima-lllite-inpainting-v2.safetensors'
with open(path, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(n))
meta = header.pop('__metadata__', {})
print('metadata:')
for k in sorted(meta):
    print(f'  {k} = {meta[k]}')
print('tensors:', len(header), 'dtypes:', dict(Counter(v['dtype'] for v in header.values())))
blocks = Counter()
for name, v in sorted(header.items()):
    m = re.match(r'lllite_dit_blocks_(\d+)_(.+?)\.', name)
    if m:
        blocks[m.group(1)] += 1
        if m.group(1) != '0':
            continue
    print(f'  {name} {v["dtype"]} {tuple(v["shape"])}')
print('tensors per block:', dict(sorted(blocks.items(), key=lambda kv: int(kv[0]))))
