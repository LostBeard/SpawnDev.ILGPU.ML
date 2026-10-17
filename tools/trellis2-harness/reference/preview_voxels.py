"""Debug preview: splat the voxel centres of a generation (base colour from the baked attributes) from 4 orthographic views into one PNG.
  python preview_voxels.py out.voxels.bin preview.png"""
import sys, struct, numpy as np
from PIL import Image
b = open(sys.argv[1], 'rb').read()
n, grid = struct.unpack('<ii', b[:8])
a = np.frombuffer(b[8:], dtype=np.dtype([('c', '<i4', 3), ('a', '<f4', 6)]), count=n)
pos = (a['c'].astype(np.float32) + 0.5) / grid - 0.5          # world, y up the volume
col = np.clip(a['a'][:, :3], 0, 1)
S = 384
views = [("front", 0, 1, 2, 1), ("side", 2, 1, 0, 1), ("top", 0, 2, 1, -1), ("back", 0, 1, 2, -1)]
canvas = np.full((S, S * 4, 3), 235, np.uint8)
for vi, (name, ax_u, ax_v, ax_d, sign) in enumerate(views):
    u = ((pos[:, ax_u] * (1 if vi != 3 else -1) + 0.55) / 1.1 * (S - 1)).astype(int)
    v = ((0.55 - pos[:, ax_v]) / 1.1 * (S - 1)).astype(int)
    d = pos[:, ax_d] * sign
    order = np.argsort(-d)                                       # far first, near last
    img = np.full((S, S, 3), 235, np.uint8)
    for k in order:
        img[max(v[k] - 1, 0):v[k] + 2, max(u[k] - 1, 0):u[k] + 2] = (col[k] * 255).astype(np.uint8)
    canvas[:, vi * S:(vi + 1) * S] = img
Image.fromarray(canvas).save(sys.argv[2])
print(n, "voxels; bounds", pos.min(0), pos.max(0))
