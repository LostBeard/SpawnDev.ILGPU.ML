"""Golden for flexible_dual_grid_to_mesh (o_voxel.convert.flexible_dual_grid), from the reference's own code.

The function needs two CUDA hashmap ops (o_voxel _C); they are replaced by sorted-key lookups, the one thing that is not
the reference's code here. Everything else (neighbour offsets, quad split rules, vertex placement, the decoder-output
post-processing from FlexiDualGridVaeDecoder.forward) is copied from the reference. The decoder output used is the
GOLDEN shape decoder output (golden_decoders.py), so this gate isolates mesh extraction.

  python golden_dualgrid.py        writes TRELLIS2_DIR/golden/dualgrid/*
"""
import os, sys, json
W = os.environ["TRELLIS2_DIR"]
D = os.path.join(W, "golden", "decoders")
OUT = os.path.join(W, "golden", "dualgrid"); os.makedirs(OUT, exist_ok=True)
import numpy as np, torch
import torch.nn.functional as F

man = json.load(open(os.path.join(D, "manifest.json")))
h = torch.from_numpy(np.fromfile(os.path.join(D, "shape_out.f32"), dtype=np.float32).reshape(man["shape_out"]))
coords = torch.from_numpy(np.fromfile(os.path.join(D, "shape_out_coords.i32"), dtype=np.int32).reshape(man["shape_out_coords"]))
grid_size = 512
voxel_margin = 0.5
aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])

# --- FlexiDualGridVaeDecoder.forward, eval branch
vertices = (1 + 2 * voxel_margin) * torch.sigmoid(h[..., 0:3]) - voxel_margin
intersected = h[..., 3:6] > 0
quad_lerp = F.softplus(h[..., 6:7])

# --- flexible_dual_grid_to_mesh (train=False), hashmap ops replaced by sorted keys
edge_neighbor_voxel_offset = torch.tensor([
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
], dtype=torch.int).unsqueeze(0)
quad_split_1 = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long)
quad_split_2 = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long)

voxel_size = (aabb[1] - aabb[0]) / grid_size
N = vertices.shape[0]
mesh_vertices = (coords.float() + vertices) * voxel_size + aabb[0].reshape(1, 3)

G = grid_size
def key(c): return (c[..., 0].long() * G + c[..., 1].long()) * G + c[..., 2].long()
keys = key(coords)
sk, order = torch.sort(keys)

edge_neighbor_voxel = coords.reshape(N, 1, 1, 3) + edge_neighbor_voxel_offset      # (N, 3, 4, 3)
connected_voxel = edge_neighbor_voxel[intersected]                                 # (M, 4, 3)
M = connected_voxel.shape[0]
cv = connected_voxel.reshape(-1, 3)
inb = ((cv >= 0) & (cv < G)).all(dim=1)
ck = key(cv)
pos = torch.clamp(torch.searchsorted(sk, ck), 0, N - 1)
hit = inb & (sk[pos] == ck)
idx = torch.full((M * 4,), -1, dtype=torch.long)
idx[hit] = order[pos[hit]]
idx = idx.reshape(M, 4)
valid = (idx >= 0).all(dim=1)
quad_indices = idx[valid]                                                          # (L, 4)

sw = quad_lerp[quad_indices]                                                       # split_weight[quad_indices]
sw = sw.reshape(quad_indices.shape[0], 4)
w02 = sw[:, 0] * sw[:, 2]
w13 = sw[:, 1] * sw[:, 3]
tri = torch.where((w02 > w13)[:, None], quad_indices[:, quad_split_1], quad_indices[:, quad_split_2]).reshape(-1, 3)

mesh_vertices.numpy().astype(np.float32).tofile(os.path.join(OUT, "vertices.f32"))
tri.numpy().astype(np.int32).tofile(os.path.join(OUT, "faces.i32"))
json.dump({"vertices": list(mesh_vertices.shape), "faces": list(tri.shape), "quads": int(quad_indices.shape[0]), "grid": G}, open(os.path.join(OUT, "manifest.json"), "w"))
print("vertices", tuple(mesh_vertices.shape), "faces", tuple(tri.shape), "quads", int(quad_indices.shape[0]), "intersected edges", M)
