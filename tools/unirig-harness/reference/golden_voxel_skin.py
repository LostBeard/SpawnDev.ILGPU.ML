"""Golden for UniRig's skinning PRIOR ("voxel skin": geodesic voxel binding, Dionne & de Lasa 2013), from the reference's own `voxel_skin()` on CPU.

  python -I golden_voxel_skin.py <mesh.obj> <rig.json> [grid=196]      writes UNIRIG_DIR/golden/voxel_skin/*

The reference voxelises with pyrender or open3d (neither is installed here). Both compute the same thing - the visual hull of six axis-aligned depth views, a voxel counting as
inside when at least two of the three axes put it between the nearest surface hits from both sides - so grid_coords here come from my own implementation of exactly that
(`visual_hull`), exported next to the result. The C# port is gated in two steps: its graph + Dijkstra + weights on THESE grid_coords against `skin.npy`, and its own voxelisation
against `visual_hull`.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_env
import numpy as np


def load_obj(path):
    v, f = [], []
    for line in open(path, encoding='utf-8'):
        t = line.split()
        if len(t) >= 4 and t[0] == 'v': v.append([float(x) for x in t[1:4]])
        elif len(t) >= 4 and t[0] == 'f': f.append([int(x.split('/')[0]) - 1 for x in t[1:4]])
    return np.array(v, np.float64), np.array(f, np.int64)


def visual_hull(v, f, grid):
    """Voxel centres (M,3) in [-1,1]^3 that lie inside along at least two axes. v is already scaled into [-1,1].
    Along an axis a voxel is inside when it lies between the smallest and the largest surface hit of the ray through it (the nearest hits from either side)."""
    centers = (np.arange(grid) + 0.5) * 2.0 / grid - 1.0
    jit = np.array([0.0127, 0.0093]) * (2.0 / grid)                  # rays at cell centres plus a fixed odd offset: no ray runs along a shared edge
    tri = v[f]
    inside = []
    for a in range(3):
        u, w = (a + 1) % 3, (a + 2) % 3
        lo = np.full((grid, grid), np.inf); hi = np.full((grid, grid), -np.inf)
        pu, pw, pa = tri[:, :, u], tri[:, :, w], tri[:, :, a]
        for t in range(len(tri)):
            x0, x1, x2 = pu[t]; y0, y1, y2 = pw[t]
            det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(det) < 1e-18: continue
            i0 = max(0, int(np.ceil((min(x0, x1, x2) - jit[0] + 1) * grid / 2 - 0.5))); i1 = min(grid - 1, int(np.floor((max(x0, x1, x2) - jit[0] + 1) * grid / 2 - 0.5)))
            j0 = max(0, int(np.ceil((min(y0, y1, y2) - jit[1] + 1) * grid / 2 - 0.5))); j1 = min(grid - 1, int(np.floor((max(y0, y1, y2) - jit[1] + 1) * grid / 2 - 0.5)))
            if i1 < i0 or j1 < j0: continue
            gx = centers[i0:i1 + 1][:, None] + jit[0]; gy = centers[j0:j1 + 1][None, :] + jit[1]
            w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / det
            w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / det; w2 = 1 - w0 - w1
            m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not m.any(): continue
            z = w0 * pa[t, 0] + w1 * pa[t, 1] + w2 * pa[t, 2]
            sl = (slice(i0, i1 + 1), slice(j0, j1 + 1))
            lo[sl] = np.where(m, np.minimum(lo[sl], z), lo[sl]); hi[sl] = np.where(m, np.maximum(hi[sl], z), hi[sl])
        # inside[a][i,j,k] with (i, j) the cell on the (u, w) plane and k along the axis
        inside.append((lo[:, :, None] <= centers[None, None, :]) & (centers[None, None, :] <= hi[:, :, None]))
    # bring each [u, w, a] array to [x, y, z]
    ix = np.transpose(inside[0], (2, 0, 1))      # axis 0: (u, w, a) = (y, z, x)  ->  (x, y, z)
    iy = np.transpose(inside[1], (1, 2, 0))      # axis 1: (u, w, a) = (z, x, y)  ->  (x, y, z)
    iz = inside[2]                               # axis 2: (u, w, a) = (x, y, z)
    mask = (ix.astype(int) + iy.astype(int) + iz.astype(int)) >= 2
    i, j, k = np.nonzero(mask)
    return np.stack([centers[i], centers[j], centers[k]], 1).astype(np.float32)


if __name__ == '__main__':
    mesh, rig = sys.argv[1], sys.argv[2]
    grid = int(sys.argv[3]) if len(sys.argv) > 3 else 196
    from src.data.vertex_group import voxel_skin
    v, f = load_obj(mesh)
    d = json.load(open(rig)); J = np.array(d['joints'], np.float64).reshape(-1, 3)
    lo, hi = v.min(0), v.max(0); center = (lo + hi) / 2; scale = (hi - lo).max() / 2
    vn, jn = (v - center) / scale, (J - center) / scale
    out = os.path.join(ref_env.UNIRIG_DIR, 'golden', 'voxel_skin'); os.makedirs(out, exist_ok=True)
    gc = visual_hull(vn, f, grid)
    print(f'{len(v)} vertices, {len(f)} faces, {len(J)} joints, grid {grid}: {len(gc)} inside voxels')
    skin = voxel_skin(grid=grid, grid_coords=gc, joints=jn, vertices=vn, faces=f, alpha=0.5, link_dis=0.00001, grid_query=7, vertex_query=1, grid_weight=3.0, mode='square')
    skin = np.nan_to_num(skin, nan=0., posinf=0., neginf=0.)
    print('skin', skin.shape, 'row sums min/max', float(skin.sum(1).min()), float(skin.sum(1).max()), ' dominant bone share of vertices:', np.bincount(skin.argmax(1), minlength=len(J)).tolist())
    np.save(out + '/vertices.npy', vn.astype(np.float32)); np.save(out + '/faces.npy', f.astype(np.int32)); np.save(out + '/joints.npy', jn.astype(np.float32))
    np.save(out + '/grid_coords.npy', gc); np.save(out + '/skin.npy', np.ascontiguousarray(skin, dtype=np.float32))   # the reference returns a transposed view: write C order
    json.dump({'grid': grid, 'mesh': mesh, 'rig': rig, 'center': center.tolist(), 'scale': float(scale)}, open(out + '/manifest.json', 'w'))
    print('wrote', out)
