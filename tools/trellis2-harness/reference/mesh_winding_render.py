"""Shows which visible faces are wound the wrong way round.

usage: python mesh_winding_render.py mesh.obj|mesh.glb out.png [more.obj ...]   (one image per mesh, side by side, same camera)

Four views per mesh. A face that faces the camera is shaded grey (headlight Lambert); a face that is the front-most surface at its pixel but whose
winding normal points AWAY from the camera is RED. On a solid wound outward the front-most surface always faces the camera, so any red is a face
a viewer with back-face culling or one-sided lighting would get wrong. Prints the share of visible pixels that are red.
"""
import sys
import numpy as np
from PIL import Image
import mesh_audit as m

S = 420


def render(pos, idx, yaw, pitch, lo=None, hi=None):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    R = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]); R2 = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    q = pos @ R.T @ R2.T                       # camera looks down -z; +z is toward the camera
    if lo is None:
        lo, hi = q.min(0), q.max(0)
    sc = (S - 16) / max(hi[0] - lo[0], hi[1] - lo[1]); c = (lo + hi) / 2
    X = (q[:, 0] - c[0]) * sc + S / 2; Y = S / 2 - (q[:, 1] - c[1]) * sc; Z = q[:, 2]
    img = np.full((S, S, 3), 235, np.uint8); zb = np.full((S, S), -1e9, np.float32)
    a, b, cc = q[idx[:, 0]], q[idx[:, 1]], q[idx[:, 2]]
    n = np.cross(b - a, cc - a); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    facing = n[:, 2]                           # > 0: winding normal points at the camera
    order = np.argsort(Z[idx].mean(1))
    for t in order:
        f = idx[t]
        x = X[f]; y = Y[f]; z = Z[f]
        x0, x1 = max(int(x.min()), 0), min(int(x.max()) + 1, S - 1); y0, y1 = max(int(y.min()), 0), min(int(y.max()) + 1, S - 1)
        if x1 < x0 or y1 < y0: continue
        det = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
        if abs(det) < 1e-9: continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w0 = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / det
        w1 = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / det; w2 = 1 - w0 - w1
        mk = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not mk.any(): continue
        zz = w0 * z[0] + w1 * z[1] + w2 * z[2]
        sub = zb[y0:y1 + 1, x0:x1 + 1]; ok = mk & (zz > sub)
        if not ok.any(): continue
        sub[ok] = zz[ok]
        col = np.array([int(40 + 190 * facing[t]) if facing[t] > 0 else 0] * 3, dtype=np.uint8)
        if facing[t] <= 0: col = np.array([220, 30, 30], dtype=np.uint8)
        else: col = np.array([int(60 + 170 * facing[t])] * 3, dtype=np.uint8)
        img[y0:y1 + 1, x0:x1 + 1][ok] = col
    red = ((img[..., 0] == 220) & (img[..., 1] == 30)).sum(); vis = (img != 235).any(2).sum()
    return img, red / max(vis, 1)


if __name__ == '__main__':
    out = sys.argv[2]
    rows = []
    for path in [sys.argv[1]] + sys.argv[3:]:
        pos, idx = m.load_glb(path)
        v, f = m.weld(pos, idx)
        views, shares = [], []
        for yaw, pitch in [(20, 20), (110, 20), (200, 20), (290, 20)]:
            im, r = render(v, f, np.radians(yaw), np.radians(pitch)); views.append(im); shares.append(r)
        print(f'{path}: {len(f)} faces, red (wrongly wound, visible) share per view: ' + ', '.join(f'{100 * s:.2f}%' for s in shares) + f'  mean {100 * np.mean(shares):.2f}%')
        rows.append(np.concatenate(views, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(out)
    print('wrote', out)
