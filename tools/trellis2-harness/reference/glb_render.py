import sys, json, struct, io
import numpy as np
from PIL import Image
d = open(sys.argv[1], 'rb').read(); out = sys.argv[2]
jl, = struct.unpack_from('<I', d, 12); j = json.loads(d[20:20+jl]); b0 = 20+jl+8
def acc(i, dt, cnt):
    a = j['accessors'][i]; bv = j['bufferViews'][a['bufferView']]
    off = bv.get('byteOffset', 0) + a.get('byteOffset', 0) + b0
    x = np.frombuffer(d, dtype=dt, count=a['count']*cnt, offset=off)
    return x.reshape(a['count'], cnt) if cnt > 1 else x
p = j['meshes'][0]['primitives'][0]
pos = acc(p['attributes']['POSITION'], np.float32, 3); uv = acc(p['attributes']['TEXCOORD_0'], np.float32, 2)
idx = acc(p['indices'], np.uint32, 1).reshape(-1, 3)
bv = j['bufferViews'][j['images'][0]['bufferView']]; o = b0 + bv.get('byteOffset', 0)
tex = np.array(Image.open(io.BytesIO(d[o:o+bv['byteLength']])).convert('RGB')); T = tex.shape[0]
S = 640
def render(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    R = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) ; R2 = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    q = pos @ R.T @ R2.T
    lo, hi = q.min(0), q.max(0); sc = (S - 20) / max(hi[0]-lo[0], hi[1]-lo[1]); c = (lo + hi) / 2
    X = (q[:, 0]-c[0]) * sc + S/2; Y = S/2 - (q[:, 1]-c[1]) * sc; Z = q[:, 2]
    img = np.full((S, S, 3), 235, np.uint8); zb = np.full((S, S), -1e9, np.float32)
    order = np.argsort(Z[idx].mean(1))  # far..near; z-buffer handles rest
    for f in idx[order]:
        x = X[f]; y = Y[f]; z = Z[f]
        x0, x1 = max(int(x.min()), 0), min(int(x.max())+1, S-1); y0, y1 = max(int(y.min()), 0), min(int(y.max())+1, S-1)
        if x1 < x0 or y1 < y0: continue
        det = (y[1]-y[2])*(x[0]-x[2]) + (x[2]-x[1])*(y[0]-y[2])
        if abs(det) < 1e-9: continue
        gx, gy = np.meshgrid(np.arange(x0, x1+1)+0.5, np.arange(y0, y1+1)+0.5)
        w0 = ((y[1]-y[2])*(gx-x[2]) + (x[2]-x[1])*(gy-y[2]))/det
        w1 = ((y[2]-y[0])*(gx-x[2]) + (x[0]-x[2])*(gy-y[2]))/det; w2 = 1-w0-w1
        m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not m.any(): continue
        zz = w0*z[0]+w1*z[1]+w2*z[2]
        sub = zb[y0:y1+1, x0:x1+1]; ok = m & (zz > sub)
        if not ok.any(): continue
        u = (w0*uv[f[0], 0]+w1*uv[f[1], 0]+w2*uv[f[2], 0]); v = (w0*uv[f[0], 1]+w1*uv[f[1], 1]+w2*uv[f[2], 1])
        tx = np.clip((u*T).astype(int), 0, T-1); ty = np.clip((v*T).astype(int), 0, T-1)
        col = tex[ty, tx]
        sub[ok] = zz[ok]; im = img[y0:y1+1, x0:x1+1]; im[ok] = col[ok]
    return img
views = [render(np.radians(a), np.radians(25)) for a in (20, 110)]
Image.fromarray(np.concatenate(views, 1)).save(out)
