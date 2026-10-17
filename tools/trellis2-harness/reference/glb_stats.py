import sys, json, struct
import numpy as np
d = open(sys.argv[1], 'rb').read()
jl, = struct.unpack_from('<I', d, 12)
j = json.loads(d[20:20 + jl])
bin0 = 20 + jl + 8
def acc(i, dt, cnt):
    a = j['accessors'][i]; bv = j['bufferViews'][a['bufferView']]
    off = bv.get('byteOffset', 0) + a.get('byteOffset', 0) + bin0
    n = a['count'] * cnt
    return np.frombuffer(d, dtype=dt, count=n, offset=off).reshape(a['count'], cnt) if cnt > 1 else np.frombuffer(d, dtype=dt, count=n, offset=off)
p = j['meshes'][0]['primitives'][0]
pos = acc(p['attributes']['POSITION'], np.float32, 3)
idx = acc(p['indices'], np.uint32, 1).reshape(-1, 3)
print('verts', len(pos), 'faces', len(idx))
# weld by position
q = np.round(pos * 4096).astype(np.int64)
_, inv = np.unique(q, axis=0, return_inverse=True)
f = inv[idx]
v = pos[np.unique(inv, return_index=True)[1]]
a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
n = np.cross(b - a, c - a); ln = np.linalg.norm(n, axis=1); n /= np.maximum(ln[:, None], 1e-20)
print('bbox', v.min(0), v.max(0))
print('area total', ln.sum() / 2, 'median face area', np.median(ln) / 2)
# edge adjacency
e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
fi = np.tile(np.arange(len(f)), 3)
k = np.sort(e, axis=1); key = k[:, 0].astype(np.int64) * (f.max() + 1) + k[:, 1]
o = np.argsort(key); key = key[o]; fi2 = fi[o]
m = key[:-1] == key[1:]
f0, f1 = fi2[:-1][m], fi2[1:][m]
dot = (n[f0] * n[f1]).sum(1)
ang = np.degrees(np.arccos(np.clip(dot, -1, 1)))
print('adjacent pairs', len(ang))
for t in (5, 10, 20, 40, 60, 90):
    print(f'  > {t} deg: {100.0 * (ang > t).mean():.2f}%')
