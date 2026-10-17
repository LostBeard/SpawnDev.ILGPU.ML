import sys, json, struct
import numpy as np
d = open(sys.argv[1], 'rb').read()
jl, = struct.unpack_from('<I', d, 12)
j = json.loads(d[20:20 + jl]); bin0 = 20 + jl + 8
def acc(i, dt, cnt):
    a = j['accessors'][i]; bv = j['bufferViews'][a['bufferView']]
    off = bv.get('byteOffset', 0) + a.get('byteOffset', 0) + bin0
    n = a['count'] * cnt
    x = np.frombuffer(d, dtype=dt, count=n, offset=off)
    return x.reshape(a['count'], cnt) if cnt > 1 else x
p = j['meshes'][0]['primitives'][0]
pos = acc(p['attributes']['POSITION'], np.float32, 3); idx = acc(p['indices'], np.uint32, 1).reshape(-1, 3)
q = np.round(pos * 4096).astype(np.int64)
_, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
inv = inv.reshape(-1)
f = inv[idx].astype(np.int32); v = pos[first].astype(np.float32)
with open(sys.argv[2], 'wb') as o:
    o.write(struct.pack('<ii', len(v), len(f))); o.write(v.tobytes()); o.write(f.tobytes())
print(len(v), len(f))
