"""Topology audit of a GLB's first mesh primitive: what a rigger / physics / 3D-print / subdivision step would trip over.

usage: python mesh_audit.py model.glb [more.glb ...]

Vertices are welded by position first (a GLB duplicates vertices along UV seams, which says nothing about the surface).
Reports: welded size, open (boundary) edges, non-manifold edges, orientation conflicts, connected components and the
share of faces in small floating fragments, degenerate and sliver triangles, signed volume, Euler characteristic.
"""
import sys, json, struct
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

CT = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


def load_obj(path):
    """Plain v/f OBJ (what the harness's TRELLIS2_MESHSTAGES writes); faces may be 'i' or 'i/t/n'."""
    v, f = [], []
    for line in open(path, encoding='utf-8'):
        if line.startswith('v '):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            f.append([int(t.split('/')[0]) - 1 for t in line.split()[1:4]])
    return np.array(v, dtype=np.float64), np.array(f, dtype=np.int64)


def load_glb(path):
    if path.lower().endswith('.obj'):
        return load_obj(path)
    d = open(path, 'rb').read()
    jl, = struct.unpack_from('<I', d, 12)
    j = json.loads(d[20:20 + jl])
    bin0 = 20 + jl + 8

    def acc(i):
        a = j['accessors'][i]
        bv = j['bufferViews'][a['bufferView']]
        off = bv.get('byteOffset', 0) + a.get('byteOffset', 0) + bin0
        c = NC[a['type']]
        arr = np.frombuffer(d, dtype=CT[a['componentType']], count=a['count'] * c, offset=off)
        return arr.reshape(a['count'], c) if c > 1 else arr

    p = j['meshes'][0]['primitives'][0]
    return acc(p['attributes']['POSITION']).astype(np.float64), acc(p['indices']).astype(np.int64).reshape(-1, 3)


def weld(pos, idx, quant=1.0 / 4096):
    q = np.round(pos / quant).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    f = inv.reshape(-1)[idx]
    return pos[first], f


def audit(path):
    pos, idx = load_glb(path)
    return audit_mesh(pos, idx, path)


def audit_mesh(pos, idx, path, quiet=False):
    v, f = weld(pos, idx)
    n_in = len(f)
    # drop collapsed faces from the weld, but count them: they are degenerate
    collapsed = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])
    f = f[~collapsed]
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cr = np.cross(b - a, c - a)
    area = np.linalg.norm(cr, axis=1) / 2
    diag = np.linalg.norm(v.max(0) - v.min(0))
    degenerate = int(collapsed.sum() + (area < 1e-9 * diag * diag).sum())

    # directed edges
    de = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    nv = len(v)
    und = np.sort(de, axis=1)
    key = und[:, 0] * nv + und[:, 1]
    ukey, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    boundary = int((cnt == 1).sum())
    nonmanifold = int((cnt > 2).sum())
    # orientation conflict: an edge shared by exactly 2 faces that traverse it in the SAME direction
    two = cnt[inv] == 2
    dk = de[:, 0] * nv + de[:, 1]
    order = np.argsort(inv[two], kind='stable')
    dk2 = dk[two][order]
    pair_same = (dk2[0::2] == dk2[1::2]).sum() if len(dk2) % 2 == 0 else -1
    # components (face adjacency through shared vertices is too loose; use shared EDGES)
    fid = np.tile(np.arange(len(f)), 3)
    ordr = np.argsort(inv, kind='stable')
    fs, ks = fid[ordr], inv[ordr]
    same = ks[:-1] == ks[1:]
    g = coo_matrix((np.ones(same.sum()), (fs[:-1][same], fs[1:][same])), shape=(len(f), len(f)))
    ncomp, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    big = sizes.max()
    small_faces = int(sizes[sizes < max(50, 0.001 * len(f))].sum())
    # triangle quality: min angle
    e0, e1, e2 = np.linalg.norm(b - c, axis=1), np.linalg.norm(a - c, axis=1), np.linalg.norm(a - b, axis=1)
    ok = area > 0
    def ang(x, y, z):
        return np.degrees(np.arccos(np.clip((y * y + z * z - x * x) / np.maximum(2 * y * z, 1e-30), -1, 1)))
    minang = np.minimum(np.minimum(ang(e0, e1, e2), ang(e1, e0, e2)), ang(e2, e0, e1))
    sliver = float((minang[ok] < 5).mean() * 100)
    vol = float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6)
    used = np.unique(f)
    euler = len(used) - len(ukey) + len(f)
    if quiet:
        return dict(boundary=boundary, nonmanifold=nonmanifold, conflicts=int(pair_same), comps=int(ncomp), small=small_faces, euler=int(euler), volume=vol)
    print(f'== {path}')
    print(f'  GLB verts {len(pos)} faces {n_in}  |  welded verts {len(used)} faces {len(f)} (collapsed by weld {int(collapsed.sum())})')
    print(f'  bbox diag {diag:.3f}  area {area.sum():.4f}  signed volume {vol:.5f} ({"inside-out" if vol < 0 else "ok"})')
    print(f'  open boundary edges {boundary}  ({100.0 * boundary / len(ukey):.3f}% of edges)   non-manifold edges {nonmanifold}   orientation conflicts {pair_same}')
    print(f'  components {ncomp} (largest {big} faces = {100.0 * big / len(f):.2f}%)  faces in small fragments {small_faces} ({100.0 * small_faces / len(f):.3f}%)')
    print(f'  degenerate faces {degenerate}  slivers (min angle < 5 deg) {sliver:.2f}%  Euler characteristic {euler}')
    return dict(boundary=boundary, nonmanifold=nonmanifold, conflicts=int(pair_same), comps=int(ncomp), small=small_faces)


def icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(x, float) / np.linalg.norm(x) for x in v]
    for _ in range(level):
        cache, nf = {}, []
        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[k] = len(v) - 1
            return cache[k]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v), np.array(f, dtype=np.int64)


def selftest():
    """The tool must report a closed sphere as clean and each injected defect as exactly that defect."""
    v, f = icosphere(4)
    ok = audit_mesh(v, f, 'sphere', quiet=True)
    assert (ok['boundary'], ok['nonmanifold'], ok['conflicts'], ok['comps'], ok['euler']) == (0, 0, 0, 1, 2) and ok['volume'] > 0, ok
    hole = audit_mesh(v, f[1:], 'hole', quiet=True)                       # one face removed: 3 open edges
    assert hole['boundary'] == 3 and hole['conflicts'] == 0, hole
    g = f.copy(); g[0] = g[0][::-1]                                         # one face flipped: its 3 edges conflict
    flip = audit_mesh(v, g, 'flip', quiet=True)
    assert flip['conflicts'] == 3 and flip['boundary'] == 0, flip
    inside = audit_mesh(v, f[:, ::-1], 'inside-out', quiet=True)           # whole sphere inverted: clean topology, negative volume
    assert inside['volume'] < 0 and inside['boundary'] == 0 and inside['conflicts'] == 0, inside
    two = np.concatenate([f, f[:20] + 0 + len(v)]); vv = np.concatenate([v, v[:0]])
    dup = audit_mesh(np.concatenate([v, v + 5]), np.concatenate([f, f + len(v)]), 'two', quiet=True)   # two separate spheres
    assert dup['comps'] == 2 and dup['euler'] == 4, dup
    third = np.concatenate([f, [[f[0][0], f[0][1], len(v)]]]); extra = np.concatenate([v, [[0.3, 0.3, 3.0]]])   # a third face on one edge
    nm = audit_mesh(extra, third, 'nonmanifold', quiet=True)
    assert nm['nonmanifold'] == 1, nm
    print('selftest: closed sphere clean; hole -> 3 boundary; flipped face -> 3 conflicts; inverted -> negative volume; 2 spheres -> 2 components; extra face -> 1 non-manifold edge: PASS')


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        selftest()
    else:
        for p in sys.argv[1:]:
            audit(p)
