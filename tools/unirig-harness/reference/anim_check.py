"""Does an animated GLB play? Reads the skin and an animation back out of the file, poses the mesh with linear blend skinning at evenly spaced frames and draws a contact sheet
(front view on top, side view below; points coloured by height).

  python -I anim_check.py animated.glb out.png [animation=0] [frames=6]

Prints, per frame, how far the feet and the head are from where they started, so a walk should show the feet alternating and the head nearly level.
"""
import sys, json, struct
import numpy as np
from PIL import Image, ImageDraw

CT = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}; NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def load(path):
    d = open(path, 'rb').read(); jl, = struct.unpack_from('<I', d, 12); j = json.loads(d[20:20 + jl]); b0 = 20 + jl + 8
    def acc(i):
        a = j['accessors'][i]; bv = j['bufferViews'][a['bufferView']]; c = NC[a['type']]
        x = np.frombuffer(d, dtype=CT[a['componentType']], count=a['count'] * c, offset=bv.get('byteOffset', 0) + a.get('byteOffset', 0) + b0)
        return x.reshape(a['count'], c) if c > 1 else x
    return j, acc


def quat_matrix(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def slerp(a, b, u):
    if a @ b < 0: b = -b
    d = min(1.0, float(a @ b))
    if d > 0.9995: q = a + u * (b - a)
    else:
        th = np.arccos(d); q = (np.sin((1 - u) * th) * a + np.sin(u * th) * b) / np.sin(th)
    return q / np.linalg.norm(q)


def main():
    path, out = sys.argv[1], sys.argv[2]
    which = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    nframes = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    j, acc = load(path)
    prim = j['meshes'][0]['primitives'][0]
    pos = acc(prim['attributes']['POSITION']).astype(np.float64); ji = acc(prim['attributes']['JOINTS_0']).astype(np.int64); wi = acc(prim['attributes']['WEIGHTS_0']).astype(np.float64)
    skin = j['skins'][0]; joints = skin['joints']; J = len(joints)
    ibm = acc(skin['inverseBindMatrices']).reshape(J, 4, 4).transpose(0, 2, 1)
    nodes = j['nodes']; local_t = np.array([nodes[n].get('translation', [0, 0, 0]) for n in joints], dtype=np.float64)
    parent = np.full(J, -1)
    for a, n in enumerate(joints):
        for c in nodes[n].get('children', []):
            if c in joints: parent[joints.index(c)] = a
    anim = j['animations'][which]
    print('animation', anim.get('name'), 'channels', len(anim['channels']))
    track = {}
    for ch in anim['channels']:
        s = anim['samplers'][ch['sampler']]; node = ch['target']['node']
        track[(joints.index(node), ch['target']['path'])] = (acc(s['input']).astype(np.float64), acc(s['output']).astype(np.float64))
    times = next(iter(track.values()))[0]; end = times[-1]
    root = int(np.where(parent < 0)[0][0])

    def pose_at(t):
        W = [None] * J
        for a in range(J):
            L = np.eye(4); L[:3, 3] = local_t[a]
            if (a, 'rotation') in track:
                tt, qq = track[(a, 'rotation')]; k = min(np.searchsorted(tt, t, side='right') - 1, len(tt) - 2); k = max(k, 0)
                u = 0 if tt[k + 1] == tt[k] else np.clip((t - tt[k]) / (tt[k + 1] - tt[k]), 0, 1)
                L[:3, :3] = quat_matrix(slerp(qq[k], qq[k + 1], u))
            if a == root and (a, 'translation') in track:
                tt, pp = track[(a, 'translation')]; k = min(np.searchsorted(tt, t, side='right') - 1, len(tt) - 2); k = max(k, 0)
                u = 0 if tt[k + 1] == tt[k] else np.clip((t - tt[k]) / (tt[k + 1] - tt[k]), 0, 1)
                L[:3, 3] = pp[k] + u * (pp[k + 1] - pp[k])
            W[a] = (W[parent[a]] if parent[a] >= 0 else np.eye(4)) @ L
        S = np.array([W[a] @ ibm[a] for a in range(J)])
        vh = np.concatenate([pos, np.ones((len(pos), 1))], 1); posed = np.zeros_like(pos)
        for k in range(4): posed += wi[:, k, None] * np.einsum('vij,vj->vi', S[ji[:, k]], vh)[:, :3]
        return posed, np.array([w[:3, 3] for w in W])

    ts = np.linspace(0, end * (1 - 1 / nframes), nframes)
    lo, hi = pos.min(0), pos.max(0); height = hi[1] - lo[1]; S0 = 300
    sheet = Image.new('RGB', (S0 * nframes, S0 * 2), (235, 235, 235))
    first_head = None
    for i, t in enumerate(ts):
        posed, jp = pose_at(t)
        top = posed[:, 1].max(); bot = posed[:, 1].min()
        if first_head is None: first_head = top
        print(f't={t:.2f}s: lowest point {bot:.3f}, highest {top:.3f}, hips at ({jp[root][0]:.3f}, {jp[root][1]:.3f}, {jp[root][2]:.3f})')
        for view, (ax, off) in enumerate((((0, 1), 0), ((2, 1), S0))):
            im = Image.new('RGB', (S0, S0), (235, 235, 235)); d = ImageDraw.Draw(im)
            c0 = (lo + hi) / 2; sc = 0.9 * S0 / max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
            travel = jp[root] - local_t[root]                                              # keep the body in frame: the side view follows the hips' forward travel
            cx = c0[0] if ax[0] == 0 else c0[2] + travel[2]
            for p in posed[::3]:
                x = (p[ax[0]] - cx) * sc + S0 / 2
                y = S0 / 2 - (p[1] - c0[1]) * sc
                k = (p[1] - lo[1]) / height; d.point((x, y), fill=(int(40 + 180 * k), 70, int(220 - 180 * k)))
            sheet.paste(im, (i * S0, off))
    sheet.save(out); print('wrote', out)


main()
