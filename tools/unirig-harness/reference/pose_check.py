"""Does a rigged GLB deform sensibly? Reads the skin back out of the file (JOINTS_0 / WEIGHTS_0, joint nodes, inverse bind matrices), rotates one joint, applies linear blend skinning
and reports where the surface moved.

  python -I pose_check.py rigged.glb out.png [joint=auto] [degrees=-60] [axis=z]

Joint 'auto' = the joint nearest the model's right shoulder region (the outermost joint on +x that has children and sits above the hips). Prints how many vertices moved, how far,
and how far the torso, head and legs moved (they should hardly move); draws rest and posed front views (x right, y up).
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


def main():
    path, out = sys.argv[1], sys.argv[2]
    deg = float(sys.argv[4]) if len(sys.argv) > 4 else -60.0
    axis = sys.argv[5] if len(sys.argv) > 5 else 'z'
    j, acc = load(path)
    prim = j['meshes'][0]['primitives'][0]
    pos = acc(prim['attributes']['POSITION']).astype(np.float64)
    ji = acc(prim['attributes']['JOINTS_0']).astype(np.int64); wi = acc(prim['attributes']['WEIGHTS_0']).astype(np.float64)
    skin = j['skins'][0]; joints = skin['joints']; J = len(joints)
    ibm = acc(skin['inverseBindMatrices']).reshape(J, 4, 4).transpose(0, 2, 1)           # glTF is column-major
    nodes = j['nodes']; local_t = np.array([nodes[n].get('translation', [0, 0, 0]) for n in joints])
    parent = np.full(J, -1)
    for a, n in enumerate(joints):
        for c in nodes[n].get('children', []):
            if c in joints: parent[joints.index(c)] = a
    rest = np.zeros((J, 3))
    for a in range(J): rest[a] = local_t[a] + (rest[parent[a]] if parent[a] >= 0 else 0)
    print(f'{len(pos)} vertices, {J} joints; weights per vertex sum min/max {wi.sum(1).min():.4f}/{wi.sum(1).max():.4f}; influences used (mean nonzero) {(wi > 0).sum(1).mean():.2f}')

    if len(sys.argv) > 3 and sys.argv[3] != 'auto': pick = int(sys.argv[3])
    else:
        kids = np.array([len([c for c in range(J) if parent[c] == a]) for a in range(J)])
        cand = [a for a in range(J) if kids[a] > 0 and rest[a, 1] > np.percentile(rest[:, 1], 40)]
        pick = max(cand, key=lambda a: (rest[a, 0] > 0.0) * 1.0 + 0.1 * rest[a, 0])        # a joint out on +x: the right shoulder / upper arm
        # the shoulder is the first joint on the way out whose x exceeds a quarter of the span
        span = rest[:, 0].max()
        for a in range(J):
            if rest[a, 0] > 0.2 * span and kids[a] > 0 and rest[parent[a], 0] <= 0.2 * span: pick = a; break
    print('posing joint', pick, 'at', rest[pick].round(3), f'by {deg} degrees about {axis}')

    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    R = {'z': np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), 'x': np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), 'y': np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])}[axis]
    W = [None] * J
    for a in range(J):                                                                    # parents precede children
        L = np.eye(4); L[:3, 3] = local_t[a]
        if a == pick: L[:3, :3] = R
        W[a] = (W[parent[a]] if parent[a] >= 0 else np.eye(4)) @ L
    S = np.array([W[a] @ ibm[a] for a in range(J)])
    vh = np.concatenate([pos, np.ones((len(pos), 1))], 1)
    posed = np.zeros_like(pos)
    for k in range(4):
        m = S[ji[:, k]]                                                                   # [V,4,4]
        posed += wi[:, k, None] * np.einsum('vij,vj->vi', m, vh)[:, :3]
    move = np.linalg.norm(posed - pos, axis=1); height = pos[:, 1].max() - pos[:, 1].min()
    sub = np.zeros(J, bool); sub[pick] = True
    for a in range(J):
        if parent[a] >= 0 and sub[parent[a]]: sub[a] = True                               # parents precede children: one pass finds the subtree
    tipweight = (wi * sub[ji]).sum(1)                                                     # how much of a vertex follows the posed subtree
    print(f'vertices that follow the posed subtree (weight > 0.5): {(tipweight > 0.5).sum()}; moved more than 2% of the height: {(move > 0.02 * height).sum()}')
    far = np.abs(pos[:, 0] - rest[pick, 0]) > 0.3 * height
    for name, sel in [('left half (x < -0.1 height, should not move)', pos[:, 0] < -0.1 * height), ('torso/head/legs (|x| < 0.08 height)', np.abs(pos[:, 0]) < 0.08 * height)]:
        print(f'  {name}: {sel.sum()} vertices, max movement {move[sel].max() / height:.3f} of the height, mean {move[sel].mean() / height:.4f}')
    # views
    S0 = 460; lo, hi = pos.min(0), pos.max(0); c0 = (lo + hi) / 2; sc = 0.46 * S0 / max(hi[0] - lo[0], hi[1] - lo[1]) * 1.0
    def draw(p, color_by):
        im = Image.new('RGB', (S0, S0), (235, 235, 235)); d = ImageDraw.Draw(im)
        for i in range(0, len(p), 3):
            x = (p[i, 0] - c0[0]) * sc + S0 / 2; y = S0 / 2 - (p[i, 1] - c0[1]) * sc
            k = color_by[i]; d.point((x, y), fill=(int(40 + 200 * k), 60, int(220 - 200 * k)))
        return im
    a, b = draw(pos, tipweight), draw(posed, tipweight)
    o = Image.new('RGB', (S0 * 2, S0)); o.paste(a, (0, 0)); o.paste(b, (S0, 0)); o.save(out)
    print('wrote', out, '(left: rest, right: posed; red = follows the posed joint)')


main()
