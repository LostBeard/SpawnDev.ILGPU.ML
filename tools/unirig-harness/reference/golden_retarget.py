"""Golden for the humanoid retargeter (Kimodo SOMA clips onto a SkinTokens-rigged character), from the numpy maths of image-to-3dlab's scripts/kimodo_retarget.py
(github.com/Bingeljell/image-to-3dlab, Apache-2.0) run as a BLACK BOX. The C# port (SpawnDev.ILGPU.ML.Animation) is its own implementation of the same published approach;
this script only produces the numbers it is gated against.

  python -I golden_retarget.py <rigged.glb | joints.npy+parents.npy base> <clip.npz> <name> [speed=1]      writes ANIM_DIR/golden/<name>_*.npy / .json

ANIM_DIR (default E:/spawndev-unirig/anim) holds ref/kimodo_retarget.py (a copy of the upstream script; not redistributed here) and clips/*.npz.
The rig is read from the GLB's joint nodes (`bone_N`, identity rotation, translation relative to the parent: what SpawnDev.ILGPU.ML's RiggedGlb writes) or from heads/parents arrays.
Up is +y. Output: mapping (SOMA index per bone), the character frame, the rest alignment, hip scale, SOMA T-pose, and per frame the local rotation quaternions (w, x, y, z)
and the root's travel, exactly what a glTF animation needs (rest rotations are identity, so basis = parentWorld^T * world).
"""
import os, sys, json, struct
import numpy as np

ANIM = os.environ.get('ANIM_DIR', 'E:/spawndev-unirig/anim')
sys.path.insert(0, os.path.join(ANIM, 'ref'))
import kimodo_retarget as kr                                    # noqa: E402


def glb_rig(path):
    d = open(path, 'rb').read(); jl, = struct.unpack_from('<I', d, 12); j = json.loads(d[20:20 + jl])
    skin = j['skins'][0]; joints = skin['joints']; nodes = j['nodes']
    idx = {n: i for i, n in enumerate(joints)}
    parent = [-1] * len(joints)
    for n in joints:
        for c in nodes[n].get('children', []):
            if c in idx: parent[idx[c]] = idx[n]
    heads = np.zeros((len(joints), 3))
    for i, n in enumerate(joints):                              # parents precede children
        t = np.array(nodes[n].get('translation', [0, 0, 0]), np.float64)
        heads[i] = t + (heads[parent[i]] if parent[i] >= 0 else 0)
    return heads, np.array(parent)


if __name__ == '__main__':
    src, clip, name = sys.argv[1], sys.argv[2], sys.argv[3]
    speed = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    out = os.path.join(ANIM, 'golden'); os.makedirs(out, exist_ok=True)
    if src.endswith('.glb'): heads, parent = glb_rig(src)
    else: heads, parent = np.load(src + '_heads.npy').astype(np.float64), np.load(src + '_parents.npy')
    J = len(parent); names = [f'bone_{i}' for i in range(J)]
    np.save(f'{out}/{name}_heads.npy', heads.astype(np.float32)); np.save(f'{out}/{name}_parents.npy', parent.astype(np.int32))
    parents = {names[i]: (names[parent[i]] if parent[i] >= 0 else None) for i in range(J)}
    hd = {names[i]: heads[i] for i in range(J)}
    up = np.array([0.0, 1.0, 0.0])
    try:
        mapping, frame = kr.map_skintokens_to_soma(parents, hd, up)
    except ValueError as e:
        json.dump({'error': str(e)}, open(f'{out}/{name}.json', 'w')); print('mapping refused:', e); sys.exit(0)
    axes = kr.soma_to_rig_axes(frame)
    g, joints0, hips = kr.load_motion(clip)
    lowest = min(heads[:, 1]); rig_hip = float(heads[0, 1] - lowest)
    scale = rig_hip / kr.soma_hip_height(joints0[None])
    tpose = kr.soma_tpose(g[:1], joints0[None])
    align = kr.rest_alignment(parents, hd, mapping, axes, tpose)
    rest = {}
    for n in names:
        m = np.eye(4); m[:3, 3] = hd[n]; rest[n] = m
    frames = list(kr.retarget_frames(rest, parents, names, mapping, axes, g, hips, scale, align, frame, 0.0, None, None))
    q = np.stack([kr.matrices_to_quaternions(np.stack([f[n][:3, :3] for n in names])) for f in frames])           # [T, J, 4] (w, x, y, z)
    travel = np.stack([frames[t][names[0]][:3, 3] for t in range(len(frames))])                                    # basis translation of the root = hips travel
    np.save(f'{out}/{name}_soma.npy', np.array([kr.SOMA_INDEX[mapping[n]] for n in names], np.int32))
    np.save(f'{out}/{name}_frame.npy', frame.astype(np.float64)); np.save(f'{out}/{name}_align.npy', np.stack([align[n] for n in names]))
    np.save(f'{out}/{name}_tpose.npy', tpose); np.save(f'{out}/{name}_quats.npy', q.astype(np.float64)); np.save(f'{out}/{name}_travel.npy', travel)
    json.dump({'clip': os.path.basename(clip), 'frames': len(frames), 'bones': J, 'hip_scale': float(scale), 'rig_hip': rig_hip,
               'mapping': {n: mapping[n] for n in names}}, open(f'{out}/{name}.json', 'w'), indent=1)
    print(name, J, 'bones,', len(frames), 'frames, hip scale', round(float(scale), 4), '| head/hands:', {k: v for k, v in mapping.items() if v in ('Head', 'LeftHand', 'RightHand', 'Hips')})
