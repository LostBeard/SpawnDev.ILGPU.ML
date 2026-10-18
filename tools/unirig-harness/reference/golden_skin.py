"""Golden for the UniRig SKIN network, from the reference's own classes: UniRigSkin + PTv3Object, the genuine spconv on the GPU (venv-cuda), float32.

  E:\\spawndev-unirig\\venv-cuda\\Scripts\\python.exe -I golden_skin.py <mesh.obj> <rig.json> <voxel_skin_dir> [seed=0]      writes UNIRIG_DIR/golden/skin/*

Inputs are exactly what the shipped pipeline hands the network: 32768 'mix' samples of the mesh (8192 vertex samples + area-weighted surface samples) with normals, the
joints of the rig, and the voxel-skin prior interpolated onto the samples (vertex samples take their vertex's row, surface samples the barycentric blend of the
face's three rows - SamplerMix). The prior comes from golden_voxel_skin.py (the reference's own voxel_skin()).

What is NOT the shipped behaviour, and why (each is recorded in manifest.json):
  - shuffle_orders=False. The shipped config permutes the four serialization orders (z, z-trans, hilbert, hilbert-trans) at random on every forward, so which order a
    block uses changes from run to run; any assignment is "valid". The port needs ONE, so the golden fixes the identity assignment (block i uses order i % 4).
  - torch.argsort is made stable. Two samples that fall in one 0.005 grid cell have equal serialization codes and the default argsort orders them arbitrarily.
  - flash_attn / torch_scatter are the shims in skin_shims.py (documented semantics). The patch attention runs in fp16 like the real kernel.
Intermediates saved: sampled points/normals/prior, grid coords, the four serialization orders, the output of the embedding, of every PTv3 stage, feat_map, the global
latents, the bone features, mesh_feat after cross-attention blocks 0/7/15, and the final skin weights.
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'trellis2-harness', 'reference'))
import skin_shims
skin_shims.install()
import numpy as np
import torch
import spconv.pytorch                                    # the real one: ref_env's mock loop will not replace a module that is already imported
import ref_env


def candidates(v, f, seed, n_surface_pool):
    """A pool of SamplerMix draws in priority order: every vertex (random order), then area-weighted surface points (weights = squared cross-product length, as the
    reference). Returns positions, normals, kind (0 vertex / 1 surface), vertex id (vertex kind), face id and barycentric weights (surface kind)."""
    rng = np.random.default_rng(seed)
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); area2 = np.linalg.norm(fn, axis=1)
    fn = fn / np.maximum(area2[:, None], 1e-30)
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn * area2[:, None])
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-30)
    pick = rng.permutation(len(v))
    cum = np.cumsum(area2 ** 2)
    face = np.searchsorted(cum, rng.random(n_surface_pool) * cum[-1])
    r = rng.random((n_surface_pool, 2, 1)); flip = r.sum(1).reshape(-1) > 1.0; r[flip] -= 1.0; r = np.abs(r)
    p = tri[face, 0] + (np.stack([tri[face, 1] - tri[face, 0], tri[face, 2] - tri[face, 0]], 1) * r).sum(1)
    w = np.concatenate([1.0 - r.sum(1), r[:, 0], r[:, 1]], 1)
    pos = np.concatenate([v[pick], p]); nrm = np.concatenate([vn[pick], fn[face]])
    kind = np.concatenate([np.zeros(len(pick), np.int8), np.ones(len(face), np.int8)])
    return pos, nrm, kind, pick, face, w


def cells_of(pts, origin, grid=0.005):
    return np.floor((pts.astype(np.float32) - origin.astype(np.float32)) / np.float32(grid)).astype(np.int64)      # the reference: torch.div(..., rounding_mode='trunc') on floats >= 0


def sample_mix_unique(v, f, seed, num_samples=32768, vertex_samples=8192, grid=0.005):
    """SamplerMix with one change: no two samples share a 0.005 grid cell (colliding samples are re-drawn from the pool). Why: with duplicate cells the genuine
    spconv picks neighbours through a racy hash insert, so its output differs from run to run (measured); without them it is deterministic and the port has an
    exact target. The origin of the cells is the per-axis minimum of the CHOSEN samples (what the reference uses), iterated until the choice agrees with it."""
    pos, nrm, kind, pick, face, w = candidates(v, f, seed, 6 * num_samples)
    n_v = min(vertex_samples, len(v))
    origin = pos.min(0)
    for it in range(40):
        key = cells_of(pos, origin, grid)
        flat = (key[:, 0] * 4096 + key[:, 1]) * 4096 + key[:, 2]
        _, first = np.unique(flat, return_index=True)
        keep = np.zeros(len(pos), bool); keep[first] = True
        vi = np.nonzero(keep & (kind == 0))[0][:n_v]
        si = np.nonzero(keep & (kind == 1))[0][:num_samples - n_v]
        sel = np.concatenate([vi, si])
        if len(sel) < num_samples: raise SystemExit(f'only {len(sel)} collision-free samples in the pool')
        new_origin = pos[sel].min(0)
        if np.array_equal(new_origin, origin): break
        origin = new_origin
    key = cells_of(pos[sel], pos[sel].min(0), grid)
    assert len(np.unique(key, axis=0)) == len(sel), 'cells are not unique under the chosen minimum'
    nv = len(vi)
    sv, sf = vi, si - len(pick)
    return pos[sel].astype(np.float32), nrm[sel].astype(np.float32), pick[sv], face[sf], w[si], it


def sample_group(g, f, pick, face, w):
    # SamplerMix._sample_barycentric: origin + sum(length_i * (v_i - origin)) = (1-a-b) g0 + a g1 + b g2
    return np.concatenate([g[pick], (g[f[face]] * w[:, :, None]).sum(1)]).astype(np.float32)


def build_skin_model():
    import src.model.michelangelo.models.tsal.sal_perceiver  # noqa: F401  (imported before install_fps patches it)
    from src.model.unirig_skin import UniRigSkin
    cfg = ref_env.load_yaml('configs/model/unirig_skin.yaml')
    cfg.pop('__target__')
    cfg.mesh_encoder.shuffle_orders = False
    cfg.global_encoder.flash = False
    ref_env.install_fps()
    return UniRigSkin(**cfg), cfg


def load_skin_weights(model):
    sys.path.insert(0, HERE)
    from safe_ckpt import load_checkpoint
    sd = load_checkpoint(os.path.join(ref_env.UNIRIG_DIR, 'ckpt', 'skin', 'model.ckpt'))['state_dict']
    sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}
    return model.load_state_dict(sd, strict=False)


if __name__ == '__main__':
    mesh, rig, vsdir = sys.argv[1], sys.argv[2], sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    out = os.path.join(ref_env.UNIRIG_DIR, 'golden', 'skin'); os.makedirs(out, exist_ok=True)
    save = lambda name, a: np.save(os.path.join(out, name + '.npy'), np.ascontiguousarray(a.detach().cpu().numpy() if torch.is_tensor(a) else a))

    stable = torch.argsort
    torch.argsort = lambda t, *a, **k: stable(t, *a, **{**k, 'stable': True})

    vn = np.load(os.path.join(vsdir, 'vertices.npy')).astype(np.float64)            # the mesh in [-1,1], exactly as the voxel-skin golden normalised it
    f = np.load(os.path.join(vsdir, 'faces.npy')).astype(np.int64)
    J = np.load(os.path.join(vsdir, 'joints.npy')).astype(np.float32)
    skin_v = np.load(os.path.join(vsdir, 'skin.npy')).astype(np.float64)             # (V, J) prior
    parents = np.array(json.load(open(rig))['parents'], np.int64)
    pts, nrm, pick, face, w, iters = sample_mix_unique(vn, f, seed)
    prior = sample_group(skin_v, f, pick, face, w)
    print('collision-free sampling converged after', iters + 1, 'rounds')
    print('samples', pts.shape, 'joints', J.shape, 'prior', prior.shape, 'row sums', float(prior.sum(1).min()), float(prior.sum(1).max()))
    for n, a in dict(points=pts, normals=nrm, prior=prior, joints=J, parents=parents, pick=pick, face=face, bary=w).items(): save(n, a)

    dev = torch.device('cuda')
    model, cfg = build_skin_model()
    miss, unexp = load_skin_weights(model)
    print('missing', miss, 'unexpected', unexp)
    model = model.to(dev).eval()

    cap = {}
    def grab(name, fn=lambda o: o):
        def hook(mod, inp, outp): cap[name] = fn(outp).detach().float().cpu().numpy()
        return hook
    feat_of = lambda p: p.feat
    me = model.mesh_encoder
    me.embedding.register_forward_hook(grab('embedding', feat_of))
    for s in range(me.num_stages):
        getattr(me.enc, f'enc{s}').register_forward_hook(grab(f'stage{s}', feat_of))
    model.feat_map.register_forward_hook(grab('feat_map'))
    model.out_proj.register_forward_hook(grab('global_latents'))
    model.bone_encoder.register_forward_hook(grab('bone_feat'))
    for i in (0, 7, 15):
        model.mesh_bone_attn[i].register_forward_hook(grab(f'cross{i}'))
    orders = {}
    orig_ser = type(me.embedding).forward
    def spy(self, point):
        orders['grid_coord'] = point.grid_coord.cpu().numpy(); orders['order'] = point.serialized_order.cpu().numpy()
        orders['code'] = point.serialized_code.cpu().numpy(); orders['depth'] = point.serialized_depth
        return orig_ser(self, point)
    type(me.embedding).forward = spy

    n_b = J.shape[0]
    batch = dict(
        num_bones=torch.tensor([n_b]), vertices=torch.from_numpy(pts)[None].to(dev), normals=torch.from_numpy(nrm)[None].to(dev),
        joints=torch.from_numpy(J)[None].to(dev), tails=torch.zeros(1, n_b, 3, device=dev), voxel_skin=torch.from_numpy(prior)[None].to(dev),
        parents=torch.from_numpy(parents)[None].to(dev), offset=torch.tensor([pts.shape[0]], device=dev))
    t0 = time.time()
    with torch.no_grad():
        res = model.predict_step(batch)
    torch.cuda.synchronize(); print('predict_step', round(time.time() - t0, 2), 's')
    skin = res[0].float().cpu().numpy()
    save('skin', skin)
    # the global encoder's eval path pre-samples 4096 points with numpy's default_rng(0) and keeps the 1024 farthest-point-sampled ones as queries
    pre, fps_idx = ref_env.FPS_CALLS[-1]
    ind = np.random.default_rng(seed=0).choice(pts.shape[0], 4 * 1024, replace=False)
    assert np.array_equal(pre.cpu().numpy(), pts[ind]), 'the encoder did not pre-sample the points the way numpy seed 0 does'
    save('presample_idx', ind); save('fps_idx', fps_idx.cpu().numpy())
    for k, a in cap.items(): save(k, a[0] if k in ('global_latents', 'bone_feat') or k.startswith('cross') else a)
    for k in ('grid_coord', 'order', 'code'): save('ser_' + k, orders[k])
    print('skin', skin.shape, 'row sums', float(skin.sum(1).min()), float(skin.sum(1).max()), 'dominant-bone histogram', np.bincount(skin.argmax(1), minlength=n_b).tolist())
    print('grid_coord max', int(orders['grid_coord'].max()), 'serialized depth', orders['depth'], 'distinct cells', len(np.unique(orders['grid_coord'], axis=0)), 'of', pts.shape[0])
    json.dump({'seed': seed, 'num_samples': int(pts.shape[0]), 'joints': int(n_b), 'serialized_depth': int(orders['depth']), 'shuffle_orders': False, 'stable_argsort': True,
               'spconv': 'real (venv-cuda)', 'torch': torch.__version__, 'missing': list(miss), 'unexpected': list(unexp)}, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print('wrote', out)
