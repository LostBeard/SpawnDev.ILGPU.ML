"""Golden for the UniRig skeleton stage, from the reference's own classes (CPU, float32, greedy).

  python -I golden_skeleton.py <mesh.glb|mesh.obj> [seed=0] [max_new_tokens=1024]     writes UNIRIG_DIR/golden/skeleton/*

Writes the exact point cloud the model saw (points, normals in [-1,1]), the encoder latents after output_proj, the generated token ids and the
detokenized skeleton (bones, parents), so the C# port can be gated stage by stage: latents (float tolerance), then greedy token ids (exact).
Greedy: num_beams=1, do_sample=False, repetition_penalty=1 - the shipped inference config samples with 15 beams at temperature 1.5, which is random.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'trellis2-harness', 'reference'))
import ref_env
import numpy as np
import torch
import mesh_audit as ma


def sample_points(v, f, seed, num_samples=65536, vertex_samples=8192):
    """The reference's 'mix' sampler (src/data/sampler.py): random vertices with vertex normals + area-weighted surface samples with face normals,
    after centring the mesh and scaling its longest side into [-1, 1] (AugmentAffine). Seeded so the golden is reproducible."""
    rng = np.random.default_rng(seed)
    lo, hi = v.min(0), v.max(0)
    v = (v - (lo + hi) / 2) / ((hi - lo).max() / 2)
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); area2 = np.linalg.norm(fn, axis=1)
    fn = fn / np.maximum(area2[:, None], 1e-30)
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn * area2[:, None])
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-30)
    pick = rng.permutation(len(v))[:min(vertex_samples, len(v))]
    n_surface = num_samples - len(pick)
    cum = np.cumsum(area2 ** 2)                                   # the reference weights by squared cross-product length
    face = np.searchsorted(cum, rng.random(n_surface) * cum[-1])
    r = rng.random((n_surface, 2, 1)); flip = r.sum(1).reshape(-1) > 1.0; r[flip] -= 1.0; r = np.abs(r)
    p = tri[face, 0] + (np.stack([tri[face, 1] - tri[face, 0], tri[face, 2] - tri[face, 0]], 1) * r).sum(1)
    return (np.concatenate([v[pick], p]).astype(np.float32), np.concatenate([vn[pick], fn[face]]).astype(np.float32))


if __name__ == '__main__':
    mesh, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
    max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    out = os.path.join(ref_env.UNIRIG_DIR, 'golden', 'skeleton'); os.makedirs(out, exist_ok=True)
    pos, idx = ma.load_glb(mesh)
    v, f = ma.weld(pos, idx)
    used = np.unique(f); remap = -np.ones(len(v), int); remap[used] = np.arange(len(used)); v, f = v[used], remap[f]
    pts, nrm = sample_points(v, f, seed)
    print('sampled', pts.shape, 'bbox', pts.min(0).round(3), pts.max(0).round(3))

    torch.manual_seed(0)
    model, tokenizer, cfg = ref_env.build_skeleton_model()
    missing, unexpected = ref_env.load_skeleton_weights(model)
    print('state dict: missing', len(missing), 'unexpected', len(unexpected), missing[:3], unexpected[:3])
    assert not unexpected, unexpected
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        lat = model.encode_mesh_cond(vertices=torch.from_numpy(pts), normals=torch.from_numpy(nrm))
        print('encoder latents', tuple(lat.shape), f'{time.time() - t0:.1f}s', 'mean', float(lat.mean()), 'std', float(lat.std()))
        # the encoder's eval path: default_rng(seed=0).choice(N, 4 * token_num, replace=False) pre-samples the points, fps keeps a quarter as the queries
        pre, fps_idx = ref_env.FPS_CALLS[-1]
        ind = np.random.default_rng(seed=0).choice(pts.shape[0], 4 * 1024, replace=False)
        assert np.array_equal(pre.numpy(), pts[ind]), 'the encoder did not pre-sample the points the way numpy seed 0 does'
        np.save(os.path.join(out, 'presample_idx.npy'), ind); np.save(os.path.join(out, 'fps_idx.npy'), fps_idx.numpy())
        t0 = time.time()
        res = model.generate(vertices=torch.from_numpy(pts), normals=torch.from_numpy(nrm), cls='articulationxl',
                             max_new_tokens=max_new, num_beams=1, do_sample=False, num_return_sequences=1)
        print(f'generate {time.time() - t0:.1f}s, tokens {len(res.tokens)}, bones {res.bones.shape}')
    np.save(os.path.join(out, 'points.npy'), pts); np.save(os.path.join(out, 'normals.npy'), nrm)
    np.save(os.path.join(out, 'latents.npy'), lat.numpy())
    np.save(os.path.join(out, 'tokens.npy'), np.asarray(res.tokens)); np.save(os.path.join(out, 'bones.npy'), np.asarray(res.bones))
    np.save(os.path.join(out, 'parents.npy'), np.asarray([-1 if p is None else p for p in res.parents]))
    json.dump({'mesh': mesh, 'seed': seed, 'max_new_tokens': max_new, 'num_tokens': int(len(res.tokens)), 'num_bones': int(res.bones.shape[0]),
               'cls': res.cls, 'vocab_size': tokenizer.vocab_size, 'bos': tokenizer.bos, 'eos': tokenizer.eos, 'pad': tokenizer.pad},
              open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print('wrote', out)
