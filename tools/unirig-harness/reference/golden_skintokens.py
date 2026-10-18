"""Golden for SkinTokens / TokenRig (VAST-AI-Research/SkinTokens, MIT), from the reference's own classes, float32 on the GPU, greedy.

  E:\\spawndev-unirig\\venv-cuda\\Scripts\\python.exe -I golden_skintokens.py <mesh.obj> [seed=0] [max_new_tokens=1400]      writes ST_DIR/golden/*

What the network sees (the shipped predict transform): the mesh scaled into [-1,1] (longest side), 54000 'mix' samples (16384 of them vertex samples) with normals.
Written, so the C# port can be gated stage by stage:
  points / normals                    the 54000 x 3 inputs
  mesh_presample_idx, mesh_fps_idx    the Michelangelo encoder's 2048 pre-sampled points (numpy default_rng(0)) and the 512 farthest-point-sampled queries
  mesh_cond                           [512, 896] prefix after output_proj (RMSNorm)
  cond_presample_idx, cond_fps_idx    the VAE cond encoder's 1536 pre-sampled points (default_rng(seed)) and the 384 FPS queries
  cond_latents                        [384, 512] after cond_quant
  ids                                 start tokens + generated ids (greedy, repetition_penalty 1, grammar on), skeleton then J x 4 skin tokens then eos
  logits_teacher                      [len(ids), 33036] raw logits of the decoder at each position (teacher forced on ids)
  allowed_skeleton.json               the grammar's allowed ids at each skeleton step
  skin_z, skin_kv0, skin_pred         FSQ -> post_quant input of the first bone [4, 512], decoder state after blocks[:-1] [388, 768], skin weights [54000, J]
  joints, parents                     the detokenized skeleton
The sampling of the shipped config is random (beams 10, top-k 10, top-p .95, temperature 1.5, repetition penalty 2): a golden needs ONE sequence, so greedy.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import st_env
import torch
import torch.nn.functional as F


def sample_points(v, f, seed, num_samples=54000, vertex_samples=16384):
    """SamplerMix on a mesh (any scale; normalised here like the 'affine normalize_into [-1,1]' augment), seeded."""
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
    cum = np.cumsum(area2 ** 2)
    face = np.searchsorted(cum, rng.random(n_surface) * cum[-1])
    r = rng.random((n_surface, 2, 1)); flip = r.sum(1).reshape(-1) > 1.0; r[flip] -= 1.0; r = np.abs(r)
    p = tri[face, 0] + (np.stack([tri[face, 1] - tri[face, 0], tri[face, 2] - tri[face, 0]], 1) * r).sum(1)
    return (np.concatenate([v[pick], p]).astype(np.float32), np.concatenate([vn[pick], fn[face]]).astype(np.float32))


def load_obj(path):
    v, f = [], []
    for line in open(path, encoding='utf-8'):
        t = line.split()
        if len(t) >= 4 and t[0] == 'v': v.append([float(x) for x in t[1:4]])
        elif len(t) >= 4 and t[0] == 'f': f.append([int(x.split('/')[0]) - 1 for x in t[1:4]])
    return np.array(v, np.float64), np.array(f, np.int64)


if __name__ == '__main__':
    mesh = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 1400
    out = os.path.join(st_env.ST_DIR, 'golden'); os.makedirs(out, exist_ok=True)
    save = lambda name, a: np.save(os.path.join(out, name + '.npy'), np.ascontiguousarray((a.detach().cpu().long().numpy() if not a.is_floating_point() else a.detach().float().cpu().numpy()) if torch.is_tensor(a) else a))

    # record every farthest-point-sampling call (positions in, indices out); the first is the VAE cond encoder's, the second the mesh encoder's... order is asserted below
    import src.model.utils as mu
    import src.model.michelangelo.models.tsal.sal_perceiver as sp
    import src.model.skin_vae.autoencoders.skin_fsq_cvae_model as sv
    import src.model.skin_vae.autoencoders.autoencoder_kl_tripo2 as at
    FPS = []
    orig_fps = mu.fps
    def rec_fps(x, batch, ratio, random_start=False):
        idx = orig_fps(x, batch, ratio, random_start)
        FPS.append((x.detach().clone(), idx.detach().clone()))
        return idx
    for m in (mu, sp, sv, at):
        m.fps = rec_fps

    v, f = load_obj(mesh)
    pts, nrm = sample_points(v, f, seed)
    print('samples', pts.shape, 'bbox', pts.min(0).round(3), pts.max(0).round(3))
    save('points', pts); save('normals', nrm)

    dev = 'cuda'
    model, hp = st_env.build_tokenrig(dev)
    tok = model.tokenizer
    P = torch.from_numpy(pts).to(dev); Nn = torch.from_numpy(nrm).to(dev)
    with torch.no_grad():
        # probes: the first outputs of the VAE's modules (rows capped), for the C# port to find where it first diverges
        PROBE = {}
        def probe(name, mod):
            def hook(m, i, o):
                out = o[0] if isinstance(o, tuple) else o
                PROBE.setdefault(name, []).append(out.detach().float().cpu().numpy()[0][:4000])
            mod.register_forward_hook(hook)
        vm = model.vae.model
        probe('cond.proj_in', vm.cond_encoder.proj_in)
        for bi, b in enumerate(vm.cond_encoder.blocks): probe(f'cond.block{bi}', b)
        probe('cond.norm_out', vm.cond_encoder.norm_out)
        probe('dec.proj_query', vm.decoder.proj_query); probe('dec.block10', vm.decoder.blocks[-1]); probe('dec.norm_out', vm.decoder.norm_out); probe('dec.proj_out', vm.decoder.proj_out)
        # 1) the cond path of the VAE (seeded here; the shipped generate() leaves the seed None = random picks)
        cond = torch.cat([P, Nn], dim=-1).unsqueeze(0)
        FPS.clear()
        _, cond_latents = model.vae.model._encode(x=None, cond=cond, num_tokens=model.tokens_per_skin, cond_tokens=model.tokens_skin_cond, seed=seed, return_z=False)
        pre, fps_idx = FPS[-1]
        ind = np.random.default_rng(seed).choice(pts.shape[0], 4 * model.tokens_skin_cond, replace=False)
        assert np.array_equal(pre.cpu().numpy(), cond[0, ind, :3].cpu().numpy()), 'cond pre-sample is not default_rng(seed).choice'
        save('cond_presample_idx', ind); save('cond_fps_idx', fps_idx); save('cond_latents', cond_latents[0])
        print('cond_latents', tuple(cond_latents.shape), 'mean', float(cond_latents.mean()), 'std', float(cond_latents.std()))

        # 2) the mesh encoder -> prefix
        import src.model.tokenrig as tr
        enc = model.mesh_encoder.encoder
        probe('mesh.cross', enc.cross_attn)
        for bi, b in enumerate(enc.self_attn.resblocks): probe(f'mesh.self{bi}', b)
        FPS.clear()
        mesh_cond = tr.encode_mesh_cond(model.mesh_encoder, model.output_proj, model.tokens_skin_cond, {'vertices': P, 'normals': Nn})
        pre, fps_idx = FPS[-1]
        ind = np.random.default_rng(seed=0).choice(pts.shape[0], 4 * 512, replace=False)
        assert np.array_equal(pre.cpu().numpy(), pts[ind]), 'mesh pre-sample is not default_rng(0).choice'
        save('mesh_presample_idx', ind); save('mesh_fps_idx', fps_idx); save('mesh_cond', mesh_cond[0])
        save('mesh_latents', model.mesh_encoder.encode_latents(pc=P.unsqueeze(0), feats=Nn.unsqueeze(0))[1][0])        # ln_post output, before output_proj
        print('mesh_cond', tuple(mesh_cond.shape), 'mean', float(mesh_cond.mean()), 'std', float(mesh_cond.std()))

        # 3) greedy generation with the grammar
        start = torch.tensor(model.make_start_tokens(cls=['articulation'])[0], device=dev).unsqueeze(0)
        emb = model.transformer.get_input_embeddings()(start)
        inputs_embeds = torch.cat([mesh_cond, emb], dim=1)
        proc = tr.get_logits_processor(tokenizer=tok, eos=model.eos, tokens_per_skin=model.tokens_per_skin, start_tokens=start[0])
        t0 = time.time()
        gen = model.transformer.generate(inputs_embeds=inputs_embeds, bos_token_id=tok.bos, eos_token_id=model.eos, pad_token_id=tok.pad, logits_processor=proc,
                                         max_new_tokens=max_new, do_sample=False, num_beams=1, num_return_sequences=1)
        print('generate', round(time.time() - t0, 1), 's', gen.shape)
        ids = torch.cat([start[0], gen[0]])
        if ids[-1].item() != model.eos: print('WARNING: the sequence did not end with eos (max_new_tokens)')
        save('ids', ids.cpu().numpy().astype(np.int64))

        # 4) teacher-forced logits over the whole sequence (the prefix rows are dropped)
        full_emb = model.transformer.get_input_embeddings()(ids.unsqueeze(0))
        logits = model.transformer(inputs_embeds=torch.cat([mesh_cond, full_emb], dim=1), use_cache=False).logits[0, mesh_cond.shape[1]:]
        save('logits_teacher', logits)

        # 5) the grammar: allowed ids at each skeleton step, for the ids that were generated
        eos_pos = int((ids == tok.eos).nonzero()[0])
        allowed = []
        for r in range(1, eos_pos + 1):                                        # the set offered after ids[:r]
            allowed.append(sorted(int(x) for x in tok.next_posible_token(ids=ids[:r].cpu().numpy())))
        json.dump({'switch_position': eos_pos, 'allowed_after_prefix_len': allowed}, open(os.path.join(out, 'allowed_skeleton.json'), 'w'))

        # 6) decode the skin tokens
        res = tr.decode(cond=cond[0], cond_latents=cond_latents[0], inputs_ids=ids, tokenizer=tok, tokens_per_skin=model.tokens_per_skin, vae=model.vae)
        d = res['detokenize_output']; skin = res['skin_pred']
        J = d.joints.shape[0]
        save('joints', d.joints); save('parents', np.array(d.parents, np.int64)); save('skin_pred', skin)
        first = ids[eos_pos + 1:eos_pos + 1 + model.tokens_per_skin].unsqueeze(0) - tok.vocab_size
        z = model.vae.model.FSQ.indices_to_codes(first).reshape(1, model.tokens_per_skin, -1)
        save('skin_z', z[0])
        zz = model.vae.model.post_quant(torch.cat([z, cond_latents], dim=1))
        h = zz
        for b in model.vae.model.decoder.blocks[:-1]:
            h = b(h)
        save('skin_kv0', h[0])
        for name, lst in PROBE.items():
            if name.startswith('mesh.'): save('probe_' + name, lst[0])
            elif name.startswith('cond.'): save('probe_' + name, lst[0] if name != 'cond.proj_in' else np.concatenate([lst[0][:384], lst[1][:3616]]))     # proj_in: 384 queries then the first keys
            else: save('probe_' + name, lst[0])
        print('J', J, 'skin_pred', tuple(skin.shape), 'min/max', float(skin.min()), float(skin.max()), 'row-sum range', float(skin.sum(1).min()), float(skin.sum(1).max()))
    json.dump({'seed': seed, 'num_samples': int(pts.shape[0]), 'J': int(J), 'sequence_length': int(ids.numel()), 'switch_position': eos_pos, 'torch': torch.__version__,
               'dtype': 'float32 (weights bf16 -> fp32 exactly)', 'attn': 'sdpa', 'repetition_penalty': 1.0, 'greedy': True}, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print('wrote', out)
