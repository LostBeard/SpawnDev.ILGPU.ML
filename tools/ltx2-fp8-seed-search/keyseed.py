import numpy as np, torch, zlib, hashlib, itertools
import os; S = os.path.dirname(os.path.abspath(__file__)) + '/'
frac, up, ok = [a.ravel() for a in np.load(S + 'sr_attn1.to_q.npy')]
N0 = 400
f = torch.tensor(frac[:N0]); u_ = torch.tensor(up[:N0]).bool(); k = torch.tensor(ok[:N0]).bool()
names = []
for pre in ['', 'model.', 'diffusion_model.', 'model.diffusion_model.']:
    for suf in ['.weight', '']:
        names.append(pre + 'transformer_blocks.1.attn1.to_q' + suf)
def seeds(s):
    b = s.encode()
    out = {'crc32': zlib.crc32(b), 'adler32': zlib.adler32(b)}
    for h in ('md5', 'sha1', 'sha256', 'blake2b'):
        d = hashlib.new(h, b).digest()
        for nb in (4, 8):
            out[f'{h}_{nb}le'] = int.from_bytes(d[:nb], 'little'); out[f'{h}_{nb}be'] = int.from_bytes(d[:nb], 'big')
        out[f'{h}_hexmod32'] = int(hashlib.new(h, b).hexdigest(), 16) % (2 ** 32)
        out[f'{h}_hexmod63'] = int(hashlib.new(h, b).hexdigest(), 16) % (2 ** 63)
        out[f'{h}_hexmod64'] = int(hashlib.new(h, b).hexdigest(), 16) % (2 ** 64)
    return out
g = torch.Generator()
best = (0, None)
for nm in names:
    for hname, sd in seeds(nm).items():
        for dt in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            try: g.manual_seed(sd)
            except Exception: continue
            r = torch.rand(N0, generator=g, dtype=dt).double()
            for on, pred in (('u<frac', r < f), ('u>=1-frac', r >= 1 - f), ('u>frac (down)', r > f)):
                agree = ((pred == u_) & k).sum().item() / k.sum().item()
                if agree > best[0]: best = (agree, (nm, hname, str(dt), on))
                if agree > 0.99: print('MATCH', agree, nm, hname, dt, on, flush=True)
print('best', best)
