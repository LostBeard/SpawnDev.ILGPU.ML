import numpy as np, torch, time, sys
import os; S = os.path.dirname(os.path.abspath(__file__)) + '/'
frac, up, ok = np.load(S + 'sr_attn1.to_q.npy')
N0 = 200
f = torch.tensor(frac.ravel()[:N0]); u_ = torch.tensor(up.ravel()[:N0]).bool(); k = torch.tensor(ok.ravel()[:N0]).bool()
lo, hi = int(sys.argv[1]) if len(sys.argv) > 1 else 0, int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
g = torch.Generator()
best = (0, None)
t0 = time.time()
for s in range(lo, hi):
    g.manual_seed(s)
    for dt in (torch.float32, torch.float64):
        if dt is torch.float64: g.manual_seed(s)
        r = torch.rand(N0, generator=g, dtype=dt).double()
        for name, pred in (('u<frac', r < f), ('u>=1-frac', r >= 1 - f)):
            agree = ((pred == u_) & k).sum().item() / k.sum().item()
            if agree > best[0]: best = (agree, (s, str(dt), name))
            if agree == 1.0:
                print('MATCH', s, dt, name, flush=True)
print('best', best, f'{time.time() - t0:.1f} s', flush=True)
