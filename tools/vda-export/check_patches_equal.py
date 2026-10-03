"""Patched (dynamic-shape) VDA vs the ORIGINAL code: same streaming outputs at several sizes.

usage: py -3.13 check_patches_equal.py <weights.pth> <out.npz> [--patched]
       py -3.13 check_patches_equal.py --compare a.npz b.npz
"""
import sys

import numpy as np


def run(weights, out, patched):
    import torch
    import export_vda_stream as e
    if patched:
        import vda_dynamic_patches
        vda_dynamic_patches.apply()
    m = e.load_model(weights)
    step = e.StreamStep(m).eval()
    res = {}
    for h, w in [(98, 168), (126, 224), (252, 448), (280, 504)]:
        g = torch.Generator().manual_seed(h * 1000 + w)
        xs = [torch.randn(1, 1, 3, h, w, generator=g) for _ in range(3)]
        d, c = e.first_frame_cache(m, xs[0])
        res[f'{h}x{w}_0'] = d.numpy()
        win = e.Window(c)
        with torch.no_grad():
            for k, x in enumerate(xs[1:], 1):
                o = step(x, *win.inputs())
                res[f'{h}x{w}_{k}'] = o[0].numpy()
                win.push(o[1:])
    np.savez(out, **res)


def compare(a, b):
    A, B = np.load(a), np.load(b)
    worst = 0.0
    for k in A.files:
        rel = float(np.abs(A[k] - B[k]).max() / (np.abs(A[k]).max() + 1e-6))
        worst = max(worst, rel)
        print(k, A[k].shape, f'{rel:.2e}')
    print('WORST', f'{worst:.2e}')


if __name__ == '__main__':
    if sys.argv[1] == '--compare':
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1], sys.argv[2], '--patched' in sys.argv)
