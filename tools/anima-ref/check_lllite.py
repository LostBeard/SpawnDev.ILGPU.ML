"""Check anima_lllite_ref.py against Lore's ComfyUI LLLite black box (anima-run4-lllite: base v1.0 + inpainting v2, strength 1;
anima-run5-control: the same graph without the LLLite). Prints max-abs / relRMS per stage, writes JSON.

Usage: python check_lllite.py [--base PATH] [--lllite PATH] [--run DIR] [--skip-full] [--out JSON]

Stages:
  control block 0 / 27 from run5's own tapped inputs (no LLLite): proves the base weights and the DiT reference on this model.
  run4 block 0 / 27 from run4's own tapped inputs WITH the LLLite, one row per picture convention (the 'alt' rows must be worse),
    and WITHOUT it (the size of the LLLite's effect, so a match is not the DiT alone).
  END-TO-END call_0 / call_1 (denoised x - sigma v vs the dumped output) for run4 with the LLLite and run5 without.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anima_ref as A  # noqa: E402
import anima_lllite_ref as LL  # noqa: E402
from st_reader import SafeTensors  # noqa: E402
from taps import find  # noqa: E402

ROWS = []


def report(stage, ours, ref, note=''):
    ours = np.asarray(ours, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    assert ours.shape == ref.shape, (stage, ours.shape, ref.shape)
    d = ours - ref
    mx = float(np.abs(d).max())
    nr = float(np.linalg.norm(ref))
    rr = float(np.linalg.norm(d) / nr) if nr > 0 else float(np.linalg.norm(d))
    ROWS.append({'stage': stage, 'max_abs': mx, 'rel_rms': rr, 'note': note})
    print(f'{stage:<66} max_abs {mx:10.3e}  relRMS {rr:10.3e}  {note}', flush=True)
    return rr


class Weights:
    def __init__(self, path):
        self.st = SafeTensors(path)
        self.cache = {}

    def __call__(self, name):
        v = self.cache.get(name)
        if v is None:
            v = self.st.get(name)
            self.cache[name] = v
        return v

    def drop(self, prefix):
        for k in [k for k in self.cache if k.startswith(prefix)]:
            del self.cache[k]


def block_from_taps(folder, i, w, cos, sin, pre=None):
    t = lambda n: find(folder, f'tap_diffusion_model_blocks_{i}_0_{n}')  # noqa: E731
    x = t('pre-in0_0')
    B = x.shape[0]
    y = A.dit_block(x.reshape(B, -1, A.D_MODEL), t('pre-in1_0')[:, 0], t('pre-in2_0'), t('pre-kw-adaln_lora_B_T_3D0')[:, 0],
                    cos, sin, w, i, pre=pre)
    return y.reshape(x.shape), t('out0')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=r'D:\Projects\anima\anima-base-v1.0.safetensors')
    ap.add_argument('--lllite', default=r'D:\Projects\anima\anima-lllite-inpainting-v2.safetensors')
    ap.add_argument('--run', default=r'D:\users\tj\Projects\_DevComms\anima-run4-lllite-2026-11-28')
    ap.add_argument('--skip-full', action='store_true')
    ap.add_argument('--out', default=os.path.join(here, '..', '..', '_scratch', 'anima-ref', 'check_lllite_results.json'))
    a = ap.parse_args()
    r4, r5 = os.path.join(a.run, 'run4-lllite'), os.path.join(a.run, 'run5-control')
    w = Weights(a.base)
    L = LL.LLLite(SafeTensors(a.lllite))
    print('lllite metadata:', {k: v for k, v in L.meta.items() if k.startswith('lllite.')})
    t0 = time.time()

    rgb = np.asarray(Image.open(os.path.join(a.run, 'compositor-a.png')).convert('RGB'))
    alpha = np.asarray(Image.open(os.path.join(a.run, 'painter-lllite-rect.png')))[..., 3]
    mask = (alpha.astype(np.float64) / 255.0 >= 0.5).astype(np.float64)
    print(f'picture {rgb.shape}, mask covers {mask.mean():.4f} of it')
    variants = ['kohya', 'black', 'nomask_rgb', 'mask01', 'invert']
    cxs = {v: L.cond(LL.inpaint_picture(rgb, mask, v)) for v in variants}
    print(f'cx {cxs["kohya"].shape} ({time.time() - t0:.1f} s)')

    cos, sin = A.dit_rope_cos_sin(A.rope_angles(1, 32, 32))
    for i in (0, 27):
        y, ref = block_from_taps(r5, i, w, cos, sin)
        report(f'control (run5) blocks.{i} no LLLite, own tapped inputs', y, ref)
        y, ref = block_from_taps(r4, i, w, cos, sin)
        report(f'run4 blocks.{i} WITHOUT the LLLite (its effect)', y, ref, 'should be worse')
        for v in variants:
            y, ref = block_from_taps(r4, i, w, cos, sin, pre=L.pre_for_block(i, cxs[v]))
            note = '' if v == 'kohya' else 'alt: should be worse'
            report(f'run4 blocks.{i} WITH LLLite, picture {v}', y, ref, note)
        if i == 0:
            for eps in (1e-6,):
                cx_e = L.cond(LL.inpaint_picture(rgb, mask, 'kohya'), eps_gn=eps, eps_ln=eps)
                y, ref = block_from_taps(r4, i, w, cos, sin, pre=L.pre_for_block(i, cx_e))
                report(f'run4 blocks.0 WITH LLLite, kohya, GN/LN eps {eps:g}', y, ref, 'alt: eps')
            y, ref = block_from_taps(r4, i, w, cos, sin, pre=L.pre_for_block(i, cxs['kohya'], strength=0.5))
            report('run4 blocks.0 WITH LLLite, kohya, strength 0.5', y, ref, 'alt: should be worse')
        w.drop(f'net.blocks.{i}.')
        L.drop_block(i)

    if not a.skip_full:
        for name, folder, cx in (('run4 LLLite', r4, cxs['kohya']), ('run5 control', r5, None)):
            for c in (0, 1):
                x = find(folder, f'call_{c}_input')
                t = find(folder, f'call_{c}_timestep')
                ctx = find(folder, f'call_{c}_c_c_crossattn')
                out = find(folder, f'call_{c}_output')

                def hook(i, h):
                    w.drop(f'net.blocks.{i}.')
                    L.drop_block(i)
                pre = None if cx is None else (lambda i, cx=cx: L.pre_for_block(i, cx))
                ts = time.time()
                v = A.dit_forward(x, t, ctx, w, block_hook=hook, pre_for_block=pre)
                report(f'END-TO-END {name} call_{c} denoised (t={t[0]:.7f})', A.denoise_flow(x, t, v), out, f'{time.time() - ts:.0f} s')
        # the LLLite's own effect end to end (a match above must not be the DiT alone)
        report('run4 call_0 output vs run5 call_0 output (effect of the LLLite)', find(r4, 'call_0_output'), find(r5, 'call_0_output'), 'info')

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(ROWS, f, indent=1)
    print(f'wrote {a.out} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
