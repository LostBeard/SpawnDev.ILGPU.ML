"""Check anima_lora_ref.py (an Anima LoRA merged into the reference's weights) against Lore's ComfyUI run 10
(_DevComms/anima-run10-turbo-lora-2026-11-28: anima-base-v1.0 + anima-turbo-lora-v0.2 via LoraLoaderModelOnly at strength 1.0 / 0.5, and a
no-LoRA control; cfg 1, seed 42; tags lora100_ / lora050_ / nolora_).

Decides: the scale (strength * B A, no alpha), whether ComfyUI's patched weight is exact or re-rounded to bf16, and that the adapter is NOT
patched (Lore: its output is the same with and without the LoRA). Usage: python check_lora.py [--base PATH] [--lora PATH] [--run DIR] [--quick]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anima_ref as A  # noqa: E402
from anima_lora_ref import LoraWeights  # noqa: E402
from check_lllite import Weights, report, ROWS  # noqa: E402
from st_reader import SafeTensors  # noqa: E402
from taps import find  # noqa: E402


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=r'D:\Projects\anima\anima-base-v1.0.safetensors')
    ap.add_argument('--lora', default=r'D:\Projects\anima\anima-turbo-lora-v0.2.safetensors')
    ap.add_argument('--run', default=r'D:\users\tj\Projects\_DevComms\anima-run10-turbo-lora-2026-11-28')
    ap.add_argument('--quick', action='store_true', help='call_0 only, exact mode only')
    ap.add_argument('--out', default=os.path.join(here, '..', '..', '_scratch', 'anima-ref', 'check_lora_results.json'))
    a = ap.parse_args()
    d = lambda tag, name: find(a.run, f'{tag}_{name}')  # noqa: E731
    base = Weights(a.base)
    lst = SafeTensors(a.lora)
    t0 = time.time()

    def dit_only(lw):
        # the LoRA on the DiT blocks only: the adapter keys are left out (ComfyUI's observable behaviour)
        lw.mods = {k: v for k, v in lw.mods.items() if not k.startswith('net.llm_adapter.')}
        return lw

    def call(tag, c, w):
        x, t, ctx, out = d(tag, f'call_{c}_input'), d(tag, f'call_{c}_timestep'), d(tag, f'call_{c}_c_c_crossattn'), d(tag, f'call_{c}_output')
        hook = lambda i, h: w.drop(f'net.blocks.{i}.')  # noqa: E731
        ts = time.time()
        v = A.dit_forward(x, t, ctx, w, block_hook=hook)
        return A.denoise_flow(x, t, v), out, time.time() - ts

    # the adapter: ComfyUI's output is the same with and without the LoRA. NOT a test of whether ComfyUI patches it: in anima-turbo-lora-v0.2 every
    # llm_adapter lora_B is ZERO (printed below), so patching it changes nothing either. The DiT-only rule follows the observation.
    q = d('pos_0', 'cond')
    ids = np.rint(d('pos_0', 't5xxl_ids')).astype(np.int64)[None]
    ad_ref = d('lora100', 'tap_diffusion_model_llm_adapter_0_out0')
    ad_b = [k for k in lst.header if 'llm_adapter' in k and k.endswith('lora_B.weight')]
    print(f'adapter lora_B tensors: {len(ad_b)}, largest |value| {max(float(np.abs(lst.get(k)).max()) for k in ad_b):.3g}')
    report('adapter, no LoRA vs lora100 tap', A.adapter_forward(q, ids, base), ad_ref)
    report('adapter WITH the LoRA on its blocks vs lora100 tap', A.adapter_forward(q, ids, LoraWeights(base, lst, 1.0)), ad_ref, 'info: lora_B all zero')
    report('nolora adapter tap vs lora100 adapter tap', d('nolora', 'tap_diffusion_model_llm_adapter_0_out0'), ad_ref, 'info: Lore 0.0')
    base.drop('net.llm_adapter.')

    y, ref, s = call('nolora', 0, base)
    report('control nolora call_0 denoised', y, ref, f'{s:.0f} s')
    cases = [('lora100', 1.0, 'exact'), ('lora050', 0.5, 'exact')]
    if not a.quick:
        cases += [('lora100', 1.0, 'bf16'), ('lora050', 0.5, 'bf16')]
    for tag, st, mode in cases:
        w = dit_only(LoraWeights(base, lst, st, mode=mode))
        y, ref, s = call(tag, 0, w)
        report(f'{tag} call_0 denoised, strength {st} x B A, merged {mode}', y, ref, f'{s:.0f} s')
    if not a.quick:
        w = dit_only(LoraWeights(base, lst, 1.0))
        y, ref, s = call('lora100', 1, w)
        report('lora100 call_1 denoised, strength 1 x B A, merged exact', y, ref, f'{s:.0f} s')
        w = dit_only(LoraWeights(base, lst, 0.5))
        y, ref, s = call('lora100', 0, w)
        report('alt: lora100 call_0 with strength 0.5', y, ref, 'should be worse')
        y, ref, s = call('lora100', 0, base)
        report('alt: lora100 call_0 without the LoRA (its effect)', y, ref, 'should be worse')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(ROWS, f, indent=1)
    print(f'wrote {a.out} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
