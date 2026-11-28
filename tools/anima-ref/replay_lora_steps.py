"""Replay run 10's 8 euler steps (anima-base-v1.0 +- anima-turbo-lora-v0.2, cfg 1) in float64 and compare the final latent with ComfyUI's.

Question (Lore, 2026-11-28): through 8 steps the GPU port lands 1.0e-3 from ComfyUI's final latent with the LoRA and 3.5e-5 without, while each single
call is 3e-5 either way. If this float64 replay lands about as far with the LoRA, the trajectory itself amplifies ComfyUI's fp32 rounding (nothing to
fix in the port); if it lands close, the port drifts.

Sampler (checked here first on ComfyUI's own numbers): euler on a flow model, x_{i+1} = x_i + (s_{i+1} - s_i) (x_i - D_i) / s_i, D = denoised = x - s v;
the last step (s = 0) leaves D. Usage: python replay_lora_steps.py [--tags lora100,nolora]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anima_ref as A  # noqa: E402
from anima_lora_ref import LoraWeights  # noqa: E402
from check_lllite import Weights, report  # noqa: E402
from st_reader import SafeTensors  # noqa: E402
from taps import find  # noqa: E402

RUN = r'D:\users\tj\Projects\_DevComms\anima-run10-turbo-lora-2026-11-28'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', default='lora100,nolora')
    ap.add_argument('--base', default=r'D:\Projects\anima\anima-base-v1.0.safetensors')
    ap.add_argument('--lora', default=r'D:\Projects\anima\anima-turbo-lora-v0.2.safetensors')
    a = ap.parse_args()
    base = Weights(a.base)
    lst = SafeTensors(a.lora)
    strengths = {'lora100': 1.0, 'lora050': 0.5, 'nolora': 0.0}
    for tag in a.tags.split(','):
        d = lambda n: find(RUN, f'{tag}_{n}')  # noqa: E731
        s = d('call_0_c_transformer_options_sample_sigmas')
        x0, ctx = d('call_0_input'), d('call_0_c_c_crossattn')
        # the sampler on ComfyUI's own numbers: call_0 (input, output) -> call_1's input
        D0 = d('call_0_output')
        x1 = x0 + (s[1] - s[0]) * (x0 - D0) / s[0]
        report(f'{tag}: euler step from ComfyUI call_0 vs ComfyUI call_1 input', x1, d('call_1_input'), 'sampler check')
        print(f'{tag}: sigmas {np.array2string(s, precision=6)}', flush=True)
        st = strengths[tag]
        w = base if st == 0 else LoraWeights(base, lst, st)
        if st != 0:
            w.mods = {k: v for k, v in w.mods.items() if not k.startswith('net.llm_adapter.')}
        x = x0.copy()
        t0 = time.time()
        for i in range(len(s) - 1):
            t = np.array([s[i]])
            v = A.dit_forward(x, t, ctx, w, block_hook=lambda b, h: w.drop(f'net.blocks.{b}.'))
            D = A.denoise_flow(x, t, v)
            if i == 1:
                report(f'{tag}: replay step 1 input vs ComfyUI call_1 input', x, d('call_1_input'))
            x = D if s[i + 1] == 0 else x + (s[i + 1] - s[i]) * (x - D) / s[i]
            print(f'  {tag} step {i} sigma {s[i]:.6f} done ({time.time() - t0:.0f} s)', flush=True)
        report(f'{tag}: float64 8-step final latent vs ComfyUI out_samples', x, d('out_samples'), f'{time.time() - t0:.0f} s')
        np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '_scratch', 'anima-ref', f'replay_{tag}_final.npy'), x)


if __name__ == '__main__':
    main()
