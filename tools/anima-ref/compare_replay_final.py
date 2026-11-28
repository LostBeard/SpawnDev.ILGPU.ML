"""Compare replay_lora_steps.py's float64 final latents with ComfyUI's out_samples in the same space.

ComfyUI's KSampler hands back latent_format.process_out(x): for Anima's 16-channel (Wan 2.1 / Qwen-Image VAE) latent that is x * std_c + mean_c per channel
(the constants below are ML's Wan/WanComfyFormat.cs LatentsMean21 / LatentsStd21). The script checks that, by a least-squares per-channel affine fit and
with the constants, then reports both runs in the model's own space.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_lllite import report  # noqa: E402
from taps import find  # noqa: E402

RUN = r'D:\users\tj\Projects\_DevComms\anima-run10-turbo-lora-2026-11-28'
MEAN = np.array([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921])
STD = np.array([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160])
here = os.path.dirname(os.path.abspath(__file__))
for tag in ('lora100', 'nolora'):
    ours = np.load(os.path.join(here, '..', '..', '_scratch', 'anima-ref', f'replay_{tag}_final.npy'))
    out = find(RUN, f'{tag}_out_samples')
    fit = []
    for c in range(16):
        a, b = np.polyfit(ours[0, c].ravel(), out[0, c].ravel(), 1)
        fit.append((a, b))
    fit = np.array(fit)
    print(f'{tag}: fitted scale vs std: max |a - std| {np.abs(fit[:, 0] - STD).max():.3e}; fitted shift vs mean: max |b - mean| {np.abs(fit[:, 1] - MEAN).max():.3e}')
    report(f'{tag}: float64 replay -> process_out vs ComfyUI out_samples', ours * STD[None, :, None, None, None] + MEAN[None, :, None, None, None], out)
    comfy = (out - MEAN[None, :, None, None, None]) / STD[None, :, None, None, None]
    report(f'{tag}: float64 replay vs process_in(out_samples) (model space)', ours, comfy)
    gpu_path = os.path.join(here, '..', '..', '_scratch', 'anima-ref', f'gpu_replay_{tag}_final_f32.raw')
    if os.path.exists(gpu_path):
        gpu = np.fromfile(gpu_path, dtype=np.float32).astype(np.float64).reshape(ours.shape)
        report(f'{tag}: GPU replay (Anima_Lora_EightStepsMatchComfyUi) vs float64 replay', gpu, ours)
        report(f'{tag}: GPU replay vs ComfyUI', gpu, comfy)
