"""Offline, PAIRED temporal-stability comparison on identical frames: DAv3-Small (onnxruntime), DAv3 + Anaglyphohol's
One Euro filter, and Video Depth Anything Small streaming (torch, VDA's own window).

Metric = Anaglyphohol's flicker probe: mean |disparity_t - disparity_t-1| over STATIC pixels (every RGB channel of the
model-resolution frame changed by <= 2 levels), disparity normalized to [0,1] by the frame's min/max. Also reported
over all pixels, and with the min/max range smoothed the way Anaglyphohol does (grow 0.5, shrink 0.05).

usage: py -3.13 compare_flicker.py <video> <h> <w> <frames> [--dav3 <model.onnx>] [--vda <weights.pth>]
"""
import argparse
import math

import cv2
import numpy as np


def load_frames(video, n, h, w):
    cap = cv2.VideoCapture(video)
    rgb, norm = [], []
    while len(rgb) < n:
        ok, f = cap.read()
        if not ok:
            break
        x = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
        rgb.append(np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8))
        norm.append(((x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)).transpose(2, 0, 1))
    return rgb, norm


def dav3_disparity(model, norm):
    import onnxruntime as ort
    sess = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
    out = []
    for x in norm:
        d = sess.run(['predicted_depth'], {'pixel_values': x[None, None].astype(np.float32)})[0][0, 0]
        out.append(1.0 / np.maximum(d, 1e-6))   # depth -> disparity
    return out


def vda_disparity(weights, norm):
    import torch
    import export_vda_stream as e
    import vda_dynamic_patches
    vda_dynamic_patches.apply()
    m = e.load_model(weights)
    step = e.StreamStep(m).eval()
    xs = [torch.from_numpy(x)[None, None] for x in norm]
    d0, c0 = e.first_frame_cache(m, xs[0])
    win = e.Window(c0)
    out = [d0[0].numpy()]
    with torch.no_grad():
        for x in xs[1:]:
            o = step(x, *win.inputs())
            out.append(o[0][0].numpy())
            win.push(o[1:])
    return out


def one_euro(disp_unit, min_cutoff=0.015, beta=0.02, d_cutoff=0.1):
    """Anaglyphohol's TemporalDepthKernels.OneEuroKernel on u = 1 + 100 * disparity(0..1)."""
    def alpha(c):
        tau = 1.0 / (2 * math.pi * c)
        return 1.0 / (1.0 + tau)
    out, x_hat, dx_hat = [], None, None
    a_d = alpha(d_cutoff)
    for d in disp_unit:
        u = 1.0 + 100.0 * d
        if x_hat is None:
            x_hat, dx_hat = u.copy(), np.zeros_like(u)
        else:
            dx = u - x_hat
            dx_hat = dx_hat + a_d * (dx - dx_hat)
            cutoff = min_cutoff + beta * np.abs(dx_hat)
            tau = 1.0 / (2 * np.pi * cutoff)
            a = 1.0 / (1.0 + tau)
            x_hat = x_hat + a * (u - x_hat)
        out.append((x_hat - 1.0) / 100.0)
    return out


def normalize(disps, smooth):
    out, lo, hi = [], None, None
    for d in disps:
        mn, mx = float(d.min()), float(d.max())
        if smooth:
            if lo is None:
                lo, hi = mn, mx
            else:
                # Anaglyphohol SmoothRangeKernel: widen fast (0.5), narrow slowly (0.05).
                lo = lo + (0.5 if mn < lo else 0.05) * (mn - lo)
                hi = hi + (0.5 if mx > hi else 0.05) * (mx - hi)
        else:
            lo, hi = mn, mx
        out.append(np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1))
    return out


def flicker(unit, rgb):
    allv, statv = [], []
    for t in range(1, len(unit)):
        diff = np.abs(unit[t] - unit[t - 1])
        static = (np.abs(rgb[t].astype(np.int16) - rgb[t - 1].astype(np.int16)) <= 2).all(axis=2)
        allv.append(diff.mean())
        if static.any():
            statv.append(diff[static].mean())
    return np.array(allv), np.array(statv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video'); ap.add_argument('h', type=int); ap.add_argument('w', type=int); ap.add_argument('frames', type=int)
    ap.add_argument('--dav3', default=r'D:\users\tj\Projects\Anaglyphohol\Anaglyphohol\Anaglyphohol\wwwroot\models\depth-anything-v3-small\onnx\model.onnx')
    ap.add_argument('--vda', default=r'..\..\..\_research\vda-weights\video_depth_anything_vits.pth')
    a = ap.parse_args()
    rgb, norm = load_frames(a.video, a.frames, a.h, a.w)
    print(f'{len(rgb)} frames at {a.w}x{a.h}')
    arms = {}
    dav3 = dav3_disparity(a.dav3, norm)
    vda = vda_disparity(a.vda, norm)
    for smooth in (False, True):
        tag = 'smoothed range' if smooth else 'per-frame range'
        d_unit = normalize(dav3, smooth)
        v_unit = normalize(vda, smooth)
        arms = {
            'DAv3 raw': d_unit,
            'DAv3 + One Euro': one_euro(d_unit),
            'VDA raw': v_unit,
            'VDA + One Euro': one_euro(v_unit),
        }
        print(f'--- {tag} (mean |d disparity| per frame, x1000) ---')
        base = None
        for name, u in arms.items():
            al, st = flicker(u, rgb)
            med, p90 = np.median(st), np.percentile(st, 90)
            base = base or med
            print(f'  {name:16s} static median {med*1000:7.3f}  p90 {p90*1000:7.3f}  mean {st.mean()*1000:7.3f} | all median {np.median(al)*1000:7.3f}   (x{med/base:.2f} of DAv3 raw)')


if __name__ == '__main__':
    main()
