"""Torch reference for a whole VDA stream: N consecutive frames through VDA's own window bookkeeping.

Writes <out>.pixels.f32 ([N,1,1,3,H,W] normalized inputs), <out>.depth.f32 ([N,H,W] torch depths) and <out>.json.
N > 41 exercises the anchor that slides once 40 frames have followed frame 0.

usage: py -3.13 make_stream_reference.py <weights.pth> <video> <h> <w> <frames> <out-basename>
"""
import json
import sys

import numpy as np
import torch

import export_vda_stream as e
import vda_dynamic_patches


def main():
    weights, video, h, w, n, out = sys.argv[1:7]
    h, w, n = int(h), int(w), int(n)
    vda_dynamic_patches.apply()
    m = e.load_model(weights)
    step = e.StreamStep(m).eval()
    import cv2
    cap = cv2.VideoCapture(video)
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        assert ok, f'clip has only {len(frames)} frames'
        frames.append(e.preprocess(f, h, w))
    d0, c0 = e.first_frame_cache(m, frames[0])
    win = e.Window(c0)
    depths = [d0]
    with torch.no_grad():
        for x in frames[1:]:
            o = step(x, *win.inputs())
            depths.append(o[0])
            win.push(o[1:])
    np.stack([f.numpy() for f in frames]).astype(np.float32).tofile(out + '.pixels.f32')
    np.stack([d.numpy() for d in depths]).astype(np.float32).tofile(out + '.depth.f32')
    json.dump({'frames': n, 'h': h, 'w': w}, open(out + '.json', 'w'))
    print('wrote', out, n, 'frames', h, 'x', w)


if __name__ == '__main__':
    main()
