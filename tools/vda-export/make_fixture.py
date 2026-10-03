"""runonnx fixture for the VDA stream step: a real streamed step (frame k of a clip, with VDA's own cache window)
and onnxruntime's outputs for it.

usage: py -3.13 make_fixture.py <weights.pth> <model.onnx> <video> <h> <w> <step> <out.json>
"""
import json
import sys

import numpy as np
import torch

import export_vda_stream as e
import vda_dynamic_patches


def main():
    weights, model, video, h, w, step_k, out = sys.argv[1:8]
    h, w, step_k = int(h), int(w), int(step_k)
    vda_dynamic_patches.apply()
    if '--kv' in sys.argv:
        e.KV = True
        vda_dynamic_patches.apply_kv()
    m = e.load_model(weights)
    step = e.StreamStep(m).eval()
    import cv2
    cap = cv2.VideoCapture(video)
    frames = []
    while len(frames) <= step_k:
        ok, f = cap.read()
        assert ok
        frames.append(e.preprocess(f, h, w))
    _, c = e.first_frame_cache(m, frames[0])
    win = e.Window(c)
    with torch.no_grad():
        for x in frames[1:step_k]:
            win.push(step(x, *win.inputs())[1:])
    x = frames[step_k]
    # step 0 = a clip's FIRST frame: zero cached frames (VDA's no-cache path).
    caches = win.inputs() if step_k > 0 else [torch.zeros(t.shape[0], 0, t.shape[2]) for t in c]
    nc = e.n_caches()

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model, so, providers=['CPUExecutionProvider'])
    feed = {'pixel_values': x.numpy()}
    for i in range(nc):
        feed[f'cache_{i}'] = caches[i].numpy()
    names = [o.name for o in sess.get_outputs()]
    res = dict(zip(names, sess.run(None, feed)))

    def t(a):
        a = np.asarray(a, np.float32)
        return {'shape': list(a.shape), 'data': [float(v) for v in a.ravel()]}
    fx = {'inputs': {k: t(v) for k, v in feed.items()}, 'outputs': {k: t(v) for k, v in res.items()}}
    with open(out, 'w') as f:
        json.dump(fx, f)
    np.savez(out.replace('.json', '.npz'), **feed, **{'out_' + k: v for k, v in res.items()})
    print('wrote', out, {k: v.shape for k, v in res.items()})


if __name__ == '__main__':
    main()
