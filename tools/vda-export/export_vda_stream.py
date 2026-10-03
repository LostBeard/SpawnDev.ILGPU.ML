"""Export Video Depth Anything SMALL (Apache-2.0) in its STREAMING form as one ONNX step graph.

StreamStep(pixel_values [1,1,3,H,W] (normalized, H and W multiples of 14),
           cache_0..cache_7 [P_i, F, C_i]: F = 31 cached frames, or 0 for a clip's FIRST frame)  ->  (depth [1,H,W], new_cache_0..new_cache_7 [P_i, 1, C_i])

The 8 caches are the INPUT hidden states of the 2 temporal attention blocks of each of the head's 4 motion modules
(VDA's streaming mode: video_depth_stream.py). The caller keeps the window: VDA uses frames [0:2] (the first two, kept as
anchors) + the last 29, i.e. 31 cached frames, and on the first frame replicates its own cache 32 times.

usage: py -3.13 export_vda_stream.py <weights.pth> <out.onnx> [--check-frames N --video path]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# The Video-Depth-Anything checkout (github.com/DepthAnything/Video-Depth-Anything). VDA_REPO overrides the default,
# which is the workspace's research clone beside this repo.
REPO = os.environ.get('VDA_REPO') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '_research', 'Video-Depth-Anything')
sys.path.insert(0, os.path.abspath(REPO))
# The repo's utils/ has no __init__.py, so any installed top-level `utils` module shadows it: pin the package.
import types  # noqa: E402
_utils = types.ModuleType('utils')
_utils.__path__ = [os.path.join(os.path.abspath(REPO), 'utils')]
sys.modules['utils'] = _utils
from video_depth_anything.video_depth_stream import VideoDepthAnything, INFER_LEN  # noqa: E402
import vda_dynamic_patches  # noqa: E402

CACHE_FRAMES = INFER_LEN - 1   # 31


class StreamStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, c0, c1, c2, c3, c4, c5, c6, c7):
        feats = self.model.forward_features(x)
        depth, new = self.model.forward_depth(feats, x.shape, cached_hidden_state_list=[c0, c1, c2, c3, c4, c5, c6, c7])
        return (depth[0],) + tuple(new)


def load_model(weights):
    m = VideoDepthAnything(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    m.load_state_dict(torch.load(weights, map_location='cpu'), strict=True)
    return m.eval()


def first_frame_cache(model, x):
    """VDA's first-frame path: no cache; returns the frame's depth and its own 8 cache entries [P,1,C]."""
    with torch.no_grad():
        feats = model.forward_features(x)
        depth, cache = model.forward_depth(feats, x.shape)
    return depth[0], list(cache)


class Window:
    """VDA's streaming window bookkeeping (video_depth_stream.infer_video_depth_one), for 8 cache tensors."""

    def __init__(self, first_cache):
        self.frames = [first_cache] * INFER_LEN
        self.ids = [0] * INFER_LEN
        self.id = 0

    def inputs(self):
        cur = self.frames[0:2] + self.frames[-INFER_LEN + 3:]
        assert len(cur) == CACHE_FRAMES
        return [torch.cat([f[i] for f in cur], dim=1) for i in range(8)]

    def push(self, new_cache):
        self.id += 1
        self.frames.append(list(new_cache))
        gap = (INFER_LEN - 10) * 2 - 1 - (10 - 8)
        if self.id + INFER_LEN > gap + 1:
            del self.frames[1]


def preprocess(frame_bgr, h, w):
    import cv2
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_CUBIC)
    rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('weights')
    ap.add_argument('out')
    ap.add_argument('--h', type=int, default=252)
    ap.add_argument('--w', type=int, default=448)
    ap.add_argument('--check-frames', type=int, default=6)
    ap.add_argument('--video', default=None)
    ap.add_argument('--no-fold', action='store_true')
    ap.add_argument('--baked', action='store_true', help='export without the dynamic-shape patches')
    args = ap.parse_args()
    if not args.baked:
        vda_dynamic_patches.apply()   # dynamic H/W (see vda_dynamic_patches.py)

    model = load_model(args.weights)
    step = StreamStep(model).eval()
    h, w = args.h, args.w
    assert h % 14 == 0 and w % 14 == 0

    # Frames: a real clip if given, else noise.
    frames = []
    if args.video:
        import cv2
        cap = cv2.VideoCapture(args.video)
        while len(frames) < args.check_frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(preprocess(f, h, w))
    while len(frames) < args.check_frames:
        frames.append(torch.randn(1, 1, 3, h, w))

    # Torch reference: VDA's own streaming bookkeeping driving the step module.
    d0, c0 = first_frame_cache(model, frames[0])
    win = Window(c0)
    shapes = [tuple(t.shape) for t in c0]
    print('cache entry shapes (one frame):', shapes)
    ref_depths = [d0]
    step_inputs = []
    with torch.no_grad():
        for x in frames[1:]:
            caches = win.inputs()
            step_inputs.append((x, caches))
            out = step(x, *caches)
            ref_depths.append(out[0])
            win.push(out[1:])
    print('torch streaming depths:', [tuple(d.shape) for d in ref_depths], 'range', float(ref_depths[-1].min()), float(ref_depths[-1].max()))

    # Export with dynamic H/W (and the cache position dims that follow them).
    x_ex, caches_ex = step_inputs[0]
    names_in = ['pixel_values'] + [f'cache_{i}' for i in range(8)]
    names_out = ['depth'] + [f'new_cache_{i}' for i in range(8)]
    dyn = {'pixel_values': {3: 'H', 4: 'W'}, 'depth': {1: 'H', 2: 'W'}}
    for i in range(8):
        dyn[f'cache_{i}'] = {0: f'P{i}', 1: 'F'}   # F = cached frames: 31 in steady state, 0 for the FIRST frame
        dyn[f'new_cache_{i}'] = {0: f'P{i}'}
    torch.onnx.export(step, (x_ex, *caches_ex), args.out, input_names=names_in, output_names=names_out,
                      dynamic_axes=dyn, opset_version=17, do_constant_folding=not args.no_fold, dynamo=False)
    print('exported', args.out, os.path.getsize(args.out), 'bytes')

    # ONNX Runtime parity on every streamed frame (same cache inputs as the torch step).
    import onnxruntime as ort
    # ORT 1.24's graph optimizer breaks on this graph (a Reshape rewrite references a removed node); the graph itself is
    # valid (onnx.checker passes, and it runs unoptimized). ILGPU.ML has its own optimizer.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(args.out, so, providers=['CPUExecutionProvider'])
    worst = 0.0
    for k, (x, caches) in enumerate(step_inputs):
        feed = {'pixel_values': x.numpy()}
        for i in range(8):
            feed[f'cache_{i}'] = caches[i].numpy()
        res = sess.run(None, feed)
        ref = ref_depths[k + 1].numpy()
        rel = float(np.abs(res[0] - ref).max() / (np.abs(ref).max() + 1e-6))
        worst = max(worst, rel)
        print(f'frame {k + 1}: ORT vs torch max rel diff {rel:.2e}')
    print('WORST', f'{worst:.2e}')

    # First frame through the SAME graph: zero cached frames is VDA's no-cache first-frame path.
    x0 = frames[0]
    feed = {'pixel_values': x0.numpy()}
    for i in range(8):
        feed[f'cache_{i}'] = np.zeros((c0[i].shape[0], 0, c0[i].shape[2]), np.float32)
    res = sess.run(None, feed)
    rel = float(np.abs(res[0] - d0.numpy()).max() / (np.abs(d0.numpy()).max() + 1e-6))
    crel = max(float(np.abs(res[1 + i] - c0[i].numpy()).max() / (np.abs(c0[i].numpy()).max() + 1e-6)) for i in range(8))
    print(f'FIRST FRAME (F=0) vs torch no-cache: depth rel {rel:.2e}, caches rel {crel:.2e}')

    # Dynamic H/W: run a DIFFERENT size through the same ONNX against torch.
    for (h2, w2) in [(98, 168), (224, 392)]:
        xs = [torch.randn(1, 1, 3, h2, w2) for _ in range(3)]
        dd, cc = first_frame_cache(model, xs[0])
        win2 = Window(cc)
        worst2 = 0.0
        try:
            with torch.no_grad():
                for x in xs[1:]:
                    caches = win2.inputs()
                    out = step(x, *caches)
                    feed = {'pixel_values': x.numpy()}
                    for i in range(8):
                        feed[f'cache_{i}'] = caches[i].numpy()
                    res = sess.run(None, feed)
                    ref = out[0].numpy()
                    worst2 = max(worst2, float(np.abs(res[0] - ref).max() / (np.abs(ref).max() + 1e-6)))
                    win2.push(out[1:])
            print(f'DYNAMIC {h2}x{w2}: ORT vs torch worst rel {worst2:.2e}, depth {res[0].shape}')
        except Exception as e:
            print(f'DYNAMIC {h2}x{w2}: FAILED {type(e).__name__}: {str(e)[:300]}')


if __name__ == '__main__':
    main()
