"""Export-time patches that keep Video Depth Anything's graph shape-generic (dynamic H/W).

The tracer bakes every Python-int shape computation into a constant. VDA has four:

1. DINOv2.interpolate_pos_encoding: bicubic resize of the 37x37 position grid to the patch grid with a FLOAT
   scale_factor ((n + 0.1) / 37) - a Python float, so the output size and the sampling positions are constants.
   Replaced by the same bicubic resampling written as two in-graph weight matrices built from the patch counts
   (Range/Floor/Equal/MatMul - standard ops). It reproduces torch's upsample_bicubic2d exactly: align_corners=False,
   the given scale used for the source coordinate, A = -0.75, edge-clamped taps, no antialias.
2. The motion module's einops rearranges (shapes computed in Python): replaced by permute/reshape on traced sizes.
3. DPTHeadTemporal.forward: `int(patch_h * 14)` - the int() detaches the size from the graph. Dropped (the value
   is already an int; under tracing it stays a traced size).
4. Tensor.unflatten(0, (B, T)) in the head and in forward_depth: torch's ONNX symbolic bakes the other dims.
"""
import inspect
import math
import textwrap

import torch
import torch.nn as nn

A = -0.75


def _cubic1(x):
    return ((A + 2) * x - (A + 3)) * x * x + 1


def _cubic2(x):
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A


def bicubic_matrix(out_n, in_n, inv_scale):
    """[out_n, in_n] weights so that M @ v == torch bicubic resize of v along one axis.

    inv_scale = 1 / scale_factor (torch casts 1.0 / scale to float, then source = inv * (dst + 0.5) - 0.5)."""
    dt = inv_scale.dtype
    i = torch.arange(out_n, dtype=dt)
    real = inv_scale * (i + 0.5) - 0.5
    i0 = torch.floor(real)
    t = real - i0
    w = [_cubic2(t + 1), _cubic1(t), _cubic1(1 - t), _cubic2(2 - t)]
    cols = torch.arange(in_n, dtype=dt).unsqueeze(0)
    m = None
    for k in range(4):
        idx = torch.clamp(i0 + (k - 1), 0, in_n - 1).unsqueeze(1)
        term = w[k].unsqueeze(1) * (idx == cols).to(dt)
        m = term if m is None else m + term
    return m


def interpolate_pos_encoding(self, x, w, h):
    # NOTE: DINOv2 names dim -2 'w' and dim -1 'h'; kept as-is (dim -2 is the image height).
    previous_dtype = x.dtype
    side = math.isqrt(int(self.pos_embed.shape[1]) - 1)   # a parameter's shape: a true constant (37)
    pos_embed = self.pos_embed.float()
    class_pos_embed = pos_embed[:, 0]
    patch = pos_embed[:, 1:].reshape(1, side, side, -1).permute(0, 3, 1, 2)       # [1, dim, side, side]
    n0 = w // self.patch_size
    n1 = h // self.patch_size
    inv0 = side / (torch.as_tensor(n0, dtype=torch.float32) + self.interpolate_offset)
    inv1 = side / (torch.as_tensor(n1, dtype=torch.float32) + self.interpolate_offset)
    m0 = bicubic_matrix(n0, side, inv0)                                             # [n0, side]
    m1 = bicubic_matrix(n1, side, inv1)                                               # [n1, side]
    out = torch.matmul(torch.matmul(m0, patch), m1.transpose(0, 1))                # [1, dim, n0, n1]
    dim = x.shape[-1]
    out = out.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), out), dim=1).to(previous_dtype)


def rearrange(x, pattern, **axes):
    """The four einops patterns the motion module uses, as permute/reshape on traced sizes (einops computes the
    target shapes in Python, which the tracer bakes)."""
    if pattern == "b c f h w -> (b f) c h w":
        return x.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[1], x.shape[3], x.shape[4])
    if pattern == "(b f) c h w -> b c f h w":
        return x.reshape(-1, axes['f'], x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
    if pattern == "(b f) d c -> (b d) f c":
        f = axes['f']
        return x.reshape(-1, f, x.shape[1], x.shape[2]).permute(0, 2, 1, 3).reshape(-1, f, x.shape[2])
    if pattern == "(b d) f c -> (b f) d c":
        d = axes['d']
        return x.reshape(-1, d, x.shape[1], x.shape[2]).permute(0, 2, 1, 3).reshape(-1, d, x.shape[2])
    raise NotImplementedError(pattern)


def apply():
    from video_depth_anything import dinov2, dpt_temporal
    from video_depth_anything.motion_module import motion_module
    motion_module.rearrange = rearrange
    dinov2.DinoVisionTransformer.interpolate_pos_encoding = interpolate_pos_encoding

    _rewrite(dpt_temporal, dpt_temporal.DPTHeadTemporal, 'forward', [
        ('(int(patch_h * 14), int(patch_w * 14))', '(patch_h * 14, patch_w * 14)', 2),
    ] + [(f'{v}.unflatten(0, (B, T))', f'{v}.reshape(_split0({v}, B, T))', 1)
         for v in ('layer_3', 'layer_4', 'path_4', 'path_3')])
    from video_depth_anything import video_depth_stream
    _rewrite(video_depth_stream, video_depth_stream.VideoDepthAnything, 'forward_depth', [
        ('depth.squeeze(1).unflatten(0, (B, T))', 'depth.squeeze(1).reshape(_split0(depth.squeeze(1), B, T))', 1),
    ])


def _split0(like, b, t):
    """Shape for splitting dim 0 of a [B*T, ...] tensor into (B, T), from TRACED sizes. torch's ONNX symbolic for
    unflatten writes the example's other dims in as constants, which freezes every later shape."""
    return [b, t] + [like.shape[i] for i in range(1, like.dim())]


def _rewrite(module, cls, name, edits):
    """Re-compiles cls.name from its source with exact text edits (each must occur the stated number of times)."""
    src = textwrap.dedent(inspect.getsource(getattr(cls, name)))
    for old, new, count in edits:
        assert src.count(old) == count, (name, old, src.count(old))
        src = src.replace(old, new)
    ns = dict(module.__dict__)
    ns['_split0'] = _split0
    exec(compile(src, module.__file__, 'exec'), ns)
    setattr(cls, name, ns[name])


def self_test():
    """The matrix resampler against torch's own bicubic, at the sizes Anaglyphohol uses."""
    import torch.nn.functional as F
    torch.manual_seed(0)
    g = torch.randn(1, 8, 37, 37)
    worst = 0.0
    for n0, n1 in [(7, 12), (12, 21), (18, 32), (36, 32), (37, 37), (40, 72)]:
        ref = F.interpolate(g, scale_factor=((n0 + 0.1) / 37, (n1 + 0.1) / 37), mode='bicubic', antialias=False)
        assert ref.shape[-2:] == (n0, n1), ref.shape
        m0 = bicubic_matrix(n0, 37, 37 / (torch.tensor(float(n0)) + 0.1))
        m1 = bicubic_matrix(n1, 37, 37 / (torch.tensor(float(n1)) + 0.1))
        mine = m0 @ g @ m1.T
        worst = max(worst, float((mine - ref).abs().max()))
    print('bicubic matrix vs torch, worst abs diff', worst)
    return worst


if __name__ == '__main__':
    self_test()
