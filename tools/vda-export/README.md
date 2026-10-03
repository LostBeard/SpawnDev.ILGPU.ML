# Video Depth Anything - streaming step export

Exports Video Depth Anything **Small** (Apache-2.0, ByteDance) as ONE ONNX graph for its streaming mode, the model that
`Pipelines/VideoDepthAnythingStream` (and `DepthEstimationPipeline`) drives:

```
pixel_values [1,1,3,H,W] (ImageNet-normalized, H and W multiples of 14)
cache_0..7   [P_i, F, C_i]   F = 31 cached frames, or 0 for a clip's FIRST frame
->
depth        [1,H,W]         relative DISPARITY (high = near)
new_cache_0..7 [P_i, 1, C_i]
```

Needs a checkout of https://github.com/DepthAnything/Video-Depth-Anything (`VDA_REPO`, default
`../../../_research/Video-Depth-Anything`) and the weights `video_depth_anything_vits.pth`
(HF `depth-anything/Video-Depth-Anything-Small`; fetch through hub.spawndev.com, never huggingface.co directly).

| script | does |
|---|---|
| `export_vda_stream.py <weights> <out.onnx> [--video clip]` | export (opset 17, dynamic H/W/F) + onnxruntime parity: streamed frames, the F=0 first frame, two other sizes |
| `vda_dynamic_patches.py` | the export-time patches that keep the graph shape-generic (see its docstring: 4 places the tracer bakes sizes) |
| `check_patches_equal.py` | patched vs ORIGINAL torch code, 4 sizes (7.6e-7) |
| `make_fixture.py` | one step as a `zipvoice-harness runonnx` fixture (ORT outputs as the reference) |
| `make_stream_reference.py` | N frames through VDA's own window bookkeeping, for `zipvoice-harness vdastream` |
| `compare_flicker.py` | offline PAIRED flicker comparison: DAv3, DAv3 + One Euro, VDA, VDA + One Euro on identical frames |

onnxruntime 1.24's graph optimizer fails on this graph; the scripts run it with `ORT_DISABLE_ALL` (the graph itself
is valid - `onnx.checker` passes).
