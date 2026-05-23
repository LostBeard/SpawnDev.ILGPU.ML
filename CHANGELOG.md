# SpawnDev.ILGPU.ML Changelog

Notable changes per release. Pre-stable; API will change between preview drops.

## 4.0.0-preview.1 (2026-05-23) — first nuget.org preview

First public preview. SpawnDev.ILGPU.ML provides native GPU neural-network inference
for .NET — C# compute kernels transpiled to WebGPU, WebGL, Wasm, CUDA, OpenCL, and CPU
via [SpawnDev.ILGPU](https://www.nuget.org/packages/SpawnDev.ILGPU). No ONNX Runtime, no
JavaScript bridge, no native binaries.

This is a **preview**: API is stabilizing but will change. Not yet recommended for
production. Ship feedback as GitHub issues — bugs that surface in your model usually
become regression tests in our PMT suite.

### What works today

- **6-backend coverage** — same kernels run on WebGPU, WebGL, Wasm, CUDA, OpenCL, CPU.
- **Universal model loading** — `InferenceSession.CreateFromFileAsync()` auto-detects
  ONNX / TFLite / GGUF / SafeTensors / TF GraphDef / PyTorch / Core ML from magic bytes.
- **16 inference pipelines** — Classification, StyleTransfer, SuperResolution,
  DepthEstimation, ObjectDetection, PoseEstimation, FaceDetection, TextClassification,
  ZeroShotClassification (CLIP), BackgroundRemoval, SpeechRecognition (Whisper),
  TextGeneration, FeatureExtraction, Diffusion (DDPM), TextToSpeech (SpeechT5),
  Image3D (TripoSR).
- **Verified demo pipelines for this preview**: depth estimation (Depth Anything V2),
  style transfer, image classification (SqueezeNet). Other pipelines build and run
  but may have rough edges. See the [live demo](https://lostbeard.github.io/SpawnDev.ILGPU.ML/).
- **Zero-copy GPU pipeline** — `ImagePreprocessKernel` → inference → `ImagePostprocessKernel`
  (incl. plasma colormap, bilinear resize for depth/masks) → `CanvasRendererFactory`.
  Data enters the GPU at pre-processing and stays until the pixel lands on the canvas.
- **Aspect-aware depth pipeline** — `DepthEstimationPipeline.EstimateAsync` /
  `EstimateGpuAsync` accept optional `outputWidth = 0, outputHeight = 0`. Default
  matches source dimensions (preserves source aspect); explicit values fit one
  axis (derives the other from aspect) or set exact size. Resize runs on the
  accelerator via bilinear interpolation.
- **HuggingFace CDN integration** via `ModelHub` with OPFS caching in the browser.
- **Streaming weight loader** for large models (GPT-2 652MB single-tensor-at-a-time).
- **30 GPU kernel files** — MatMul (tiled 16×16 shared mem, ~92-101 GFLOPS validated),
  Conv2D, FWHT, TurboQuant KV cache compression, RoPE, QKNorm, GroupNorm, SelectiveScan
  (Mamba-3), MarchingCubes, SpatialMemoryUnit, and more.

### Known rough edges

- This is a preview — pipelines other than the three named above are not all verified
  end-to-end across every backend.
- Some operators have backend-specific limitations on WebGL (no shared memory / atomics
  / barriers). `AcceleratorRequirements` in the underlying SpawnDev.ILGPU lets consumers
  declare requirements and have an incapable backend filtered out at selection time.
- Memory pressure on Wasm is higher than other backends due to the 2 GiB heap ceiling.
  Large models (GPT-2 scale) may exceed this on Wasm; prefer WebGPU for those.

### Dependencies

- `SpawnDev.ILGPU` 4.9.7 — the underlying transpiler and runtime.
- `SpawnDev.WebTorrent` 2.3.1 — for the optional P2P model delivery code path.
  (Will track WebTorrent 3.x in a follow-up preview once interop is verified.)
- `Microsoft.AspNetCore.Components.Web` 10.0.4 — for the demo's Blazor surface.

### Credits

The SpawnDev Crew:

- **LostBeard** (Todd Tanner) — captain, library author, vision.
- **Data** — operations officer, ML library lead.
- **Geordi** — chief engineer, ILGPU internals and GPU kernels.
- **Riker** — first officer, WebRTC / WebTorrent / BlazorJS plumbing.
- **Tuvok** — security/research officer, codecs, design review.
