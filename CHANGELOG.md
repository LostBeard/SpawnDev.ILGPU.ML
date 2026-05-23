# SpawnDev.ILGPU.ML Changelog

Notable changes per release. Pre-stable; API will change between preview drops.

## 4.0.0-preview.2 (2026-05-23) — dep bump + DepthEstimationPipeline aspect-ratio fix + /remove-bg demo cleanup

### Fixes
- **DepthEstimationPipeline.EstimateAsync / EstimateGpuAsync** gained optional
  `outputWidth = 0, outputHeight = 0`. Default `(0, 0)` matches source image dimensions
  (preserves aspect — previously the depth result came back at the model's square
  input size and visibly squished against the source). `(w, 0)` and `(0, h)` fit
  one axis and derive the other from source aspect; `(w, h)` is exact. Resize runs
  on the accelerator via a new `ImagePostprocessKernel.ResizeBilinear` — no CPU
  readback of the raw depth tensor.
- **CoreML/ONNX format detection false positive** closed. `CoreMLParser.IsCoreML`
  now refuses when the next protobuf tag after `specificationVersion` is `0x3A`
  (the ONNX graph field-7 tag). Without this guard, ONNX models with no producer
  string (e.g. SqueezeNet 1.1) were misclassified as CoreML and routed through
  the CoreML placeholder graph — surfaced as `Inference failed: KeyNotFoundWithKey, output`
  in the `/classify` demo.
- **/remove-bg demo backend selector** now honors the dropdown (was always
  WebGPU regardless of pick), and the **Transparent / White / Blur buttons** now
  actually composite the result on mode change.

### Dependency bumps
- `SpawnDev.ILGPU` 4.9.7 → **4.9.8** — WebGL device probe no longer leaks an
  `OffscreenCanvas` + `WebGL2RenderingContext` per registration. This is what
  caused Chrome's "too many WebGL contexts" warning in v4.0.0-preview.1 even
  when the WebGL backend was never selected.

### Known rough edges (unchanged from preview.1)

- RMBG-1.4 (`/remove-bg`) on WebGPU is sluggish during load + compile and the
  output mask is still being investigated. WebGL works for this model. Other
  pipelines (depth, classify, style) are smooth on WebGPU.
- Wasm has a tighter memory ceiling than other backends; large models may exceed
  it on Wasm.

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
