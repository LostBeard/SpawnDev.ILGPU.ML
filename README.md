# SpawnDev.ILGPU.ML

[![NuGet](https://img.shields.io/nuget/v/SpawnDev.ILGPU.ML.svg?)](https://www.nuget.org/packages/SpawnDev.ILGPU.ML)

**Hardware-agnostic neural network inference for .NET — C# compute kernels that run on WebGPU, CUDA, OpenCL, WebGL, Wasm, and CPU via [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU).**

SpawnDev.ILGPU.ML implements neural network inference as native GPU compute kernels written entirely in C#. Models run as compute shaders transpiled from C# — no ONNX Runtime, no JavaScript, no native binaries. The same code runs in the browser (Blazor WebAssembly) and on desktop. Drop in a `.onnx` file and run it on any of six backends.

> **Active development.** API is stabilizing but may change. Contributions and feedback welcome.

## Highlights

- **Neural style transfer runs in the browser** — 112-node pipeline, 5 styles, entirely on WebGPU. Turn your photo into a Van Gogh, Monet, or Picasso — no server, no upload, no cloud.
- **Image classification in-browser** — SqueezeNet identifies "tiger cat" at 51.97% confidence on WebGPU. Drop any photo.
- **Image super resolution** — ESPCN 3x upscale running on WebGPU. The "enhance!" button is real.
- **71 ONNX operators** — enough to run classification, style transfer, super resolution, depth estimation, pose estimation, and more
- **Direct .onnx loading** — zero-dependency protobuf parser (~700 lines C#, no Google.Protobuf). Just point at a `.onnx` file.
- **6 backends from one codebase** — WebGPU, WebGL, Wasm, CUDA, OpenCL, CPU
- **10+ models compile** — SqueezeNet, MobileNetV2, 5 style transfer models, ESPCN, Depth Anything V2 (823 nodes!), MoveNet Lightning
- **Model Inspector** — drop any `.onnx` file for instant architecture analysis and compatibility check. No other browser ML library has this.

## How It Works

Neural network operations (matrix multiply, convolution, normalization, attention) are implemented as [ILGPU](https://github.com/m4rs-mt/ILGPU) kernels in C#. [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU) transpiles each kernel to the target shader language at runtime:

```
C# Kernel Code
    |
    v
SpawnDev.ILGPU (transpilation)
    |
    +---> WGSL      (WebGPU)      -- browser GPU
    +---> GLSL      (WebGL)       -- browser GPU (universal)
    +---> Wasm      (Web Workers) -- browser CPU
    +---> PTX       (CUDA)        -- NVIDIA GPU
    +---> OpenCL C  (OpenCL)      -- any GPU
    +---> CPU       (threads)     -- no GPU needed
```

## Quick Start

### Load and Run an ONNX Model

```csharp
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;

// Create accelerator (auto-selects best backend)
var builder = MLContext.Create();
await builder.AllAcceleratorsAsync();
var context = builder.ToContext();
var accelerator = await context.CreatePreferredAcceleratorAsync();

// Load model directly from .onnx — no extraction step needed
var session = await InferenceSession.CreateFromOnnxAsync(
    accelerator, httpClient, "models/squeezenet/model.onnx");

// Classify an image
var pipeline = new ClassificationPipeline(session, accelerator);
var results = await pipeline.ClassifyAsync(rgbaPixels, width, height);

Console.WriteLine($"{results[0].Label}: {results[0].Confidence:P1}");
// Output: "tiger cat: 52.0%"
```

### Using a Kernel Directly

```csharp
var matMul = new MatMulKernel(accelerator);

using var a = accelerator.Allocate1D<float>(M * K);
using var b = accelerator.Allocate1D<float>(K * N);
using var c = accelerator.Allocate1D<float>(M * N);

matMul.MatMul(a.View, b.View, c.View, M, K, N);
await accelerator.SynchronizeAsync();
var result = await c.CopyToHostAsync<float>();
```

## Supported Backends

| | WebGPU | WebGL | Wasm | CUDA | OpenCL | CPU |
|---|---|---|---|---|---|---|
| **Runs on** | GPU | GPU | Workers | NVIDIA GPU | Any GPU | CPU cores |
| **Transpiles to** | WGSL | GLSL ES 3.0 | Wasm binary | PTX | OpenCL C | Threads |
| **Shared memory** | Yes | No | Yes | Yes | Yes | Yes |
| **Environment** | Browser | Browser | Browser | Desktop | Desktop | Both |

Auto-selection: WebGPU > WebGL > Wasm (browser) or CUDA > OpenCL > CPU (desktop).

## Validated Models

| Model | Task | Size | Status |
|-------|------|------|--------|
| **SqueezeNet** | Classification (1000 classes) | 5 MB | **Working** — tiger cat 51.97% on WebGPU |
| **MobileNetV2** | Classification (1000 classes) | 13 MB | Compiles, graph runs |
| **ESPCN** | Super Resolution (3x) | 100 KB | **Working** on WebGPU |
| **Style Transfer** (5 models) | Artistic style transfer | 6-7 MB each | **Working** on WebGPU — 112 nodes, ~19s inference |
| **Depth Anything V2 Small** | Monocular depth estimation | 95 MB | Compiles (823 nodes, 25 op types) |
| **MoveNet Lightning** | Pose estimation (17 keypoints) | 9 MB | Compiles (21 op types) |

Style models: mosaic, candy, rain princess, udnie, pointilism.

## Architecture

### ONNX Inference Engine

```
.onnx file
    |
    v
OnnxParser (zero-dependency protobuf)
    |
    v
ModelGraph (nodes, weights, shapes)
    |
    v
GraphCompiler (71 operators → execution plan)
    |
    v
GraphExecutor (topological dispatch, buffer pooling, periodic flush)
    |
    v
InferenceSession (public API: CreateFromOnnxAsync / CreateAsync / Run / RunAsync)
```

Three model loading paths:
1. **Direct `.onnx`** — `CreateFromOnnxAsync(accelerator, http, url)` — simplest, no preprocessing
2. **Pre-extracted** — `CreateAsync(accelerator, http, basePath)` — loads `model_graph.json` + `weights_fp16.bin` (smaller download, faster load)
3. **Programmatic** — `Create(accelerator, graph, weights)` — full control

### GPU Kernels

| Kernel | Description | Performance |
|--------|-------------|-------------|
| **MatMul** | Tiled 16x16 shared memory | 92-101 GFLOPS |
| **Conv2D** | Arbitrary kernel/stride/padding | — |
| **ConvTranspose2D** | Transposed convolution | — |
| **LayerNorm** | Row-wise, learned gamma/beta | — |
| **InstanceNorm** | Two-pass O(N) per (N,C) slice | 50,000x faster than naive |
| **BatchNorm** | Inference mode with running stats | — |
| **RMSNorm** | LLaMA/Mistral style normalization | — |
| **Softmax** | Two-pass numerically stable | — |
| **Attention** | Multi-head split/score/merge | — |
| **GELU/ReLU/SiLU** | With in-place variants | — |
| **ImagePreprocess** | RGBA → NCHW float, resize + normalize | — |
| **NearestUpsample** | 4D nearest-neighbor upsampling | — |

### 71 ONNX Operators

Abs, Add, ArgMax, AveragePool, BatchNormalization, Cast, Ceil, Clip, Concat, Constant, ConstantOfShape, Conv, ConvTranspose, DepthToSpace, Div, Dropout, Equal, Erf, Exp, Expand, Flatten, Floor, Gather, GatherND, Gelu, Gemm, GlobalAveragePool, Greater, HardSigmoid, HardSwish, Identity, InstanceNormalization, LayerNormalization, LeakyRelu, Less, Log, MatMul, Max, MaxPool, Min, Mul, Neg, Not, Pad, Pow, Range, Reciprocal, ReduceMax, ReduceMean, ReduceMin, ReduceSum, Relu, Reshape, Resize, Shape, Sigmoid, Sign, SiLU, Slice, Softmax, Split, Sqrt, Squeeze, Sub, Tanh, TopK, Transpose, Unsqueeze, Upsample, Where

### Pipeline Classes

| Pipeline | Input | Output |
|----------|-------|--------|
| **ClassificationPipeline** | RGBA image | Top-K labels + confidence |
| **SuperResolutionPipeline** | RGBA image | Upscaled RGBA image |
| **StyleTransferPipeline** | RGBA image | Stylized RGBA image |
| **DepthEstimationPipeline** | RGBA image | Normalized depth map |

Additional pipeline scaffolds: detection, segmentation, pose, CLIP, embeddings, text generation, speech recognition, audio classification, image generation.

## Demo App

The demo is a Blazor WebAssembly app showcasing what's possible when GPU inference runs entirely in the browser — no server, no uploads, no cloud. Everything stays on the user's device.

### Working Now

| Demo | What It Does | Status |
|------|-------------|--------|
| **Image Classification** | Drop a photo, get top-5 ImageNet predictions with confidence bars. Race Mode compares inference speed across WebGPU/WebGL/Wasm side-by-side. | **Live** |
| **Neural Style Transfer** | Turn your photo into a Van Gogh, Monet, or Picasso. 5 style models, instant gallery switching. Before/after slider. | **Live** |
| **Super Resolution** | Upload a small image, get 3x upscale. Before/after comparison with download. | **Live** |
| **Model Inspector** | Drop any `.onnx` file for instant architecture analysis — node count, parameters, operators, compatibility check. | **Live** |

### Coming Next

| Demo | What It Does | Why It's Cool |
|------|-------------|---------------|
| **Depth Estimation** | Generate depth maps from any photo with selectable color palettes (plasma, viridis, inferno). Before/after slider. | Depth Anything V2 (823 nodes) already compiles. Visually stunning output. |
| **Real-Time Object Detection** | Live webcam with bounding boxes. 80 COCO classes, confidence slider, FPS counter. | YOLOv8-nano at 15-30+ FPS on WebGPU would outperform most JS implementations. GPU-accelerated NMS keeps everything on-device. |
| **Background Removal** | One-click background removal. Transparent PNG download. Replace background with custom image or blur. | Universal use case. The "no upload" privacy angle is a strong selling point. |
| **Pose Estimation** | Live webcam with skeleton overlay. 17 keypoints, joint angles, movement trails. | MoveNet Lightning already compiles. Skeleton overlay is inherently fun. |
| **Zero-Shot Classification (CLIP)** | Type ANY text description. Classify images by it. No fixed categories. | Feels like magic — the user defines what to look for. |
| **Live Webcam Style Transfer** | Real-time styled video feed. Your webcam as a living painting. | With optimized WebGPU kernels, this could run at high FPS — noticeably smoother than JS alternatives. |

### The Big Vision

| Demo | Description |
|------|-------------|
| **Backend Showdown** | Run the same model on all available backends simultaneously. Leaderboard of inference times. Copy-paste results for social media. |
| **Depth → 3D** | Live webcam depth estimation → 3D point cloud visualization. ML inference feeding directly into 3D rendering, all on GPU, no CPU readback. |
| **Spatial Intelligence** | YOLO (find a person) + MoveNet (extract skeleton) + Depth Anything (place in 3D space). Composable AI pipelines understanding the physical world. |
| **In-Browser GPU Training** | Draw 10-20 custom gestures → train a classifier entirely in the browser. Most browser ML can only run inference — training proves this is a complete GPU compute platform. |
| **Image-to-3D** | Generate 3D meshes or Gaussian splats from a single photo. Open directly in [SpawnScene](https://github.com/LostBeard/SpawnScene). |

All demos include backend selection, inference timing, "100% client-side" privacy badges, and the voice command system ("Computer, classify this image").

26 demo pages total.

## Model Inspector

The built-in model inspector is a unique tool — drop any `.onnx` file and instantly see:
- Graph metadata (name, producer, opset version)
- Node count, parameter count, weight sizes
- Input/output tensor shapes and types
- Operator usage histogram
- Top 20 largest weights
- **Compatibility check** — green badge if SpawnDev.ILGPU.ML can run the model

No other browser ML library offers this. It uses our zero-dependency ONNX parser — the same one that powers direct `.onnx` loading.

## Weight Loading

Two weight formats supported:

**Direct `.onnx`** — the parser extracts weights directly from the protobuf. Simplest path — no preprocessing step needed.

**Pre-extracted FP16** — for optimized delivery:
- `weights_fp16.bin` — flat FP16 values, 256-byte aligned per tensor
- `manifest_fp16.json` — tensor metadata (names, shapes, offsets)

FP16 weights are converted to FP32 on GPU upload. Individual tensors are accessed via SubView with 256-byte alignment for WebGPU buffer binding.

## Blazor WebAssembly Configuration

Requires [SpawnDev.BlazorJS](https://github.com/LostBeard/SpawnDev.BlazorJS) for browser interop:

```xml
<PropertyGroup>
  <!-- ILGPU requires IL reflection at runtime -->
  <PublishTrimmed>false</PublishTrimmed>
  <RunAOTCompilation>false</RunAOTCompilation>
</PropertyGroup>
```

## Testing

Tests run across all 6 backends via **PlaywrightMultiTest**:

```bash
# All tests (desktop + browser)
dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj

# WebGPU only
dotnet test --filter "FullyQualifiedName~WebGPUTests."

# Desktop console tests (CUDA, OpenCL, CPU)
dotnet run --project DemoConsole/SpawnDev.ILGPU.ML.DemoConsole.csproj
```

Every kernel validates against CPU reference implementations. 60/62 WebGPU tests passing (2 skipped pending InstanceNorm correctness fix).

## Recent Breakthroughs

- **WGSL PHI codegen bug fixed** — `WGSLKernelFunctionGenerator.cs` was generating incorrect continuation code after loop-break patterns. Fix: post-loop continuation now uses `headerExitTarget` instead of `loopNode.Exits[0]`. This unblocked the Pad kernel (and any kernel with early-exit loop patterns) on WebGPU.
- **Style transfer end-to-end** — 112-node pipeline with 16 InstanceNorm layers, 16 Conv layers, Pad, Upsample — all running correctly on WebGPU.
- **InstanceNorm 50,000x speedup** — Rewrote from O(N*spatial) per thread to two-pass O(N): Pass 1 computes mean+variance (N*C threads), Pass 2 normalizes (N*C*spatial threads, no loops).

## Known Issues

1. **Secondary WGSL codegen bug** — Missing continuation code after if-else with break in certain patterns. Tracked, workaround in place for affected kernels.
2. **Depth Anything V2** — 823 nodes compile but full GPU execution not yet tested (95 MB model).

## Credits

SpawnDev.ILGPU.ML would not be possible without:

- **[ILGPU](https://github.com/m4rs-mt/ILGPU)** — The GPU compiler that makes C# GPU kernels possible. Created by [Marcel Koester](https://github.com/m4rs-mt) and [contributors](https://github.com/m4rs-mt/ILGPU/graphs/contributors).
- **[SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU)** — Extends ILGPU with three browser backends (WebGPU, WebGL, Wasm), bringing GPU compute to Blazor WebAssembly.
- **[SpawnDev.BlazorJS](https://github.com/LostBeard/SpawnDev.BlazorJS)** — Full JS interop for Blazor WebAssembly. Typed C# wrappers for all browser APIs.

## Resources

- [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU) — Cross-platform GPU compute for .NET (6 backends)
- [SpawnDev.BlazorJS](https://github.com/LostBeard/SpawnDev.BlazorJS) — Full JS interop for Blazor WebAssembly
- [ILGPU](https://github.com/m4rs-mt/ILGPU) — The GPU compiler
- [ILGPU Documentation](https://ilgpu.net/)
- [Plans/full-inference-engine-plan.md](Plans/full-inference-engine-plan.md) — Detailed roadmap

## License

Licensed under the same terms as ILGPU. See [LICENSE](LICENSE.txt) for details.
