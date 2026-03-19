# SpawnDev.ILGPU.ML

[![NuGet](https://img.shields.io/nuget/v/SpawnDev.ILGPU.ML.svg?)](https://www.nuget.org/packages/SpawnDev.ILGPU.ML)

**Hardware-agnostic neural network inference for .NET — C# compute kernels that run on WebGPU, CUDA, OpenCL, WebGL, Wasm, and CPU via [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU).**

SpawnDev.ILGPU.ML implements neural network inference as native GPU compute kernels written entirely in C#. Models run as compute shaders transpiled from C# — no ONNX Runtime, no JavaScript, no native binaries. The same code runs in the browser (Blazor WebAssembly) and on desktop. Drop in a `.onnx` file and run it on any of six backends.

> **Active development.** API is stabilizing but may change. Contributions and feedback welcome.

## Highlights

- **71 ONNX operators** — classification, depth estimation, style transfer, super resolution, pose estimation, and more
- **Direct .onnx loading** — zero-dependency protobuf parser (~700 lines C#, no Google.Protobuf). Just point at a `.onnx` file.
- **6 backends from one codebase** — WebGPU, WebGL, Wasm, CUDA, OpenCL, CPU
- **Browser-verified** — SqueezeNet classifies a cat photo as "tiger cat" at 51.97% confidence, running entirely in the browser via WebGPU
- **10+ models compile** — SqueezeNet, MobileNetV2, 5 style transfer models, ESPCN super-resolution, Depth Anything V2, MoveNet Lightning
- **60/62 WebGPU tests passing**

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
| **Style Transfer** (5 models) | Artistic style transfer | 6-7 MB each | Runs, output pending InstanceNorm fix |
| **Depth Anything V2 Small** | Monocular depth estimation | 95 MB | Compiles (823 nodes, 25 op types) |
| **MoveNet Lightning** | Pose estimation (17 keypoints) | 9 MB | Compiles (21 op types) |

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

The demo is a Blazor WebAssembly app with interactive pages for each pipeline:

- **/classify** — Image classification with backend selector and race mode (**working**)
- **/super-res** — Image super resolution with before/after slider (**working**)
- **/style** — Neural style transfer with 6 style gallery (pending InstanceNorm fix)
- **/depth** — Monocular depth estimation with color palettes (pending full pipeline test)
- **/inspector** — Drop any `.onnx` file for instant architecture analysis and compatibility check (**working**)
- **/models** — Browse available demo models
- **/tests** — Run unit tests in-browser across WebGPU, WebGL, and Wasm

26 demo pages total, with shared components: BackendSelector, ConfidenceBars, BeforeAfterSlider, ImageDropZone, InferenceTimer, ModelLoadProgress, PrivacyBadge, and more.

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

## Known Issues

1. **InstanceNorm correctness** — Two-pass kernel runs (50,000x faster than original) but produces uniform output on WebGPU due to shared buffer race condition. Fix identified: switch to scalar kernel parameters.
2. **Style transfer** — Blocked by InstanceNorm. All other operators work correctly.

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
