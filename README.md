# SpawnDev.ILGPU.ML

[![NuGet](https://img.shields.io/nuget/v/SpawnDev.ILGPU.ML.svg?)](https://www.nuget.org/packages/SpawnDev.ILGPU.ML)

[**Live Demo**](https://lostbeard.github.io/SpawnDev.ILGPU.ML/) — Try image classification, style transfer, and super resolution in your browser right now.

**Hardware-agnostic neural network inference for .NET — C# compute kernels that run on WebGPU, CUDA, OpenCL, WebGL, Wasm, and CPU via [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU).**

SpawnDev.ILGPU.ML implements neural network inference as native GPU compute kernels written entirely in C#. Models run as compute shaders transpiled from C# — no ONNX Runtime, no JavaScript, no native binaries. The same code runs in the browser (Blazor WebAssembly) and on desktop. Drop in a model file — ONNX, TFLite, GGUF, or any of 7 supported formats — and run it on any of six backends.

> **Active development.** API is stabilizing but may change. Contributions and feedback welcome.

## Highlights

- **Neural style transfer runs in the browser** — 112-node pipeline, 5 styles, entirely on WebGPU. Turn your photo into a Van Gogh, Monet, or Picasso — no server, no upload, no cloud.
- **Image classification in-browser** — SqueezeNet identifies "tiger cat" at 51.97% confidence on WebGPU. Drop any photo.
- **Image super resolution** — ESPCN 3x upscale running on WebGPU. The "enhance!" button is real.
- **71 ONNX operators** — enough to run classification, style transfer, super resolution, depth estimation, pose estimation, and more
- **7 model formats** — ONNX, TFLite, GGUF, SafeTensors, TF GraphDef, PyTorch, and CoreML. Zero-dependency parsers for all. Load models from any ML ecosystem through one API: `CreateFromFileAsync()` auto-detects the format.
- **6 backends from one codebase** — WebGPU, WebGL, Wasm, CUDA, OpenCL, CPU
- **10+ models compile** — SqueezeNet, MobileNetV2, 5 style transfer models, ESPCN, Depth Anything V2 (823 nodes!), MoveNet Lightning
- **Model Inspector** — drop any model file (ONNX, TFLite, GGUF, SafeTensors, and more) for instant architecture analysis and compatibility check. No other browser ML library has this.

## Universal Model Loading

One API loads models from any ML ecosystem. Format is auto-detected from magic bytes — no configuration needed.

| Format | Ecosystem | What It Opens |
|--------|-----------|--------------|
| **ONNX** (.onnx) | PyTorch, ONNX Model Zoo | Industry standard. Most exported models. |
| **TFLite** (.tflite) | TensorFlow, MediaPipe, Google | Mobile/edge models. Face detection, pose, classification. |
| **GGUF** (.gguf) | llama.cpp, HuggingFace | Quantized LLMs. Llama, Mistral, Phi, SmolLM. |
| **SafeTensors** (.safetensors) | HuggingFace | Safe weight format. Nearly every HF model. |
| **TF GraphDef** (.pb) | TensorFlow 1.x/2.x | Frozen graphs, TF Hub models. |
| **PyTorch** (.pt/.pth) | PyTorch research | Weight extraction from checkpoints. |
| **Core ML** (.mlmodel) | Apple, iOS/macOS | Apple's Neural Engine models. |

```csharp
// All of these work — format detected automatically
var session = await InferenceSession.CreateFromFileAsync(accelerator, http, "model.onnx");
var session = await InferenceSession.CreateFromFileAsync(accelerator, http, "model.tflite");
var session = await InferenceSession.CreateFromFileAsync(accelerator, http, "model.gguf");
```

Every format produces the same `ModelGraph` intermediate representation. All 71 operators, all 30 GPU kernels, all 6 backends, and the full graph optimizer work identically regardless of source format. **Write one pipeline, load from any ecosystem.**

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

### Load and Run Any Model

```csharp
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;

// Create accelerator (auto-selects best backend)
var builder = MLContext.Create();
await builder.AllAcceleratorsAsync();
var context = builder.ToContext();
var accelerator = await context.CreatePreferredAcceleratorAsync();

// Load any model from any URL — format auto-detected from magic bytes
var session = await InferenceSession.CreateFromFileAsync(
    accelerator, httpClient, "models/squeezenet/model.onnx");
// Works with any URL and any supported format:
//   "models/blaze-face/model.tflite"                                    — local TFLite
//   "https://huggingface.co/org/repo/resolve/main/model.onnx"          — HuggingFace
//   "https://storage.googleapis.com/mediapipe-models/.../model.tflite"  — Google CDN

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
| **Environment** | Browser | Browser | Browser | Desktop | Desktop | Desktop |

Auto-selection: WebGPU > WebGL > Wasm (browser) or CUDA > OpenCL > CPU (desktop).

## Validated Models

| Model | Task | Size | Status |
|-------|------|------|--------|
| **SqueezeNet** | Classification (1000 classes) | 5 MB | **Working** — tiger cat 51.97% on WebGPU |
| **MobileNetV2** | Classification (1000 classes) | 13 MB | Compiles, graph runs |
| **ESPCN** | Super Resolution (3x) | 100 KB | **Working** on WebGPU |
| **Style Transfer** (5 models) | Artistic style transfer | 6-7 MB each | **Working** on WebGPU — 112 nodes, 3.9s inference |
| **Depth Anything V2 Small** | Monocular depth estimation | 95 MB | Compiles (823 nodes, 25 op types) |
| **MoveNet Lightning** | Pose estimation (17 keypoints) | 9 MB | Compiles (21 op types) |

Style models: mosaic, candy, rain princess, udnie, pointilism.

**TFLite models:**

| Model | Task | Size | Format |
|-------|------|------|--------|
| **BlazeFace** | Face detection | 229 KB | TFLite (MediaPipe) |
| **EfficientNet-Lite0** | Classification (1000 classes) | 17.7 MB | TFLite (MediaPipe) |
| **YOLOv8 Nano** | Object detection (80 classes) | 12.2 MB | ONNX |

## Architecture

### Multi-Format Inference Engine

```
Any model file (.onnx, .tflite, .gguf, .safetensors, .pb, .pt, .mlmodel)
    |
    v
Format auto-detection (magic bytes) → appropriate parser
    |
    v
ModelGraph (shared IR — nodes, weights, shapes)
    |
    v
GraphOptimizer (6 passes: constant fold, identity elim, linear fusion,
                scaled matmul fusion, strength reduction, dead node elim)
    |
    v
GraphCompiler (71 operators + fused ops → execution plan)
    |
    v
GraphExecutor (topological dispatch, buffer recycling, periodic flush)
    |
    v
InferenceSession (public API: CreateFromFileAsync / Run / RunAsync)
```

**Model loading** — one API, any format:
```csharp
// Auto-detect format from magic bytes
var session = await InferenceSession.CreateFromFileAsync(accelerator, http, "model.onnx");
var session = await InferenceSession.CreateFromFileAsync(accelerator, http, "model.tflite");
var session = await InferenceSession.CreateFromFileAsync(accelerator, http, "model.gguf");
```

Or use format-specific methods: `CreateFromOnnxAsync`, `CreateFromTFLiteAsync`, `CreateFromGGUFAsync`, `CreateAsync` (pre-extracted), `Create` (programmatic).

All formats produce the same `ModelGraph` IR — every operator, kernel, optimizer pass, and backend works identically regardless of source format.

### Graph Optimizer (automatic, 6 passes)

Every model is automatically optimized during compilation:

| Pass | What It Does | Impact |
|------|-------------|--------|
| **Constant folding** | Evaluates Shape→Gather→Cast→Floor chains at compile time | ~30% fewer nodes for style transfer |
| **Identity elimination** | Removes Identity/Dropout no-ops | 10 fewer nodes for SqueezeNet |
| **Linear fusion** | MatMul + Add + Activation → single FusedLinear dispatch | 2/3 less memory bandwidth |
| **Scaled MatMul fusion** | MatMul + Scale → FusedScaledMatMul | Attention optimization |
| **Strength reduction** | Div→Mul, eliminate Mul×1 and Add+0 | Cheaper operations |
| **Dead node elimination** | Removes orphaned nodes after fusion | Clean graph |

### GPU Kernels (30 files)

| Kernel | Description | Performance |
|--------|-------------|-------------|
| **MatMul** | Tiled 16x16 shared memory | 92-101 GFLOPS |
| **RegisterBlockedMatMul** | 4x4 register blocking, 64x64 tiles | Target: 200+ GFLOPS |
| **FusedLinear** | MatMul + Bias + Activation in 1 dispatch | 3x less memory bandwidth |
| **Conv2D / ConvTranspose2D** | Arbitrary kernel/stride/padding | — |
| **InstanceNorm** | Two-pass O(N) per (N,C) slice | 50,000x faster than naive |
| **LayerNorm / BatchNorm / RMSNorm** | All normalization variants | — |
| **Softmax** | Two-pass numerically stable | — |
| **Attention** | Multi-head split/score/merge | — |
| **GELU/ReLU/SiLU** | With in-place variants | — |
| **ImagePreprocess** | RGBA → NCHW, resize + normalize, Y-channel | GPU preprocessing |
| **ImagePostprocess** | NCHW float → packed RGBA on GPU | Zero-copy output |
| **DepthColormap** | Depth float → colored RGBA via GPU LUT | GPU visualization |
| **PostProcessing** | YOLO decode, NMS filter, cosine similarity, L2 norm | GPU postprocessing |
| **ColorConversion** | RGB↔YCbCr, grayscale, BGR on GPU | — |
| **ImageTransform** | GPU resize, crop, flip | — |
| **TensorLayout** | NCHW↔NHWC, interleaved↔planar on GPU | — |

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
| **Model Inspector** | Drop any model file (ONNX, TFLite, GGUF, SafeTensors...) for instant architecture analysis — node count, parameters, operators, compatibility check. | **Live** |

### Vision Demos

| Demo | What It Does |
|------|-------------|
| **Depth Estimation** | Generate depth maps from any photo. Selectable color palettes (plasma, viridis, inferno). Depth Anything V2 (823 nodes) already compiles. |
| **Real-Time Object Detection** | Live webcam with bounding boxes. 80 COCO classes, confidence slider, FPS counter. GPU-accelerated NMS. |
| **Background Removal** | One-click background removal. Transparent PNG download. Replace background with custom image or blur. |
| **Pose Estimation** | Live webcam with skeleton overlay. 17 keypoints, joint angles, movement trails. MoveNet Lightning already compiles. |
| **Face Detection** | Face detection with landmarks and confidence visualization. |
| **Zero-Shot (CLIP)** | Type ANY text description. Classify images by it. No fixed categories — the user defines what to look for. |

### Language & Audio Demos

| Demo | What It Does |
|------|-------------|
| **Speech to Text** | Whisper-powered transcription. Upload audio or use the microphone — transcription runs on your GPU, never leaves your device. |
| **Semantic Search** | Generate text embeddings. Find similar passages, rank by relevance — all computed locally. |
| **Text Generation** | GPT-style text generation with greedy/top-K/top-P sampling, temperature control, and tokens/second counter. |

### Experimental & Fun Demos

| Demo | What It Does | Why It's Special |
|------|-------------|-----------------|
| **AI Assistant** | Remember Clippy, Merlin, and Robby? They're back — but now they actually think. Choose from 6 classic MS Agent-style characters (Merlin, Robby, Clippy, Peedy, Rocky, Links), talk to them via voice or text, and they respond with AI-generated text and speech. All local. | Microsoft's animated assistants had personality but no intelligence. Now imagine them powered by a local LLM on your GPU with speech recognition (Whisper) and TTS — all in the browser, all private. |
| **Comic Chat AI** | A comic strip chat room where every character is an AI running locally. Add characters, give them personalities, and watch them converse in comic panel format. Inspired by Microsoft Comic Chat (1996), reimagined with local AI. | Multiple AI characters chatting with each other, each with distinct personality, rendered as a comic strip — all on your GPU. Pure nostalgia meets cutting-edge tech. |
| **Inside the Network** | Peek inside the neural network. See feature maps, attention patterns, and activation heatmaps as the model processes your image — layer by layer. Scrub through layers to see what the GPU "sees." | Educational and mesmerizing. Shows that neural networks aren't magic — they're math running on your GPU, and you can watch it happen. |
| **Draw to Train** | Draw 10-20 custom gestures → train a classifier entirely in the browser. | Most browser ML can only do inference. Training proves this is a complete GPU compute platform. |
| **Voice Collaboration** | Talk to your AI dev team. Speech-to-text (Whisper on GPU or Web Speech API) transcribes your voice, routes to AI agents with distinct personas, agents respond via text-to-speech. Full transcript with speaker labels. Hybrid: Claude API for reasoning now, local GGUF model on WebGPU later. | Multi-agent voice chat in a browser tab. Whisper STT running on your GPU, multiple AI agents with distinct voices, real-time conversation. The future of AI-assisted development — no install, no server (except LLM API). |

### Generative & 3D Demos

| Demo | What It Does |
|------|-------------|
| **Image Generation** | Stable Diffusion-style image generation. Prompt, negative prompt, steps, guidance scale, seed, resolution — all running on WebGPU. |
| **Image to 3D** | Generate 3D meshes, Gaussian splats, or point clouds from a single photo. Open directly in [SpawnScene](https://github.com/LostBeard/SpawnScene). |
| **Depth Voxel** | Live webcam depth → 3D point cloud visualization. ML inference feeding directly into 3D rendering, all on GPU, no CPU readback. |

### Infrastructure Demos

| Demo | What It Does |
|------|-------------|
| **Backend Showdown** | Run the same model on all available backends simultaneously. Leaderboard of inference times. Copy-paste shareable results. |
| **Model Inspector** | Drop any model file for instant architecture analysis and compatibility check. All 7 formats supported. |
| **Model Gallery** | Browse all available demo models. Load custom models from HuggingFace. |
| **Getting Started** | 5-step interactive tutorial with code examples. |

All demos include backend selection, inference timing, "100% client-side" privacy badges, keyboard shortcuts (`?` for help, `Space` = run, `D` = download), and the voice command system ("Computer, classify this image").

27 demo pages. Everything runs on YOUR GPU, in YOUR browser.

### The Wow Factor

These are the things that make people stop scrolling:

- **Backend Race Mode** — Run the same model on WebGPU, WebGL, and Wasm simultaneously. Live timing bars with medals. "Copy Results" formatted for social media. No other library can do this — this IS the differentiator.
- **"How Fast Is Your Device?"** — A dedicated benchmark page. MatMul throughput, model load time, inference speed. Like Cinebench for browser ML. Developers love posting benchmark scores.
- **Pipeline Composer** — Drag-and-drop model chaining: Image → Depth → Colorize → Download. Or: Webcam → Detect → Crop Faces → Classify Each. Shows this isn't just single models — it's a composable GPU pipeline.
- **Progressive Enhancement** — Start with Wasm (slow), switch to WebGL (faster), switch to WebGPU (fastest). Animated bars showing the speedup. Tells the story of "why WebGPU matters" in 10 seconds.
- **Offline Mode** — Toggle airplane mode. Inference still runs. "Your AI doesn't need the cloud."
- **Collaborative Canvas** — Multiple users on different devices, all running the same model, real-time via WebRTC (using SpawnDev.BlazorJS). Multi-device ML collaboration, all in-browser.
- **Model-to-Model Pipeline** — Photo → depth estimation → 3D point cloud → style transfer on the texture → render. Three ML models + 3D rendering, all on GPU, no server, one C# codebase. The ultimate SpawnDev ecosystem demo.
- **Real-Time Audio + Video Fusion** — Webcam (pose + face landmarks) + microphone (speech + emotion) simultaneously: "Person speaking with happy expression, arms raised." Multi-modal real-time inference from two input streams.
- **Screenshot Sharing** — One-click capture of demo result + timing as a shareable image card, pre-formatted for X/Twitter.

## Model Inspector

Drop any model file — **ONNX, TFLite, GGUF, SafeTensors, or any supported format** — and instantly see:
- Graph metadata (name, producer, opset version)
- Node count, parameter count, weight sizes
- Input/output tensor shapes and types
- Operator usage histogram
- Top 20 largest weights
- **Compatibility check** — green badge if SpawnDev.ILGPU.ML can run the model
- **GGUF models** — architecture info (layers, heads, context length, vocab size)

Format is auto-detected from magic bytes. All parsing happens in-browser with zero dependencies.

## Weight Loading

Weights are extracted automatically from any supported format:

| Format | Weight Types | Notes |
|--------|-------------|-------|
| **ONNX** | F32, F16 | Extracted from protobuf |
| **TFLite** | F32, F16, INT8, UINT8 | Auto-dequantized with quantization params |
| **GGUF** | F32, F16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1 | Block dequantization for quantized LLMs |
| **SafeTensors** | F32, F16, BF16, F64, I32, I16, I8, U8 | Zero-copy JSON header + raw data |
| **Pre-extracted FP16** | F16 → F32 | `weights_fp16.bin` + `manifest_fp16.json` (optimized web delivery) |

All weight types are converted to F32 on GPU upload. Pre-extracted FP16 uses 256-byte alignment for WebGPU buffer binding requirements.

## Blazor WebAssembly Configuration

Requires [SpawnDev.BlazorJS](https://github.com/LostBeard/SpawnDev.BlazorJS) for browser interop:

```xml
<PropertyGroup>
  <!-- ILGPU requires IL reflection at runtime -->
  <PublishTrimmed>false</PublishTrimmed>
  <RunAOTCompilation>false</RunAOTCompilation>
</PropertyGroup>
```

## Recent Breakthroughs

- **7 model format parsers** — ONNX, TFLite, GGUF, SafeTensors, TF GraphDef, PyTorch, CoreML. All zero-dependency, all auto-detected. One API loads any format.
- **6-pass graph optimizer** — constant folding, identity elimination, linear fusion, scaled MatMul fusion, strength reduction, dead node elimination. Automatically reduces node count by ~30% on style transfer models.
- **Fused linear kernel** — `MatMul + Bias + Activation` in a single GPU dispatch. Eliminates 2/3 of memory bandwidth for every linear layer in every model.
- **Zero-copy style transfer** — entire pipeline (preprocess → inference → postprocess) stays on GPU. No CPU pixel loops.
- **All 4 pipelines GPU-preprocessed** — Classification, StyleTransfer, SuperResolution, Depth all use GPU kernels for image preprocessing.
- **WGSL/GLSL codegen bugs fixed** — 4 codegen bugs found and fixed in SpawnDev.ILGPU. All 6 backends green: 1450 pass / 0 fail.
- **InstanceNorm 50,000x speedup** — Two-pass O(N) kernel. Style transfer went from infinite hang to 3.9 seconds.
- **Register-blocked MatMul** — 4x4 per thread, 64x64 tiles. Targeting 200+ GFLOPS (current tiled: 92-101).

## Testing

Tests run across all 6 backends via **PlaywrightMultiTest**:

```bash
# All tests (desktop + browser)
dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj
```

**SpawnDev.ILGPU: 1450 pass / 0 fail** across all 6 backends. Wasm backend: **179 pass / 0 fail / 55 skip** (fiber refactor complete — all RadixSort, scan, barrier, and sort tests pass).
**SpawnDev.ILGPU.ML: 78/78 WebGPU, 70/70 CUDA, 70/70 OpenCL.**

Every kernel validates against CPU reference implementations.

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

## Why this exists

This project was born out of 72 hours of "Architectural Vengeance" because the industry standard has a fundamental WebGPU device-sharing bug that has gone ignored for over 6 months:  

**See: microsoft/onnxruntime#26107**

