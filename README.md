<div align="center">

# SpawnDev.ILGPU.ML

**Neural network inference and training in C#, on any GPU, in the browser or on the desktop.**

Compute kernels written in C# run as WGSL, GLSL, PTX, OpenCL C, WebAssembly or SIMD CPU code through [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU).
No ONNX Runtime. No native inference binaries. No server.

[![NuGet](https://img.shields.io/nuget/v/SpawnDev.ILGPU.ML.svg?label=nuget&logo=nuget)](https://www.nuget.org/packages/SpawnDev.ILGPU.ML)
[![Downloads](https://img.shields.io/nuget/dt/SpawnDev.ILGPU.ML.svg?label=downloads)](https://www.nuget.org/packages/SpawnDev.ILGPU.ML)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.txt)
[![.NET 10](https://img.shields.io/badge/.NET-10-512bd4.svg)](https://dotnet.microsoft.com/)
[![Live demo](https://img.shields.io/badge/live%20demo-GitHub%20Pages-2ea44f.svg)](https://lostbeard.github.io/SpawnDev.ILGPU.ML/)
[![Sponsor](https://img.shields.io/badge/sponsor-LostBeard-ea4aaa.svg?logo=githubsponsors)](https://github.com/sponsors/LostBeard)

[**Live demo**](https://lostbeard.github.io/SpawnDev.ILGPU.ML/) | [Docs](Docs/README.md) | [Examples](Examples/README.md) | [Changelog](CHANGELOG.md) | [What is verified](Docs/DEMO_AND_MODEL_STATUS.md)

<img src="https://raw.githubusercontent.com/LostBeard/SpawnDev.ILGPU.ML/master/SpawnDev.ILGPU.ML.Demo/wwwroot/screenshots/2026-10-21_TextTo3D-Rig-Animate-Astronaut-FLUX3-1024.gif" alt="A text prompt becomes a rigged, animated 3D astronaut, entirely in the browser" width="720">

<sub>One prompt, one browser tab, no server: Qwen3 writes the caption, FLUX 3 Image Klein draws it, TRELLIS.2 builds the textured mesh, SkinTokens rigs it, Kimodo clips animate it. All of it runs on the visitor's GPU.</sub>

</div>

---

## What it is

SpawnDev.ILGPU.ML is an inference and training engine for .NET whose every layer, from matrix multiply to a 5.6B-parameter image model, is a C# kernel that [ILGPU](https://github.com/m4rs-mt/ILGPU) compiles for the device in front of it. The same library, the same kernels and the same models run:

- **in the browser** (Blazor WebAssembly) on WebGPU, WebGL or WebAssembly, with nothing leaving the user's machine, and
- **on the desktop** (Windows, Linux, macOS) on CUDA, OpenCL or a SIMD CPU backend.

You hand it a model file. It detects the format, builds a graph, optimizes it and runs it on whatever accelerator you give it.

## Highlights

- **One codebase, six backends.** WebGPU, WebGL, WebAssembly, CUDA, OpenCL and CPU (the Velocity SIMD backend by default on 64-bit desktop). Kernels are tested on all of them.
- **Seven model formats, auto-detected.** ONNX, TFLite, GGUF, SafeTensors, TensorFlow GraphDef, PyTorch checkpoints and Core ML, all lowered to one graph IR.
- **200+ ONNX operators**, a graph optimizer (constant folding, fusion, strength reduction, dead-node elimination) and fused kernels for linear layers, attention, quantized matmul and more.
- **Generative models, ported and checked.** FLUX 3 Image Klein, FLUX.2 klein, SD-Turbo, TRELLIS.2 (text and image to textured 3D), SkinTokens and UniRig (auto-rigging), Kimodo (character animation). Stages are gated against the reference implementations.
- **Language and speech.** Quantized GGUF LLMs with KV-cache decode and TurboQuant cache compression, Whisper speech-to-text, Kokoro and ZipVoice text-to-speech with voice cloning.
- **Training, not only inference.** Backpropagation, SGD and Adam as GPU kernels, so a small network can be trained in a browser tab.
- **Zero-copy by design.** Pixels, activations and weights stay on the GPU. In the browser, bulk data streams from `fetch` and OPFS straight into GPU buffers without touching the .NET heap.
- **Trim and AOT safe.** The trim analyzer runs on every build of the library.

## Try it

The [live demo](https://lostbeard.github.io/SpawnDev.ILGPU.ML/) is a Blazor WebAssembly app. Everything runs on your GPU; models download once from [`models.spawndev.com`](https://models.spawndev.com) and are cached in your browser.

| Demo | What it does | Model |
|------|--------------|-------|
| [Text to 3D](https://lostbeard.github.io/SpawnDev.ILGPU.ML/text-to-3d) / [Image to 3D](https://lostbeard.github.io/SpawnDev.ILGPU.ML/image-to-3d) | A prompt or a photo becomes a textured PBR `.glb`; rig it, animate it with 22 preset moves | FLUX 3 / FLUX.2 / SD-Turbo, RMBG-1.4, TRELLIS.2, SkinTokens, Kimodo |
| [Image generation](https://lostbeard.github.io/SpawnDev.ILGPU.ML/generate) | Text to image, and edits of up to 10 reference pictures | FLUX 3 Image Klein, FLUX.2 klein, SD-Turbo |
| [Video generation](https://lostbeard.github.io/SpawnDev.ILGPU.ML/video) | Text to video, or animate a picture (the first frame); about 13 GB of weights on first use | Wan 2.2 TI2V-5B |
| [On-device chat](https://lostbeard.github.io/SpawnDev.ILGPU.ML/ai-chat) | Pick a `.gguf` and chat with it | Qwen, Gemma, Llama and others |
| [Speech to text](https://lostbeard.github.io/SpawnDev.ILGPU.ML/whisper) | Microphone to text | Whisper |
| [Text to speech](https://lostbeard.github.io/SpawnDev.ILGPU.ML/tts) | Built-in voices, or clone a voice from a short clip | Kokoro, ZipVoice |
| [Depth](https://lostbeard.github.io/SpawnDev.ILGPU.ML/depth), [style transfer](https://lostbeard.github.io/SpawnDev.ILGPU.ML/style), [background removal](https://lostbeard.github.io/SpawnDev.ILGPU.ML/remove-bg), [super-resolution](https://lostbeard.github.io/SpawnDev.ILGPU.ML/super-res), [pose](https://lostbeard.github.io/SpawnDev.ILGPU.ML/pose), [CLIP](https://lostbeard.github.io/SpawnDev.ILGPU.ML/clip), [classification](https://lostbeard.github.io/SpawnDev.ILGPU.ML/classify) | Vision pipelines, with a race mode that runs one model on every backend side by side | Depth Anything V2, ESPCN, RMBG, MoveNet, CLIP, SqueezeNet |
| [Model inspector](https://lostbeard.github.io/SpawnDev.ILGPU.ML/inspector) | Drop any model file (or paste a URL) for its architecture and an operator compatibility check, streaming only the header | any supported format |
| [Benchmark](https://lostbeard.github.io/SpawnDev.ILGPU.ML/benchmark), [train](https://lostbeard.github.io/SpawnDev.ILGPU.ML/train), [snake](https://lostbeard.github.io/SpawnDev.ILGPU.ML/snake) | Throughput on your device, on-device training, a small decision-head demo | n/a |

Not every page is finished. [**Docs/DEMO_AND_MODEL_STATUS.md**](Docs/DEMO_AND_MODEL_STATUS.md) lists each demo and model as verified, partial or work in progress, with the test that proves it. If a statement here disagrees with that file, that file wins.

## Install

```bash
dotnet add package SpawnDev.ILGPU.ML
```

Requires **.NET 10**. Targets the browser (Blazor WebAssembly), Windows, Linux and macOS. Depends on [SpawnDev.ILGPU](https://www.nuget.org/packages/SpawnDev.ILGPU) and [SpawnDev.SpawnJS](https://www.nuget.org/packages/SpawnDev.SpawnJS).

## Quick start

### 1. Create an accelerator

The application owns the accelerator. Library code never disposes it.

```csharp
// Desktop: the best device available (CUDA, then OpenCL, then CPU)
using var context = MLContext.CreateContext();
using var accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);
```

```csharp
// Blazor WebAssembly: WebGPU, falling back to WebGL, then Wasm
using SpawnDev.ILGPU;   // AllAcceleratorsAsync, CreatePreferredAcceleratorAsync

var builder = MLContext.Create();
await builder.AllAcceleratorsAsync();
var context = builder.ToContext();
var accelerator = await context.CreatePreferredAcceleratorAsync();
```

### 2. Run any model

The format (ONNX, TFLite, GGUF, SafeTensors, ...) is detected from the file's magic bytes.

```csharp
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Tensors;

using var session = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes("squeezenet.onnx"));

// Inputs are named tensors; outputs come back as a disposable map of GPU-resident tensors you own.
using var input = OwnedTensor<float>.FromHost(accelerator, pixels, new[] { 1, 3, 224, 224 });
using var outputs = await session.RunOwnedAsync(new Dictionary<string, Tensor<float>>
{
    [session.InputNames[0]] = input,
});

float[] logits = await outputs[session.OutputNames[0]].ToHostAsync();
```

Large models should not be read into a `byte[]`. `InferenceSession.CreateFromStreamAsync(accelerator, stream)` takes any seekable stream (a file, an OPFS entry, a ranged HTTP stream) and sends ONNX and GGUF weights to the GPU tensor by tensor. Pipelines such as `ClassificationPipeline`, `DepthEstimationPipeline`, `SpeechRecognitionPipeline` (Whisper) and `TextGenerationPipeline` wrap a session with the pre- and post-processing for a task.

### 3. Generate images

`ImageGenerator` puts FLUX 3 Image Klein, FLUX.2 klein and SD-Turbo behind one call. Weights stream from the models host on first use and are cached on disk (OPFS in the browser).

```csharp
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;

using var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
var source = new HttpClientModelSource(http, FileModelStore.Default());

using var gen = new ImageGenerator(accelerator, source, (repo, file, progress) => source.OpenAsync(repo, file, progress));
await gen.LoadAsync(ImageEngine.Flux2);

var pictures = await gen.GenerateAsync(new ImageGenerationRequest
{
    Prompt = "a red fox in fresh snow",
    Engine = ImageEngine.Flux2,
    Size = 512,
    Seed = 1,
});
// pictures[0].Rgba is width x height x 4 bytes
```

### 4. Prompt to rigged, animated character

`AssetGenerator` chains the whole 3D pipeline on a desktop GPU: draw, cut out, build the mesh, rig, animate.

```csharp
using SpawnDev.ILGPU.ML.Assets;

var (motions, ids) = AssetGenerator.MotionsFromDirectory("motions");
using var gen = new AssetGenerator(accelerator, source, motions);

var pictures = await gen.DrawAsync("a knight in plate armor, full body", new[] { 11, 12, 13, 14 });   // FLUX.2 klein
var (cutOut, _) = await gen.CutOutAsync(pictures[0]);                                                  // RMBG-1.4
var glb    = await gen.BuildModelAsync(cutOut, pictures[0].Width, pictures[0].Height);                 // TRELLIS.2
var rigged = await gen.RigAsync(glb);                                                                  // SkinTokens
var moves  = await gen.AnimateAsync(rigged, new[] { "walk", "dance", "wave" });                        // Kimodo
```

More: [Docs/assetgen.md](Docs/assetgen.md), [Docs/scenes.md](Docs/scenes.md) (a whole scene from one sentence).

## What you can run

| Area | What is in the box |
|------|--------------------|
| **Vision** | Classification, object detection (YOLOv8), face detection (BlazeFace), pose (MoveNet), depth (Depth Anything V2, multi-view and streaming variants), style transfer, super-resolution, background removal (RMBG-1.4, IS-Net), zero-shot classification (CLIP) |
| **Image generation** | FLUX 3 Image Klein 5.6B (text to image, edits with up to 10 references, Qwen3 caption writer), FLUX.2 klein 4B, SD-Turbo, DDPM |
| **3D** | TRELLIS.2 text and image to textured PBR mesh (`Trellis2ImageTo3D`, `Trellis2TextTo3D`), mesh repair, remeshing, simplification, UV atlas and texture baking on the device; glTF, OBJ, PLY and SPZ import and export |
| **Rigging and animation** | `SkinTokensRigger` (learned skeleton and skin weights), `UniRigRigger`, `HumanoidRetargeter` and `AnimatedGlb` (Kimodo clips on any rigged humanoid) |
| **Language** | GGUF LLM inference (Llama, Qwen, Gemma, Mistral, Phi, LFM2, gpt-oss and more) with KV-cache decode and TurboQuant cache compression, sentiment (DistilBERT), embeddings (MiniLM), text generation, multimodal Gemma 4 |
| **Speech** | Whisper and Parakeet speech-to-text, Silero voice-activity detection, smart-turn detection, Kokoro and ZipVoice text-to-speech with voice cloning |
| **Training** | `TrainableModel` with softmax cross-entropy, Conv2D, ReLU, max-pool and linear backward passes, SGD and Adam; `SystemOneDecisionHead` for small typed decision models |
| **Tooling** | Model inspector (header-only streaming), op-compatibility reports, per-node timing, an Ollama-compatible GGUF server (Example 06) |

Pipelines, models and their verification status are tracked in [Docs/DEMO_AND_MODEL_STATUS.md](Docs/DEMO_AND_MODEL_STATUS.md). "The loader reads format X" and "model Y runs end to end" are different claims, and that file keeps them apart.

## Model formats

One API loads a model from any ecosystem. Every loader produces the same `ModelGraph`, so each operator, kernel, optimizer pass and backend works the same whatever the source.

| Format | Ecosystem | Notes |
|--------|-----------|-------|
| **ONNX** (`.onnx`) | PyTorch, ONNX Model Zoo | Streaming parser, external data, 200+ operators |
| **TFLite** (`.tflite`) | TensorFlow, MediaPipe | NHWC graphs, fused activations, INT8 and UINT8 dequantized on load |
| **GGUF** (`.gguf`) | llama.cpp, Ollama | Architecture-aware graph builder; fused GPU dequantizing matmul for Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q6_K and MXFP4, plus F16, BF16 and F32 weights. Other quantization types (for example Q5_K or Q2_K) are refused with a clear error rather than decoded on the CPU: re-quantize the model |
| **SafeTensors** (`.safetensors`) | Hugging Face | Config-driven builders; also the weight format of the generative models in this repo |
| **TF GraphDef** (`.pb`) | TensorFlow | Protobuf parser with op mapping |
| **PyTorch** (`.pt`, `.pth`) | PyTorch | Checkpoint weight extraction |
| **Core ML** (`.mlmodel`) | Apple | Neural-network layer mapping |

The supported ONNX operator set is the single source of truth `OperatorRegistry.BuiltinOpTypes`, described in [Docs/operators.md](Docs/operators.md), including which operators are complete and which are partial.

## Backends

| Backend | Runs on | Compiles C# kernels to | Where |
|---------|---------|------------------------|-------|
| **WebGPU** | GPU | WGSL | Browser |
| **WebGL** | GPU | GLSL ES 3.0 | Browser |
| **Wasm** | Web Workers | WebAssembly | Browser |
| **CUDA** | NVIDIA GPU | PTX | Desktop |
| **OpenCL** | Any OpenCL GPU | OpenCL C | Desktop |
| **CPU** | CPU cores | Velocity SIMD (default on 64-bit) or the plain interpreting CPU accelerator | Desktop |

`MLContext.CreateCpuAccelerator(context)` returns the Velocity accelerator where the context has one and the plain CPU accelerator otherwise. `MLContext.CreatePlainCpuContext()` gives the slow, exact reference accelerator that tests compare against.

## How it works

```
  model file (.onnx .tflite .gguf .safetensors .pb .pt .mlmodel)
        |  magic-byte detection
        v
  format parser ---> ModelGraph (nodes, weights, shapes)
        |
        v
  GraphOptimizer     constant folding, identity elimination, linear / attention /
        |            layer-norm fusion, strength reduction, dead-node elimination
        v
  GraphCompiler ---> execution plan (200+ operators and fused ops)
        |
        v
  GraphExecutor      topological dispatch, buffer recycling, batched submission
        |
        v
  C# kernels  --- SpawnDev.ILGPU --->  WGSL | GLSL | Wasm | PTX | OpenCL C | SIMD CPU
```

- **Kernels** live in [`SpawnDev.ILGPU.ML/Kernels`](SpawnDev.ILGPU.ML/Kernels): tiled and register-blocked matmul, fused dequantizing GEMV for quantized LLMs, flash-style attention, convolution, normalization variants, RoPE, selective scan (Mamba), gated delta nets, mixture of experts, FWHT and TurboQuant, image pre- and post-processing, and the training backward passes.
- **Decode fast paths.** Autoregressive decode records its dispatch plan once and replays it per token (WebGPU command replay, CUDA graphs), so the host does almost nothing per token.
- **Tensors** are three small types: `Tensor<T>` (a shape-tracked view), `OwnedTensor<T>` (owns its GPU buffer) and `TensorView<T>` (a blittable struct kernels take directly).
- **Model delivery** is plain HTTP from `models.spawndev.com`, a static host with free egress and range requests, with Hugging Face as the automatic fallback. `HubModelSource` caches in OPFS in the browser; `HttpClientModelSource` with `FileModelStore` does it on disk with resumable downloads. The optional [`SpawnDev.ILGPU.ML.WebTorrent`](SpawnDev.ILGPU.ML.WebTorrent) package adds peer-to-peer delivery behind the same `IModelSource` interface.

## Performance

Measured on an RTX 4070 12 GB, models cached. Full methodology and the other numbers are in the linked docs.

| Task | Where | Result | Source |
|------|-------|--------|--------|
| Photo to textured 3D mesh (TRELLIS.2, 512 pipeline) | CUDA | about 40 s | [trellis2.md](Docs/trellis2.md) |
| Photo to textured 3D mesh | OpenCL | about 83 s | [trellis2.md](Docs/trellis2.md) |
| Photo to textured 3D mesh | Chrome, WebGPU, AOT build | 134 to 148 s | [trellis2.md](Docs/trellis2.md) |
| Prompt to textured 3D mesh | Chrome, WebGPU, AOT build | about 180 s | [trellis2.md](Docs/trellis2.md) |
| Photo to 1M-triangle mesh, 4096 texture (TRELLIS.2 1024 cascade) | CUDA | 201 s (the reference on the same card: 461 s) | [trellis2.md](Docs/trellis2.md) |
| The same 1024 cascade | Chrome, WebGPU | 1403 s | [trellis2.md](Docs/trellis2.md) |
| Prompt to rigged, animated character, end to end | CUDA | 77 s | [assetgen.md](Docs/assetgen.md) |
| FLUX 3 Image Klein, 512 px, 16 steps | CUDA | 26.1 s (4-bit), 32.8 s (8-bit) | [flux3.md](Docs/flux3.md) |
| Auto-rig a character (SkinTokens) | CUDA | about 6 s | [rigging.md](Docs/rigging.md) |

Correctness comes first: kernels are checked against CPU references, vision and language models against ONNX Runtime outputs, and the generative models stage by stage against their reference PyTorch implementations (for example SkinTokens: 295 of 296 generated tokens identical, skin weights within 5.7e-5).

## Examples

Self-contained projects under [`Examples/`](Examples/README.md), each runnable with `dotnet run --project Examples/<name>`.

| # | Example | Shows |
|---|---------|-------|
| 02 | [Model inspector](Examples/02.ModelInspector.Console) | Header-only inspection of a local file, URL or Ollama model |
| 03 | [Image generation](Examples/03.ImageGen.Console) | FLUX 3, FLUX.2 and SD-Turbo, reference images, interactive session |
| 04 | [GGUF text generation](Examples/04.GGUFTextGen.Console) | KV-cache decode with per-token performance probes |
| 05 | [Gemma 4 multimodal](Examples/05.Gemma4Multimodal.Console) | Vision plus text (verification pending) |
| 06 | [Ollama-compatible server](Examples/06.OllamaServer.Console) | OpenAI, Ollama and Anthropic Messages APIs over your local GGUF models (work in progress) |
| 07 | [Asset generation](Examples/07.AssetGen.Console) | Prompt to rigged, animated 3D character |
| 08 | [Asset generation (WPF)](Examples/08.AssetGen.Wpf) | The same, as a Windows app with a 3D preview |
| 09 | [Scene generation](Examples/09.SceneGen.Console) | A scene from one sentence: LLM layout, then one generated asset per object |
| 10 | [Image generation (WPF)](Examples/10.ImageGen.Wpf) | The image page as a Windows app |
| 11 | [Video generation](Examples/11.VideoGen.Console) | Wan 2.2 TI2V-5B text or image to video, PNG frames and an animated WebP ([Docs/video.md](Docs/video.md)) |
| 12 | [Video generation (WPF)](Examples/12.VideoGen.Wpf) | The same as a Windows app with a looping player |

## Repository layout

| Path | Contents |
|------|----------|
| `SpawnDev.ILGPU.ML/` | The library (NuGet package `SpawnDev.ILGPU.ML`) |
| `SpawnDev.ILGPU.ML.WebTorrent/` | Optional peer-to-peer model delivery (separate package) |
| `SpawnDev.Phonemizer/` | Dependency-free English grapheme-to-phoneme for text-to-speech, MIT |
| `SpawnDev.ILGPU.ML.Demo/` | The Blazor WebAssembly demo app |
| `SpawnDev.ILGPU.ML.Demo.Shared/` | Shared demo code and the test suite |
| `SpawnDev.ILGPU.ML.DemoConsole/` | Desktop host for the shared test suite and console tools |
| `SpawnDev.ILGPU.ML.Tests/` | xUnit tests for kernels, operators and sessions |
| `Examples/` | Console and WPF examples |
| `Docs/` | Topic documentation |
| `PlaywrightMultiTest/` | The test runner (desktop and browser, every backend) |
| `tools/` | Reference harnesses, weight exporters, capture and audit scripts |

## Documentation

| | |
|---|---|
| [Demo and model status](Docs/DEMO_AND_MODEL_STATUS.md) | What is verified, partial or work in progress, with evidence |
| [Operators](Docs/operators.md) | ONNX operator support and implementation tiers |
| [TRELLIS.2](Docs/trellis2.md) | Text and image to 3D: pipeline, API, browser design, numbers |
| [FLUX 3 Image Klein](Docs/flux3.md) / [FLUX.2 klein](Docs/flux2.md) | Image generation and reference editing |
| [Rigging and animation](Docs/rigging.md) | SkinTokens, UniRig, Kimodo retargeting |
| [Asset generation](Docs/assetgen.md) / [Scenes](Docs/scenes.md) | Prompt to character, sentence to scene |
| [System One](Docs/system-one.md) | Typed decision heads |
| [Changelog](CHANGELOG.md) | Every release, in detail |

## Build and test

```bash
dotnet build SpawnDev.ILGPU.ML.slnx
```

Tests run through **PlaywrightMultiTest**, which drives the desktop backends and a real Chromium for the browser backends from one command:

```bash
dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj -c Release
```

Scope a run with environment variables, for example `PMT_LANES=cuda,webgpu` for the fast backends or `PMT_FILTER=Trellis2_` for one area. Heavy real-model tests are excluded by default and are opt-in by category.

The demo is published with `dotnet publish SpawnDev.ILGPU.ML.Demo -c Release -p:DemoAot=true` (AOT keeps the CPU-bound stages of the 3D pipeline fast in the browser).

## Model licenses

The code in this repository is MIT. Model weights keep their own licenses, which travel with the files on the models host and are listed in each model's doc. Notably, **FLUX 3 Image Klein weights are under the FLUX Kommunity License (non-commercial)**, FLUX.2 klein is Apache-2.0, TRELLIS.2, SkinTokens and UniRig are MIT, and the bundled Kimodo motion clips carry the terms in [`SpawnDev.ILGPU.ML.Demo/wwwroot/motions/NOTICE.md`](SpawnDev.ILGPU.ML.Demo/wwwroot/motions/NOTICE.md). Check them before commercial use.

## Contributing

Issues and pull requests are welcome. A bug in this library, in SpawnDev.ILGPU or in SpawnDev.SpawnJS is fixed in that library, not worked around in a consumer, so a good bug report is a failing test or a minimal repro on a named backend. Run the tests for the lanes you touched before you open a pull request.

## Support the project

SpawnDev is built by one developer and a small team of AI agents. If it is useful to you, [sponsoring on GitHub](https://github.com/sponsors/LostBeard) directly funds more kernels, more verified models and more backends. Stars, issues and sharing what you build with it help just as much.

## Credits

- [**ILGPU**](https://github.com/m4rs-mt/ILGPU) by Marcel Koester and contributors, the compiler that makes C# GPU kernels possible.
- [**SpawnDev.ILGPU**](https://github.com/LostBeard/SpawnDev.ILGPU), which adds the WebGPU, WebGL and Wasm backends and the Velocity CPU backend.
- [**SpawnDev.SpawnJS**](https://github.com/LostBeard/SpawnDev.SpawnJS), typed C# access to the browser's APIs.
- The authors and licensors of the models ported here: Black Forest Labs (FLUX), Microsoft (TRELLIS.2), VAST AI Research (SkinTokens, UniRig), NVIDIA (Kimodo), OpenAI (Whisper), Stability AI (SD-Turbo), Alibaba (Qwen), and the many others named in the docs.

Created by Todd Tanner ([@LostBeard](https://github.com/LostBeard)) together with a crew of Claude AI agents: Riker, Data, Tuvok, Geordi and Seven. Live long and prosper. 🖖

## License

[MIT](LICENSE.txt). SpawnDev.ILGPU.ML includes code derived from ILGPU under the University of Illinois Open Source License; see [LICENSE.txt](LICENSE.txt) for the full text.
