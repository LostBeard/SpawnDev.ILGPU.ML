# SpawnDev.ILGPU.ML

[![NuGet](https://img.shields.io/nuget/v/SpawnDev.ILGPU.ML.svg?)](https://www.nuget.org/packages/SpawnDev.ILGPU.ML)

**Hardware-agnostic neural network inference for .NET — C# compute kernels that run on WebGPU, CUDA, OpenCL, WebGL, Wasm, and CPU via [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU).**

SpawnDev.ILGPU.ML implements neural network operations as native GPU compute kernels written entirely in C#. Models run as compute shaders transpiled from C# — no ONNX Runtime dependency, no JavaScript, no native binaries. The same inference code runs in the browser (Blazor WebAssembly) and on desktop.

> **Early development.** This library is under active development and the API is not yet stable. Contributions and feedback are welcome.

## How It Works

Neural network operations (matrix multiply, convolution, normalization, activation functions, attention) are implemented as [ILGPU](https://github.com/m4rs-mt/ILGPU) kernels in C#. [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU) transpiles these kernels to the appropriate shader language or binary for each backend at runtime. The result is native GPU inference without any external runtime dependencies.

```
C# Kernel Code
    |
    v
SpawnDev.ILGPU (transpilation)
    |
    +---> WGSL    (WebGPU)
    +---> GLSL    (WebGL)
    +---> Wasm    (Web Workers)
    +---> PTX     (CUDA)
    +---> OpenCL C (OpenCL)
    +---> CPU     (multi-threaded)
```

## Supported Backends

**Browser (Blazor WebAssembly)**

| | WebGPU | WebGL | Wasm |
|---|---|---|---|
| **Runs on** | GPU | GPU | Web Workers |
| **Transpiles to** | WGSL | GLSL ES 3.0 | Wasm binary |
| **Shared memory** | Yes | No | Yes |
| **Best for** | GPU inference (modern browsers) | GPU inference (universal) | CPU inference |

**Desktop (Console, WPF, ASP.NET)**

| | CUDA | OpenCL | CPU |
|---|---|---|---|
| **Runs on** | NVIDIA GPU | NVIDIA/AMD/Intel GPU | CPU cores |
| **Transpiles to** | PTX | OpenCL C | Multi-threaded |
| **Shared memory** | Yes | Yes | Yes |
| **Best for** | NVIDIA hardware | Cross-vendor GPU | No GPU available |

Auto-selection picks the best available backend: WebGPU > WebGL > Wasm (browser) or CUDA > OpenCL > CPU (desktop).

## Implemented Operations

### Kernels

| Category | Operations |
|----------|-----------|
| **Linear algebra** | MatMul (tiled 16x16 shared memory), batched MatMul |
| **Convolution** | Conv2D (arbitrary kernel/stride/padding), ConvTranspose2D |
| **Normalization** | LayerNorm (row-wise, learned gamma/beta), Softmax (two-pass, numerically stable) |
| **Activations** | GELU (erf-based with clamping), ReLU, GELU in-place, ReLU in-place |
| **Element-wise** | Add, Mul, Sub, Scale, AddBias, BroadcastMul, Transpose |
| **In-place variants** | GELUInPlace, ReLUInPlace, ScaleInPlace, AddInPlace |
| **Spatial** | BilinearUpsample (align_corners true/false), ConcatLastDim |
| **Attention** | Multi-head attention (split heads, Q*K^T, softmax, *V, merge heads) |

### Composed Operations

| Operation | Description |
|-----------|-------------|
| **TransformerBlock** | Full Vision Transformer block (pre-LayerNorm, MHSA, MLP, residual connections, LayerScale) |
| **DPT Head** | Dense Prediction Transformer head (multi-scale feature refinement, RefineNet fusion) |

## Planned Features

SpawnDev.ILGPU.ML is being developed toward a complete inference engine. Planned additions include:

- **Tensor abstraction** with shape tracking and buffer pooling
- **ONNX model loading** for standard model interchange
- **Graph execution engine** with topological ordering and automatic buffer management
- **Expanded operator library** covering the most common ONNX operators (~55 ops across 3 tiers)
- **Additional kernels:** BatchNorm, GroupNorm, RMSNorm, Sigmoid, Tanh, SiLU, pooling, reductions, gather/scatter, general transpose, padding, resize
- **Graph optimizations:** constant folding, dead node elimination, operator fusion, in-place optimization
- **Pipeline abstractions** for common tasks (depth estimation, image classification, object detection)
- **Quantization support** (INT8/INT4) for reduced memory and faster inference
- **KV cache** for autoregressive text generation
- **InferenceSession API** similar to ONNX Runtime's familiar interface

See [Plans/full-inference-engine-plan.md](Plans/full-inference-engine-plan.md) for the detailed roadmap.

## Quick Start

### Installation

```bash
dotnet add package SpawnDev.ILGPU.ML
```

### Using a Kernel Directly

```csharp
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;

// Create accelerator (picks best available backend)
using var context = await Context.CreateAsync(builder => builder.AllAcceleratorsAsync());
using var accelerator = await context.CreatePreferredAcceleratorAsync();

// Create a kernel instance
var matMul = new MatMulKernel(accelerator);

// Allocate GPU buffers
int M = 128, K = 64, N = 256;
using var a = accelerator.Allocate1D<float>(M * K);
using var b = accelerator.Allocate1D<float>(K * N);
using var c = accelerator.Allocate1D<float>(M * N);

// Run matrix multiplication on GPU
matMul.MatMul(a.View, b.View, c.View, M, K, N);

// Wait for completion and read results
await accelerator.SynchronizeAsync();
var result = await c.CopyToHostAsync<float>();
```

## Weight Loading

SpawnDev.ILGPU.ML supports loading pre-extracted model weights from a binary format optimized for web delivery:

- **`weights_fp16.bin`** — flat FP16 values, 256-byte aligned per tensor
- **`manifest_fp16.json`** — JSON manifest with tensor names, shapes, offsets, and data types

FP16 weights are converted to FP32 on upload and stored in a single GPU buffer. Individual tensors are accessed via `SubView` with 256-byte alignment to satisfy WebGPU buffer binding requirements.

## Blazor WebAssembly Configuration

SpawnDev.ILGPU.ML requires [SpawnDev.BlazorJS](https://github.com/LostBeard/SpawnDev.BlazorJS) for browser interop and must be configured with IL trimming and AOT compilation disabled:

```xml
<PropertyGroup>
  <!-- ILGPU requires IL reflection at runtime -->
  <PublishTrimmed>false</PublishTrimmed>
  <RunAOTCompilation>false</RunAOTCompilation>
</PropertyGroup>
```

## Testing

Tests run across all 6 backends using the **PlaywrightMultiTest** infrastructure (same approach used by [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU)):

```bash
# Run all tests (desktop + browser)
dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj

# Run only WebGPU tests
dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj \
  --filter "FullyQualifiedName~WebGPUTests."

# Run only CPU tests
dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj \
  --filter "FullyQualifiedName~CPUTests."
```

Every kernel includes validation against CPU reference implementations to ensure cross-backend correctness.

## Issues and Feature Requests

Please report bugs and request features via [GitHub Issues](https://github.com/LostBeard/SpawnDev.ILGPU.ML/issues).

## Credits

SpawnDev.ILGPU.ML would not be possible without the foundational work of the [ILGPU](https://github.com/m4rs-mt/ILGPU) project. ILGPU provides the IL-to-GPU compiler that makes it possible to write GPU kernels in C# and have them execute on CUDA, OpenCL, and CPU backends. The engineering effort behind ILGPU — building a complete GPU compiler in managed .NET — is remarkable, and this project is deeply grateful for it.

- **ILGPU:** [https://github.com/m4rs-mt/ILGPU](https://github.com/m4rs-mt/ILGPU)
- **ILGPU Authors:** [Marcel Koester](https://github.com/m4rs-mt) and the [ILGPU contributors](https://github.com/m4rs-mt/ILGPU/graphs/contributors)

[SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU) extends ILGPU with three browser backends (WebGPU, WebGL, Wasm), bringing GPU compute to Blazor WebAssembly. SpawnDev.ILGPU.ML builds on top of both projects to bring neural network inference to every platform that .NET supports.

## Resources

- [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU) — Cross-platform GPU compute for .NET (6 backends)
- [SpawnDev.BlazorJS](https://github.com/LostBeard/SpawnDev.BlazorJS) — Full JS interop for Blazor WebAssembly
- [ILGPU](https://github.com/m4rs-mt/ILGPU) — The GPU compiler this project is built upon
- [ILGPU Documentation](https://ilgpu.net/)
- [GitHub Repository](https://github.com/LostBeard/SpawnDev.ILGPU.ML)

## License

This project is licensed under the same terms as ILGPU. See [LICENSE](LICENSE.txt) for details.
