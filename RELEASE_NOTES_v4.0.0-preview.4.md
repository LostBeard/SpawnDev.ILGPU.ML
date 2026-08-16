# SpawnDev.ILGPU.ML 4.0.0-preview.4

> **Native C# neural-network inference on six GPU backends from one codebase.** WebGPU. WebGL. WebAssembly. CUDA. OpenCL. CPU. No ONNX Runtime, no JavaScript bridge, no native binaries. The same kernels run in your browser, on your laptop, and in your server.

This is the **biggest API milestone** of the preview series - a Transformers.js-style `Tensor<T>` API surface, GPU-direct rendering all the way to the page, and a 5-demo gallery of fully-working inference pipelines you can run in your browser right now.

📦 **NuGet:** [`SpawnDev.ILGPU.ML 4.0.0-preview.4`](https://www.nuget.org/packages/SpawnDev.ILGPU.ML/4.0.0-preview.4)
🌐 **Live demos:** [lostbeard.github.io/SpawnDev.ILGPU.ML](https://lostbeard.github.io/SpawnDev.ILGPU.ML/)
💚 **Sponsor the crew:** [github.com/sponsors/LostBeard](https://github.com/sponsors/LostBeard)

---

## What works today (5 verified pipelines)

Each of these runs end-to-end on **WebGPU, WebGL, Wasm, CPU, CUDA, and OpenCL** from a single C# pipeline:

| Demo | Model | Pipeline |
|------|-------|----------|
| ![Classification](SpawnDev.ILGPU.ML.Demo/wwwroot/screenshots/2026-05-23-11-37_Classification-Cat.jpg) | **Image classification** - SqueezeNet 1.1 | `ClassificationPipeline` |
| ![Depth](SpawnDev.ILGPU.ML.Demo/wwwroot/screenshots/2026-05-23-11-37_DepthAnythingV2-House.jpg) | **Monocular depth estimation** - Depth Anything V2 Small (95MB) | `DepthEstimationPipeline` |
| ![Style transfer](SpawnDev.ILGPU.ML.Demo/wwwroot/screenshots/2026-05-23-11-37_StyleTransfer-CatMosaic.jpg) | **Neural style transfer** - Mosaic (ONNX Model Zoo) | `StyleTransferPipeline` |
| ![Background removal](SpawnDev.ILGPU.ML.Demo/wwwroot/screenshots/2026-05-23-11-37_BackgroundRemoval-Person.jpg) | **Background removal** - RMBG-1.4 | `BackgroundRemovalPipeline` |
| ![Super resolution](SpawnDev.ILGPU.ML.Demo/wwwroot/screenshots/2026-05-23-11-37_SuperResolution-Tree.jpg) | **3× super-resolution** - ESPCN, tile-based with color and source-aspect preservation | `SuperResolutionPipeline` |

Every result above is rendered straight from a GPU buffer to an HTML `<canvas>` via the library's `ICanvasRenderer` - no PNG encode, no base64 data URL, no host readback of pixel data. The depth and super-res pipelines preserve source aspect ratio. Super-res uses tile-based inference so the full source resolution gets the model's enhancement.

11 more pipelines exist in the codebase (object detection, pose, face, NLP, diffusion, TTS, single-image-to-3D) but aren't all verified end-to-end on every backend yet. That's the work ahead.

---

## Headline change - Transformers.js-style API

If you've used [Transformers.js](https://huggingface.co/docs/transformers.js) or ONNX Runtime, the new tensor API will feel immediately familiar:

```csharp
using var session = await InferenceSession.CreateFromFileAsync(
    accelerator, http, "models/squeezenet/model.onnx");

// Allocate the input as an OwnedTensor - wraps a fresh GPU buffer.
using var input = OwnedTensor<float>.FromHost(
    accelerator, pixels, new[] { 1, 3, 224, 224 });

// Transformers.js-style call. Inputs are Tensor<float>; OwnedTensor converts implicitly.
// Outputs come back as an OwnedTensorMap<float> - each output tensor lives in its own
// freshly-allocated GPU buffer, independent of the session's pool. Run B will not mutate
// Run A's outputs. The `using` disposes every contained tensor in one go.
using var outputs = await session.RunOwnedAsync(new Dictionary<string, Tensor<float>>
{
    [session.InputNames[0]] = input,
});

var logits = outputs[session.OutputNames[0]];   // OwnedTensor<float>
var hostLogits = await logits.ToHostAsync();    // copy back to CPU only when needed
```

Under the hood there are three types, mirroring the split ILGPU itself uses between `MemoryBuffer<T>` (host, lifetime-managing) and `ArrayView<T>` (kernel-passable struct):

- **`Tensor<T>`** - host-side, generic, zero-copy `Reshape` / `Slice` / `SubTensor`.
- **`OwnedTensor<T>`** - `IDisposable`, owns a `MemoryBuffer1D<T>`. What pipelines return. Implicit conversions to `Tensor<T>` and `TensorView<T>` mean you never have to write `.AsTensor` or `.View` at a call site.
- **`TensorView<T>`** - blittable struct, passes directly into ILGPU kernels. Inline `D0..D3 + Rank`. Replaces the old "pass an ArrayView plus four shape ints" idiom. Kernel authors write `Get4D(n, c, h, w)` instead of doing manual row-major stride math.

```csharp
// Kernel takes the tensor directly - no separate W/H scalar parameters.
private static void DoubleKernel(Index1D idx,
    TensorView<float> input, TensorView<float> output)
{
    int w = idx % input.D3;
    int h = (idx / input.D3) % input.D2;
    int c = (idx / (input.D3 * input.D2)) % input.D1;
    int n = idx / (input.D3 * input.D2 * input.D1);
    output.Set4D(n, c, h, w, input.Get4D(n, c, h, w) * 2f);
}
```

Generic over `T : unmanaged` - `Tensor<float>`, `Tensor<int>`, `Tensor<Half>` (FP16) all work through the same kernel pipeline. The legacy non-generic `Tensor` is preserved for backwards compatibility and now exposes a `.View` property so existing call sites can opt into kernel-passable views without changing their type.

## Other highlights

- **`InferenceSession.RunOwnedAsync`** - outputs are caller-owned GPU buffers, GPU-to-GPU copied off the executor's internal pool. Subsequent runs cannot mutate previously-returned tensors. Existing `RunAsync` continues to work unchanged.
- **`/depth` palette swap is one accelerator dispatch.** The dropdown's plasma / viridis / inferno / grayscale toggle re-runs the colormap kernel on the cached raw depth and re-presents to the canvas. No re-inference. No host readback.
- **`/depth` and `/super-res` render via `ICanvasRenderer`** - backend-optimized GPU→canvas (WebGPU texture copy / WebGL blit / CPU `putImageData`). Zero PNG encode, zero base64 data URL.
- **Tile-based super-resolution** - ESPCN now tiles the source through the model with overlap-averaged stitching on the accelerator, preserving full source resolution + color + aspect ratio. The 4× single-thumbnail-then-bilinear behavior is gone.
- **First two Phase 2 kernel migrations** (`ResizeBilinear`, `DepthToColormapPalette`) - kernels read shape from `TensorView<T>.D0/D1` instead of taking H/W as separate scalar parameters. More kernel families migrate in subsequent previews.

Full changelog: [CHANGELOG.md](CHANGELOG.md)

## Dependencies (unchanged)

- `SpawnDev.ILGPU 4.9.8`
- `SpawnDev.WebTorrent 2.3.1`
- `Microsoft.AspNetCore.Components.Web 10.0.4`

---

## 💚 Support the project - help us hit warp 10 again

This library exists because **one person and a small team** have spent months hand-writing native C# GPU kernels, six-backend transpilers, and ML pipelines while running on a **$20/month** budget.

When the budget allows it, peak output looks like **410 commits in a single day** across SpawnDev.ILGPU, SpawnDev.ILGPU.ML, SpawnDev.SpawnJS, SpawnDev.RTC, SpawnDev.WebTorrent, and the rest of the SpawnDev stack. When it doesn't, work slows to whatever individual evenings can spare.

**We're asking for $200/month total in GitHub Sponsorships to put the full crew back on the ship.** That's the gap between this preview and the next ten - every remaining operator family migrated to the new Tensor API, every pipeline verified end-to-end on every backend, FP16 attention, Flash Attention on WebGPU, Llama and Phi-4 LLM inference, full text-to-image diffusion through SD-Turbo, P2P distributed compute through SpawnDev.WebTorrent. It's all in flight; the bottleneck is hours, not ideas.

[**→ Sponsor on GitHub**](https://github.com/sponsors/LostBeard) - any amount helps. Even a $5/month sponsorship is a vote of confidence that puts wind behind the work.

If sponsorship isn't an option, you can:
- ⭐ Star the [repo](https://github.com/LostBeard/SpawnDev.ILGPU.ML) - visibility is the second-most-valuable thing after dollars.
- 🐛 File issues from your own models - every model that doesn't load is a kernel or operator we can ship next preview.
- 🛠️ Contribute kernel migrations to the new `TensorView<T>` API - most have a clear pattern now.
- 📣 Talk about the project anywhere developers gather. .NET community, ML community, Blazor circles - all good.

🖖🚀
