# SpawnDev.ILGPU.ML — Examples

A progressive ladder of **individual, self-contained** example projects — from a minimal inspect to a
full chat app. Each one reads top-to-bottom (its own boilerplate, on purpose), takes its primary input
as an **argument or an interactive prompt**, and supports a **`--ci`** flag so it doubles as an
assertable smoke test. Each showcases a real SpawnDev gear.

Run any example with:

```bash
dotnet run --project Examples/<name>/<name>.csproj -- <args>
```

## The ladder

| # | Example | What it shows | Status |
|---|---------|---------------|--------|
| 01 | `HelloPipeline.Console` | minimal: load a model, one forward pass, print output | planned |
| 02 | **`ModelInspector.Console`** | stream a model's **header only** (no full download) → architecture, operators, tensors, quantization, GGUF metadata + tensor templates, engine compatibility | ✅ shipped |
| 03 | **`ImageGen.Console`** | everything the browser `/generate` page does, on native GPU kernels: FLUX 3 Image Klein (+ the Qwen3 caption writer), FLUX.2 klein, SD-Turbo; reference images to edit or combine; 1-N pictures; sizes; seed; prompt → `.png` | ✅ shipped |
| 04 | **`GGUFTextGen.Console`** | GGUF autoregressive decode + KV cache (qwen2.5-coder, gemma4); decode/prefill perf probes | ✅ shipped |
| 05 | **`Gemma4Multimodal.Console`** | gemma4:12b vision+text (mmproj image tokens) | ⏳ written, verify pending |
| 06 | **`OllamaServer.Console`** | **drop-in Ollama replacement** — native-GPU GGUF server (OpenAI + Ollama + Anthropic APIs) for Claude CLI / Pi / Codex / etc., zero-copy from your `~/.ollama` cache | 🚧 WIP (works; perf + huge-context decode are the active work) |
| 07 | **`AssetGen.Console`** | **a prompt to a rigged, animated 3D character** on your own GPU (CUDA / OpenCL): FLUX.2 klein draws it, RMBG-1.4 cuts it out, TRELLIS.2 builds the textured mesh, SkinTokens rigs it, Kimodo moves animate it; `.png` + `.glb` files out | ✅ shipped |
| 08 | **`AssetGen.Wpf`** | the browser's Text and Image to 3D page as a Windows app: FLUX 3 / FLUX.2 / SD-Turbo (with the Qwen3 caption writer) or your own picture, auto-pick or draw 4 / 8 and pick (and draw more), triangles and texture size, rig, animate (arm clearance), preview in a WebGL viewer (WebView2), save the .glb | ✅ shipped |
| 10 | **`ImageGen.Wpf`** | the browser `/generate` page as a Windows app: the same models and settings, reference images, a gallery with Save, the live caption | ✅ shipped |
| 11 | **`VideoGen.Console`** | **text or image to video**: Wan 2.2 TI2V-5B on CUDA / OpenCL, 8-bit weights, PNG frames and an animated WebP (`--turbo` for the 4-step distilled model, `--size`, `--frames`, `--steps`, `--image`, `--models`); see [Docs/video.md](../Docs/video.md) | ✅ shipped |
| 12 | **`VideoGen.Wpf`** | the browser `/video` page as a Windows app: settings, first-frame picture, a looping player, Save frames | ✅ shipped |
| 09+ | `*.BlazorWasm` | browser mirrors (zero-copy canvas for image-gen) | planned (the main demo already is one) |

## 02 · ModelInspector.Console

Drop a **local path**, an **http(s) URL**, or an **Ollama `name:tag`** and see the model's structure —
streaming only the metadata header, so a multi-GB model inspects from a few KB.

```bash
dotnet run --project Examples/02.ModelInspector.Console/ModelInspector.Console.csproj -- model.onnx
dotnet run --project Examples/02.ModelInspector.Console/ModelInspector.Console.csproj -- gemma4:12b
dotnet run --project Examples/02.ModelInspector.Console/ModelInspector.Console.csproj -- --ci
```

No accelerator needed — inspection is pure header parsing.

## 03 · ImageGen.Console

The browser demo's Image Generation page on the command line: **FLUX 3 Image Klein 5.6B** (16 steps; Qwen3 4B writes the full scene caption it draws from, `--no-caption` to skip), **FLUX.2 klein 4B** (4 steps) or **SD-Turbo** (the default, one step),
up to 10 **reference images** (`--ref`, FLUX) to edit or combine, any number of pictures per prompt (consecutive seeds), sizes 512 / 768 / 1024, 4-bit or 8-bit weights, SD-Turbo's steps / guidance / negative prompt. With no prompt it is an
interactive session (`/model`, `/size`, `/pictures`, `/ref`, `/seed`, ... see `/help`): the model loads once. The accelerator is created in-app; weights stream from the SpawnDev models host
(Hugging Face as the fallback) on first run (SD-Turbo 2.5 GB, FLUX.2 4.6 GB, FLUX 3 5.7 GB + 2.4 GB for the caption writer; cached after, resumable). Output is a `.png` (`.bmp` if `--out` ends in `.bmp`). FLUX 3's weights are under the FLUX Kommunity License (non-commercial use).

```bash
dotnet run --project Examples/03.ImageGen.Console/ImageGen.Console.csproj -- a photo of a cat
dotnet run --project Examples/03.ImageGen.Console/ImageGen.Console.csproj -- "a watercolor fox" --seed 7 --out fox.png
dotnet run --project Examples/03.ImageGen.Console/ImageGen.Console.csproj -- "a red fox in snow" --model flux3 --pictures 4 --size 768
dotnet run --project Examples/03.ImageGen.Console/ImageGen.Console.csproj -- "turn the cat into a tiger" --model flux2 --ref cat.png
dotnet run --project Examples/03.ImageGen.Console/ImageGen.Console.csproj -- --ci
```

## 10 · ImageGen.Wpf

The same page as a Windows app (WPF): prompt, model (FLUX 3 + caption writer / FLUX.2 / SD-Turbo), size, pictures 1 / 4 / 8, seed, 4-bit or 8-bit weights, SD-Turbo steps / guidance / negative prompt, reference images (Add...), a gallery with Save, the caption as it is written.
Both this and 03 are thin shells over `SpawnDev.ILGPU.ML.Pipelines.ImageGenerator`. Set `IMAGEGEN_CACHE` to keep the models on a big drive.

```bash
dotnet run --project Examples/10.ImageGen.Wpf
```

## 06 · OllamaServer.Console 🚧 WIP

A **drop-in Ollama replacement**: a native-GPU GGUF inference server that prebuilt agentic frontends
(Claude CLI, Pi, Codex, OpenCode, Continue, …) talk to with zero reconfiguration. It serves the GGUF models
**already in your `~/.ollama` cache, zero-copy** — no re-download, no duplicate files — and speaks three wire
protocols at once: **OpenAI** (`/v1/chat/completions`), **Ollama-native** (`/api/chat`, `/api/tags`,
`/api/show`), and **Anthropic Messages** (`/v1/messages`, for Claude CLI).

```bash
dotnet run --project Examples/06.OllamaServer.Console -c Release            # serve on :11434
dotnet run --project Examples/06.OllamaServer.Console -- --list             # list servable cached models
dotnet run --project Examples/06.OllamaServer.Console -- --chat qwen2.5-coder:7b "Hi"
```

Point a client at it (see the example's own [README](06.OllamaServer.Console/README.md) for the full table):
`ANTHROPIC_BASE_URL=http://localhost:11434 claude`, or any OpenAI-compatible client at `…/v1`.

**Status (WIP):** functional end-to-end on qwen2.5-coder:7b and gemma4:12b (greedy + KV-cache decode, streaming
detokenize, tool-calling, ChatML/Llama3/gemma templates). The active work is **performance** (decode is
GPU-compute-bound; beating Ollama needs vectorized-load + fused GEMV — in progress) and a large-context decode
path for 12B models. Validate against real Ollama as an oracle (same GGUF blob → first-divergence token +
tokens/sec), not byte-identity.

## Conventions

- **Self-contained** — no shared helper library; a little duplication keeps each example readable alone.
- **Arg-or-prompt** — primary input from `args`; if absent and interactive, prompt; if absent and `--ci`,
  use a fixed default + fixed seed.
- **`--ci`** — non-interactive, deterministic, asserts a known result, exits `0` on success.

## 09 · SceneGen.Console (a scene from a sentence)

```bash
dotnet run --project Examples/09.SceneGen.Console -- "a blacksmith's workshop" --out workshop
```

A local LLM (`--llm qwen3:14b`, an Ollama name or a .gguf) writes a layout, every distinct object is generated with `AssetGenerator`, and `GltfSceneComposer` puts them in one glTF scene. Details and measurements: [Docs/scenes.md](../Docs/scenes.md).

## 07 · AssetGen.Console and 08 · AssetGen.Wpf (text to character)

```bash
dotnet run --project Examples/07.AssetGen.Console -- "a knight in plate armor, full body, standing" --cache D:\models
dotnet run --project Examples/07.AssetGen.Console -- "a friendly robot" --pictures 4 --pick --no-animate   # at the prompt: a number to build that picture, or m to draw more
dotnet run --project Examples/07.AssetGen.Console -- "a dragon" --image-model flux3                         # FLUX 3 Image Klein + the Qwen3 caption writer
dotnet run --project Examples/07.AssetGen.Console -- --image crown.png --no-rig                              # image to 3D from your own picture
dotnet run --project Examples/08.AssetGen.Wpf        # set ASSETGEN_CACHE to keep the models on a big drive
```

Both are thin shells over `SpawnDev.ILGPU.ML.Assets.AssetGenerator` (draw -> cut out -> 3D -> rig -> animate, each stage loaded on its own so a 12 GB card is enough). The first run downloads about 17 GB of models into the cache folder
(FLUX.2 klein 4.4 GB, TRELLIS.2 11 GB, SkinTokens 1.2 GB, RMBG 44 MB) and resumes if interrupted. RTX 4070, CUDA, models cached: prompt to animated GLB in about 80 s; the pictures themselves take 2-3 s each after the first. See `Docs/assetgen.md`.
