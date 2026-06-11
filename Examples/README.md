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
| 03 | **`ImageGen.Console`** | SD-Turbo text→image on native GPU kernels; prompt → `.bmp` | ⏳ written, verify pending |
| 04 | `TextGen.Console` | GPT-2 autoregressive decode + KV cache | planned |
| 05 | `AIChat.Console` | full chat loop + history (Gemma) | planned |
| 06 | `Precompiled.Console` | same workload via precompiled shaders — startup/perf delta | planned |
| 07–09 | `*.BlazorWasm` | browser mirrors (zero-copy canvas for image-gen) | planned |

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

SD-Turbo (single-step diffusion) text→image. The accelerator is created in-app; weights stream from the
SpawnDev hub on first run (~2.5 GB, cached after). Output is a `.bmp`.

```bash
dotnet run --project Examples/03.ImageGen.Console/ImageGen.Console.csproj -- a photo of a cat
dotnet run --project Examples/03.ImageGen.Console/ImageGen.Console.csproj -- "a watercolor fox" --seed 7 --out fox.bmp
dotnet run --project Examples/03.ImageGen.Console/ImageGen.Console.csproj -- --ci
```

## Conventions

- **Self-contained** — no shared helper library; a little duplication keeps each example readable alone.
- **Arg-or-prompt** — primary input from `args`; if absent and interactive, prompt; if absent and `--ci`,
  use a fixed default + fixed seed.
- **`--ci`** — non-interactive, deterministic, asserts a known result, exits `0` on success.
