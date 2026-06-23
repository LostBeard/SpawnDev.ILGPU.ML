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
| 04 | **`GGUFTextGen.Console`** | GGUF autoregressive decode + KV cache (qwen2.5-coder, gemma4); decode/prefill perf probes | ✅ shipped |
| 05 | **`Gemma4Multimodal.Console`** | gemma4:12b vision+text (mmproj image tokens) | ⏳ written, verify pending |
| 06 | **`OllamaServer.Console`** | **drop-in Ollama replacement** — native-GPU GGUF server (OpenAI + Ollama + Anthropic APIs) for Claude CLI / Pi / Codex / etc., zero-copy from your `~/.ollama` cache | 🚧 WIP (works; perf + huge-context decode are the active work) |
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
