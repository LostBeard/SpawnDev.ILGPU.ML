# SpawnDev.ILGPU.ML — Examples Folder Plan

**Date:** 2026-06-11
**Author:** Tuvok (lead editor, all lanes)
**Status:** Design agreed with Captain. Build deferred until ILGPU 4.10.0 ships (so example csprojs pin a stable version).

## Goal

A progressive ladder of **individual example projects** in the solution, from a minimal single-pipeline
console app up to a full AI chat console app, plus separate Blazor WASM mirrors. Examples double as
runnable, assertable tests (Rule 5) and each one **showcases a SpawnDev gear** we built (Rule 4).

## Principles (Captain's calls, 2026-06-11)

- **Self-contained.** Each example reads top-to-bottom with its own DI/accelerator/model-acquisition
  boilerplate. A little duplication is a *feature* here — the value is opening ONE file and understanding
  it. No shared `Examples.Common` library.
- **Separate projects.** One project per example; Blazor examples are separate projects too (mirror the
  console ladder), not one multi-page app.
- **Test-first dual purpose.** Every console example supports `--ci` (or no-TTY detection) → runs
  non-interactive with a fixed seed → asserts a known result. That makes each example a smoke test.
- **Arg-or-prompt UX.** Take the primary input from `args[0]`; if absent and interactive, prompt on the
  console; if absent and `--ci`, use a fixed default + fixed seed.

## The ladder

| # | Project | Kind | Showcases | Shaders | CI assertion |
|---|---------|------|-----------|---------|--------------|
| 01 | `HelloPipeline.Console` | console | minimal load → one forward pass | runtime | output tensor matches reference |
| 02 | `ModelInspector.Console` | console | **streaming `InspectAsync`** — path *or* HF/Ollama URL → arch/ops/tensors/quant/compatibility, header-only (no full download) | n/a | known arch + op/tensor counts for a fixed model |
| 03 | `TextGen.Console` | console | GPT-2 autoregressive decode + KV cache | runtime | greedy tokens == reference for a fixed prompt |
| 04 | `ImageGen.Console` | console | `ImageGenerationPipeline` (SD-Turbo), arg-or-prompt → PNG | runtime | image non-black / non-flat (existing assertion) |
| 05 | `AIChat.Console` | console | full chat loop + history + chat template (Gemma 4 lands here) | runtime | scripted turn produces expected token(s) |
| 06 | `Precompiled.Console` | console | same workload as an earlier rung via **precompiled shaders** (Geordi's L1-3) — startup/perf delta | precompiled | parity with the runtime rung |
| 07 | `HelloPipeline.BlazorWasm` | blazor | minimal browser pipeline | runtime | — |
| 08 | `ImageGen.BlazorWasm` | blazor | SD-Turbo in browser, **zero-copy `CanvasRendererFactory`** GPU→canvas | runtime | — |
| 09 | `AIChat.BlazorWasm` | blazor | chat UI in browser | runtime | — |

`02 ModelInspector` is the recommended first build: simplest (parse + print, no inference), highest
wow-factor (characterize a multi-GB model from a streamed header), and its Blazor mirror already exists
(`InspectorPage.razor`) so 02's browser twin is mostly lift-and-trim.

## Shared conventions (re-implemented per example, intentionally)

- **Accelerator pick:** desktop console → CUDA → OpenCL → CPU fallback; lift the exact bootstrap from the
  real `DemoConsole`/`Tests` (do not fabricate the factory call).
- **Model acquisition:** via the hub (`HubModelStream`) — HuggingFace today, **Ollama once `OllamaProxy`
  lands** (see `gemma4-gguf-bringup-2026-06-11.md` §2). Client stays dumb: ask hub → get a stream.
- **Determinism in `--ci`:** fixed prompt + fixed seed so the assertion is stable across runs/backends.

## Testing integration

Each console example exposes its core as a method the Tests project (and PMT) can invoke headless with
`--ci`, asserting reference values (never liveness — see the GPT-2 lesson: assert real tokens, not
"non-NaN / >=1 token"). The image-gen rung reuses the existing non-black/non-flat check.

## Relationship to the existing Demo

The Demo is the **showroom** (everything wired together, UI-rich). The Examples are the **workbench
samples** — minimal, single-purpose, copy-paste-able, each proving one capability end-to-end. They are not
a replacement for the Demo; they are the "how do I do *just this one thing*" answer for a consumer.

## Build sequencing

Build against **stable ILGPU 4.10.0** once Geordi ships it (avoid pinning example csprojs to a version
about to change). Order: 02 (inspector) → 04 (image-gen, pipeline already exists) → 03 (text-gen) →
05 (chat / Gemma) → 06 (precompiled) → Blazor mirrors.
