# Demo & Model Status — what's VERIFIED vs WIP

**This table is the source of truth. If a demo isn't marked ✅ VERIFIED, treat it as work-in-progress and don't expect it to fully work yet.** We would rather under-promise here than have you click something and get nothing.

**What the statuses mean:**
- **✅ VERIFIED** — has a passing end-to-end test in the suite (most match ONNX Runtime numerically, the strongest bar), and/or was confirmed running live. The cited test is the evidence.
- **🟡 PARTIAL** — the real pipeline runs, but something is incomplete: only a subset of the path is tested, it needs an external API, or it works only when a model is loaded (placeholder otherwise).
- **🚧 WIP** — the page exists but the core action is a stub/no-op, or there is no end-to-end test yet. **Don't expect it to work.**
- **Meta / Doc** — not an inference demo (tooling, onboarding, model browser).

> Evidence basis: status reflects the cited E2E test (in `SpawnDev.ILGPU.ML.Demo.Shared/UnitTests/`) plus recent green runs. The **canonical** pass/fail at any moment is the latest `PlaywrightMultiTest` results JSON - run PMT to re-confirm before a release.
>
> ⚠️ **A cited test only counts if it RUNS THE MODEL.** Three tests here were named `Pipeline_*_Reference_*` and documented as validating "end-to-end pipeline correctness" while only asserting that a reference JSON was internally consistent - they would pass with the ML library deleted, and two of them were this table's evidence. Before citing a test, check that it constructs an `InferenceSession` or calls a pipeline. Fixture-integrity tests are useful, but they are named `ReferenceData_*_FixtureIsWellFormed` and they are not evidence that a demo works.
>
> **Live page smoke (2026-09-26):** `tools/drive-ml-pages-smoke.cs` against a Release publish — **32/32 routes mounted**, 0 pageerrors. Inference gates: `tools/drive-ml-pages.cs` — `/sentiment` ✅ POSITIVE; `/embeddings` re-pointed to all-MiniLM-L6-v2 (was DistilBertSST2 classifier — threw live until fixed this session).

## Demos

| Route | Demo | Status | Evidence / why |
|-------|------|--------|----------------|
| `/classify` | Image classification | ✅ **VERIFIED** | `CreateFromFile_SqueezeNet_CatClassification`, `OptimizedPipeline_SqueezeNet_SameResult`; MobileNetV2 graph tests |
| `/style` | Neural style transfer | ✅ **VERIFIED** | ORT-matched 5 styles: `Reference_StyleMosaic/Candy/Pointilism/RainPrincess/Udnie_MatchesOnnxRuntime` |
| `/depth` | Depth estimation (Depth Anything) | ✅ **VERIFIED** | `Reference_DepthAnything_MatchesOnnxRuntime`, `DA3Small_DepthMap_NotFlat`, `CreateFromFile_DepthAnything_Inference` |
| `/detect` | Object detection (YOLOv8) | ✅ **VERIFIED** | `Pipeline_YOLOv8_Reference_MatchesOnnxRuntime`, `Pipeline_YOLOv8_DetectsObjects` |
| `/pose` | Pose estimation (MoveNet) | ✅ **VERIFIED** | `Reference_MoveNetLightning_MatchesOnnxRuntime`, `Pipeline_MoveNet_DetectsKeypoints` (asymmetric-pad decode fixed) |
| `/clip` | Zero-shot classification (CLIP) | ✅ **VERIFIED** | `Reference_CLIPVision_MatchesOnnxRuntime` (runs the model against ORT). ⚠️ Previously also cited `Pipeline_CLIP_Reference_CatIsTopMatch`, which never ran the model - it asserted the reference JSON was well formed, and is now named `ReferenceData_CLIP_FixtureIsWellFormed` |
| `/remove-bg` | Background removal (RMBG) | ✅ **VERIFIED** | `Pipeline_BackgroundRemoval_RealImage_ProducesVaryingMask` (perf caveats on WebGPU compile) |
| `/super-res` | Super-resolution (ESPCN) | ✅ **VERIFIED** | `CreateFromFile_SuperResolution_ESPCN`, `HF_DownloadAndLoadSession_SuperResolution` |
| `/face` | Face detection (BlazeFace) | ✅ **VERIFIED** | `Pipeline_BlazeFace_Reference_MatchesOnnxRuntime` (finite regressors). Page loads on first Capture (opt-in). Live smoke mounts; webcam E2E is manual |
| `/sentiment` | Sentiment (DistilBERT-SST2) | ✅ **VERIFIED** | `Sentiment_DistilBertSST2_ClassifiesPositiveAndNegative`; **live gate 2026-09-26** `drive-ml-pages.cs` → `.sentiment-verdict` = POSITIVE |
| `/embeddings` | Sentence embeddings / semantic search | ✅ **VERIFIED** | `Embeddings_RealTokenizer_RelatedScoresHigherThanUnrelated` on **all-MiniLM-L6-v2** (384-dim, all six backends). **Page re-pointed 2026-09-26** off DistilBertSST2 (classifier `logits` [1,2] — threw at runtime). Gate: `drive-ml-pages.cs` `/embeddings` |
| `/text-gen` | Text generation (DistilGPT-2) | ✅ **VERIFIED** | `Pipeline_TextGeneration_ProducesTokens` + `Sampler_*` suite; **confirmed live on GH Pages 2026-06-04**. WebGPU verified; ~0.2 tok/s (perf WIP). NOTE: superseded by `/ai-chat` for real LLM chat — candidate to retire |
| `/whisper` | Speech-to-text (Whisper) | ✅ **VERIFIED** | **mic → text works end to end.** `Pipeline_Whisper_TranscribesKnownSpeech` — same answer on **all six backends**: "All legal box recordings are in the public domain". `MLTestBase.ResamplerTests`, `MicrophoneCaptureTests`, `tools/drive-mic-capture.cs --transcribe`. Nav: no badge (VERIFIED) |
| `/snake` | System One Classic Snake | ✅ **VERIFIED** | Package: generic `SystemOneDecisionHead`. Demo-only: game/teacher/`SnakeSystemOneSpec` under `Demo.Shared`. Evidence: `SystemOne_Snake_BehavioralClone_AgreesWithTeacher`, `SystemOne_Weights_RoundTrip_PreservesProbs`. Nav: no badge (VERIFIED). Docs: [`system-one.md`](system-one.md) |
| `/inspector` | Model Inspector (structure + compat) | ✅ **VERIFIED** | Streams ONNX structure-only; GPT-2 100% compat after registry fix; inspect-by-URL live-hub test |
| `/benchmark` | GPU benchmark | ✅ **VERIFIED** | MatMul / perf kernels (92-101 GFLOPS validated) |
| `/ai-chat` | On-device LLM chat (GGUF, multi-model) | 🟡 **PARTIAL** | `GgufTextGenerationPipeline`: pick a `.gguf` → stream → chat on WebGPU. Engine verified on CUDA (qwen/smollm2/gemma3 coherent). Page mounts clean. In-browser file-pick→generate E2E = manual confirm (no model delivery in PMT yet) |
| `/gemma-chat` | Gemma 4 multimodal chat | 🟡 **PARTIAL** | Opt-in large download from hub/ollama registry; mounts clean. Full multimodal E2E in-browser not gated in PMT |
| `/depth-voxel` | Depth → 3D voxels | 🟡 **PARTIAL** | Depth pipeline runs (see `/depth`); the 3D voxel/gaussian-splat scene was never built (nav-hidden 2026-06-30). Route still mounts; shows 2D depth colormap |
| `/explain` | Model explainability | 🟡 **PARTIAL** | Intercepts the executor; works on a limited set of models |
| `/assistant` | AI assistant (chat) | 🟡 **PARTIAL** | Real DistilGPT-2 when a model is loaded; **falls back to `GetPlaceholderResponse` when none loaded**. README claims of Phi-4 14B / SpeechT5 voice are **aspirational — not what this page does today** |
| `/comic-chat` | Multi-character chat | 🟡 **PARTIAL** | Same as assistant — real text-gen when loaded, `GetPlaceholderComicResponse` otherwise. No Phi-4 tiering on this page |
| `/train` | On-device training (Draw to Learn) | 🟡 **PARTIAL** | Gesture CNN path still PARTIAL; **System One BC train is verified** — see `/snake` |
| `/generate` | Image generation (SD-Turbo) | ✅ **VERIFIED** | `SDTurbo_Generate_E2E` (WebGPU/CUDA/OpenCL; Wasm skipped — weight paging tracked). Same `ImageGenerationPipeline` as SpawnDev.AI `AiImageEngine`. Demo page: opt-in load on first Generate (was Coming-soon stub that never called `LoadModelAsync`). |
| `/image-to-3d` | Image → 3D model | 🚧 **WIP** | Coming-soon banner honest. `GenerateModel()` is a **no-op**; download/open empty |
| `/voice-collab` | Voice collaboration | 🚧 **WIP** | Coming-soon banner honest. Phase 1: browser Web Speech API; **GPU Whisper path disabled** |
| `/` | Home | Meta | Landing (operator count live from `OperatorRegistry.BuiltinOpTypes` — currently **204**) |
| `/pipelines` | Pipeline catalog | Meta | Status badges must match this doc — corrected 2026-09-26 |
| `/getting-started` | Getting started | Doc | Install + first-run walkthrough |
| `/models` | Model browser | Meta | HuggingFace hub browser |
| `/cache` | Model cache | Meta | OPFS cache admin |
| `/tests` | Test runner | Meta | Hosts the PlaywrightMultiTest UI |

## Models (loaders vs verified inference)

| Model | Loads? | Inference verified? | Note |
|-------|--------|---------------------|------|
| SqueezeNet / MobileNetV2 | ✅ | ✅ classification | ORT-aligned |
| Style-transfer (5 styles) | ✅ | ✅ | ORT-matched |
| Depth Anything V2 Small | ✅ | ✅ | ORT-matched |
| YOLOv8-nano | ✅ | ✅ | ORT-matched |
| MoveNet Lightning | ✅ | ✅ | ORT-matched |
| BlazeFace | ✅ | ✅ | `Pipeline_BlazeFace_Reference_MatchesOnnxRuntime` |
| CLIP (vision) | ✅ | ✅ | ORT-matched |
| ESPCN super-res | ✅ | ✅ | ORT-matched |
| DistilGPT-2 / DistilBERT-SST2 | ✅ | ✅ text-gen / sentiment | DistilBERT-SST2 is the **SST-2 classifier** — one output `logits` [batch,2], **not** an embedding model |
| all-MiniLM-L6-v2 | ✅ | ✅ embeddings | 384-dim `last_hidden_state`; **the model `/embeddings` uses** |
| Whisper | ✅ | ✅ full STT E2E | `Pipeline_Whisper_TranscribesKnownSpeech` (encoder+decoder) on all six backends |
| SpeechT5 (TTS) | ✅ | 🟡 | `Pipeline_TTS_ReferenceTokensProduceAudio`; not wired into a verified demo page |
| SD-Turbo | ✅ | ✅ E2E | `SDTurbo_Generate_E2E` + WebGPU multi-gen gates; `/generate` wired |
| GGUF LLMs (Qwen/Gemma/Llama/SmolLM) | ✅ **runs** (desktop + browser) | 🟡 coherent, oracle-matched on qwen | Autoregressive KV-cache decode verified on desktop; browser `/ai-chat` streams GGUF (manual confirm for full file-pick→generate) |

## Keeping this honest

- **Demo PAGES have a browser gate now**, separate from the unit tests:
  - `tools/drive-ml-pages-smoke.cs` — every route mounts (no pageerror).
  - `tools/drive-ml-pages.cs` — verified routes that can seed inputs and assert a **result element**.
  Two rules the driver encodes the hard way: a page's logging is incidental (EmbeddingsPage logs ONLY on error), and a page that validates its inputs no-ops silently when you seed only the first one. Only routes marked ✅ VERIFIED belong in the inference gate table.
- **Operator count** is rendered live from `OperatorRegistry.BuiltinOpTypes` — never hardcode it again.
- **Before any "N tests passing" claim**, cite the latest PMT results JSON, not a memorized number.
- **A demo graduates to ✅ VERIFIED only when a passing E2E test is cited here.** Adding a page is not the same as verifying it.
- **`/pipelines` badges, nav menu badges, and README demo blurbs must not contradict this table.** Nav mapping: VERIFIED = no badge, PARTIAL = `beta`, WIP = `soon`. If they diverge, this file wins — fix the page/README/nav.
