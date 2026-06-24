# Demo & Model Status — what's VERIFIED vs WIP

**This table is the source of truth. If a demo isn't marked ✅ VERIFIED, treat it as work-in-progress and don't expect it to fully work yet.** We would rather under-promise here than have you click something and get nothing.

**What the statuses mean:**
- **✅ VERIFIED** — has a passing end-to-end test in the suite (most match ONNX Runtime numerically, the strongest bar), and/or was confirmed running live. The cited test is the evidence.
- **🟡 PARTIAL** — the real pipeline runs, but something is incomplete: only a subset of the path is tested, it needs an external API, or it works only when a model is loaded (placeholder otherwise).
- **🚧 WIP** — the page exists but the core action is a stub/no-op, or there is no end-to-end test yet. **Don't expect it to work.**
- **Meta / Doc** — not an inference demo (tooling, onboarding, model browser).

> Evidence basis: status reflects the cited E2E test (in `SpawnDev.ILGPU.ML.Demo.Shared/UnitTests/`) plus recent green runs. The **canonical** pass/fail at any moment is the latest `PlaywrightMultiTest` results JSON — run PMT to re-confirm before a release.

## Demos

| Route | Demo | Status | Evidence / why |
|-------|------|--------|----------------|
| `/classify` | Image classification | ✅ **VERIFIED** | `CreateFromFile_SqueezeNet_CatClassification`, `OptimizedPipeline_SqueezeNet_SameResult`; MobileNetV2 graph tests |
| `/style` | Neural style transfer | ✅ **VERIFIED** | ORT-matched 5 styles: `Reference_StyleMosaic/Candy/Pointilism/RainPrincess/Udnie_MatchesOnnxRuntime` |
| `/depth` | Depth estimation (Depth Anything) | ✅ **VERIFIED** | `Reference_DepthAnything_MatchesOnnxRuntime`, `DA3Small_DepthMap_NotFlat`, `CreateFromFile_DepthAnything_Inference` |
| `/detect` | Object detection (YOLOv8) | ✅ **VERIFIED** | `Pipeline_YOLOv8_Reference_MatchesOnnxRuntime`, `Pipeline_YOLOv8_DetectsObjects` |
| `/pose` | Pose estimation (MoveNet) | ✅ **VERIFIED** | `Reference_MoveNetLightning_MatchesOnnxRuntime`, `Pipeline_MoveNet_DetectsKeypoints` (asymmetric-pad decode fixed) |
| `/clip` | Zero-shot classification (CLIP) | ✅ **VERIFIED** | `Pipeline_CLIP_Reference_CatIsTopMatch`, `Reference_CLIPVision_MatchesOnnxRuntime` |
| `/remove-bg` | Background removal (RMBG) | ✅ **VERIFIED** | `Pipeline_BackgroundRemoval_RealImage_ProducesVaryingMask` (perf caveats on WebGPU compile) |
| `/super-res` | Super-resolution (ESPCN) | ✅ **VERIFIED** | `CreateFromFile_SuperResolution_ESPCN`, `HF_DownloadAndLoadSession_SuperResolution` |
| `/text-gen` | Text generation (DistilGPT-2) | ✅ **VERIFIED** | `Pipeline_TextGeneration_ProducesTokens` + `Sampler_*` suite; **confirmed live on GH Pages 2026-06-04**. WebGPU verified; ~0.2 tok/s (perf WIP) |
| `/embeddings` | Sentence embeddings / semantic search | ✅ **VERIFIED** | `Pipeline_SemanticSearch_SimilarSentencesCloser` (DistilBERT) |
| `/inspector` | Model Inspector (structure + compat) | ✅ **VERIFIED** | Streams ONNX structure-only; GPT-2 100% compat after registry fix; inspect-by-URL live-hub test |
| `/benchmark` | GPU benchmark | ✅ **VERIFIED** | MatMul / perf kernels (92-101 GFLOPS validated) |
| `/whisper` | Speech-to-text (Whisper) | 🟡 **PARTIAL** | `Pipeline_WhisperDecoder_Reference_440HzTone` covers the decoder; full mic→text browser E2E not yet a green test |
| `/depth-voxel` | Depth → 3D voxels | 🟡 **PARTIAL** | Depth pipeline runs (see `/depth`); the 3D voxel viewer UI is still placeholder |
| `/explain` | Model explainability | 🟡 **PARTIAL** | Intercepts the executor; works on a limited set of models |
| `/assistant` | AI assistant (chat) | 🟡 **PARTIAL** | Real DistilGPT-2 when a model is loaded; **falls back to `GetPlaceholderResponse` when none loaded** |
| `/comic-chat` | Multi-character chat | 🟡 **PARTIAL** | Same as assistant — real text-gen when loaded, `GetPlaceholderComicResponse` otherwise |
| `/generate` | Image generation (SD-Turbo) | 🚧 **WIP** | Pipeline wired, but only diffusion **math** is tested (`Diffusion_BetaSchedule`, `_GaussianNoise_Statistics`) — **no SD-Turbo end-to-end image test** |
| `/voice-collab` | Voice collaboration | 🚧 **WIP** | "Phase 1: Web Speech API" (browser built-in); the **GPU Whisper option is disabled**. Not the on-device GPU voice stack the name implies |
| `/image-to-3d` | Image → 3D model | 🚧 **WIP** | `GenerateModel()` is a **no-op** (`=> Task.CompletedTask`); `DownloadMesh()`/`OpenInSpawnScene()` are empty. The button does nothing yet |
| `/train` | On-device training | 🟡 **PARTIAL** | `TrainableModel.TrainStepAsync` present; flag as PARTIAL until a green end-to-end training test is cited here |
| `/` | Home | Meta | Landing page (operator count now rendered live from the registry) |
| `/tests` | Test runner | Meta | Hosts the PlaywrightMultiTest UI |
| `/models` | Model browser | Meta | HuggingFace hub browser |
| `/getting-started` | Getting started | Doc | Install + first-run walkthrough |

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
| DistilGPT-2 / DistilBERT | ✅ | ✅ | text-gen + embeddings verified |
| Whisper | ✅ | 🟡 decoder only | full speech E2E pending |
| SpeechT5 (TTS) | ✅ | 🟡 | `Pipeline_TTS_ReferenceTokensProduceAudio`; not wired into a verified demo page |
| SD-Turbo | ✅ | 🚧 | no end-to-end image test |
| GGUF LLMs (Qwen/Gemma/Llama) | ✅ **runs** (desktop) | 🟡 kernels verified | **Autoregressive KV-cache decode VERIFIED** — Example 06 Ollama-compatible server (OpenAI/Ollama/Anthropic APIs, Claude CLI) E2E on CUDA/OpenCL; Ollama-oracle byte-identical; ~51 tok/s on qwen2.5-coder:7b Q4_K_M (4070). Browser: decode kernels pass PMT on WebGPU (`GGUFDecodeKVCache` incremental==full-recompute), full in-browser LLM demo is WIP |

## Keeping this honest
- **Operator count** is rendered live from `OperatorRegistry.BuiltinOpTypes` (the documented single source of truth) — never hardcode it again.
- **Before any "N tests passing" claim**, cite the latest PMT results JSON, not a memorized number.
- **A demo graduates to ✅ VERIFIED only when a passing E2E test is cited here.** Adding a page is not the same as verifying it.
