# Pipelines Section — "show how, not just show off"

**Goal (TJ):** a Pipelines section with a demo for EVERY pipeline we support (and plan to), benchmarked against
Transformers.js. Each demo (1) runs the pipeline live and (2) shows a **minimal how-to** — ideally the actual
demo/pipeline code. Where an existing demo IS a pipeline demo, MOVE it into the Pipelines section and add the
how-to panel (don't rebuild it).

**Standing mandate:** match or BEAT Transformers.js pipeline coverage. This plan maps their full task list to ours.

---

## 1. Coverage matrix — Transformers.js task → our pipeline → our demo

Status: ✅ have demo · 🟦 have pipeline CLASS, NO demo (gap = wire a demo) · 🏆 we BEAT (they don't have it) · — n/a

### NLP
| Transformers.js task | Our pipeline class | Demo route | Status |
|---|---|---|---|
| `feature-extraction` | `FeatureExtractionPipeline` | `/embeddings` | ✅ |
| `sentence-similarity` | `FeatureExtractionPipeline` (cosine) | `/embeddings` (Similarity tab) | ✅ |
| `text-classification` / `sentiment-analysis` | `TextClassificationPipeline` + `NLPPipelines` | — | 🟦 gap |
| `text-generation` | `TextGenerationPipeline` (ONNX) + `GgufTextGenerationPipeline` (GGUF) | `/text-gen`, `/ai-chat` | 🏆 (GGUF LLM chat — they have no GGUF/llama.cpp-class local LLM) |
| `text2text-generation` | — (T5-style not wired) | — | 🟦 gap (low priority) |
| `fill-mask` | `NLPPipelines` (`fill-mask`) | — | 🟦 gap |
| `question-answering` | `NLPPipelines` (`question-answering`) | — | 🟦 gap |
| `summarization` | `NLPPipelines` (`summarization`) | — | 🟦 gap |
| `token-classification` / `ner` | `NLPPipelines` (`token-classification`) | — | 🟦 gap |
| `translation` | `NLPPipelines` (`translation`) | — | 🟦 gap |
| `zero-shot-classification` | `ZeroShotClassificationPipeline` | `/clip` | ✅ |

### Vision
| Transformers.js task | Our pipeline class | Demo route | Status |
|---|---|---|---|
| `image-classification` | `ClassificationPipeline` | `/classify` | ✅ |
| `object-detection` | `ObjectDetectionPipeline` | `/detect` | ✅ |
| `depth-estimation` | `DepthEstimationPipeline` / `AsyncDepthPipeline` | `/depth`, `/depth-voxel` | 🏆 (depth → live 3D voxels) |
| `background-removal` | `BackgroundRemovalPipeline` | `/remove-bg` | ✅ |
| `image-to-image` | `StyleTransferPipeline`, `SuperResolutionPipeline`/`SuperResGPUPipeline` | `/style`, `/super-res` | ✅ (two flavors) |
| `image-segmentation` | `NLPPipelines` (`image-segmentation`) — RMBG is mask-based | — | 🟦 gap (dedicated seg demo) |
| `image-feature-extraction` | CLIP vision (in `ZeroShotClassificationPipeline`) | partial | 🟦 gap |

### Audio
| Transformers.js task | Our pipeline class | Demo route | Status |
|---|---|---|---|
| `automatic-speech-recognition` | `SpeechRecognitionPipeline` | `/whisper` | ✅ |
| `text-to-speech` | `TextToSpeechPipeline` (SpeechT5) | — (not wired to a page) | 🟦 gap |
| `audio-classification` | `AudioPipelines` (partial) | — | 🟦 gap (verify support) |

### Multimodal
| Transformers.js task | Our pipeline class | Demo route | Status |
|---|---|---|---|
| `zero-shot-image-classification` | `ZeroShotClassificationPipeline` (CLIP) | `/clip` | ✅ |
| `image-to-text` | `Gemma4MultimodalPipeline` (image+audio+text → text) | `/gemma-chat` | 🏆 (Transformers.js has NO image-text-to-text pipeline; ours does image+AUDIO+text) |
| `document-question-answering` | — | — | 🟦 gap (low priority) |
| `zero-shot-object-detection` | — (OWL-ViT) | — | 🟦 gap |
| `zero-shot-audio-classification` | — (CLAP) | — | 🟦 gap (low priority) |

### Pipelines WE have that Transformers.js does NOT (pure 🏆 — lead with these)
| Capability | Our pipeline class | Demo route | Notes |
|---|---|---|---|
| Text-to-image (Stable Diffusion / SD-Turbo) | `ImageGenerationPipeline` / `DiffusionPipeline` | `/generate` | WIP (Coming Soon) but the pipeline class is real |
| Image → 3D (mesh / gaussian splat) | `Image3DPipeline` | `/image-to-3d` | WIP; SpawnScene render is end-to-end-only-we-can-do |
| Pose estimation (MoveNet) | `PoseEstimationPipeline` | `/pose` | not a Transformers.js task |
| Face detection + landmarks (BlazeFace) | `FaceDetectionPipeline` | `/face` | not a Transformers.js task |
| On-device TRAINING (backprop on GPU) | (training engine) | `/train` | Transformers.js is inference-only |

**Headline for the section:** we match Transformers.js's pipeline surface AND go beyond it — local GGUF LLM
chat, depth→3D, multimodal image+audio chat, image generation, image-to-3D, and on-device training.

---

## 2. Section design

**Nav:** a new top-level **"Pipelines"** section (the existing Vision/Language/Generative groups become the
sub-grouping WITHIN it, or stay as-is and Pipelines is a landing index — see below). Plus a **Pipelines landing
page** (`/pipelines`) styled like a task grid (à la the HF tasks page): each card = task name + modality + status
badge (✅/🏆/Coming soon) + "Try it" → the demo. The 🏆 ones get a "beyond Transformers.js" ribbon.

**Each pipeline demo page gains a "How to" panel** (collapsible, below the live demo): the minimal code to run
that pipeline. Two-tab: **(a) our API** and **(b) Transformers.js equivalent** side-by-side, so the parity/lead is
explicit. Example for image-classification:
```csharp
// SpawnDev.ILGPU.ML — runs on YOUR GPU via WebGPU, no server, no ONNX Runtime
using var session = InferenceSession.CreateFromFile(accelerator, modelBytes);
var pipe = new ClassificationPipeline(session, accelerator);
var (labels, ms) = await pipe.ClassifyAsync(rgbaPixels, w, h);
```
```js
// Transformers.js
const pipe = await pipeline('image-classification', 'Xenova/...', { device: 'webgpu' });
const out = await pipe(imageUrl);
```
Source the C# snippet from the ACTUAL page where possible (a `[PipelineSnippet]` region the page renders) so the
"how-to" can't drift from reality (honesty-pass principle).

**Unified factory (design lever — makes every how-to a one-liner + directly matches Transformers.js):** consider a
`Pipelines.CreateAsync(task, modelRefOrStream, opts)` Transformers.js-shaped entry that table-dispatches on the
task string (we ALREADY did this for `GgufTextGenerationPipeline` + `NLPPipelines` has the task strings). If we
expose `await Pipelines.CreateAsync("image-classification", model)` the how-to code becomes identical in shape to
theirs — the strongest "match or beat" story. Scope this as its own task; the section can ship with class-based
snippets first and adopt the factory as it lands.

---

## 3. MOVE existing demos into Pipelines (TJ: "if existing demos fit, move it in there")

These pages ARE pipeline demos — relabel under Pipelines + add the How-to panel; do NOT rebuild:
`/classify` (image-classification), `/detect` (object-detection), `/depth` (depth-estimation), `/remove-bg`
(background-removal), `/style` (image-to-image), `/super-res` (image-to-image), `/clip` (zero-shot-image /
zero-shot-classification), `/embeddings` (feature-extraction + sentence-similarity), `/whisper` (ASR),
`/text-gen` + `/ai-chat` (text-generation), `/gemma-chat` (image-to-text/multimodal), `/pose`, `/face`,
`/generate`, `/image-to-3d`, `/depth-voxel`, `/train`.

(`/text-gen` is redundant with `/ai-chat` for text-generation — candidate to retire OR keep as the ONNX-path
example vs `/ai-chat`'s GGUF path. TJ's call.)

## 4. NEW demos to wire (the 🟦 gaps — pipeline class/tasks exist, just no page)
Priority order (cheapest + highest "show how" value first), all reuse `NLPPipelines` task strings:
1. **Text classification / sentiment** (`text-classification`) — tiny, instant, classic Transformers.js parity demo.
2. **Fill-mask** (`fill-mask`) — quick, visual, parity.
3. **Question answering** (`question-answering`) — context box + question.
4. **Token classification / NER** (`token-classification`) — highlight entities.
5. **Summarization** (`summarization`).
6. **Translation** (`translation`).
7. **Text-to-speech** (`text-to-speech`, SpeechT5) — pipeline exists, wire audio output.
8. **Image segmentation** (`image-segmentation`) — dedicated seg demo (beyond RMBG).
Lower priority / needs a model: text2text-generation, image-feature-extraction, audio-classification,
zero-shot-object-detection (OWL-ViT), document-QA, zero-shot-audio (CLAP).

### ⚠ PREREQUISITE found 2026-06-30 — WordPiece tokenizer (blocks the BERT NLP demos)
We have **BPETokenizer + SentencePieceTokenizer** but **NO WordPiece** tokenizer. `TokenizerLoader` parses
`tokenizer.json` / `vocab.json+merges.txt` but always builds a **BPE** tokenizer (`NLPPipelines.cs`
`TextClassificationPipeline.ClassifySimpleAsync` literally comments "For real use, a WordPiece tokenizer should
be used"; `/embeddings` ships a hash-based `SimpleTokenize` = approximate results, a standing honesty caveat).
So the BERT-family gap demos — **sentiment/text-classification, fill-mask, question-answering, NER** — CANNOT be
demoed honestly yet (they'd produce meaningless tokens → garbage labels). **Do not ship them with fake
tokenization.** Prerequisite (Rule 2, fix the library): add a real **WordPiece** tokenizer + wire
`TokenizerLoader` to build it when `tokenizer.json` `model.type == "WordPiece"`; then these demos (and honest
`/embeddings`) become feasible. Demos that DON'T need WordPiece can go first: translation (NLLB → SentencePiece),
summarization (DistilBART → BPE), TTS (SpeechT5), image-segmentation. The text-generation / GGUF demos are
already honest (BPE/SentencePiece via GGUF).

## 5. Sequencing
1. Build the `/pipelines` landing grid + the reusable **How-to panel component** (renders our C# + the TJS
   equivalent; pulls the C# from a page-declared snippet so it stays honest).
2. Add the How-to panel to the existing pipeline demos + group them under Pipelines (the "move in" step).
3. Wire the NLP gap demos (1–6 above — they share one text-in/result-out shell + the task string).
4. TTS + segmentation demos.
5. (Parallel track) the `Pipelines.CreateAsync(task,...)` unified factory → collapse every how-to to a one-liner.

**Honesty:** mark each card with its real status (✅ verified / 🏆 / Coming soon) per `Docs/DEMO_AND_MODEL_STATUS.md`
— the Pipelines grid is the natural home for that truth, and ties into the badge/banner work already shipped.
