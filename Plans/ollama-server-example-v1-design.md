# Example 06: Ollama-compatible inference server (`06.OllamaServer.Console`)

**Goal (TJ, 2026-06-21):** Make SpawnDev.ILGPU.ML a drop-in replacement for Ollama so any
agentic coding CLI / LLM front-end (Codex, OpenCode, Continue, Hermes, Pi, etc.) can use our
native-GPU GGUF inference with zero reconfiguration. Reuse Ollama's existing model cache (no
redundant copies).

**v1 scope (TJ-decided):**
- General **any-GGUF** chat server (not gemma4-only) via a GGUF chat-template engine.
- Chat completions + streaming + Ollama-cache reuse + multimodal (image+text).
- **Tool / function-calling = phase 2.** Embeddings = phase 3.

This is HTTP (not WebSocket): SSE streaming on the OpenAI routes, NDJSON on the Ollama routes.

---

## Architecture

ASP.NET Core (Kestrel) minimal-API console host on **port 11434** (Ollama's default → existing
clients need no config). One inference engine behind two endpoint families. A request queue
serializes decode (single GPU session = one in-flight decode; Ollama behaves the same).

```
06.OllamaServer.Console/
  Program.cs                 — Kestrel host, DI, route registration, port 11434
  OllamaModelStore.cs        — read ~/.ollama/models cache (manifests + blobs), zero-copy
  ModelRegistry.cs           — resolve model name -> loaded InferenceSession (LRU, lazy)
  RequestQueue.cs            — serialize decode across concurrent HTTP requests
  Api/OpenAiEndpoints.cs     — /v1/chat/completions (SSE), /v1/completions, /v1/models
  Api/OllamaEndpoints.cs     — /api/chat, /api/generate, /api/tags, /api/show, /api/version
  Api/Dtos.cs                — request/response records for both protocols
```

### Endpoint map (v1)

| OpenAI-compat | Ollama-native | Backed by |
|---|---|---|
| `POST /v1/chat/completions` (SSE) | `POST /api/chat` (NDJSON) | template engine + KV-decode loop |
| `POST /v1/completions` | `POST /api/generate` | raw-prompt KV-decode loop |
| `GET /v1/models` | `GET /api/tags` | `OllamaModelStore.List()` |
| — | `POST /api/show` | `GGUFParser.ParseHeaderAsync` metadata |
| — | `GET /api/version` | static |
| (phase 3) `/v1/embeddings` | (phase 3) `/api/embeddings` | needs GGUF pooled hidden-state output |
| (phase 2) tool_calls in the above | (phase 2) | template tool-inject + output parse |

### THIRD endpoint family — Anthropic Messages API (TJ's favorite client = Claude CLI)
TJ's favorite agentic frontend is **Claude CLI (Claude Code)**, which natively speaks the **Anthropic
Messages API** (`POST /v1/messages` + Anthropic's own SSE event stream: `message_start`,
`content_block_start`, `content_block_delta`, `message_delta`, `message_stop`), NOT the OpenAI
`/v1/chat/completions` shape. So a third endpoint family is first-class, prioritized because it's his
favorite. **Pi** is "very similar to Claude CLI, works well" (verify its protocol). **Aider** is
disliked (forces a git repo / refuses to code) — deprioritize.

**VERIFIED Claude Code spec (claude-code-guide agent, sourced from code.claude.com/docs + platform.claude.com/docs, 2026-06-21):**
- **No OpenAI mode.** Claude Code ONLY speaks Anthropic Messages. It will NOT call `/v1/chat/completions`.
  LiteLLM/claude-code-proxy exist solely to bridge that gap; implementing Anthropic natively deletes them.
- **Connect:** `ANTHROPIC_BASE_URL=http://localhost:11434` (it appends `/v1/messages` etc.).
  Auth: `ANTHROPIC_AUTH_TOKEN` → `Authorization: Bearer …` (preferred) OR `ANTHROPIC_API_KEY` → `x-api-key`.
  Server must **accept arbitrary model strings** (whatever `ANTHROPIC_MODEL` is set to — base URL changes
  destination, not the model name sent). To light up the picker for a non-`claude*` id:
  `ANTHROPIC_CUSTOM_MODEL_OPTION` + `…_SUPPORTED_CAPABILITIES`.
- **Required endpoints:** `POST /v1/messages` (JSON + `"stream":true` SSE) and `POST /v1/messages/count_tokens`
  (returns `{"input_tokens":N}` — back it with `tokenizer.Encode(...).Length` over the formatted prompt).
  `GET /v1/models` OPTIONAL (only if `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`; picker only adds ids
  starting `claude`/`anthropic`).
- **Headers to accept:** `anthropic-version` (e.g. `2023-06-01`), `anthropic-beta` (accept + ignore),
  auth header. Informational (log-only): `X-Claude-Code-Session-Id` etc.
- **SSE event order (text):** `message_start` (empty content + `usage.input_tokens`) → `content_block_start`
  (`{"type":"text","text":""}`) → `content_block_delta` (`text_delta`) × N → `content_block_stop` →
  `message_delta` (`delta.stop_reason="end_turn"`, CUMULATIVE `usage.output_tokens`) → `message_stop`.
  `ping` may interleave. Each event has both an SSE `event:` name and a matching `type` in its data JSON.
- **Tool use (PHASE 2):** `content_block` type `tool_use`; deltas are `input_json_delta` carrying
  `partial_json` fragments (concatenate → the JSON arg object); final `stop_reason="tool_use"`; incoming
  requests then carry `tool_use`/`tool_result` blocks + `tools` in the body. Lines up with our phase-2
  tool-calling decision → v1 `/v1/messages` is text-only.
- **Thinking blocks:** `thinking_delta` + `signature_delta` — only if we advertise the `thinking`
  capability. v1 = don't advertise, never emit. (Our gemma4 "thought" channel is the MODEL's text, not
  Anthropic thinking blocks — keep it as plain text content in v1.)
- **Verification approach (agent's caveat):** the exact `anthropic-beta` flags + `/v1/models` discovery
  schema Claude Code emits aren't fully public. Build with `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` +
  discovery OFF first, point Claude Code at a **logging stub** to capture the REAL request bytes its
  version sends, then harden against the captured traffic (Rule 4b — capture, don't guess).

| Anthropic-compat (v1) | Backed by |
|---|---|
| `POST /v1/messages` (JSON + Anthropic SSE, text only) | template engine + KV-decode loop |
| `POST /v1/messages/count_tokens` → `{"input_tokens":N}` | `tokenizer.Encode(prompt).Length` |
| `GET /v1/models` (optional, discovery flag) | `OllamaModelStore.List()` |

---

## Verified facts (read, not guessed — 2026-06-21)

### Ollama cache layout (confirmed on TJ's machine: `C:\Users\TJ\.ollama\models`)
OCI/Docker content-addressed store:
```
~/.ollama/models/
  manifests/registry.ollama.ai/<namespace>/<model>/<tag>   — Docker manifest v2 JSON
  blobs/sha256-<hex>                                        — content-addressed layers
```
- `library/` namespace is shown by Ollama as just `<model>:<tag>` (e.g. `qwen2.5-coder:latest`).
- Manifest `layers[]` mediaTypes seen: `application/vnd.ollama.image.` + one of:
  `model` (the GGUF), `projector` (mmproj / vision encoder), `params` (JSON sampling defaults),
  `template` (Ollama Go text/template), `system` (system prompt text), `license`.
- Blob path = `blobs/sha256-<digest-with-dash-not-colon>`. Digest in manifest is `sha256:<hex>`;
  on disk the file is `sha256-<hex>`.
- Example resolved (`gemma4:12b`): model layer `sha256-1278394b…` = 7.4 GB GGUF; projector
  `sha256-675ad6e…` = mmproj-gemma-4-12B (vision); params `{"temperature":1,"top_k":64,"top_p":0.95}`.
- TJ's cache already holds: gemma4, gpt-oss, qwen2.5-coder (7b/14b), qwen3/qwen3.5/qwen3.6,
  deepseek-r1, llama3.1, nemotron-3-nano, plus custom `spawndev-coder` models.

### Chat-template source varies → engine targets GGUF `tokenizer.chat_template`
- `qwen2.5-coder` HAS an Ollama `template` layer (Go template) + `system` layer.
- `gemma4` has NO template layer — newer Ollama uses a built-in named `renderer`/`parser`
  (`config` blob: `"renderer":"gemma4","parser":"gemma4"`). Its template lives in GGUF metadata.
- **Universal source = the GGUF metadata key `tokenizer.chat_template` (Jinja2).** Embedded in
  every modern instruct GGUF. The engine renders that; per-arch hardcoded fallbacks (gemma4 done)
  cover GGUFs that lack it. Ollama `params`/`system` blobs are read as defaults/overrides.

### Library API surface to reuse (file:line in `SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/`)
- Load: `InferenceSession.CreateFromGGUFFileAsync(accel, path, onProgress?)` — `InferenceSession.cs:1545`
  (only >2 GB-capable path; streams to GPU).
- Metadata: `GGUFParser.ParseHeaderAsync(stream)` — `GGUF/GGUFParser.cs:169` → `GGUFModel`
  (`Architecture`, `Name`, `ContextLength`, `VocabSize`, generic `GetMetadataString(...)` incl.
  `tokenizer.chat_template`). `GGUF/GGUFModel.cs:9`.
- Tokenizer: `SentencePieceTokenizer.FromGGUF(gm)` — `Preprocessing/SentencePieceTokenizer.cs:167`
  (`Encode`/`Decode`, `BosId`/`EosId`, `TryGetId(token, out id)`).
- Decode (production O(n) KV path): `InferenceSession.EnableGGUFDecode(GGUFDecodeKVCache)` `:2150`
  + `RunDecodeStepAsync(dict)` `:2169` + `DecodePastLen` `:2143`. KV cache geometry via
  `GGUFGraphBuilder.GetLayerAttnConfig`. Pipeline example: `Pipelines/GgufTextGenerationPipeline.cs:61`.
- Sampling: `TextGenerationSampler` (`Greedy`/`TopK`/`TopP`/`ApplyRepetitionPenalty`/`ApplyTemperature`)
  + `GenerationConfig` — `Preprocessing/TextGenerationSampler.cs`.
- Multimodal: `Gemma4MultimodalPipeline.CreateAsync(accel, textGguf, mmproj)` +
  `GenerateAsync(prompt, images, audio, …, onToken)` — `Pipelines/Gemma4MultimodalPipeline.cs`.
  `ImageInput(byte[] Rgb, int W, int H)` (caller decodes PNG/JPEG).
- Model pull (phase: `/api/pull`): `HuggingFaceClient.DownloadFileAsync(...)` — `Hub/HuggingFaceClient.cs:161`
  (desktop-usable; `ModelHub` OPFS cache is browser-only, NOT usable here).

### Gaps to BUILD IN THE LIBRARY (general inference features — Rule 2, not workarounds)
1. **Incremental UTF-8-safe streaming detokenizer.** Today both pipelines full-re-decode the whole
   id list per token and string-diff (`GgufTextGenerationPipeline.cs:133`,
   `Gemma4MultimodalPipeline.cs:287`). A server needs a stateful piece-by-piece decoder that holds
   partial UTF-8 / partial SentencePiece pieces until a full glyph is ready.
   - **DESIGN (verified against `SentencePieceTokenizer.Decode` `:137-162`):** stateful `Push(int id)
     -> string delta`. Per token, convert to RAW BYTES and append to a pending `List<byte>`:
     control token (type==2) -> no bytes; byte-fallback `<0xHH>` -> the single literal byte;
     normal piece -> `Encoding.UTF8.GetBytes(piece.Replace('▁',' '))`. Then decode the pending
     buffer as UTF-8 up to the last COMPLETE code point, emit that delta, keep the incomplete
     trailing multi-byte sequence pending for the next token. Trim the one leading space once.
   - **LATENT BUG this also fixes (real, verified by reading `:151`):** current `Decode` does
     `sb.Append((char)b)` per byte-fallback byte = Latin-1 interpretation. A non-ASCII glyph emitted
     as a SEQUENCE of `<0xHH>` tokens (emoji/CJK/accented, e.g. "é"=0xC3 0xA9) decodes to mojibake
     ("Ã©") instead of "é". The byte-accumulate-then-UTF-8-decode design above fixes it. Build the
     incremental decoder as the correct path and route both pipelines through it (Rule 1).
2. **GGUF chat-template engine.** Render `messages[]` (system/user/assistant + multi-turn) via the
   model's `tokenizer.chat_template` (Jinja2 subset). Today `ChatTemplates` is hardcoded & gemma4-centric
   (`Preprocessing/ChatTemplates.cs`). Keep gemma4 fast-path; add the generic renderer.
3. **Arbitrary stop sequences** in the decode loop (today hardcoded gemma4 turn-close + EOS).
4. **Arch-dispatched generic text generation** above the gemma4-specific `GgufTextGenerationPipeline`.

### Build IN THE EXAMPLE (server concerns)
- `OllamaModelStore` — parse manifests + resolve blobs (zero-copy load).
- `ModelRegistry` — name → loaded `InferenceSession` (LRU; lazy load on first request).
- `RequestQueue` — serialize decode (single session). Concurrency note: `InferenceSession` is NOT
  thread-safe — `DecodePastLen`/`_decodeCache` are single mutable cursors, no locks. One in-flight
  decode at a time.
- Kestrel host + the two endpoint families.

### Concurrency reality
`InferenceSession` is single-decode-at-a-time (no locks; mutable KV cursor). v1 = one shared
session per model + a serial request queue (Ollama's default behavior too). Multi-model concurrency
later = one session per model; true intra-model concurrency would need per-request decode state.

---

## Phase plan
- **v1:** cache reader → registry/queue → template engine + incremental detok + stop sequences →
  OpenAI + Ollama chat/generate/tags/show endpoints → verify with a real coding CLI against the cache.
- **Phase 2:** tool / function-calling (per-arch tool-def injection + tool-call output parsing).
- **Phase 3:** embeddings (`/v1/embeddings`, `/api/embeddings`) — needs a pooled hidden-state output
  on the GGUF graph. `/api/pull` via `HuggingFaceClient`.

## Verification (Rule 5 — no demos handed to TJ untested)
- `OllamaModelStore`: unit-validate manifest/blob resolution against TJ's real cache (done first, no GPU).
- Template engine + incremental detok: unit tests vs. known-good full-decode reference, all 6 backends
  where GPU is involved; CPU-only for the pure-string template/detok logic.
- End-to-end: drive the running server with an HTTP client (and a real coding CLI) against a cached
  model; confirm streamed output matches the equivalent direct-pipeline generation byte-for-byte.

### Ollama as a differential oracle (TJ's idea, 2026-06-21) — correctness AND speed
We load the SAME GGUF blob real Ollama (llama.cpp) loads, so Ollama is a reference for both:
- **Correctness:** run identical prompt+model+params through real Ollama and our server.
  Do NOT expect byte-identity — our ILGPU kernels vs llama.cpp differ by tiny dequant/argmax numerics
  that accumulate at temp=0. Honest metrics: **first-divergence token index** (greedy, temp=0) and
  **per-step logit agreement** (cosine / top-k overlap of the logit vectors). A high-agreement, late-
  first-divergence run is a strong correctness signal; a 1st-token divergence flags a real bug.
- **Speed (Rule 4 ceiling):** **tokens/sec head-to-head**, same model + prompt + backend class
  (our CUDA vs Ollama CUDA). Report prompt-eval and decode tok/s separately.
- Drive both over HTTP (real Ollama on its port vs our server on 11434) with one harness so the
  comparison is the production path, not a hand-rolled loop.
