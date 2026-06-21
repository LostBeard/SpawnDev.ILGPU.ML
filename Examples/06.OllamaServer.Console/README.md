# Example 06 — Ollama-compatible inference server

Turn **SpawnDev.ILGPU.ML** into a drop-in **Ollama** replacement: a native-GPU GGUF inference server that
prebuilt agentic frontends (Claude CLI, Pi, Codex, OpenCode, Continue, …) can talk to with zero
reconfiguration. It serves the GGUF models **already in your Ollama cache, zero-copy** — no re-downloading,
no duplicate files.

## Run

```bash
# Serves on http://localhost:11434 (Ollama's default). Use OLLAMA_PORT to change it.
dotnet run --project Examples/06.OllamaServer.Console -c Release

# Utility subcommands:
dotnet run --project Examples/06.OllamaServer.Console -- --list                 # list servable cached models
dotnet run --project Examples/06.OllamaServer.Console -- --chat gemma4:12b "Hi" # one-shot CLI generate
dotnet run --project Examples/06.OllamaServer.Console -- --template qwen2.5-coder:latest  # dump chat template
```

It reads `~/.ollama/models` (or `$OLLAMA_MODELS`) — the same store the real `ollama` CLI uses. If real Ollama
is already bound to 11434, run ours on another port (`OLLAMA_PORT=11435`).

## Point a client at it

| Client | How |
|---|---|
| **Claude CLI** (Anthropic API) | `ANTHROPIC_BASE_URL=http://localhost:11434  ANTHROPIC_MODEL=gemma4:12b  claude` |
| **OpenAI-compatible** (Codex, OpenCode, Continue, …) | base URL `http://localhost:11434/v1`, any API key, model = a cached name |
| **Ollama-aware** | `OLLAMA_HOST=http://localhost:11434` |

Any auth token/key is accepted (local inference — there's nothing to authenticate).

## Endpoints

| Protocol | Endpoints | Streaming |
|---|---|---|
| OpenAI-compat | `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models` | SSE |
| Ollama-native | `POST /api/chat`, `POST /api/generate`, `GET /api/tags`, `GET /api/version` | NDJSON |
| Anthropic Messages (Claude CLI) | `POST /v1/messages`, `POST /v1/messages/count_tokens` | SSE |

All three drive the same engine: the model is loaded once (lazily, on first request) and generation is
serialized through one gate (`InferenceSession` is single-decode-at-a-time — same as Ollama on one GPU).

## Model support (v1)

Works today (verified end-to-end): **gemma4** (SentencePiece tokenizer), **qwen2 / qwen2.5-coder / qwen3**
and **llama3** (byte-level BPE tokenizer). The server auto-detects the chat format (ChatML / Llama3 / gemma4)
from each model's own `tokenizer.chat_template` and the tokenizer family from `tokenizer.ggml.model`.

Quantization: Q4_0/1, Q5_0/1, Q8_0, Q2_K–Q6_K, MXFP4. IQ-quants are not yet supported (the loader will report
an error rather than serve garbage).

**v1 is text-only.** Tool/function-calling, vision (the mmproj is already read from the cache), and a full
Jinja2-from-GGUF template renderer are on the v2 roadmap.

## How it fits

This Example is also a **stress-test harness** for SpawnDev.ILGPU.ML: a real agentic frontend exercises the
full inference pipeline far harder than a demo, and surfaces real bugs (it already caught + drove fixes for a
quantized-FusedLinear crash and byte-level-BPE tokenization). Make the engine correct and fast here, and every
consumer — including the browser PWA path — benefits.

---
Part of the SpawnDev.ILGPU.ML examples. See the repo root for the full crew + project docs.
