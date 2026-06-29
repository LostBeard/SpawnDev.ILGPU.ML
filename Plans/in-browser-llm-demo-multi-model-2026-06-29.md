# In-browser LLM demo + multi-model text-generation pipeline (2026-06-29)

Goal: a Blazor WASM in-browser LLM demo ("AI Chat") that runs small quantized GGUF models on WebGPU,
modeled on the Transformers.js `pipeline()` ergonomics, with a switchable model selector. Dual purpose:
the demo doubles as a library stress test for archs/quants we don't yet fully support (CLAUDE.md
"Dual Purpose" section). Local-file loading from `D:\users\tj\Projects\_Models` first; WebTorrent/hub
delivery wired afterward.

## Anchor models (in `_Models`, also ollama-cached)
| Model | Arch | Quant | Status (desktop CUDA oracle) |
|---|---|---|---|
| qwen2.5:0.5b-instruct-q8_0 | qwen2 | Q8_0 | ✅ "The capital of France is Paris." (47.8 tok/s) |
| qwen2.5:0.5b-instruct-q4_K_M | qwen2 | Q5_0+Q4_K+Q6_K+Q8_0 | ✅ after Q5_0 kernel added (this session) |
| gemma3:270m | gemma3 | Q8_0 | ❌ garbage output — gemma3 arch gap |
| smollm2:360m | llama | F16 | ⚠️ "A:" + immediate EOS — template/special-token suspect |

## Done this session
1. **Generalized `GgufTextGenerationPipeline`** (was gemma4-hardcoded) → architecture-agnostic, wraps
   `GgufGenerator`, auto-detects chat format via `ChatTemplates.DetectChatFormat`/`BuildChatPrompt`
   (ChatML/Llama3/gemma4), accepts a string OR multi-turn `(role,content)` messages, streams tokens.
   One-call factories `CreateFromFileAsync` / `CreateFromStreamAsync` (browser/hub). Transformers.js shape.
   Verified on CUDA via new Example 06 `--chat-pipe <file|name> "<prompt>"`.
2. **Q5_0 fused dequant kernel** (Rule 2 library fix; q4_K_M qwen needs it for 132 tensors). Added to
   `FusedDequantMatMul` (DecodeQ5_0Element + GEMV + M>1 GEMM + dispatch + Supports) and `FusedDequantGather`
   (case + GatherQ5_0Impl). Q5_0 = 22B/block `[d:f16][qh:u32][16 nibbles]` = Q4_0 + a 5th bit from the qh
   mask, `value=((nib|xh)-16)·d`. Oracle tests added (GEMV M=1, GEMM M>1, Gather) vs pre-existing `RefQ5_0`
   ggml reference — run across all 6 backends via PMT (`PMT_FILTER=Q5_0`).

## Transformers.js API model (research, for ongoing design)
- One `pipeline(task, model, options)` entry point; table-driven dispatch on the task string; returns a
  CALLABLE object. We already have a per-task `Pipelines/` dir (Classification, Depth, …) — the shape exists.
- `text-generation` accepts string | chat-messages | batches; applies the chat template AUTOMATICALLY;
  greedy by default; streaming via an injected callback. Per-task strongly-typed pipelines share a base —
  deliberately NO universal any-to-any type (processor-vs-tokenizer is the multimodal seam).
- Transformers.js has NO prompt+image (image-text-to-text) pipeline — our gemma4 multimodal chat is AHEAD
  of upstream. Selling point for the demo.

## Remaining
- **gemma3 arch** (task): garbage output. gemma family handled generically; gemma3 specifics to verify —
  sliding-window/local-vs-global RoPE theta, qk-norm presence, per-layer head_dim, no final logit softcap.
- **smollm2/llama-small** (task): "A:"+EOS. Dump its `tokenizer.chat_template` (Example 06 `--template`),
  compare to the ChatML we emit; check BOS/EOS special-token ids.
- **Browser PMT generate test**: prove a small GGUF generates byte-identical on WebGPU (needs the model
  served to the browser — deferred with WebTorrent/hub delivery, per TJ).
- **AI Chat demo page** (Phase 2): generalize GemmaChatPage → model selector (sizes shown), opt-in load,
  local-file path first, then hub delivery; use the pipeline. Rename "Gemma 4 Chat" → "AI Chat".
- **Honesty/polish** (Phase 3): DEMO_AND_MODEL_STATUS row, retire DistilGPT-2 `/text-gen`, NavMenu.
