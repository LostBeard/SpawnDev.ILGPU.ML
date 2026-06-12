# Gemma 4 (gemma4:12b) GGUF Bring-Up Plan

**Date:** 2026-06-11
**Author:** Tuvok (lead editor, all lanes)
**Status:** Investigation complete — architecture fully characterized from the real GGUF. Implementation not started.
**Target:** `gemma4:12b` from the Ollama library (Q4_K_M, ~7.38 GB main blob).

> Method note: every fact below was read from the actual GGUF metadata + tensor table over the
> network (range GET of the Ollama blob), NOT from memory. gemma4 is post knowledge-cutoff.
> Inspector script: `D:\users\tj\Projects\_gguf_header_dump.cs` (seed of the Ollama-GGUF tooling).

---

## 1. Verified model facts

| Property | Value |
|---|---|
| `general.architecture` | **`gemma4`** |
| Blob | `sha256:1278394b…` 7,381,382,048 bytes; vision projector `mmproj-gemma-4-12B-it-bf16.gguf` (175 MB) is a separate layer |
| Modality | Text + Image (multimodal). **This plan is TEXT-ONLY**; vision/mmproj is a separate milestone. |
| Quantization | Q4_K_M (`file_type=15`) → 284 × Q4_K + 45 × Q6_K + 338 × F32 (norms/scales) |
| Layers | 48 |
| Embedding dim | 3840 |
| FFN intermediate | 15360 (GeGLU) |
| Query heads | 16 |
| Vocab | 262144 (SentencePiece; tokenizer model = `gemma4`; embedded tokens/scores/merges + Jinja `chat_template`) |
| Context | 262144 |
| RMSNorm eps | 1e-6 |
| Final logit soft-cap | **30** (`logits = 30·tanh(logits/30)`) — no attention soft-cap |
| LM head | **Tied** to `token_embd` (no `output.weight`) |

### Interleaved attention (the defining structural feature)

`sliding_window_pattern = [T,T,T,T,T,F, …]` repeating over 48 layers → **5 sliding : 1 global**.
Global layers at indices **5, 11, 17, 23, 29, 35, 41, 47** (8 global, 40 sliding).

| | Sliding layers (40) | Global layers (8) |
|---|---|---|
| Attention | windowed, `sliding_window = 1024` | full causal |
| KV heads (`head_count_kv[i]`) | **8** | **1** |
| Head dim (`key_length`/`value_length`) | **256** | **512** |
| RoPE base | `freq_base_swa = 10000` | `freq_base = 1e6` |
| RoPE dims | 256 | 512 |

Per-layer geometry is real and differs — confirmed by weight shapes:
- `blk.0` (sliding): `attn_q [3840,4096]`=16×256, `attn_k/v [3840,2048]`=8×256, `attn_output [4096,3840]`.
- global layers instead carry 1 KV head × 512. **Read shapes per-layer; do not compute one global head_dim.**

### Per-layer tensor inventory (17 templates)

```
attn_norm.weight            [3840]         F32   (pre-attention RMSNorm)
attn_q.weight               [3840, q*hd]   Q4_K
attn_k.weight               [3840, kv*hd]  Q4_K
attn_v.weight               [3840, kv*hd]  Q6_K
attn_q_norm.weight          [head_dim]     F32   (QK-norm on Q, pre-RoPE)
attn_k_norm.weight          [head_dim]     F32   (QK-norm on K, pre-RoPE)
attn_output.weight          [q*hd, 3840]   Q4_K
post_attention_norm.weight  [3840]         F32   (post-attention RMSNorm, before residual add)
ffn_norm.weight             [3840]         F32   (pre-FFN RMSNorm)
ffn_gate.weight             [3840, 15360]  Q4_K  (GeGLU gate)
ffn_up.weight               [3840, 15360]  Q4_K
ffn_down.weight             [15360, 3840]  Q6_K
post_ffw_norm.weight        [3840]         F32   (post-FFN RMSNorm, before residual add)
layer_output_scale.weight   [1]            F32   (per-layer scalar)
--- model-level ---
token_embd.weight           [3840, 262144] Q6_K  (tied: also the LM head)
output_norm.weight          [3840]         F32
rope_freqs.weight           [256]          F32
```

Block dataflow (Gemma 2/3 norm-sandwich):
```
h = x
h = x + post_attention_norm( Attention( attn_norm(x) ) )       # pre AND post norm around attn
y = h + post_ffw_norm( GeGLU( ffn_norm(h) ) )                   # pre AND post norm around ffn
(× layer_output_scale where applicable)
```
Attention sublayer: q = attn_q(·); k = attn_k(·); v = attn_v(·);
q = RoPE(q_norm(q)); k = RoPE(k_norm(k)); softmax(qkᵀ·scale [+ window mask]) · v; · attn_output.
Final: `logits = 30·tanh( (token_embdᵀ · output_norm(h)) / 30 )`.

---

## 2. Acquisition — hub `OllamaProxy` (Captain's architecture call)

Model acquisition belongs on the **hub server as a pluggable source layer**, beside the existing
`HuggingFaceProxy` (`hub.spawndev.com:44365`). The client stays dumb: asks the hub for a model ref,
gets a seekable stream / WebTorrent. Future sources (civitai, direct-URL, S3) drop in as more adapters
behind one interface.

Ollama manifest→blob contract (walked by hand, works end-to-end):
1. `GET https://registry.ollama.ai/v2/library/{model}/manifests/{tag}` → JSON manifest.
2. Pick the layer with mediaType `application/vnd.ollama.image.model` → its `digest`.
   (Vision models also carry `…image.projector`; ignore for text.)
3. `GET https://registry.ollama.ai/v2/library/{model}/blobs/{digest}` → the GGUF bytes
   (HTTP range supported — `206 Partial Content`, so seekable / torrent-able).

Lane: hub/WebTorrent.Server (Riker's), covered by the 2026-06-01 lead-editor-all-lanes directive.

---

## 3. Current capability — what we already have ✅

- **K-quant dequant exists ONLY on the CPU path — the GPU fused MatMul path is a LANDMINE.** ⚠
  CORRECTION (Seven, 2026-06-11, verified by reading the code): `DequantizeQ4_K`/`Q6_K` +
  `GetTensorFloat32` (CPU) are correct, BUT the production GGUF MatMul path is NOT. `ExtractWeight`
  (`GGUFGraphBuilder.cs:224-231`) stores quantized weights as RAW BYTES and **drops the GGML type**;
  `MatMulOperator` (`StructuredOperators.cs:40-46`) sends EVERY quantized weight to
  `FusedDequantMatMul`, which decodes **Q4_0 ONLY** (`FusedDequantMatMul.cs:19-23`). A Q4_K block
  (144B/256-val) decoded as Q4_0 = noise → **gemma4 produces garbage on its first quantized MatMul.**
  This was previously (wrongly) listed here as "covered" — it is gap #0 below. No end-to-end GGUF test
  exists to catch it (all GGUF tests are parse-only).
- **GGUF parser** — sync + async + streaming (`GGUFParser.cs`).
- **`GGUFGraphBuilder`** — generic decoder-only graph synthesis (embedding → per-layer norm/QKV/attn/FFN/residual → final norm → LM head).
- **RoPE kernel** (`Kernels/RoPEKernel.cs`), **SentencePiece tokenizer** + `TokenizerLoader`, decode loop, KV cache.

## 4. Gap analysis — what gemma4 needs that the generic builder doesn't model ❌

Each maps to the code site that must change. Items 3–7 are one coherent "per-layer attention config" change.

> **PROGRESS (2026-06-11):**
> - ✅ **#0** K-quant type-routed fused dequant — Seven P1 (`2bf6934`), verified by Tuvok (68/68 all backends).
> - ✅ **#11** tied-embedding LM head + compressed `Gather`-table (Q6_K embed stays in VRAM) — Seven P1.
> - ✅ **#1** arch-tag + ✅ **#8** GeGLU + ✅ **#2** `(1+weight)` RMSNorm + ✅ **#3** 4-norm sandwich +
>   ✅ **#10** logit soft-cap — Tuvok graph-wiring (`6651b73`/`c7112a5`/`c05567d`/`f814c19`/`fb2a97a`).
> - ✅ **FLOOR (2026-06-11 late, Tuvok):** true **RMSNormalization operator** + weightless variant + registration;
>   builder now emits `RMSNormalization` (not the mean-centered `LayerNormalization` it used to — a never-correct
>   path hidden by structural-only tests). CPU-oracle EXECUTION tests + a mean-centering discriminator
>   (`MLTestBase.RMSNormTests.cs`).
> - ✅ **#4 QK-norm + #5 per-layer geometry + #6 SWA/global + #7 dual-base RoPE + #9 layer_output_scale**
>   — Tuvok attention emission (RoPE+QK-norm+FusedAttention via Seven's ops `0be3092`; freq_factors via S1
>   `26b8444`; V=Kcur for global layers; weightless V-norm). Verbatim-matched to llama.cpp gemma4.cpp.
>   Structural tests in `Gemma4Tests.cs` (`Gemma4_Attn_*`).
> - ⏳ **Remaining:** gemma4 **E2E** vs a llama.cpp reference (needs the local gguf + the exact `f_attention_scale`
>   confirmed at node-bisect). Everything upstream of E2E is in + PMT-validated.

0. **⚠ BLOCKER — K-quant GPU MatMul decodes everything as Q4_0** (the §3 landmine). Carry `GGMLType`
   with the quantized bytes (the loader dict drops it) → route only implemented types to the fused kernel,
   CPU-dequant-to-F32 fallback otherwise (correct immediately); then add **Q4_K / Q6_K / Q8_0 fused
   kernels** (mandatory for a 12B model — full F32 fallback is ~48 GB). Hits on the FIRST quantized MatMul,
   so nothing else in this plan can be validated until it's fixed. **Seven volunteered to own this item**
   (kernels + loader change; he has kernel sketches in `_scratch/seven-ml-survey-2026-06-11.md`) — pending
   Captain's cross-lane ack; Tuvok owns verification + integration. Needs the first real end-to-end GGUF
   correctness test (current GGUF tests are parse-only).
1. **arch tag** — add `gemma4` to RMSNorm/activation selection (`GGUFGraphBuilder.cs:44-45`).
2. **Gemma `(1+weight)` RMSNorm** + verify the `LayerNormalization` 2-input form is true RMSNorm
   (no mean-centering) in the executor (`AddNorm` `:264-269`).
3. **4-norm sandwich** — add `post_attention_norm` + `post_ffw_norm`, applied to the sublayer output
   *before* the residual add. Builder currently does only `attn_norm` + `ffn_norm`.
4. **QK-norm** — RMSNorm on Q and K (dim = head_dim) *before* RoPE. New per-head norm step.
5. **Per-layer attention geometry** — read q/k/v/output shapes per layer; per-layer KV heads (8 vs 1)
   and head dim (256 vs 512). `head_count_kv` is a 48-element array, not a scalar.
6. **Interleaved sliding-window (1024) vs global attention** — per-layer mask selection from
   `sliding_window_pattern`. Builder does uniform full attention.
7. **Dual RoPE base** — 10000 (sliding) vs 1e6 (global), per-layer dim (256/512); consume `rope_freqs.weight`.
   Verify `RoPEKernel` accepts a configurable `freq_base`.
8. **GeGLU FFN** — gate activation is **GELU-gated**, not SiLU. Builder default `useSiLU=true` is likely
   wrong for the gemma family — verify against reference.
9. **`layer_output_scale`** — per-layer scalar multiply on the block output.
10. **Final logit soft-cap = 30** — `30·tanh(x/30)` after the LM head.
11. **Tied embeddings** — LM head reuses `token_embd` (Q6_K); dequant needed for both Gather and the
    final matmul. No `output.weight`.
12. **gemma4 tokenizer** — load SentencePiece (262K) from GGUF metadata: tokens/scores/merges + special
    ids (bos=2, eos=1, unk=3, pad=0, mask=4) + `chat_template`. Verify `TokenizerLoader` maps `gemma4`.

## 5. Honest framing vs the GPT-2 work

The decode loop, sampling, executor, quant path, and tokenizer base transfer directly. But gemma4 is the
**most architecturally elaborate decoder this project will have run** — per-layer KV/head-dim, dual-base
RoPE, sliding/global interleave, QK-norm, 4-norm sandwich, per-layer scale, and logit soft-cap are all
things GPT-2 (uniform full attention, single head dim, learned positions, no quant, no cap) never exercised.
That is exactly why it is a *good* target: it stress-tests the GGUF + executor path as the project's
"inference engine + library stress test" mandate intends.

## 6. Bring-up order

1. **Tokenizer** — load gemma4 SentencePiece from GGUF; verify encode/decode vs a llama.cpp reference on
   a fixed prompt (assert token ids, not liveness).
2. **Per-layer attention config** (gaps 3–7) as one change in `GGUFGraphBuilder` + the executor attention
   path: per-layer geometry, QK-norm, sliding/global mask, dual RoPE.
3. **Norm correctness** (gaps 2, 3) — `(1+weight)` RMSNorm + the post-norms.
4. **FFN + scale + soft-cap + tied head** (gaps 8–11).
5. **End-to-end greedy** — assert argmax token sequence matches a llama.cpp reference forward pass for a
   fixed prompt (the GPT-2 method: real reference values, full-tensor compare, relative tolerance).

## 7. Verification method

- Reference: `llama.cpp` / `ollama run` on `gemma4:12b` with deterministic settings; capture first-N
  argmax tokens + a mid-graph hidden-state dump for a fixed prompt.
- Node-bisect any divergence (compare FULL tensors, relative tolerance) — same harness that closed GPT-2.
- All correctness on CUDA first (oracle), then the backend matrix.

## 8. Open items / risks

- **RESOLVED (2026-06-11 late, Tuvok) — global-layer structure + a template-view trap.** The collapsed
  `TensorTemplates` inspector view HID per-layer shape variance (merged differently-shaped `blk.*.attn_q`
  into one row), which made the earlier §8 read backwards. Raw per-layer dump (new inspector `--tensors`
  flag, streamed from the real Ollama blob) + verbatim `llama.cpp src/models/gemma4.cpp` give the truth:
  - **Sliding layers (40):** head_dim **256**, 8 KV heads, own `attn_v` (`[3840,2048]`). RoPE base 1e4, no freq_factors.
  - **Global layers (8: 5,11,17,23,29,35,41,47):** head_dim **512**, **1** KV head, **NO `attn_v`** →
    `Vcur = wv ? wv·x : Kcur` (V reuses the RAW K projection). RoPE base 1e6 + `rope_freqs.weight` (NTK).
  - **All layers:** Q/K get QK-norm (weighted RMS over head_dim) then RoPE; **V gets a WEIGHTLESS RMS-norm**
    (`ggml_rms_norm`, no scale) and NO RoPE. `f_attention_scale` is a stored scalar (no `query_pre_attn_scalar`
    in metadata) — exact value deferred to E2E node-bisect.
  - `GetLayerAttnConfig` (`fb2a97a`) was already correct. The top-level `headDim = embedDim/nHeads = 240` was
    NOT used for the reshape — emission uses `cfg.HeadDim` per layer. Inspector gear-fix (template-variance
    split) is queued.
- `mmproj` vision path (image input) — deferred, separate milestone.
- Whether `RoPEKernel` already supports per-call `freq_base` / the `rope_freqs.weight` precompute — code-read pending.
- `layer_output_scale` exact semantics (residual scale vs logit/attn scale) — confirm against reference numerics.
- K-quant dequant numerical parity at 12B scale — validate a few tensors vs llama.cpp dequant.

## 9. Tooling follow-up — fix the gear, don't keep a parallel script

> **STATUS: DONE 2026-06-11 (uncommitted, pending 4.10.0 rebuild + PMT).** `ModelInspectorHelper` extended
> additively — `InspectionResult.Metadata` (full GGUF KV map, arrays summarized with real element type) +
> `InspectionResult.TensorTemplates` (blk.N collapsed, surfaces small norms/scales). Build clean; **validated
> end-to-end via streaming `InspectAsync` against the live gemma4 Ollama blob** (all decisive KVs + the small
> tensors now visible). Regression test `ModelInspector_GGUF_SurfacesMetadataKVsAndTensorTemplates` added
> (synthetic GGUF, asserts typed arrays + template collapse). Scratch `_gguf_header_dump.cs` retired.
> **Bonus finding:** the template view exposed `attn_v.weight ×40` (not ×48) — see §8.


We already have `ModelInspectorHelper.InspectAsync(Stream)` which reads ONLY the GGUF header (no weight
blobs) over any seekable stream incl. HTTP/WebTorrent — so inspection never needs the full 7.4 GB download.
It is the right tool, but `BuildGGUFResult` under-reports for a frontier arch in two ways the gemma4
characterization walked straight into:

1. **`LargestWeights.Take(20)`** surfaces only the big quantized matmuls; the small-but-decisive tensors
   (`attn_q_norm`/`attn_k_norm`, `post_attention_norm`, `post_ffw_norm`, `layer_output_scale`, `rope_freqs`)
   are among the smallest and never appear → QK-norm, the 4-norm sandwich, and the per-layer scale are invisible.
2. **Only ~5 summary hyperparams** are surfaced; the full GGUF metadata KV map (which carries
   `sliding_window_pattern`, per-layer `head_count_kv`, `freq_base`/`freq_base_swa`, `final_logit_softcapping`,
   `key_length`) is parsed into `GGUFModel` but never placed in `InspectionResult`.

**Action (fix the gear):** extend the inspector to surface (a) the full GGUF metadata KV map (arrays
summarized) and (b) the distinct tensor-name templates (`blk.N.*`) with per-template dtype, so small
decisive tensors are never hidden. Then **retire the throwaway `_gguf_header_dump.cs`** — its capability
belongs in `ModelInspectorHelper`, validated via the streaming `InspectAsync` path against the Ollama blob
URL (no full download). This closes a gap that benefits every future GGUF/Ollama target, not just gemma4.
