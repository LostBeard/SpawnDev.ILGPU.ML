# GGUF decode KV-cache — design plan (2026-06-12, Tuvok)

**Status:** scoped, not started. Written while the PMT harness was occupied (Seven running). Execute
when the harness frees + after a coordination note to Seven (touches `QuantizedKVCache`).

**Context:** the gemma4:12b GGUF forward is now CORRECT end-to-end (see memory
`project-gemma4-forward-correct-2026-06-12`). Next is PERFORMANCE (Rule 4). `Examples/04` greedy decode
currently runs ~7 s/token: every step re-feeds the WHOLE growing sequence and recomputes all 48 layers'
attention over all positions — O(n²) compute + a graph recompile per length. KV-cache makes it O(n).

## Current state (measured / read)
- `GGUFGraphBuilder` emits a FLAT full-sequence graph: `FusedAttention` over [1..seq], `kv_offset = 0`,
  no `past_key_values`/`present` graph I/O.
- `InferenceSession.RunAsync` + dynamic-shape recompile re-runs the full graph each step (the verbose log
  shows "Recompiled for shapes [input_ids:[1,N]]" every token).
- Existing KV-cache infra (`KVCacheAnalyzer` + `QuantizedKVCache`, `GraphExecutor._kvCache`) is the
  **ONNX explicit** pattern: it detects `past_key_values.N.key/value` INPUTS + `present.N.key/value`
  OUTPUTS in the graph and TurboQuant-quantizes the present outputs. Built for EXPORTED decoder models
  (GPT-2/Whisper). The GGUF builder produces none of those tensors, so this path is dormant for GGUF.

## Assets already in place (the kernel is ready)
- `FusedAttention` ALREADY supports decode shape: seqQ ≠ seqKV + `kv_offset` (Seven's
  `AttnOp_FusedAttention_..._DecodeShape` is green on all 6 backends). So Q can be [1 token] attending a
  cached K/V of [n tokens].
- `RoPE` ALREADY supports `kv_offset` (position = row/rows_per_position + kv_offset) — a 1-token decode
  step rotates by the correct absolute position.
- Per-layer SWA/global config, scale=1.0, freq_factors — all correct now.

## The gap
1. The graph recomputes K/V for ALL positions every step. Need to compute K/V for ONLY the new token and
   CONCATENATE onto a persistent per-layer K/V buffer.
2. No per-layer K/V state is carried across `RunAsync` calls for the GGUF path.

## Two approaches
**(A) ONNX-style: emit `past_key_values.N`/`present.N` in the GGUF graph.** Reuses the existing
`KVCacheAnalyzer`/cache wiring. BUT that path quantizes (TurboQuant) the cache — a correctness risk we
do NOT want for the first usable decoder — and requires the FusedAttention node to consume a past-K/V
input + concat. Heavier graph surgery; couples to Seven's QuantizedKVCache.

**(B) GGUF-native incremental decode (RECOMMENDED).** Keep K/V in FULL precision in per-layer
session-owned GPU buffers, outside the graph quantizer:
   - Add a decode mode to the GGUF executor: for step > 0, input is the single new token (seq=1).
   - Each layer computes Qcur/Kcur/Vcur for the 1 new token, appends Kcur/Vcur to the layer's persistent
     K/V buffer (grow-only, like `QuantizedKVCache`'s member-buffer pattern), then FusedAttention runs
     seqQ=1 vs seqKV=cacheLen+1 with `kv_offset = cacheLen` (RoPE on the new token uses absolute position
     = cacheLen). Prefill (step 0) fills the cache for the whole prompt in one pass.
   - No graph recompile per step (seq=1 fixed after prefill) → also kills the per-step recompile cost.
   - This is the llama.cpp decode model. Full-precision cache = no correctness risk; TurboQuant becomes an
     OPT-IN compression layer later.

## Risks / coordination
- `QuantizedKVCache.cs` is shared and Seven touched it recently (af7bcfc leak fix) — approach B AVOIDS it
  (new full-precision per-layer buffers), minimizing conflict, but post a heads-up before editing the
  executor's KV region.
- Browser backends: per-layer K/V buffers must be session-owned (dispose-after-drain rule — see CLAUDE.md
  Wasm note); grow-only buffers retired at Dispose.
- Validation: a CPU-oracle decode-vs-prefill equivalence test (greedy decode K tokens == single full
  forward argmax at each position) + the llama-server teacher-forcing harness already built in `Examples/04`.

## Step plan (execute when harness free)
1. Per-layer full-precision K/V ring/grow buffers in the GGUF executor (or a `GGUFDecodeState`).
2. Prefill pass (seq=prompt) populates the cache; capture per-layer Kcur/Vcur.
3. Decode step: seq=1 input, `kv_offset=cacheLen`, FusedAttention seqKV=cacheLen+1, append.
4. `Examples/04` GGUF_GEN switches to incremental decode; assert it matches the current full-recompute
   output token-for-token (regression guard) and measure tokens/s.
5. PMT: add a small CPU-backed decode-equivalence test at gemma4 geometry (synthetic tiny model).
6. Later/optional: route the cache through TurboQuant as opt-in compression (re-enables QuantizedKVCache).
