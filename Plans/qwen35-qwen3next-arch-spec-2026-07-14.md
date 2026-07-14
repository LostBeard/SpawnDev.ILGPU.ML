# qwen3.5 (Qwen3-Next / GGUF arch "qwen35") - implementation spec

Source-backed spec (llama.cpp `src/models/qwen3next.cpp` + `delta-net-base.cpp`, HF
`modeling_qwen3_next.py`, GGUF metadata). Tuvok 2026-07-14. Test blob: ollama qwen3.5:9b (6.14GB)
`~/.ollama/models/blobs/sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c`.

## Model shape (from GGUF metadata)
- arch=qwen35, embd=4096, 32 blocks, eps=1e-6 (RMSNorm everywhere), dense SwiGLU FFN (ffn dim 12288, NO MoE
  in the 9b - no `ffn_gate_inp`), lm-head TIED to token_embd (no `output.weight`).
- **Layer routing**: `full_attention_interval=4`. Layer `il` is FULL-ATTENTION iff `(il+1) % 4 == 0`
  (layers 3,7,11,15,19,23,27,31); else DELTA-NET (linear attention). Equivalently the per-layer
  `qwen35.attention.head_count_kv = [0,0,0,4,0,0,0,4,...]`: kv>0 → full-attn, kv==0 → delta-net. Route by
  tensor presence too: `blk.N.attn_qkv.weight` present → delta-net; `blk.N.attn_q.weight` present → full-attn.
- Per-layer norms: `attn_norm` (pre-mixer RMSNorm) + `post_attention_norm` (pre-FFN RMSNorm; standard, NOT a
  gemma norm-sandwich). Residuals: `h = h + mixer(attn_norm(h))`, then `h = h + ffn(post_attention_norm(h))`.

## FULL-ATTENTION layers (8: il=3,7,11,15,19,23,27,31)
Standard GQA. Tensors: attn_q[4096,8192], attn_k[4096,1024], attn_v[4096,1024], attn_q_norm[256],
attn_k_norm[256], attn_output[4096,4096].
- head_dim = 256 (key_length=value_length=256). **32 query heads** (8192/256), **4 KV heads** (1024/256),
  GQA group 8. Per-head QK-RMSNorm(256) on q and k BEFORE rope. Scale = 1/sqrt(256).
- **RoPE**: PARTIAL - `rope.dimension_count=64` (only first 64 of 256 head dims rotated), freq_base=1e7,
  NeoX. **mrope** (multimodal, sections [11,11,10]) - for TEXT-ONLY all 3 position components equal the text
  position, so mrope == standard rope on n_rot=64. (UNVERIFIED: confirm mrope→plain-rope reduction for text;
  the existing RoPE op must support n_rot<head_dim partial rotation - CHECK it does, LFM2/qwen3 used full.)
- Likely reuses the existing FusedAttention path once head_dim=256/kv=4/partial-rope are wired. attn_q being
  8192 (=32*256) not 16*256: metadata head_count=16 is misleading; treat q as 32 heads from the 8192 width.

## DELTA-NET layers (24: the rest) - Gated DeltaNet linear attention
Tensors: attn_qkv[4096,8192] (fused q+k+v), attn_gate[4096,4096] (z gate), ssm_conv1d[4,8192] (F32, k=4),
ssm_alpha[4096,32] (a proj), ssm_beta[4096,32] (b proj), ssm_a[32] (A_log), ssm_dt[32] (dt_bias),
ssm_norm[128] (gated RMSNorm), ssm_out[4096,4096] (out proj).
Dims: num_k_heads=16, head_k_dim=128; num_v_heads=32, head_v_dim=128 (state_size=128). v replicates 2x over
k (v-head j uses k-head j/2). qkv split of the 8192: q=16*128=2048, k=2048, v=32*128=4096.

Forward (per token t; batch=1):
1. `qkv = attn_qkv @ h` → [8192]; `z = attn_gate @ h` → [4096]; `a = ssm_alpha @ h` → [32]; `b = ssm_beta @ h` → [32].
2. **conv**: `mixed = silu(causal_conv1d(qkv, ssm_conv1d, k=4, causal, depthwise-per-channel))`. Conv over the
   8192 channels of the CONCATENATED [q;k;v], kernel 4, causal (pad-left 3 / zero-pad at seq start), THEN SiLU.
   → REUSE the LFM2 ShortConv/ShortConvStateCache pattern with L=4, BUT: (a) no B*x gating (plain depthwise
   conv on the raw channel), (b) add a SiLU on the output, (c) conv weight ne=[4,8192]=[k,channels] (L-contig
   → weight[c*4+k]). The conv-state cache is identical in shape (last 3 rows of the 8192-wide input).
3. split mixed → q[16,128], k[16,128], v[32,128]. **L2-normalize** q and k per head (over 128, eps 1e-6):
   `x *= rsqrt(sum(x^2)+eps)`.
4. gates: `g = -exp(ssm_a) * softplus(a + ssm_dt)` (per v-head, 32); `beta = sigmoid(b)` (per v-head, 32).
   NOTE g is a log-decay; the recurrence uses `exp(g)` (a<=1 decay). a,b are per-v-head scalars this token.
5. **recurrence** (per v-head hv in 0..31; its k-head = hv/2; state S[hv] is [head_k_dim=128, head_v_dim=128]):
   ```
   S = S * exp(g[hv])                         # decay (scalar * matrix)
   kv[v] = sum_k S[k,v] * k_t[k]              # [128]  (k_t = key of k-head hv/2)
   delta[v] = (v_t[v] - kv[v]) * beta[hv]     # [128]
   S[k,v] += k_t[k] * delta[v]                # outer-product update
   out_t[hv, v] = sum_k S[k,v] * q_t[k]       # [128]  (q_t = query of k-head hv/2)
   ```
   State S per layer = [32, 128, 128] f32 = 2MB/layer * 24 = 48MB decode cache.
6. `out` [32*128=4096] → **gated RMSNorm**: `Qwen3NextRMSNormGated(out, z) = rmsnorm(out, ssm_norm) * silu(z)`
   (ssm_norm weight [128] = per head_v_dim; applied per v-head then flattened). Then `ssm_out @ out` → [4096].

## Prefill vs decode
- Prefill: run the recurrence over all seq tokens (S starts at 0, carries across tokens), keep final S as the
  decode-start state. Conv zero-pads at seq start.
- Decode (1 token/step): S carried from prefill/prior step (per-layer state cache, analogous to
  ShortConvStateCache but holding the [32,128,128] recurrent state + the conv's last-3 input rows). The
  recurrence is INHERENTLY sequential (matches the 1-token decode step) - this is why linear attention decodes
  cheaply. Full-recompute-vs-decode equivalence gate (GGUF_DECODE_EQUIV) is the correctness test.

## Implementation plan (order)
1. Graph builder: route by tensor presence; wire the 8 full-attn layers first (head_dim 256, kv 4, partial
   rope) + dense FFN + norms. Verify partial forward (finite logits) - even if delta-net layers are stubbed.
2. Confirm the existing RoPE op supports partial n_rot=64 and mrope-for-text; fix if not (library-first).
3. New `GatedDeltaNet` op + kernel: conv (reuse ShortConv-family, add k=4+SiLU plain-conv variant), L2norm,
   the per-v-head recurrence scan, gated-RMSNorm, out_proj. A recurrent-state cache (new, like
   ShortConvStateCache but for the [32,128,128] state) for decode.
4. Numpy reference (like lfm2_ref.py) to bit-verify per-layer; then GGUF_DECODE_EQUIV gate on CUDA/OpenCL.
5. Vision tower (v.blk.*) LAST - only for multimodal; text gen ignores it.

## Open/UNVERIFIED (confirm before/while implementing)
- mrope→plain-rope reduction for text-only (sections [11,11,10], n_rot 64). Read the ggml mrope op / HF
  get_rope_index for the text case.
- Exact `fix_query_key_value_ordering` layout in the GGUF attn_qkv (is it [q|k|v] contiguous, or interleaved
  per head?). llama.cpp's split in build_qkvz is the ground truth - re-read it.
- Whether the conv SiLU is inside the conv op or separate; and whether conv applies to [q;k;v] or includes z.
  HF cats (q,k,v) only (z is separate) - conv_dim = 2048+2048+4096 = 8192 = ssm_conv1d channels ✓.
- `ggml_gated_delta_net` fused-kernel exact form (CUDA/metal) as a cross-check on the HF recurrence above.
