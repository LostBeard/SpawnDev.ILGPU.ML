# Anima DiT + LLM adapter - implementation spec (for the C# / ILGPU port)

Model: Anima (circlestone-labs) = Cosmos-Predict2-2B MiniTrainDIT + Anima's own "LLM adapter" (`net.llm_adapter.*`), one file
`anima-preview3-base.safetensors` (685 tensors, all BF16, names `net.*`). Every convention below was checked numerically against
Lore's ComfyUI black-box taps (fp32 math on bf16 weights) with the float64 numpy reference in this folder
(`anima_ref.py`, checked by `check_anima.py`; numbers at the end). DiT semantics came from Diffusers `transformer_cosmos.py`
(Apache-2.0); the adapter was derived ONLY from the weight header and the taps (no source read; ComfyUI is GPL-3 and was not opened).

Notation: `Linear(x, W[, b])` = `x @ W^T (+ b)`, W stored `[out, in]` as in the file. All weights are BF16 in the file; widen to fp32
exactly (`bits << 16`). Nothing has a bias unless a `.bias` tensor is named.

## 0. Things that differ from a naive reading (read these first)

1. **The DiT timestep is sigma itself** (0.9999667, 0.9, ...), not sigma*1000. The sinusoid puts **cos first, then sin**.
2. **The dumped ComfyUI "output" is the DENOISED latent `x - sigma * v`**, not the DiT output. The DiT itself returns the velocity
   `v`. (Comparing `v` to the dump directly gives relRMS 3.8; `x - sigma*v` matches to 2.4e-6.)
3. **q/k RMSNorm eps = 1e-6** in the DiT. Diffusers' `Attention` default (1e-5, because `CosmosTransformerBlock` does not pass eps)
   is WRONG for this checkpoint: it costs 2.3e-4 relRMS per block.
4. **DiT RoPE pairs dim j with dim j+64** inside each 128-wide head (rotate-half), NOT (2j, 2j+1). ComfyUI's `[1024, 64, 2, 2]`
   rotation matrices are indexed by the pair j and act on the vector `(x[j], x[j+64])`. Interleaved pairing gives relRMS 0.39.
5. **DiT RoPE uses an NTK-scaled theta for rows and columns**: `theta_hw = 10000 * 4^(42/40)` (= rope_scale 4 for h and w), not
   Diffusers' default rope_scale (2, 1, 1). Pairs 0..21 (temporal) are the identity for images (t = 0).
6. **The context is padded to 512 rows with zeros and the padding is NOT masked.** Zero rows give k = v = 0 (score 0) and still take
   softmax mass in every DiT cross-attention. Attending over only the real rows gives relRMS 1.5 on block 0. Pad each batch row
   separately (call_0: row 0 has 27 real rows, row 1 has 26).
7. **The residual stream is huge**: |x| reaches 2.46e5 at block 13's input (1.1e4 at block 27). That is past the fp16 range (65504),
   so the residual cannot be fp16. Recommendation (not measured here): keep the residual and LayerNorm statistics in fp32 and compute
   the variance two-pass (mean first) or Welford rather than E[x^2]-E[x]^2, which cancels badly at that magnitude.
8. **Adapter: out_proj runs BEFORE the final norm** (`norm(out_proj(x))`), and `in_proj` is the Identity (no weights in the file).
9. **Adapter cross-attention applies RoPE to both sides**: q with the T5-side positions 0..M-1, k with the Qwen3-side positions
   0..N-1. Without it relRMS 0.04 per block.
10. Activations: GELU is the **exact erf GELU** everywhere (DiT MLP and adapter MLP); tanh GELU costs 3.7e-4 (DiT) / 1.2e-4 (adapter).
    The t-embedder uses SiLU.

## 1. Shapes and constants

| | value |
|---|---|
| DiT width D | 2048 |
| heads x head_dim | 16 x 128 |
| blocks | 28 |
| MLP | 2048 -> 8192 -> 2048, GELU exact, no bias |
| adaLN-LoRA rank | 256 |
| context | 512 x 1024 (padded with zero rows) |
| patch | (p_t, p_h, p_w) = (1, 2, 2); in 16 + 1 mask channel = 17 -> 68 features |
| latent | [B, 16, T, H, W]; 512x512 image -> [B, 16, 1, 64, 64] -> 32 x 32 = 1024 tokens |
| adapter width | 1024, 16 heads x 64, 6 blocks, MLP 1024 -> 4096 -> 1024 with bias |
| eps | every RMSNorm and LayerNorm in the model: 1e-6 |

## 2. DiT forward

Inputs: `x [B,16,T,H,W]` (fp32 latent as the sampler hands it, no scaling), `sigma [B]`, `ctx [B,512,1024]`.
Output: `v [B,16,T,H,W]`. ComfyUI's denoised prediction is `x - sigma[b] * v`.

### 2.1 Padding-mask channel + patch embed (`net.x_embedder.proj.1.weight` [2048, 68])
- `x17 = concat(x, zeros[B,1,T,H,W])` along channels (channel 16 = padding mask = 0 for a full image).
- Token `(t, hp, wp)` (Hp = H/2, Wp = W/2), token index `L = (t*Hp + hp)*Wp + wp`.
- Feature index `f = c*4 + ipt*4 + iph*2 + ipw` (order (c, p_t, p_h, p_w), c slowest; ipt = 0 here) takes
  `x17[b, c, t + ipt, 2*hp + iph, 2*wp + ipw]`.
- `h[b, L, :] = Linear(feat, W)` (no bias). Result `[B, L, 2048]`.

### 2.2 Timestep embedding
- `freq_k = exp(-ln(10000) * k / 1024)`, k = 0..1023.
- `t_proj[b] = [cos(sigma_b * freq_0..1023), sin(sigma_b * freq_0..1023)]` (2048; cos half FIRST).
- `adaln[b] = Linear(SiLU(Linear(t_proj, net.t_embedder.1.linear_1.weight [2048,2048])), net.t_embedder.1.linear_2.weight [6144,2048])`
  -> 6144 (the shared adaLN-LoRA vector).
- `emb[b] = RMSNorm(t_proj, net.t_embedding_norm.weight [2048], eps 1e-6)` (RMSNorm of the RAW sinusoid, not of the MLP output).
- RMSNorm(x, w, eps) = `x / sqrt(mean(x^2) + eps) * w`.
- (ComfyUI's `t_embedder.1` returns `(t_proj, adaln)`; the first output is its input unchanged.)

### 2.3 RoPE (self-attention only)
Per token angles `a[L, j]`, j = 0..63:
- j = 0..21 (temporal): `t * ft[j]`, `ft[j] = theta_t^(-2j/44)`. For images t = 0, so these are 0 (identity). theta_t is not
  observable from these taps (see section 6).
- j = 22..42 (row): `hp * fh[j-22]`; j = 43..63 (column): `wp * fw[j-43]`;
  `fh[k] = fw[k] = theta_hw^(-2k/42)`, k = 0..20, `theta_hw = 10000 * 4^(42/40)` (= 42870.9385...). Compute the frequencies in at least fp32 from that formula (the taps match it to
  1.2e-7 relRMS, i.e. fp32 rounding).
- cos/sin of each angle; the matrix form is `R[L, j] = [[cos, -sin], [sin, cos]]`.
- Apply to q and k per head (128 dims), AFTER the q/k RMSNorm:
  `y[j] = x[j]*cos(a_j) - x[j+64]*sin(a_j)`, `y[j+64] = x[j+64]*cos(a_j) + x[j]*sin(a_j)`, j = 0..63.
  (Equivalently cos/sin duplicated to 128 as `cat(a, a)` and `y = x*cos + rotate_half(x)*sin`, `rotate_half(x) = cat(-x[64:], x[:64])`.)

### 2.4 Block i (x28), all weights `net.blocks.{i}.*`
For each of the three sub-layers `s` in (self_attn, cross_attn, mlp), with `ad = adaln_modulation_{s}`:
- `m = Linear(Linear(SiLU(emb), ad.1.weight [256,2048]), ad.2.weight [6144,256]) + adaln` -> 6144;
  `shift = m[0:2048], scale = m[2048:4096], gate = m[4096:6144]` (per batch row, broadcast over tokens).
- `n = LayerNorm(x, eps 1e-6, no affine) * (1 + scale) + shift`.

Then:
1. Self-attention on n (`self_attn.*`): `q = Linear(n, q_proj)`, `k = Linear(n, k_proj)`, `v = Linear(n, v_proj)` (all [2048,2048],
   no bias); split heads as `[B, L, 16, 128]` (head h = columns h*128..h*128+127); `q = RMSNorm(q, q_norm.weight [128], 1e-6)`,
   same for k with `k_norm.weight` (per head, over the 128 dims); RoPE on q and k (2.3); attention
   `softmax(q k^T / sqrt(128)) v`, no mask; merge heads back to 2048 (head-major); `o = Linear(., output_proj.weight)`.
   `x = x + gate_sa * o`.
2. Cross-attention (`cross_attn.*`), with n from the cross_attn modulation: `q = Linear(n, q_proj [2048,2048])`,
   `k = Linear(ctx, k_proj [2048,1024])`, `v = Linear(ctx, v_proj [2048,1024])`; q/k RMSNorm per head (eps 1e-6); NO RoPE;
   attention over all 512 context rows, no mask; `output_proj`. `x = x + gate_ca * o`.
3. MLP, n from the mlp modulation: `x = x + gate_mlp * Linear(GELU_exact(Linear(n, mlp.layer1.weight [8192,2048])), mlp.layer2.weight [2048,8192])`.
   GELU_exact(z) = `0.5*z*(1 + erf(z/sqrt(2)))`.

No absolute position embedding (extra_per_block_pos_emb is None), no gate on anything else, no final residual scale.

### 2.5 Final layer
- `m = Linear(Linear(SiLU(emb), net.final_layer.adaln_modulation.1.weight [256,2048]), net.final_layer.adaln_modulation.2.weight [4096,256]) + adaln[0:4096]`;
  `shift = m[0:2048]`, `scale = m[2048:4096]`.
- `y = Linear(LayerNorm(x, 1e-6, no affine) * (1 + scale) + shift, net.final_layer.linear.weight [64, 2048])` -> `[B, L, 64]`.

### 2.6 Unpatchify
`v[b, c, t*1 + ipt, 2*hp + iph, 2*wp + ipw] = y[b, L(t,hp,wp), ((iph*2 + ipw)*1 + ipt)*16 + c]`
i.e. the 64 outputs are ordered (p_h, p_w, p_t, C) with C FASTEST - the opposite of the patchify order (C slowest).

## 3. LLM adapter (`net.llm_adapter.*`)

Inputs: `qwen [1, N, 1024]` = Qwen3-0.6B's last hidden state AFTER its final norm (Qwen ids without BOS/EOS), `ids [1, M]` = T5
tokenizer ids WITH the EOS (id 1) last. Output `[1, M, 1024]` (unpadded). No attention masks anywhere.

1. `x = embed.weight[ids]` (`[32128, 1024]` row gather). `in_proj` = Identity.
2. Rotary tables (HF-Llama style, head_dim 64, base 10000): `inv_k = 10000^(-2k/64)`, k = 0..31; for positions p:
   `cos[p] = cos(cat(p*inv, p*inv))`, `sin[p]` likewise ([., 64]). T5 side: p = 0..M-1; Qwen3 side: p = 0..N-1 (same formula).
   Applied rotate-half within each 64-dim head: dims (k, k+32): `y = x*cos + rotate_half(x)*sin`, `rotate_half(x) = cat(-x[32:], x[:32])`.
3. Block i (x6), weights `net.llm_adapter.blocks.{i}.*`, pre-norm, plain residuals (no gates, no timestep):
   - `n = RMSNorm(x, norm_self_attn.weight [1024], 1e-6)`; self-attention: `q, k, v = Linear(n, {q,k,v}_proj [1024,1024])` (no bias),
     heads 16 x 64, `q = RMSNorm(q, self_attn.q_norm.weight [64], 1e-6)`, `k = RMSNorm(k, self_attn.k_norm.weight [64], 1e-6)`,
     RoPE T5-side tables on q and on k, `softmax(q k^T / 8) v`, `o_proj [1024,1024]` (no bias); `x = x + o`.
   - `n = RMSNorm(x, norm_cross_attn.weight, 1e-6)`; cross-attention: `q = Linear(n, cross_attn.q_proj)`,
     `k = Linear(qwen, cross_attn.k_proj)`, `v = Linear(qwen, cross_attn.v_proj)` (qwen states RAW, no norm; all [1024,1024], no bias);
     q/k RMSNorm per head (cross_attn.q_norm / k_norm, 1e-6); RoPE: **q with the T5-side tables (0..M-1), k with the Qwen3-side
     tables (0..N-1)**; `softmax(q k^T / 8) v`; `cross_attn.o_proj`; `x = x + o`.
   - `n = RMSNorm(x, norm_mlp.weight, 1e-6)`; `x = x + Linear(GELU_exact(Linear(n, mlp.0.weight [4096,1024], mlp.0.bias)), mlp.2.weight [1024,4096], mlp.2.bias)`.
4. `y = Linear(x, out_proj.weight [1024,1024], out_proj.bias)`; `out = RMSNorm(y, norm.weight [1024], 1e-6)`.

### 3.1 Context for the DiT
`ctx[b] = [adapter(qwen_b, ids_b) (M_b rows); zeros (512 - M_b rows)]`. ComfyUI's CFG batch is `[uncond (negative), cond (positive)]`;
call_0's row 1 rows 0..25 equal the positive adapter output exactly, rows 26..511 are 0.0; row 0 = negative (27 ids), rows 27.. = 0.

## 4. Sampler-side facts seen in the dumps (not part of the DiT)
- ComfyUI feeds the DiT `x` unscaled (CONST / flow), timestep = sigma; the model output it records = `x - sigma * v`.
- The preview run used sigmas 1, 0.9, 0.75, 0.5, 0 (flow shift 3); the first call's timestep is 0.9999667 (not 1.0).

## 5. Achieved errors (float64 reference vs ComfyUI fp32 taps)

Inputs per stage = that stage's own tapped inputs unless the row says END-TO-END / chained. relRMS = ||ours - ref|| / ||ref||.
"alt:" rows are red-checks with a wrong convention (they must be large; they prove the taps tell the conventions apart).
"delta" rows compare the block's update (out - in), because the residual stream's outliers (up to 2.46e5) hide errors in the full
output. Regenerate: `python check_anima.py` (155 s on this machine; weights are memory-mapped and widened per block) then `python results_table.py`.

| stage | max abs | relRMS | note |
|---|---|---|---|
| t_embedder.0 sinusoid (t = sigma, [cos, sin]) | 6.779e-08 | 2.063e-08 |  |
| t_embedder.1 out1 adaLN-LoRA vector [B,6144] | 1.916e-05 | 2.581e-07 |  |
| t_embedder.1 out0 (= its input, t_proj) | 6.779e-08 | 2.063e-08 |  |
| t_embedding_norm (RMSNorm eps 1e-6) from tapped input | 4.003e-08 | 4.243e-08 |  |
| alt: t_embedding_norm eps 1e-5 | 3.623e-06 | 8.972e-06 | should be worse |
| alt: t sinusoid with t*1000 | 2.000e+00 | 1.267e+00 | should be worse |
| x_embedder.proj (patch 1x2x2, (c,pt,ph,pw) order) | 8.948e-07 | 1.444e-07 |  |
| pos_embedder rotation matrices [1024,64,2,2] | 9.878e-07 | 1.176e-07 |  |
| alt: rope matrices with Diffusers default scale (1,1) | 1.362e+00 | 3.233e-01 | should be worse |
| blocks.0 (own tapped inputs) both rows | 4.208e-05 | 5.601e-07 |  |
| blocks.0 row 0 (uncond) | 4.099e-05 | 5.558e-07 |  |
| blocks.0 row 1 (cond) | 4.208e-05 | 5.643e-07 |  |
| blocks.0 delta (out - in) | 4.208e-05 | 5.738e-07 | \|x\|max 2.83 |
| alt: blocks.0 q/k RMSNorm eps 1e-5 (Diffusers default) | 5.107e-03 | 2.283e-04 | should be worse |
| alt: blocks.0 interleaved RoPE pairs (2j,2j+1) | 9.578e+00 | 3.906e-01 | should be worse |
| alt: blocks.0 GELU tanh | 1.296e-02 | 3.688e-04 | should be worse |
| alt: blocks.0 context NOT padded (real rows [27, 26] only) | 1.867e+01 | 1.517e+00 | should be worse |
| alt: blocks.0 adaLN LayerNorm eps 1e-5 | 3.297e-03 | 1.120e-04 | should be worse |
| blocks.13 (own tapped inputs) both rows | 1.990e-02 | 6.024e-08 |  |
| blocks.13 row 0 (uncond) | 1.407e-02 | 5.223e-08 |  |
| blocks.13 row 1 (cond) | 1.990e-02 | 6.763e-08 |  |
| blocks.13 delta (out - in) | 1.990e-02 | 5.869e-07 | \|x\|max 2.46e+05 |
| blocks.27 (own tapped inputs) both rows | 1.200e-02 | 1.405e-07 |  |
| blocks.27 row 0 (uncond) | 1.200e-02 | 1.424e-07 |  |
| blocks.27 row 1 (cond) | 1.080e-02 | 1.387e-07 |  |
| blocks.27 delta (out - in) | 1.200e-02 | 1.710e-07 | \|x\|max 1.14e+04 |
| final_layer (own tapped inputs) [2,1,32,32,64] | 1.744e-06 | 2.336e-07 |  |
| call_0 chained: after blocks.0 | 1.006e-04 | 2.535e-06 |  |
| call_0 chained: after blocks.13 | 3.918e+00 | 6.415e-06 |  |
| call_0 chained: after blocks.27 | 1.354e-01 | 6.036e-07 |  |
| call_0 chained: DiT raw output vs unpatchify(final_layer tap) | 3.989e-06 | 7.014e-07 |  |
| DiT END-TO-END call_0 denoised x - t*v vs dumped output (t=0.9999667) | 3.957e-06 | 2.380e-06 |  |
| DiT END-TO-END call_0 row 0 (uncond) denoised | 3.957e-06 | 2.690e-06 |  |
| DiT END-TO-END call_0 row 1 (cond) denoised | 2.649e-06 | 2.008e-06 |  |
| DiT END-TO-END call_0 raw v vs implied (input - output)/t | 3.957e-06 | 7.018e-07 | fp32 cancellation in the dump |
| alt: call_0 raw v vs dumped output directly | 4.929e+00 | 3.800e+00 | should be worse (dump is denoised) |
| DiT END-TO-END call_1 denoised x - t*v vs dumped output (t=0.9000000) | 2.635e-06 | 1.152e-06 |  |
| DiT END-TO-END call_1 row 0 (uncond) denoised | 2.306e-06 | 1.164e-06 |  |
| DiT END-TO-END call_1 row 1 (cond) denoised | 2.635e-06 | 1.141e-06 |  |
| DiT END-TO-END call_1 raw v vs implied (input - output)/t | 2.928e-06 | 5.693e-07 | fp32 cancellation in the dump |
| alt: call_1 raw v vs dumped output directly | 5.447e+00 | 2.819e+00 | should be worse (dump is denoised) |
| t_embedding_norm chained from t | 4.753e-08 | 4.592e-08 |  |
| pad_mask_channel(call_0 input) vs x_embedder input | 0.000e+00 | 0.000e+00 |  |
| adapter.embed (gather rows of embed.weight) | 0.000e+00 | 0.000e+00 |  |
| adapter.in_proj (Identity) | 0.000e+00 | 0.000e+00 |  |
| adapter.rotary_emb cos [1,M,64] (T5 side) | 5.909e-07 | 6.655e-08 |  |
| adapter.rotary_emb sin [1,M,64] (T5 side) | 7.195e-07 | 1.409e-07 |  |
| adapter rotary cos [1,N,64] (Qwen3 side) | 5.909e-07 | 6.080e-08 |  |
| adapter rotary sin [1,N,64] (Qwen3 side) | 6.430e-07 | 1.285e-07 |  |
| adapter.blocks.0 (own tapped inputs) | 5.514e-06 | 3.972e-07 |  |
| adapter.blocks.0 delta (out - in) | 5.514e-06 | 3.139e-07 |  |
| adapter.blocks.1 (own tapped inputs) | 2.964e-06 | 3.507e-07 |  |
| adapter.blocks.1 delta (out - in) | 2.964e-06 | 4.092e-07 |  |
| adapter.blocks.2 (own tapped inputs) | 2.579e-06 | 3.821e-07 |  |
| adapter.blocks.2 delta (out - in) | 2.579e-06 | 4.873e-07 |  |
| adapter.blocks.3 (own tapped inputs) | 2.482e-06 | 3.913e-07 |  |
| adapter.blocks.3 delta (out - in) | 2.482e-06 | 5.626e-07 |  |
| adapter.blocks.4 (own tapped inputs) | 2.917e-06 | 4.242e-07 |  |
| adapter.blocks.4 delta (out - in) | 2.917e-06 | 6.439e-07 |  |
| adapter.blocks.5 (own tapped inputs) | 8.758e-06 | 2.962e-07 |  |
| adapter.blocks.5 delta (out - in) | 8.758e-06 | 3.428e-07 |  |
| adapter.out_proj (Linear + bias) | 8.211e-05 | 1.674e-07 |  |
| adapter.norm (RMSNorm eps 1e-6, AFTER out_proj) | 1.147e-07 | 5.286e-08 |  |
| alt: adapter.blocks.0 eps 1e-5 | 4.908e-05 | 2.810e-06 | should be worse |
| alt: adapter.blocks.0 GELU tanh | 1.627e-03 | 1.227e-04 | should be worse |
| alt: adapter.blocks.0 no RoPE in cross-attn | 8.508e-01 | 4.005e-02 | should be worse |
| alt: adapter.norm eps 1e-5 | 4.098e-06 | 3.001e-06 | should be worse |
| adapter END-TO-END positive vs adapter tap out | 1.578e-06 | 1.225e-06 |  |
| adapter END-TO-END positive vs c_crossattn[1, :26] | 1.578e-06 | 1.225e-06 |  |
| adapter positive padded to 512 vs c_crossattn[1] | 1.578e-06 | 1.225e-06 |  |
| adapter END-TO-END negative vs c_crossattn[0, :27] | 1.548e-06 | 1.306e-06 |  |
| adapter negative padded to 512 vs c_crossattn[0] | 1.548e-06 | 1.306e-06 |  |

## 6. Unresolved / assumptions
- **Temporal RoPE theta** (pairs 0..21): unobservable with T = 1 (angle = 0 for any theta). `anima_ref.py` uses rope_scale_t = 1.0
  as a placeholder; it only matters for video (T > 1), which Anima does not use.
- The adapter's block-norm eps is less strongly pinned than the others because the block inputs are large (mean-square 0.6..15):
  1e-6 is best at every block (mean delta relRMS 4.6e-7; 1e-5 gives 1.8e-6, 1e-7 gives 4.9e-7, 0 gives 5.0e-7). The q/k norm and final
  norm eps (1e-6) are pinned clearly.
- Remaining errors (~1e-7 .. 2.5e-6) are consistent with the taps' own fp32 rounding: e.g. block 0 on its tapped inputs is 5.6e-7;
  swapping in our float64 x_embedder output gives 5.6e-7, our emb 6.5e-7, our adaLN-LoRA vector 2.39e-6, all three 2.54e-6
  (`_scratch/anima-ref/probe6.py`). The adaLN vector reaches 72 (fp32 ulp ~8e-6) and differs from the tap by 2.6e-7 relRMS, and
  the block amplifies that ~10x. No stage is above 1e-5.

## 7. Files
- `anima_ref.py` - the float64 reference (one function per stage; `dit_forward`, `denoise_flow`, `adapter_forward`, `pad_context`).
- `check_anima.py` - runs every stage against the taps / dumps, prints the table above, writes `_scratch/anima-ref/check_results.json`
  (`--only adapter|dit|full`, `--skip-full`).
- `st_reader.py` - safetensors header parser + memory-mapped BF16 -> float64 tensor reader (no safetensors package needed).
- `taps.py` - loads Lore's raw tap / dump files (shape and dtype from the file name).
- `dump_header.py` - prints the weight header (non-block tensors + block 0 + adapter block 0).
- `results_table.py` - renders check_results.json as the Markdown table in section 5.
