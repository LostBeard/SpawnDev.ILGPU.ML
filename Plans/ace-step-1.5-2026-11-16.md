# ACE-Step 1.5 (music) on ILGPU.ML: next family for SpawnGen

Decided 2026-11-16 (Tuvok; TJ: no machine over 32 GB, park models whose ComfyUI reference cannot load). ACE-Step 1.5 fits on both sides: ComfyUI's
float32 CPU reference loads in SpawnGen's 32 GB, and ours runs on the 16 GB boxes. It is also a new modality for SpawnGen.
Licence: ACE-Step/Ace-Step1.5 is MIT (LICENSE text read in the clone); ACE-Step-v1-3.5B is Apache-2.0 (Lore read it); Comfy-Org's repacks state none.

## ComfyUI template SpawnGen measures (Lore, 11-16)
`audio_ace_step_1_5_split`: acestep_v1.5_turbo (4.8 GB), DualCLIPLoader type ace with qwen_0.6b_ace15 (1.2 GB) + qwen_1.7b_ace15 (3.7 GB),
ace_1.5_vae (0.34 GB); 8 steps, cfg 1, AuraFlow shift 3, negative = ConditioningZeroOut. TextEncodeAceStepAudio1.5 has generate_audio_codes (default on):
the 1.7B LM SAMPLES audio codes (seed, cfg 2, temperature 0.85, top_p 0.9), which is not bit-reproducible outside torch. So two dumps: codes OFF
(deterministic conditioning, the clean gate) and codes ON (the recorded codes; our LM gated teacher-forced on them).

## Architecture facts (from ACE-Step 1.5's MIT source, E:/spawndev-acestep/refsrc/ACE-Step-1.5; T = acestep/models/turbo/modeling_acestep_v15_turbo.py)
Defaults from the config class; VERIFY every size against the safetensors shapes, and every behaviour against SpawnGen's ComfyUI taps (ComfyUI's own
implementation is what the template runs; this source is the port's reference, the taps are the truth).
- DiT `AceStepDiTModel` (T:1243): hidden 2048, 24 layers, 16 q heads / 8 kv heads (GQA), head 128, SwiGLU MLP 6144 no bias, no attention bias;
  RMSNorm eps 1e-6 everywhere, per-head q/k norm before RoPE. Even layers: bidirectional sliding window |i-j| <= 128 tokens; odd layers: full.
- Block: temb6 = scale_shift_table[1,6,D] + timestep_proj; chunks shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate. x += gate_msa *
  SA(RMS(x)(1+scale)+shift); x += CA(RMS(x), enc) (no adaLN, no gate, no RoPE; k_norm on encoder K); x += c_gate * MLP(RMS(x)(1+c_scale)+c_shift).
- RoPE: 1 axis, positions over patch tokens, theta 1e6, half-split (rotate_half) layout, self-attention only.
- Time: sinusoid of t * 1000, 256 dims [cos, sin]; linear_1, SiLU, linear_2 = temb; time_proj(SiLU(temb)) -> 6D. time_embed(t) + time_embed_r(t - r),
  r = t at inference (so time_embed_r sees 0). Head: scale_shift_table[1,2,D] + temb; RMS(norm_out)(1+scale)+shift.
- Input: cat(context 128 = [src_latents 64, chunk_mask 64], xt 64) = 192 ch; time padded to even; proj_in Conv1d(192->2048, k2 s2); proj_out
  ConvTranspose1d(2048->64, k2 s2), cropped. 25 latent frames per second at 48 kHz.
- Conditioning encoder: text_projector Linear 1024->2048 (caption = Qwen3-Embedding-0.6B last_hidden_state of the SFT_GEN_PROMPT template, max 256);
  lyrics = 0.6B's embed_tokens ONLY, then the DiT's own 8-layer lyric encoder (Linear 1024->2048 + bias, bidirectional layers, norm); timbre encoder
  (Linear 64->2048, 4 layers, norm, frame-0 output; text-to-music input = silence_latent[:, :750]); packed [lyrics | timbre | text]; DiT
  condition_embedder Linear 2048->2048 + bias. No masks reach the DiT's attention.
- Text-to-music context: src_latents = silence_latent tiled (silence_latent.pt ships with the DiT checkpoint), chunk_mask all ones; min 128 frames.
- Prediction: v = noise - x0, x_t = t noise + (1 - t) x0. Turbo sampler: t_i = 1 - i/8 shifted 3t/(1+2t) = [1, .9545, .9, .8333, .75, .6429, .5, .3];
  Euler x -= v (t_i - t_{i+1}); last x0 = x - v t. No CFG (distilled). ACE-Step's own pipeline adds a DCW wavelet correction on turbo by default
  (Haar, scalers 0.05 / 0.02): CHECK whether ComfyUI's template applies anything like it (expected not: its sampler is plain Euler + AuraFlow shift).
- Audio codes: the 1.7B "5 Hz LM" (Qwen3 chat template, CoT metadata then <|audio_code_N|>, N < 64000) -> ResidualFSQ levels [8,8,8,5,5,5]
  get_output_from_indices -> AudioTokenDetokenizer (Linear, repeat x5 + special_tokens, 2 layers over each 5-group, norm, Linear 2048->64) -> 25 Hz
  hints that replace src_latents (is_covers). FSQ decode lives in vector_quantize_pytorch (not in the clone): read the file's quantizer tensors.
- VAE: diffusers AutoencoderOobleck (Snake activations x + 1/(exp(b)+1e-9) sin^2(exp(a) x); residual units dilations 1/3/9; weight-norm convs to
  fuse; decoder Conv k7 64->2048, blocks, Snake, Conv k7 -> 2 ch, no bias). Stereo 48 kHz, 64 latent channels, no latent scale / shift. Hop: 1920
  per the handler vs 2048 from the MLX defaults: take the ratios from the VAE file. Peak-normalised after decode only if the peak exceeds 1.

## Measured
- Files verified against the facts: DiT 24 layers, GQA 16/8 x 128, SwiGLU 6144, time_embed + time_embed_r, proj_in Conv1d [2048, 192, 2]; FSQ project_in [6, 2048];
  Qwen3 0.6B (28 layers, vocab 151669, width 1024) and 1.7B (28 layers, vocab 217204 = Qwen + audio codes, width 2048), both with per-head q/k norm.
  VAE decoder: ConvTranspose kernels 20 / 12 / 8 / 8 / 4 = strides 10 / 6 / 4 / 4 / 2 = 1920 samples per frame (Lore's audio: 480000 / 250).
- 11-16 23:52 CUDA: `OobleckDecoder` vs ComfyUI's VAEDecodeAudio 3.1e-6 (peaks equal) (`AceStep_VaeDecode_MatchesComfyUi`). Stage 2 done.

## Stages (each gated against SpawnGen's ComfyUI dumps, parity first)
1. Files: the four ComfyUI files (models host mirror, standing go); read tensor names / shapes / dtypes; map to the facts above.
2. VAE decode (smallest, independent): ComfyUI's latent in -> audio out.
3. Text side, codes off: Qwen3-Embedding-0.6B last hidden state (our Qwen3 path) + lyric embeddings; the conditioning tensor ComfyUI hands the model.
4. DiT: the call-0 taps, then both calls of the 2-step graph.
5. Codes on: the 1.7B LM teacher-forced on the recorded codes; FSQ + detokenizer -> hints.
6. Sampler + whole graph vs ComfyUI's final latent and audio.
