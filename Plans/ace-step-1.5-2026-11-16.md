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
- 11-17 CUDA, codes off: `AceStepDiT` vs ComfyUI's decoder tap (call 0) 1.5e-4; `AceStepConditionEncoder` vs the encoder tap 6.3e-6 (lyrics 7.5e-5, timbre
  2.2e-6, text 3.8e-6) - ONLY with FULL attention: ComfyUI's implementation runs no sliding window (ACE-Step's own |i - j| <= 128 puts the 350 lyric tokens 0.47 off),
  so both default to SlidingWindow = -1. Text: the caption ids through Qwen3-0.6B (Flux2TextEncoder, last layer + final norm) 2.1e-5, with ComfyUI's padding rule: the
  prompt is real up to its FIRST <|endoftext|> (151643), every key from there on is padding (rows still output); every key valid misses at 0.10 (numpy rebuild:
  _tuvok-handoff/ace_qwen06_variants.py). Lyrics = the 0.6B's embed_tokens rows exactly (7.7e-9). Stages 3 and 4 done.
- 11-17 CUDA: the WHOLE codes-off graph (`AceStepGenerator`: ComfyUI's ids, call-0 noise and silence latent in; 2 Euler steps 1 -> 0.75 -> 0; VAE) vs ComfyUI: latent
  1.8e-4, audio 3.9e-4 (2 steps 2.8 s, decode 10.5 s for 10 s). ComfyUI's model call outputs the denoised x - sigma v; its step is plain Euler (no DCW).
- SILENCE LATENT: ComfyUI's is 4 distinct frames then ONE steady row tiled, values exactly 4-decimal (a stored rounded table); ACE-Step's MIT silence_latent.pt
  (acestep-v15-turbo, [64, 15000], downloaded to F:/spawndev-acestep/hf-acestep15) matches its first 4 frames to 4 decimals, but its later frames vary (mean within 0.009
  of the steady row). ComfyUI's table is never copied into the library: the generator takes the silence latent as an input. To match from MIT sources: port the Oobleck
  ENCODER, encode silence, round to 4 decimals, compare with ComfyUI's measured rows (also needed for audio-to-audio / covers).
- 11-17: OobleckEncoder ported (shared OobleckLayers; the decoder unchanged, 3.1e-6). Encoding digital silence gives the right STRUCTURE (4 lead-in frames, then constant)
  but is 2.5e-2 off ComfyUI's table (and off the MIT tensor's first frames, which equal ComfyUI's): either the encoder port differs or ACE-Step's silence latent was not
  made from zeros. Asked Lore for ComfyUI VAEEncodeAudio of (a) zeros, (b) ace_g0_audio. Test MLTestBase.AceStepSilenceTests.cs stays uncommitted until then.
- 11-17 CUDA: the 1.7B code LM (Flux2TextEncoder on qwen_1.7b_ace15) teacher-forced on ComfyUI's run: prompt pass 2.4e-5, the 11 tapped code steps 2.5e-6 .. 5.9e-6
  (positive row; the CFG negative row is LEFT-padded with 151643: not run yet). Stage 5a done; next 5b: sampling (CFG 2 on logits, temperature 0.85, top-p 0.9 -
  not bit-reproducible vs torch, so gated by teacher forcing) + codes -> FSQ -> detokenizer -> hints vs ComfyUI's detokenizer tap (ace_g1).

## Stages (each gated against SpawnGen's ComfyUI dumps, parity first)
1. Files: the four ComfyUI files (models host mirror, standing go); read tensor names / shapes / dtypes; map to the facts above.
2. VAE decode (smallest, independent): ComfyUI's latent in -> audio out.
3. Text side, codes off: Qwen3-Embedding-0.6B last hidden state (our Qwen3 path) + lyric embeddings; the conditioning tensor ComfyUI hands the model.
4. DiT: the call-0 taps, then both calls of the 2-step graph.
5. Codes on: the 1.7B LM teacher-forced on the recorded codes; FSQ + detokenizer -> hints.
6. Sampler + whole graph vs ComfyUI's final latent and audio.
- 11-17 CUDA: FSQ decode = digit scaled (d - (L - 1) / 2) / ((L - 1) / 2), mixed radix [8,8,8,5,5,5] first level fastest (5.9e-8 vs ComfyUI's detokenizer input; the
  half-width reading is 0.13 off). `AceStepCodeDecoder` (FSQ, project_out, detokenizer: 5-token groups, positions 0..4) vs ComfyUI's detokenizer output 1.2e-5.
  WHOLE codes-on graph (ComfyUI's 50 codes in; hints replace the silence latent as the source latents) vs ComfyUI: latent 3.4e-4, audio 8.7e-4. Stage 5b + 6 done;
  left: the LM's SAMPLING loop (CFG 2 with the left-padded negative, temperature 0.85, top-p 0.9: own RNG, gated by teacher forcing + statistics), tokenizer + templates.
- 11-17 CUDA: `OobleckEncoder` vs ComfyUI's VAEEncodeAudio (Lore's dumps): posterior 3.0e-6 on decoded audio, 1.6e-6 on silence. ComfyUI SAMPLES the posterior (its latent
  is 4e-4 .. 6e-4 from its own mean), and its silence table is NOT its own encode of zeros (3.6e-2 apart in ComfyUI itself).
- SILENCE LATENT, settled: all 7 official ACE-Step 1.5 checkpoints ship the SAME silence_latent.pt (md5 34809d60); ACE-Step's MIT source hard-codes no table; so ComfyUI's
  4-decimal table is ComfyUI's own data (GPL: never shipped here). `AceStepSilenceLatent` reads the MIT .pt (no torch). Whole codes-off run with it vs ComfyUI: latent
  1.6e-2, audio 3.4e-2 (vs 1.8e-4 / 3.9e-4 with ComfyUI's table): the 2.2e-3 input difference is amplified ~10x. That is the parity SpawnGen gets on MIT inputs.
- 11-17 CUDA: `AceStepCodeLm` (two `Qwen3KvDecoder` caches, logits over the 64000 code tokens = all the mass at every measured step, cfg / temperature / top-p with its own
  seeded RNG): teacher-forced through the KV path, prompt 5.8e-5 (negative 3.8e-5: ComfyUI's LEFT-padded negative = the unpadded one from position 0), steps <= 9.2e-7.
  The SkinTokens RoPE table rounds frequencies to bf16 (1.7e-2 off a float32 Qwen3): the decoder uses exact host tables. Left: prompt templates (caption / lyrics / LM).
- 11-17: PROMPTS measured from ComfyUI's ids (decoded with ACE-Step's MIT tokenizer): `AceStepPrompts` (caption, lyrics, LM positive with the metas in <think>, LM negative
  with an empty <think>) + `Qwen2Tokenizer` re-encode to ComfyUI's ids EXACTLY (127 / 350 / 478 / 451). ComfyUI's instruction is "Generate audio semantic tokens based on the
  given conditions:" in BOTH prompts (not ACE-Step's "Fill the audio semantic mask").
- 11-17 (Lore's template runs): NO lyric cut (the split template's own lyrics end mid-word); lyric prompt = lyrics as given + <|endoftext|><|endoftext|>; the LM prompt
  strips them; duration prints in the caption and LM; bpm / time signature / key are always set (no "none" in the node); language "unknown" prints on the lyric side only.
  AceStep_Prompts_MatchComfyUiIds: all 24 id lists identical (6 runs x caption / lyrics / LM+ / LM-).
