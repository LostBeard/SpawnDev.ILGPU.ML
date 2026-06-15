# Gemma 4 12B "Unified" — Multimodal Bring-up Spec

Goal (TJ, 2026-06-15): **gemma4 100% working — all parts, not just text gen** + a new `Examples/` console app
that exercises the full multimodal model.

This is the no-guessing implementation spec. Every concrete number below is verified against **llama.cpp
`master` `tools/mtmd/`** (file:function cited) and cross-checked against the actual GGUF tensor shapes/metadata
we inspected locally. Items that could NOT be verified are listed under "Unverified".

## What gemma4 12B actually is
- **Encoder-free, "Unified"** any-to-text model. Inputs: text, image, audio, video. **Output: text only.**
- No SigLIP vision tower, no Conformer audio encoder. Raw image patches and raw audio frames are projected
  **directly** into the 3840-dim LLM embedding space by lightweight linear layers (`block_count=0` for both).
- This makes the multimodal path matmuls + norms + preprocessing + splicing — all on kernels we already have.

## Local assets
- Text decoder: `D:\users\tj\Projects\gemma4-12b-Q4_K_M.gguf` (6.9 GB, 48 layers, 3840d, 16 heads,
  vocab 262144). **Text gen VERIFIED CORRECT on CUDA** (2026-06-15: "capital of France" → Paris, ~390ms/tok).
- Multimodal projector: `D:\users\tj\Projects\mmproj-gemma-4-12B-it-bf16.gguf` (167 MB, `arch=clip`,
  `type=mmproj`, 52.4M params, 11 tensors, bf16 projections = GGUF type 30 → dequant to F32 on load).

## mmproj tensors (the only 11)
```
v.patch_embd.weight       [6912, 3840]      bf16   vision patch projection (matmul)
v.patch_embd.bias         [3840]            f32
v.patch_norm.1.weight/bias[6912]            f32    LayerNorm over 6912 (pre-projection)
v.patch_norm.2.weight/bias[3840]            f32    LayerNorm over 3840 (post-projection)
v.patch_norm.3.weight/bias[3840]            f32    LayerNorm over 3840 (post-pos-embd / "pos_norm")
v.position_embd.weight    [3840, 1120, 2]   f32    factorized 2D pos: tbl_x (axis0) + tbl_y (axis1), 1120 pos each
mm.input_projection.weight[3840, 3840]      bf16   vision -> LLM embed (matmul, no bias/act)
mm.a.input_projection.weight[640, 3840]     bf16   audio  -> LLM embed (matmul, no bias/act)
```
NOTE two **weightless RMSNorms** also exist in the graph (no params, so not in the tensor list): one before the
vision final projection, one in the audio path.

---

## VISION path (projector type `gemma4uv`)
### Preprocess (llama.cpp `mtmd-image.cpp`, `clip-impl.h`)
1. Effective patch = **48×48** (load-time: patch_size 16 × n_merge 3, then n_merge→1; `clip.cpp:1424-1428`).
   Token-merging is baked into the bigger patch, so flat patch dim = 48*48*3 = **6912**.
2. Resize **aspect-ratio-preserving**, both W and H forced to **multiples of 48**, total pixels clamped to
   **[92160, 645120]** (= [40, 280] tokens). NOT a fixed 224² square, NOT tiled crops. Bilinear.
   (`calc_size_preserved_ratio(align=48, min=92160, max=645120)`, `mtmd-image.cpp:145-169,897-901`.)
3. RGB, channel-planar (CHW) `inp_raw [nx,ny,3]`. Normalize = **pixel/255** only (GGUF mean=0, std=1).
4. `im2col` kernel [48,48,3] stride 48, no pad → **[6912, n_patches]**. Within a 6912 vector: R plane (2304),
   then G, then B (channel-planar; inner kx/ky nesting = ggml convention, see Unverified #2).
5. **N tokens = (W/48) * (H/48)**, dynamic, bounded [40,280].

### Forward graph (llama.cpp `models/gemma4uv.cpp`) — exact order
1. im2col → `[6912, N]`
2. **patch_norm.1** — LayerNorm over 6912, **eps 1e-5**, *(x-mean)/std * w + b* (these norms have BIAS = real
   PyTorch LayerNorm, NOT RMSNorm; `gemma4uv.cpp:7-8`)
3. `mm.input... ` NO — first matmul is **v.patch_embd** → `[3840, N]`, then **+ v.patch_embd.bias**
4. **patch_norm.2** — LayerNorm over 3840
5. **+ position embd**: factorized 2D. n_cols = W/48. For patch i: pos_x = i % n_cols, pos_y = i / n_cols.
   `emb_x = tbl_x[:, pos_x]`, `emb_y = tbl_y[:, pos_y]` (tbl_x = position_embd[...,0], tbl_y = [...,1]). Add BOTH.
6. **patch_norm.3** — LayerNorm over 3840 (the "pos_norm", after pos add)
7. **weightless RMSNorm** (embedding_pre_projection_norm, eps = model eps)
8. **mm.input_projection** matmul → `[3840, N]`. No bias, no activation.
9. Wrap in prompt with text tokens **`<|image>` … `<image|>`**; the N rows occupy reserved slots between them.

---

## AUDIO path (projector type `gemma4ua`) — raw waveform, NO mel
### Preprocess (llama.cpp `mtmd-audio.cpp:946-981`)
- **No STFT, no mel filterbank, no log-mel.** `num_mel_bins=640` is a misnomer = raw frame length.
- 16 kHz. Chunk waveform into **640-sample frames (40 ms), non-overlapping** (hop=640), last frame zero-padded.
- **n_tokens = ceil(n_samples / 640) = 25 tokens/sec.** Storage frame-major → tensor `[n_tokens, 640]`.

### Forward graph (`models/gemma4ua.cpp`)
1. inp_raw `[n_tokens, 640]`
2. permute → `[640, n_tokens]`
3. **weightless RMSNorm** over 640, **eps 1e-6**
4. **mm.a.input_projection** matmul `[640,3840]` → `[3840, n_tokens]`. No conv/attention/bias/activation.
5. Wrap with text tokens **`<|audio>` … `<audio|>`**.

(Do NOT confuse `gemma4ua` with `gemma4a` — the latter is a separate mel/Conformer encoder, not our file.)

---

## SPLICING + chat template
- Per media item, mtmd emits **3 chunks**: `[text: begin marker]`, `[media chunk reserving N empty slots]`,
  `[text: end marker]`. The projected `[3840, N]` embeddings overwrite the N reserved positions
  (`mtmd.cpp:1033-1221,1402-1456`). One chunk per image for gemma4 (no llava-uhd slicing).
- **DEFINITIVE — splice media embeddings RAW (no sqrt(n_embd) scale).** `src/models/gemma4.cpp:181-182`:
  `inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);` — the embedding scale applies ONLY
  to gathered token embeddings; provided multimodal embeddings (the `ubatch.embd` path) get ×1.0. mtmd encode
  adds no scaling either. So our integration: keep the graph's `input_ids → Gather(token_embd) → ×sqrt(n_embd)`
  for text positions; OVERWRITE the media positions with the RAW projector output. (gemma4 `f_embedding_scale==0`,
  so the Granite-style extra scale is also a no-op.)
- Chat template (`google/gemma-4-12B-it/chat_template.jinja`, verified verbatim — NOT Gemma 3's):
  - turn open `<|turn>` + role + `\n`; turn close `<turn|>\n`; gen prompt `<|turn>model\n` (assistant role = `model`)
  - media markers in prompt text: **`<|image|>` / `<|audio|>` / `<|video|>`** (pipes both sides)
  - thinking: `<|think|>` to enable; reasoning channel `<|channel>thought\n…\n<channel|>`

## VIDEO
- Helper-level only, model-agnostic. Default sample **fps = 4.0**; each frame → the gemma4uv image path.
  Timestamp text chunks (e.g. `[10m50.5s]`) interleaved every 5 s. No gemma4-specific temporal merge.

---

## Implementation order (fast backends first: CUDA/OpenCL/WebGPU, then WebGL/Wasm)
1. **mmproj loader** — parse the 11 tensors + clip.* metadata; dequant bf16→F32; weights to GPU. (Task #3)
2. **Vision input** — preprocess + the 9-step graph above; splice. CPU reference + fast-backend equivalence. (#4)
3. **Audio input** — raw-frame preprocess + permute/RMSNorm/matmul; splice. (#5)
4. **Video** — frame sampler → image path. (#6)
5. **Example console app** (`Examples/05.Gemma4Multimodal.Console`) — text + image/audio/video → text. (#7)
6. **PMT correctness tests** per modality vs reference, fast backends first. (#8)

## Reference oracle
llama.cpp `llama-mtmd-cli` / `mtmd` produces the per-media embedding tensors and the full generation — use it as
the node-level + end-to-end oracle (same pattern as the text bring-up used llama-server). Teacher-force identical
inputs and compare the `[3840, N]` projected embeddings, then the generated tokens.

## RESOLVED (was Unverified) — verbatim snippets in `Plans/gemma4-llamacpp-reference-snippets.md`
1. **Resize formula** (`mtmd-image.cpp:145-169` `calc_size_preserved_ratio`): `round_by_factor(x)=round(x/48)*48`
   (round = half-AWAY-from-zero → C# `MidpointRounding.AwayFromZero`); `h_bar=max(48,round(h))`, `w_bar=max(48,round(w))`;
   if `h_bar*w_bar > max_pixels(645120)`: `beta=sqrt(h*w/max)`, `h_bar=max(48,floor(h/beta/48)*48)`, same w;
   else if `< min_pixels(92160)`: `beta=sqrt(min/(h*w))`, `h_bar=ceil(h*beta/48)*48`, same w. Bilinear.
   Then **PAD_CEIL** (`mtmd-image.cpp:58-99`): `scale=min(scale_w,scale_h)`, `new=min(ceil(src*scale),target)`,
   black fill, CENTER composite — must implement (independent W/H rounding can leave a thin black border; a plain
   stretch diverges).
2. **im2col within-patch order** (`ggml-cpu/ops.cpp:6207-6280`): `dst[iic*(KH*KW) + ikh*KW + ikw]` =
   `[channel][ky][kx]`, channel-outermost, row-major within channel (R 48×48 block, then G, then B). Patch order
   `p = gy*n_cols + gx` (gx fastest). pos_x=p%n_cols, pos_y=p/n_cols, n_cols=resized_W/48.
3. **Media-embedding scale** (`models/gemma4.cpp:181-182`): RAW, ×1.0 — see SPLICING section above. (mtmd encode
   adds no scaling.)
4. (informational) HF `preprocessor_config.json` for `google/gemma-4-12B` still not independently fetched (repo
   gated); llama.cpp values match our GGUF metadata anchors exactly, so the llama.cpp path IS the reference.

## Notes
- mmproj projections are bf16 (GGUF type 30). Our GGUF loader already dequants to F32 — not blocked on Geordi's
  in-flight `ILGPU.BFloat16` work (that's a separate perf/precision track).
- Source refs the research agent pulled live: `tools/mtmd/{clip.cpp,clip-impl.h,clip-model.h,mtmd.cpp,
  mtmd-image.cpp,mtmd-audio.cpp,mtmd-helper.cpp}` and `models/{gemma4uv,gemma4ua,gemma4v,gemma4a}.cpp`,
  ggml-org/llama.cpp master.
