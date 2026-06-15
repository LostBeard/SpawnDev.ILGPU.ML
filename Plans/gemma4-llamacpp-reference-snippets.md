# Gemma 4 12B "Unified" (gemma4uv) Multimodal - llama.cpp VERBATIM Reference Snippets

Source: ggml-org/llama.cpp `master`, cloned and read directly (not paraphrased).
Captured 2026-06-15. All snippets are exact copies with original line numbers.

These three details define the gemma4uv image-input front-end so a C# port reproduces it bit-exact.

---

## Context: how gemma4uv differs from gemma4v (CONFIG, load-time)

`tools/mtmd/clip.cpp` lines 1417-1432 (in `load_hparams`):

```cpp
                case PROJECTOR_TYPE_GEMMA4V:
                case PROJECTOR_TYPE_GEMMA4UV:
                    {
                        hparams.rope_theta = 100.0f;
                        hparams.n_merge = 3; // pooling_kernel_size
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        if (model.proj_type == PROJECTOR_TYPE_GEMMA4UV) {
                            // for "unified" variant, we directly use a bigger patch size, because the "token merging" is done directly on conv layer
                            hparams.patch_size = hparams.patch_size * hparams.n_merge;
                            hparams.n_merge = 1;
                        }
                        // @ngxson : the model performs quite poor with small images, we need to bump minimum image tokens to 40 to avoid that
                        hparams.set_limit_image_tokens(40, 280);
                        hparams.set_warmup_n_tokens(256); // avoid OOM on warmup
                    } break;
```

Plain English for the C# port:
- For GEMMA4UV: base patch_size (16) is MULTIPLIED by n_merge (3) -> **patch_size becomes 48**, and **n_merge is reset to 1**. The 3x token merge is baked into the conv/patch size, NOT a separate merge step.
- `set_limit_image_tokens(40, 280)` is what produces min_pixels / max_pixels in detail #1 (40 * 48*48 = 92160, 280 * 48*48 = 645120).
- Resize algorithm is **RESIZE_ALGO_BILINEAR**.
- This branch does NOT set `image_resize_pad`, so it stays at the struct default `PAD_CEIL` (see detail #1).

---

## DETAIL 1 - Exact image resize formula (smart_resize)

### 1a. The function itself

`tools/mtmd/mtmd-image.cpp` lines 142-169 (inside `struct img_tool`):

```cpp
    // calculate the size of the **resized** image, while preserving the aspect ratio
    // the calculated size will have min_pixels <= W*H <= max_pixels
    // this is referred as "smart_resize" in transformers code
    static clip_image_size calc_size_preserved_ratio(const clip_image_size & inp_size, const int align_size, const int min_pixels, const int max_pixels) {
        GGML_ASSERT(align_size > 0);
        const int width  = inp_size.width;
        const int height = inp_size.height;

        auto round_by_factor = [f = align_size](float x) { return static_cast<int>(std::round(x / static_cast<float>(f))) * f; };
        auto ceil_by_factor  = [f = align_size](float x) { return static_cast<int>(std::ceil(x / static_cast<float>(f))) * f; };
        auto floor_by_factor = [f = align_size](float x) { return static_cast<int>(std::floor(x / static_cast<float>(f))) * f; };

        // always align up first
        int h_bar = std::max(align_size, round_by_factor(height));
        int w_bar = std::max(align_size, round_by_factor(width));

        if (h_bar * w_bar > max_pixels) {
            const auto beta = std::sqrt(static_cast<float>(height * width) / max_pixels);
            h_bar = std::max(align_size, floor_by_factor(height / beta));
            w_bar = std::max(align_size, floor_by_factor(width  / beta));
        } else if (h_bar * w_bar < min_pixels) {
            const auto beta = std::sqrt(static_cast<float>(min_pixels) / (height * width));
            h_bar = ceil_by_factor(height * beta);
            w_bar = ceil_by_factor(width * beta);
        }

        return {w_bar, h_bar};
    }
```

Exact rounding semantics (match these in C# - note all the `/ static_cast<float>(f)` divisions are FLOAT, then `std::round/ceil/floor`, then truncating cast to int, then `* f`):
- `round_by_factor(x) = (int)std::round(x / (float)f) * f`  (C#: `(int)MathF.Round(x / f) * f` -- BUT see banker's-rounding note below)
- `ceil_by_factor(x)  = (int)std::ceil (x / (float)f) * f`
- `floor_by_factor(x) = (int)std::floor(x / (float)f) * f`
- Step 1 ("always align up first"): `h_bar = max(align_size, round_by_factor(height))`, same for w_bar.
- Step 2: if `h_bar*w_bar > max_pixels` -> over-max branch: `beta = sqrt((float)(height*width) / max_pixels)`, then `h_bar = max(align_size, floor_by_factor(height / beta))`, `w_bar = max(align_size, floor_by_factor(width / beta))`.
- Else if `h_bar*w_bar < min_pixels` -> under-min branch: `beta = sqrt((float)min_pixels / (height*width))`, then `h_bar = ceil_by_factor(height * beta)`, `w_bar = ceil_by_factor(width * beta)` (NOTE: no `max(align_size, ...)` clamp on the under-min branch).
- Returns `{w_bar, h_bar}` i.e. {width, height}.

**ROUNDING WARNING for C# port:** C++ `std::round` rounds half AWAY from zero. C# `Math.Round`/`MathF.Round` default to banker's rounding (half to even). Use `MathF.Round(x, MidpointRounding.AwayFromZero)` to match `std::round` exactly. `Math.Ceiling`/`Math.Floor` match `std::ceil`/`std::floor`.

For gemma4uv the call args are: `align_size = patch_size * cur_merge`. After the load-time config above, patch_size=48 and n_merge=1, so in the preprocessor `cur_merge = (n_merge==0 ? 1 : n_merge) = 1`, giving **align_size = 48**. min_pixels=92160, max_pixels=645120.

### 1b. The call site for gemma4uv (preprocessor = dyn_size)

gemma4uv selects `mtmd_image_preprocessor_dyn_size`. `tools/mtmd/mtmd.cpp` lines 606-613:

```cpp
            case PROJECTOR_TYPE_GEMMA4V:
            case PROJECTOR_TYPE_GEMMA4UV:
                {
                    // <|image> ... (image embeddings) ... <image|>
                    img_beg = "<|image>";
                    img_end = "<image|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
```

`tools/mtmd/mtmd-image.cpp` lines 891-910 (`mtmd_image_preprocessor_dyn_size::preprocess`):

```cpp
bool mtmd_image_preprocessor_dyn_size::preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) {
    GGML_ASSERT(hparams.image_min_pixels > 0 && hparams.image_max_pixels > 0);
    clip_image_u8 resized_image;
    const clip_image_size original_size = img.get_size();
    // the original pixtral model doesn't have n_merge
    const int cur_merge = hparams.n_merge == 0 ? 1 : hparams.n_merge;
    const clip_image_size target_size = img_tool::calc_size_preserved_ratio(
        original_size,
        hparams.patch_size * cur_merge,
        hparams.image_min_pixels,
        hparams.image_max_pixels);
    img_tool::resize(img, resized_image, target_size,
                        hparams.image_resize_algo,
                        hparams.image_resize_pad,
                        hparams.image_pad_color);
    clip_image_f32_ptr img_f32(clip_image_f32_init());
    img_u8_to_f32(resized_image, *img_f32, hparams.image_mean, hparams.image_std);
    output.entries.push_back(std::move(img_f32));
    return true;
}
```

### 1c. The resize() function - bilinear + PAD_CEIL behavior

`tools/mtmd/mtmd-image.cpp` lines 23-101 (`img_tool::resize`). The default `padding = PAD_CEIL`. The gemma4uv branch does NOT override `image_resize_pad`, and `clip-model.h` line 66 confirms the struct default:

`tools/mtmd/clip-model.h` line 66:
```cpp
    pad_style image_resize_pad = PAD_CEIL; // padding style when resizing
```

`tools/mtmd/mtmd-image.cpp` lines 23-101:

```cpp
    static void resize(
            const clip_image_u8 & src,
            clip_image_u8 & dst,
            const clip_image_size & target_resolution,
            resize_algo algo,
            pad_style padding = PAD_CEIL,
            std::array<uint8_t, 3> pad_color = {0, 0, 0}) {
        dst.set_size(target_resolution, src.is_placeholder());

        if (src.is_placeholder()) {
            // no-op for placeholder image, just set the size and return
            return;
        }

        if (dst.get_size() == src.get_size()) {
            // no resize needed, simple copy
            dst.cpy_buf(src.get_ro_buf());
            return;
        }

        if (padding == PAD_NONE) {
            // direct resize
            switch (algo) {
                case RESIZE_ALGO_BILINEAR:
                    resize_bilinear(src, dst, target_resolution.width, target_resolution.height);
                    break;
                case RESIZE_ALGO_BICUBIC:
                    resize_bicubic(src, dst, target_resolution.width, target_resolution.height);
                    break;
                case RESIZE_ALGO_BICUBIC_PILLOW:
                    resize_bicubic_pillow(src, dst, target_resolution.width, target_resolution.height);
                    break;
                default:
                    throw std::runtime_error("Unsupported resize algorithm");
            }
        } else {
            // resize with padding
            clip_image_u8 resized_image;
            float scale_w = static_cast<float>(target_resolution.width) / src.get_size().width;
            float scale_h = static_cast<float>(target_resolution.height) / src.get_size().height;
            float scale = std::min(scale_w, scale_h);

            int new_width, new_height;
            if (padding == PAD_NEAREST) {
                new_width  = std::min(static_cast<int>(std::round(src.get_size().width * scale)), target_resolution.width);
                new_height = std::min(static_cast<int>(std::round(src.get_size().height * scale)), target_resolution.height);
            } else {
                new_width  = std::min(static_cast<int>(std::ceil(src.get_size().width * scale)), target_resolution.width);
                new_height = std::min(static_cast<int>(std::ceil(src.get_size().height * scale)), target_resolution.height);
            }

            switch (algo) {
                case RESIZE_ALGO_BILINEAR:
                    resize_bilinear(src, resized_image, new_width, new_height);
                    break;
                case RESIZE_ALGO_BICUBIC:
                    resize_bicubic(src, resized_image, new_width, new_height);
                    break;
                case RESIZE_ALGO_BICUBIC_PILLOW:
                    resize_bicubic_pillow(src, resized_image, new_width, new_height);
                    break;
                default:
                    throw std::runtime_error("Unsupported resize algorithm");
            }

            // fill dst with pad_color
            fill(dst, pad_color);

            int offset_x, offset_y;
            if (padding == PAD_NEAREST) {
                offset_x = static_cast<int>(std::round((target_resolution.width  - new_width)  / 2.0f));
                offset_y = static_cast<int>(std::round((target_resolution.height - new_height) / 2.0f));
            } else {
                offset_x = (target_resolution.width  - new_width)  / 2;
                offset_y = (target_resolution.height - new_height) / 2;
            }
            composite(dst, resized_image, offset_x, offset_y);
        }
    }
```

**Is resize bilinear?** YES - `image_resize_algo = RESIZE_ALGO_BILINEAR` for gemma4uv.

**Is there padding AFTER resize (PAD_CEIL)?** It is NOT bypassed - gemma4uv uses the default `PAD_CEIL`, so the `else` (padding) branch runs, NOT a no-op. BUT the smart-resize target is built so that the padded region is normally ZERO/negligible:
- `scale = min(target_w/src_w, target_h/src_h)`, then `new_w = min(ceil(src_w*scale), target_w)`, `new_h = min(ceil(src_h*scale), target_h)`.
- Because `calc_size_preserved_ratio` already chose `target_w,target_h` to closely match the source aspect ratio (each dim independently rounded to a multiple of align_size=48), `new_w ~= target_w` and `new_h ~= target_h`. When they are exactly equal, the pad region is zero and `composite` fills the whole dst.
- HOWEVER it is NOT guaranteed bit-exact zero-pad: independent rounding of W and H to multiples of 48 can leave the target aspect ratio slightly off from source, so `min(scale_w,scale_h)` can yield `new_w < target_w` OR `new_h < target_h` by up to a few pixels, producing a thin symmetric black border (offset = (target-new)/2, integer division, centered) on ONE axis. The final tensor dims are always the target (multiples of 48).

**C# port requirement:** You MUST implement the PAD_CEIL path (scale=min, ceil, clamp, center-composite onto a black-filled target-sized canvas), not a plain stretch-to-target. A plain stretch will diverge by a few pixels whenever the rounded target aspect != source aspect. The resize kernel itself must be bilinear matching `resize_bilinear`.

---

## DETAIL 2 - im2col / patch flattening byte layout

### 2a. The gemma4uv graph - how im2col is invoked

`tools/mtmd/models/gemma4uv.cpp` lines 4-27:

```cpp
ggml_cgraph * clip_graph_gemma4uv::build() {
    ggml_tensor * inp_raw = build_inp_raw();

    // Gemma4UnifiedVisionEmbedder uses default pytorch LayerNorm, not RMSNorm
    float eps = 1e-5f; // default eps for pytorch LayerNorm

    ggml_tensor * inp = nullptr;
    {
        // note: we cannot use ggml_conv_2d here because we need to apply norm after im2col
        auto c = inp_raw->ne[2];
        ggml_tensor * kernel = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, patch_size, patch_size, c);
        inp = ggml_im2col(ctx0, kernel, inp_raw, patch_size, patch_size, 0, 0, 1, 1, true, inp_raw->type);
        // inp shape: [patch_size * patch_size * c, n_patches_w, n_patches_h]

        inp = ggml_reshape_2d(ctx0, inp, inp->ne[0], inp->ne[1] * inp->ne[2] * inp->ne[3]);
        inp = build_norm(inp, model.patch_norm_1_w, model.patch_norm_1_b, NORM_TYPE_NORMAL, eps, -1);
        // inp shape: [patch_size * patch_size * c, n_patches]

        inp = ggml_mul_mat(ctx0, model.patch_embeddings_0, inp);
        inp = ggml_add(ctx0, inp, model.patch_bias);
        // inp shape: [n_embd, n_patches]

        inp = build_norm(inp, model.patch_norm_2_w, model.patch_norm_2_b, NORM_TYPE_NORMAL, eps, -1);
    }
```

Notes: `ggml_im2col(ctx0, kernel, inp_raw, s0=patch_size, s1=patch_size, p0=0, p1=0, d0=1, d1=1, is_2D=true, dst_type)`. Kernel is `[patch_size, patch_size, c]` = [48,48,3]. inp_raw is channel-planar `[nx, ny, 3]` (ne0=W, ne1=H, ne2=C). Note the patch-vector length comment says `patch_size*patch_size*c` = 48*48*3 = **6912**, ordered with **c as the OUTER block** (see the index math below - destination index is `iic*(KH*KW) + ikh*KW + ikw`).

### 2b. The exact im2col destination index (CPU reference)

`ggml/src/ggml-cpu/ops.cpp` lines 6207-6280 (`ggml_compute_forward_im2col_f32`):

```cpp
// ggml_compute_forward_im2col_f32
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
static void ggml_compute_forward_im2col_f32(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[5];
    const bool is_2D = ((const int32_t *)(dst->op_params))[6] == 1;

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t N  = is_2D ? ne13 : ne12;
    const int64_t IC = is_2D ? ne12 : ne11;
    const int64_t IH = is_2D ? ne11 : 1;
    const int64_t IW = ne10;

    const int64_t KH = is_2D ? ne01 : 1;
    const int64_t KW = ne00;

    const int64_t OH = is_2D ? ne2 : 1;
    const int64_t OW = ne1;

    int ofs0 = is_2D ? nb13 : nb12;
    int ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == sizeof(float));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        float * const wdata = (float *) dst->data;

        for (int64_t in = 0; in < N; in++) {
            for (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                for (int64_t iow = 0; iow < OW; iow++) {
                    for (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        float * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        const float * const src_data = (float *)((char *) src1->data + in*ofs0 + iic*ofs1); // [IH, IW]

                        for (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            for (int64_t ikw = 0; ikw < KW; ikw++) {
                                const int64_t iiw = iow*s0 + ikw*d0 - p0;
                                const int64_t iih = ioh*s1 + ikh*d1 - p1;

                                if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } else {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = (src_data[iih*IW + iiw]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
```

**Within-patch element order (the 6912-vector):** destination offset within a patch =
`iic*(KH*KW) + ikh*KW + ikw`
= **channel-outermost, then kernel-row (ky=ikh), then kernel-col (kx=ikw) innermost**.

So the patch vector is: all 2304 (=48*48) values of channel 0 (R) as a 48x48 **row-major** block (ky outer, kx inner), then all 2304 of channel 1 (G), then channel 2 (B). This is exactly `[channel][ky][kx]` with channel outermost, row-major within each channel block. Confirmed.

For gemma4uv: KH=KW=48, IC=3, stride s0=s1=48, pad p0=p1=0, dilation d0=d1=1. Source pixel read is `src_data[iih*IW + iiw]` where `iih = ioh*48 + ikh`, `iiw = iow*48 + ikw` (since stride=patch=48, patches are non-overlapping, tiling the image exactly). The src1 channel plane is selected by `iic*ofs1` (channel-planar layout).

**Patch (position) iteration order producing patch index p:** The output is indexed `(in*OH*OW + ioh*OW + iow)`. With OH = n_patches_h (rows of patches), OW = n_patches_w (cols), the linear patch index is:
`p = ioh*OW + iow`  -> **row-major over the patch grid, iow (gx) fastest, ioh (gy) slowest**.
i.e. `p = gy*n_cols + gx`, gx fastest. This matches what the position-embedding loader assumes (detail 2c).

### 2c. Position-embedding assignment confirms patch order

`tools/mtmd/clip.cpp` lines 4062-4074 (in `set_input` for gemma4):

```cpp
        case PROJECTOR_TYPE_GEMMA4V:
        case PROJECTOR_TYPE_GEMMA4UV:
            {
                // set (col, row) patch positions for learned positional embedding
                const int n_cols = image_size_width  / patch_size;
                std::vector<int> pos_x(num_patches), pos_y(num_patches);
                for (int i = 0; i < num_patches; i++) {
                    pos_x[i] = i % n_cols;
                    pos_y[i] = i / n_cols;
                }
                set_input_i32("pos_x", pos_x);
                set_input_i32("pos_y", pos_y);
            } break;
```

Plain English: patch index `i` maps to grid `(gx = i % n_cols, gy = i / n_cols)`, with `n_cols = resized_width / 48`. This is row-major, gx (column) fastest, EXACTLY matching the im2col output order `p = gy*n_cols + gx`. So patch p gets position-embedding row pos_x = p % n_cols (the x/column learned table) added, and pos_y = p / n_cols (the y/row learned table) added - two separate learned lookup tables, both added (see gemma4uv.cpp lines 37-57). The position embeddings are stored as one tensor split into x-table (offset 0) and y-table (offset pos_size*nb1).

---

## DETAIL 3 - Does the gemma sqrt(n_embd) scale apply to multimodal (provided) embeddings?

### 3a. The text-model gemma4 graph scale - THE KEY LINE

`src/models/gemma4.cpp` lines 172-183:

```cpp
llama_model_gemma4::graph::graph(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params),
        model(model),
        n_embd_per_layer(model.hparams.n_embd_per_layer) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // important: do not normalize weights for raw embeddings input (i.e. encoded image emdeddings)
    inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);
    cb(inpL, "inp_scaled", -1);
```

**ANSWER: the sqrt(n_embd) scale is applied ONLY when `ubatch.token` is set (the gathered-token path). For provided embeddings (multimodal, `ubatch.embd` path, `ubatch.token == nullptr`) the scale is `1.0f` - i.e. NO scaling.** The comment is explicit: "do not normalize weights for raw embeddings input (i.e. encoded image embeddings)."

### 3b. build_inp_embd - token branch vs embd branch

`src/llama-graph.cpp` lines (function `llm_graph_context::build_inp_embd`, ~1820-1900 region; the scale-relevant tail shown). The two inputs are built and selected by `ubatch.token ? 0 : 1`:

Token branch: `cur = ggml_get_rows(ctx0, tok_embd, inp->tokens)` (+ optional LoRA, + optional pad when n_embd_inp != n_embd).
Embd branch: `cur = inp->embd` (the provided F32 [n_embd_inp, n_tokens] input, used RAW).
Selection: `ggml_build_forward_select(gf, inps.data(), inps.size(), ubatch.token ? 0 : 1)`.

The Granite/deepstack auto-scale inside build_inp_embd (lines 1891-1895) is GATED off for gemma4 because gemma4 has `f_embedding_scale == 0.0f` (gemma4 applies its own scale externally at gemma4.cpp:182 instead):

`src/llama-graph.cpp` lines 1891-1895:
```cpp
    if (hparams.f_embedding_scale != 0.0f && (ubatch.token || hparams.n_deepstack_layers == 0)) {
        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }
```
(For gemma4 `f_embedding_scale` is 0, so this block is a no-op; the real scale is the explicit `gemma4.cpp:182` line, which uses the SAME `ubatch.token ? sqrtf(n_embd) : 1.0f` token-only gating.)

Sibling confirmation (same token-only gating pattern across the gemma family):
- `src/models/gemma3.cpp:93`:  `inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);`
- `src/models/gemma-embedding.cpp:85`: `inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);`
- `src/models/gemma3n.cpp:104`: `inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);`
- (gemma.cpp:49 and gemma2.cpp:70 are the OLDER text-only gemma1/2 and always scale `sqrtf(n_embd)` because they never take provided embeddings.)

### 3c. mtmd encode - does it pre-scale the projected embeddings? NO.

`tools/mtmd/mtmd.cpp` lines 1402-1460 (`mtmd_encode_impl`): it resizes `out_embd` and copies the clip projection output straight in - no `ggml_scale`, no `sqrtf`, no multiply:

```cpp
static int32_t mtmd_encode_impl(mtmd_context * ctx, const mtmd_image_tokens * image_tokens, std::vector<float> & out_embd) {
    clip_ctx * ctx_clip = ctx->ctx_v;
    if (!ctx_clip) {
        LOG_ERR("%s: this API does not support non-vision input, please use mtmd_encode_chunk instead\n", __func__);
        return 1;
    }
    auto proj_type = clip_get_projector_type(ctx_clip);

    int n_embd_out = ctx->n_embd_out();
    auto n_tokens_out = image_tokens->n_tokens();
    out_embd.resize((size_t)n_embd_out * n_tokens_out);

    bool ok = false;

    if (clip_is_llava(ctx_clip)
        || proj_type == PROJECTOR_TYPE_MINICPMV
        || proj_type == PROJECTOR_TYPE_GLM_EDGE
        || proj_type == PROJECTOR_TYPE_INTERNVL
        || proj_type == PROJECTOR_TYPE_DEEPSEEKOCR2
        || proj_type == PROJECTOR_TYPE_GRANITE4_VISION) {
        ...
            std::copy(tmp_embd.begin(), tmp_embd.end(), out_embd.begin() + offset);
        ...
    } else {
        ...
        ok = clip_image_batch_encode(
            ctx_clip,
            ctx->n_threads,
            &image_tokens->batch_f32,
            out_embd);
    }

    return ok ? 0 : 1;
}
```

(gemma4uv takes the `else` branch -> `clip_image_batch_encode` writes the projected embeddings directly into `out_embd`. The gemma4uv graph's last op is `build_mm(model.mm_input_proj_w, cur)` after an RMSNorm - that projected output is what gets returned, unscaled. See gemma4uv.cpp lines 59-70.)

`tools/mtmd/models/gemma4uv.cpp` lines 59-71 (tail - the projection that produces the returned embeddings):
```cpp
    auto cur = inp;

    // Gemma4UnifiedMultimodalEmbedder
    {
        // embedding_pre_projection_norm
        cur = ggml_rms_norm(ctx0, cur, hparams.eps);
        cur = build_mm(model.mm_input_proj_w, cur);
        cb(cur, "projected", -1);
    }

    ggml_build_forward_expand(gf, cur);
    return gf;
}
```

### 3d. CONCLUSION for the C# port

**Splice the mtmd-projected image embeddings RAW. Do NOT pre-scale them by sqrt(3840) (~61.97).**
- The text gemma4 graph applies `sqrtf(n_embd)` ONLY to gathered token embeddings (`ubatch.token` set); image/audio (provided-embedding) rows are multiplied by `1.0f`.
- mtmd encode returns the projected embeddings with NO additional scaling.
- So in your C# pipeline: token embeddings get `* sqrt(n_embd)`; spliced image embeddings get `* 1.0` (left as-is). If you bake the `sqrt(n_embd)` into your token-embedding lookup, make sure you do NOT apply it to the image-embedding rows.

---

## One-line summaries

1. **Resize:** `calc_size_preserved_ratio` smart-resize (align=48, min=92160, max=645120): round each dim to nearest mult of 48 first; if area>max scale down by `floor((dim/beta) to mult of 48)`, beta=sqrt(area/max); if area<min scale up by `ceil((dim*beta) to mult of 48)`, beta=sqrt(min/area). C# must use AwayFromZero rounding for the initial round step. Resize is BILINEAR; PAD_CEIL is NOT bypassed (implement scale=min + ceil + centered black-pad composite), though the pad is usually ~0 because target nearly matches source aspect.
2. **im2col layout:** each 6912 patch vector is `[channel][ky][kx]` (channel-outermost: all 48x48 row-major R, then G, then B). Patches iterate row-major over the 48px grid: `p = gy*n_cols + gx`, gx (column) fastest. Patch p gets learned pos-x table row (p % n_cols) and pos-y table row (p / n_cols) added.
3. **sqrt(n_embd) scale:** Applied ONLY to token embeddings (`ubatch.token ? sqrtf(n_embd) : 1.0f` at gemma4.cpp:182). Multimodal/provided embeddings are scaled by 1.0 (RAW). mtmd encode adds no scaling. => splice image embeddings RAW, do NOT multiply by sqrt(3840).
