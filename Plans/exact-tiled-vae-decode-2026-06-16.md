# Exact-stat (seam-free) tiled VAE decode — design + build plan (2026-06-16)

Goal: tiled VAE decode that bounds GPU peak to ~one tile (small-card / browser low-VRAM) AND is bit-near-
identical to the full decode (NO brightness seams). The approx tiling (`VAE_TILE_LATENT`, shipped `7284915`)
cut peak LIVE 896→450 but has visible seams because each tile's GroupNorm uses PER-TILE stats. The fix is
**global GroupNorm stats across tiles**, which forces **layer-synchronous** tiled execution (every tile reaches
each norm together so their partial stats combine into the global before any tile proceeds).

## Decoder structure (verified, Explore agent 2026-06-16)
`post_quant_conv(1×1) → conv_in(3×3) → mid_block → up_blocks.0 → .1 → .2 → .3 → conv_norm_out → conv_act(SiLU)
→ conv_out(3×3, →3ch)`. 525 nodes, **clean linear topology**.
- **mid_block** (resnet, **attention**, resnet) runs at **64×64** (pre-upsample), `[1,512,64,64]` ≈ 8 MiB →
  RUN WHOLE, do NOT tile. The single Softmax/attention lives here at 64×64 = not a tiling concern.
- **up_blocks 0/1/2** each = 3 resnets + **upsampler** (nearest **2×** + 3×3 conv). up_blocks.3 = 3 resnets, NO
  upsampler. Resolution chain 64→128→256→512 (×2 at each of the 3 upsamplers). Channels 512→512→256→128.
- **30 InstanceNormalization sync points** (groups=32, eps=1e-6), decomposed `Reshape[0,32,-1] →
  InstanceNormalization(scale=ones[32], bias=zeros[32]) → Reshape(back) → Mul(γ[C,1,1]) → Add(β[C,1,1])`. The
  InstanceNorm is the ONLY global-stat point; γ/β are pointwise (no sync). 5 are in mid_block (run whole) →
  **25 sync points in the tiled region** (6 per up_block × 4 + conv_norm_out).
- **All spatial convs are 3×3 SAME (pad 1) → 1px halo each.** Only 3 convs are 1×1/halo-free (post_quant + the
  two resnets.0 conv_shortcut in up_blocks.2/.3). Resize is nearest-2× (local, halo-free, but DOUBLES the halo).

## ✅ Foundation BUILT + verified (commit `68f8b0f`)
`NormalizationKernels`: `InstanceNormPartialStats(input, sums, sumSqs, N, C, spatial)` (per-slice sum+sumSq,
double-accum) + `InstanceNormApplyWithStats(data, scale, bias, means, invStds, …)` (in-place, single
read_write binding). Gate `TiledStatSync` 8/0 all backends: 2-tile partial-stat combine == full InstanceNorm,
BYTE-EQUAL. This is the math at each of the 25 sync points: each tile contributes (sum, sumSq, count) over its
NON-overlap core → global mean/var per group → every tile applies the SAME global stats.

## Engine design — layer-synchronous tiled up-block decode
Phase A (monolithic head): run `post_quant_conv → conv_in → mid_block` at full 64×64 (cheap) via the existing
session → mid output `[1,512,64,64]` on GPU.
Phase B (tiled tail): tile that 64×64 into an R×R grid (e.g. 2×2 = 32×32 tiles → 256² tile peak ≈ ¼ area ≈
¼ peak; finer grid for smaller cards), then run `up_blocks.* → conv_norm_out → conv_act → conv_out` TILED.

### TiledFeatureMap (the core data structure)
- Holds an R×R grid of tiles, each a CPU float[] (offloaded) of `[C, coreH+2*halo, coreW+2*halo]` — a
  persistent halo margin. Only ONE tile resident on GPU at a time (the offload that bounds peak).
- `RefreshHalos()`: copy each tile's `halo`-px boundary rows/cols from its 4 neighbors' cores (zero-pad at
  image edges). Called before every 3×3 conv (the conv consumes the halo; output halo becomes stale).
- Tracks current channels + core spatial; grows on Resize (×2 spatial).

### Per-op tiled execution (run the up-block ops over the grid)
- **Conv 3×3 SAME**: `RefreshHalos()` → for each tile: upload padded tile, `Conv2D` (existing kernel, pad
  computed so the core stays same-size), download core. (1×1 convs/shortcuts: no halo refresh.)
- **SiLU (Sigmoid·Mul) / Add (residual)**: pointwise, per-tile, no halo.
- **Resize nearest-2×**: per-tile, local; doubles core + halo. (NearestUpsample kernel exists.)
- **GroupNorm (the 25 sync points)**: (1) per tile: reshape-to-groups view, `InstanceNormPartialStats` over the
  CORE only (exclude halo to avoid double-count) → accumulate global sum/sumSq/count per group; (2) global
  mean/invStd per group; (3) per tile: `InstanceNormApplyWithStats` (global) then the pointwise Mul(γ)+Add(β).
- **Residual shortcut**: the resnet input is held as a (tiled) tensor and added back after conv2 (1×1 shortcut
  conv if channel-change). Standard resnet wiring, tiled.

### Memory
GPU peak = one tile's largest feature map (+ the small per-group global-stat buffers + γ/β). For 2×2 tiling the
256² tile peak ≈ 450 MiB → finer tiling (3×3/4×4) → lower. CPU holds the R²−1 offloaded tiles (a few hundred MiB
of host RAM — cheap). Verify peak LIVE drops AND image is bit-near-identical to the full decode (no seams).

## Implementation vehicle — DECISION
Two options, both reuse the foundation kernels:
- **(A) Reimplement the up-block forward in C#** (resnet/upsampler structure is known exactly) using the
  existing kernels (Conv2D, the norm primitives, SiLU, NearestUpsample). Needs the up-block WEIGHTS extracted
  from the VAE session (conv kernels + γ/β). Structurally simplest; the weight extraction/mapping is the plumbing.
- **(B) Tiled mode in GraphExecutor**: represent each up-block tensor as a `TiledFeatureMap`, process each
  graph NODE over the grid, InstanceNorm does the combine. Reuses graph+weights+operators (no extraction) but
  needs the executor to carry tiled tensors + halo + offload through node iteration.
LEAN (A) for a focused, self-contained `TiledVaeUpDecoder` (no executor surgery; the up-block structure is
regular + fully mapped). Extract weights once from the session at load. Keep approx tiling as the fast path.

## Build order (each a verified, committed unit)
1. ✅ Foundation stat kernels (`68f8b0f`).
2. `TiledFeatureMap` + `RefreshHalos` + a unit test (halo refresh == a full-tensor 3×3 conv on the recombined
   grid, byte-equal — proves halo correctness in isolation).
3. Tiled resnet forward (norm-sync + silu + conv + halo) — test one resnet tiled == full, byte-near.
4. Tiled up_block (3 resnets + upsampler) — test one up_block tiled == full.
5. Full `TiledVaeUpDecoder` (Phase A whole + Phase B tiled) wired behind `VAE_TILE_LATENT` (replacing the approx
   path, or a new `VAE_TILE_EXACT=1`). Verify: peak LIVE < 896, image bit-near-identical to full (no seams),
   on CUDA/WebGPU first.
