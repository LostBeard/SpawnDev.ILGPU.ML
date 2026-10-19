# Scene generation, local 3D editing and a scene composer: research notes (2026-10-19)

First pass for SpawnDarkly's needs. Everything here comes from search results and abstracts, not from reading code or checking licences: **verify each licence and weight release on the official repo before building on it.** Local, open-weight only (no hosted services: see the memory note `fb-no-paid-services`).

## 1. Local-region editing of a generated asset

TRELLIS's project page shows editing a region of a 3D asset from a text or image prompt. That is a TRELLIS v1 feature; the TRELLIS.2 repo has no edit code. Training-free methods built on TRELLIS-style flow models exist:

| Method | Idea | Status (from search) |
|---|---|---|
| [VoxHammer](https://huanngzh.github.io/VoxHammer-Page/) ([paper](https://arxiv.org/html/2508.19247)) | Edit in TRELLIS's native 3D latent space: invert the model, cache inverted latents and key-value tokens, replace features in the preserved region while the edited region is regenerated. No training. | Code release and licence not confirmed |
| [Nano3D](https://github.com/JAMESYJL/Nano3D) ([paper](https://huggingface.co/papers/2510.15019)), ICLR 2026 | FlowEdit on the TRELLIS flow transformers, guided by an edited image; no mask needed | Repo exists; licence not shown. Its guide images come from Nano Banana (a hosted service): would be replaced by a local image editor |
| EditFlow3D, TanGO, ShapeUP, EditVerse3D (2026 arXiv) | Newer variants (trajectory preservation, tangent-space guidance, scalable image-conditioned edits) | Not read |

Why it fits ML: TRELLIS.2 is the same shape of model (flow transformers on sparse structured latents; `FlowSampler`, `FlowDiT` already ported and gated). The simplest edit is RePaint-style: hold the latents outside a 3D mask, regenerate the ones inside under a new prompt or image, project back every step. Inversion-based methods (VoxHammer) preserve the untouched part better and need the reverse ODE, a handful of extra DiT passes. Cost is of the order of the existing generation (~2 min on a desktop GPU).

Open questions: how a region is chosen (box / sphere / brush in 3D; or by part), and whether to re-run the texture stage only (recolour) or shape + texture.

## 2. Large scene generation

Two families:

1. **Layout, then assets (works with what we have).** A local LLM (Gemma is already in ML) writes a layout: list of objects with names, sizes, positions, rotations. `AssetGenerator` makes each object (FLUX.2 -> RMBG -> TRELLIS.2, ~2.5 min each), a composer places them. Large scenes stay affordable by generating a **kit** (a few dozen assets) and **instancing with variation** (scale, rotation, recolour), not by generating every object. Terrain, roads, vegetation scattering and chunk streaming are procedural, not ML.
2. **Native scene models.**
   - [SceneGen](https://github.com/Mengmouxu/SceneGen) (3DV 2026): single-image multi-object scene in one feed-forward pass; repo reports training code, inference code and pretrained models released.
   - [One2Scene](https://github.com/Wang-pengfei/One2Scene) (ICLR 2026): explorable 360-degree scene from one image; code and model reported released Feb 2026.
   - HunyuanWorld, WorldGen, WorldClaw (agentic, coarse-to-fine open-world generation, Aug 2026): weight release **not confirmed** in the results.
   - Building blocks: TRELLIS.2 (MIT), Pixal3D (MIT, TRELLIS.2 backbone), PartCrafter (MIT, part-decomposed meshes, relevant to multi-object scenes).

Open questions: output representation of the native models (meshes, Gaussian splats, panorama layers) and whether SpawnDarkly renders it; model size against the 12 GB card; licences.

## 3. Scene composer

A scene is glTF nodes with transforms referencing generated GLBs (plus lights, spawn points, collision proxies). Composition of generated assets is also on the TRELLIS page.

## 4. Where each piece belongs (my recommendation)

| Piece | Home |
|---|---|
| Edit-a-region pipeline on TRELLIS.2 (the flow/inversion maths) | **ML** (model pipeline, like `Trellis2ImageTo3D`) |
| Native scene-model ports (SceneGen etc.), if licences and size allow | **ML** |
| LLM layout step (prompt -> layout JSON) | **ML** (uses the existing Gemma pipeline) |
| Layout JSON -> glTF scene with node transforms (a data layer, like `AnimatedGlb`) | **ML**, thin |
| Interactive editor: selecting a region, placing and moving objects, brush UI | **SpawnDarkly / the demo**, not the library |
| Streaming, LOD, culling, instancing, physics, terrain, rendering | **SpawnDarkly** (engine) |

## 5. Suggested order

1. Verify licences and weights of the candidates above (SceneGen, One2Scene, Nano3D, VoxHammer).
2. Prototype RePaint-style region regeneration on the TRELLIS.2 texture stage (recolour a part), measure; then the shape stage.
3. Layout step with Gemma plus a layout-to-glTF writer, driven by `AssetGenerator`, as a first end-to-end small scene.
4. Evaluate a native scene model against that baseline.
