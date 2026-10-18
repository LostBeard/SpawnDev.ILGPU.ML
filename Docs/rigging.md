# Auto-rigging: skeleton and skin weights for a generated character (SkinTokens, UniRig)

A mesh from `Trellis2ImageTo3D` is a statue. `SkinTokensRigger` gives it a skeleton and learned skin weights so it can be posed and animated: a GLB goes in, the same GLB with joint nodes, a skin and per-vertex
`JOINTS_0` / `WEIGHTS_0` comes out (textures and materials untouched). It runs on CUDA, OpenCL and in the browser on WebGPU, on the same ILGPU kernels as the rest of the library.

Two riggers ship, both re-implemented on ILGPU from the published checkpoints' tensor shapes (no PyTorch, no ONNX Runtime):

| | `SkinTokensRigger` (`SpawnDev.ILGPU.ML.SkinTokens`) | `UniRigRigger` (`SpawnDev.ILGPU.ML.UniRig`) |
|---|---|---|
| Model | VAST AI Research **SkinTokens / TokenRig** (MIT), the successor to UniRig | VAST AI Research **UniRig** (MIT) skeleton stage |
| Skeleton | generated in the same token sequence as the skin | OPT-350m decoder under a grammar |
| Skin weights | **learned** (an FSQ skin VAE decodes 4 tokens per bone into weights at every surface sample) | UniRig's skinning *prior* only (geodesic voxel binding), not the learned network |
| Weights file | 1.15 GB bf16 (`LostBeard/SkinTokens-bf16` on the hub) | 0.7 GB bf16 (export it yourself, see below) |
| Use it | the default | a smaller, faster fallback with a hand-built skin prior |

The demo's **Rig this model** button (`/text-to-3d`, `/image-to-3d`) uses SkinTokens.

## API

```csharp
// desktop (CUDA / OpenCL): from a local file
using var weights = GpuWeights.LoadFile(accelerator, "skintokens.safetensors");
// browser or desktop: stream it from the hub / any IModelSource (cached in OPFS by the demo's RangeStreamModelSource)
using var weights = await SkinTokensWeights.LoadAsync(accelerator, modelSource, "LostBeard/SkinTokens-bf16");

using var rigger = new SkinTokensRigger(accelerator, weights);
var result = await rigger.RigGlbAsync(glbBytes, seed: 0);
File.WriteAllBytes("rigged.glb", result.Glb);       // result.Skeleton.Joints / .Parents: the joints in the mesh's own space
```

`rigger.StageDone` reports progress (mesh encoder, cond encoder, prefill, skeleton tokens, skin tokens, skin decode). Decoding is **greedy**: the published inference samples (beam search 10, top-k 10,
temperature 1.5), which is random; greedy is deterministic and is what the tests pin. `UniRigRigger` has the same shape (`RigGlbAsync`, weights from `export_weights.py skeleton`).

Best on one standing humanoid in a T or A pose (what the models were trained on); a prop-heavy or quadruped input produces a plausible but not animator-grade rig.

## How SkinTokens works here

```
points    54,000 surface samples of the mesh (16,384 vertex samples + area-weighted surface points) with normals, scaled into [-1,1]
prefix    Michelangelo shape encoder (8 layers, 512 farthest-point queries) -> output_proj + RMSNorm                          [512, 896]
cond      skin-VAE cond encoder (384 queries over all 54,000 points, 3 blocks) -> cond_quant                                  [384, 512]
tokens    Qwen3-style decoder (28 layers, 16/8 GQA heads of 128, qk RMS norm, rotary, SwiGLU, KV cache), greedy:
          bos, class, skeleton tokens under UniRig's grammar, SWITCH, then 4 FSQ skin tokens per bone
skin      VAE decoder per bone (10 self-attention blocks over [4 token latents | cond], then one cross-attention block with the 54,000 query points) -> sigmoid
transfer  every mesh vertex takes the inverse-distance blend of its 8 nearest samples; the GLB keeps the strongest 4 influences per vertex, renormalised
```

## Measured

All against the real PyTorch reference classes in float32 (`tools/unirig-harness/reference/golden_skintokens.py`, MIT SkinTokens `grpo_1400`), TRELLIS.2 humanoid and the golden's cloud, RTX 4070:

| | |
|---|---|
| Exact kernels (fp32, no tensor cores) | **295 of 296 generated tokens identical** to the reference, skin weights **5.7e-5** relRMS |
| Fast kernels (CUDA tensor cores, bf16 matmul inputs) | same 35 bones; activations drift ~1e-2 relRMS over 8-28 layers (2e-4 with the fast paths off), so near-tie tokens may differ from the reference |
| Rig time, CUDA | **5.8 s warm**, 7.1 s cold (mesh encoder 0.03, cond 0.05, prefill 0.2, 154 skeleton tokens 1.7, 140 skin tokens 1.7, skin decode 35 bones x 54,000 points 2.2) |
| Browser (WebGPU, test host) | same 35 bones; 26.6 s cold of which 12.4 s is first-use shader compilation |
| Demo page, real Chrome | TRELLIS.2 witch sample: weights 87 s first download through the hub, **59 joints in 40 s**, rigged GLB passes the glTF validator with 0 errors |
| TRELLIS.2 humanoid GLB | 42 joints; posing the right shoulder moves 18,082 of 67,186 vertices, none on the left half, 16 on the torso |

A single-token step is a coalesced GEMV (`SkinTokensKernels.Gemv`: one 128-thread group per output, bf16 pairs as coalesced 32-bit loads): the general GEMM launches too few blocks at M = 1 (43 -> 11 ms per token).

## Two quirks of the published model, reproduced or avoided on purpose

1. **bf16-rounded tables.** The VAE's frequency / phase tables and Qwen3's `inv_freq` are non-persistent buffers computed in float32 and then rounded to bf16 by the model's `.to(torch.bfloat16)`
   (`inv_freq[1]` 0.80584 -> 0.8046875, the VAE's pi * 2^7 moves by ~0.1 rad). The published model runs with the rounded values, so this port builds the same bf16-rounded tables; using exact
   float32 values costs 0.3-5% relRMS. (`nn.RMSNorm(eps=None)` likewise uses bf16's epsilon 0.0078125.)
2. **The last skin token.** The published logits processor counts the switch token among the skin tokens, so it ends the sequence one token early and decodes the last bone with the END id
   (33035 - 267 = 32768, which the FSQ digits wrap to code 0). This port writes all four tokens of every bone.

## Weights and tooling

- SkinTokens weights: `LostBeard/SkinTokens-bf16` (public, MIT, converted from VAST-AI/SkinTokens `grpo_1400.ckpt`; served by the hub at `/hf/LostBeard/SkinTokens-bf16/skintokens.safetensors`).
  Rebuild with `tools/unirig-harness/reference/export_weights.py skintokens` (fuses Qwen q/k/v and gate/up, regroups the Michelangelo and Tripo2 attention rows; the model card lists every change).
- UniRig weights: `export_weights.py skeleton`. `tools/unirig-harness/` holds the golden scripts (`golden_skintokens.py`, `golden_skeleton.py`, `golden_voxel_skin.py`), the reference environments
  (`st_env.py`, `ref_env.py`) and `serve-files.mjs`, a CORS + range file server for running the browser-lane tests against local weights.
- The UniRig PTv3 *learned* skin network (`golden_skin.py`, `skin_shims.py`, `export_weights.py skin`) is reference tooling only: it is not ported, SkinTokens replaced it. Measured while building it:
  the genuine spconv `SubMConv3d` is **not deterministic** when two points share a grid cell (a racy hash insert), so any reference run on duplicate cells differs run to run.

## Tests

`SkinTokens_*` (needs the export and goldens under `UNIRIG_DIR`, default `E:/spawndev-unirig`; skipped otherwise): mesh prefix, decoder logits (teacher forced, argmax under the grammar), cond latents,
VAE stage by stage, skin weights from the reference's own tokens, the exact-kernel end to end (token identity), fast kernels (valid rig), GLB in / rigged GLB out with a posed arm, and an HTTP-sourced
run for the browser lane. `UniRig_*`: tokenizer, voxel skin, sampler, rigged-GLB writer, encoder / decoder / skeleton against the reference, and the GLB rigger.

## Licences

SkinTokens and UniRig are MIT (VAST AI Research); the SkinTokens weights here are an MIT format conversion with attribution. Both reference projects vendor the Michelangelo shape encoder
(UniRig's copy is GPL-3.0); this port is an independent re-implementation from tensor shapes and the published behaviour and shares no code with it. Training data and any downstream use of the
generated rigs are the user's responsibility; check the licence of the mesh you rig.
