# UniRig harness (auto-rigging port)

References the real UniRig (github.com/VAST-AI-Research/UniRig, MIT; `src/model/michelangelo/` is GPL-3 upstream Michelangelo, so the C# port re-implements the
architecture independently and is checked against the reference as a black box). Everything big lives on **E:\spawndev-unirig** (C: and D: are nearly full):
`ckpt/` (the two released checkpoints), `ref/` (git clone), `venv/` (Python 3.12, CPU torch), `golden/`, `inputs/`, `weights/`.

- `reference/safe_ckpt.py` loads a checkpoint through an allowlist unpickler (only the names the two files import). Run Python with `-I`.
- `reference/ref_env.py` imports the reference on CPU: mocks for CUDA-only packages (`spconv`, `torch_scatter`, `torch_cluster`, `bpy`, ...), a pure-torch farthest-point
  sampler in place of `torch_cluster.fps`, CUDA attention paths switched off. A mock hides a call, so configs here set `flash False` / eager attention explicitly.
- `reference/golden_skeleton.py <mesh.glb> [seed] [max_new_tokens]` writes `golden/skeleton/` (points, normals, presample + fps indices, encoder latents, greedy token ids, bones).

Skeleton stage facts (MEASURED on a TRELLIS-generated T-pose humanoid, CPU float32): eval-mode encoder pre-samples 4096 of the 65536 points with numpy `default_rng(0).choice`,
farthest-point-samples 1024 of them (start index 0) as query tokens, cross-attends to the full cloud, 16 self-attention layers -> [1, 1024, 1024] prefix for OPT-350m;
greedy decode with the tokenizer grammar gave 93 tokens = 25 bones (spine, neck, head, arms, legs, feet) in 20 s on CPU; encoder 4.7 s.

## Rigging a character (all C#, no Python at run time)

`UniRigHarness riggl <in.glb> <out.glb> [seed]` = `UniRigRigger.RigGlbAsync`: welds seam duplicates, UniRig skeleton stage on the device (28 joints for the TRELLIS humanoid),
UniRig's skinning prior (geodesic voxel binding), then `RiggedGlb.Add` patches joints / skin / JOINTS_0 / WEIGHTS_0 into the file (textures untouched). 5.6 s cold on an RTX 4070.
Checks: `reference/validate_glb.mjs` (npm `gltf-validator`, installed under E:\spawndev-unirig\tools: 0 errors / 0 warnings) and `reference/pose_check.py` (rotates a shoulder, applies
linear blend skinning to the weights read back from the file: 17,987 vertices follow the right arm, the left half moves 0.000, torso/head/legs 0.05% of the height).
The weights are the PRIOR only; the learned skin network (PTv3 + attention, in the skin checkpoint) refines them and is not ported yet.
