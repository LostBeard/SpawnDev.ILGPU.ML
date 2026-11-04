# Real training in SpawnDev.ILGPU.ML: graph autograd, LoRA first (2026-11-04)

## Direction

TJ (2026-11-04, via Seven, `_DevComms/2026-11-04-0937-Seven-TO-Tuvok-real-training-in-ML.md`): "Training is something we
have promised in SpawnDev.ILGPU.ML since day 1 and it is currently VERY limited in that area, and I would like to change
that. I think we could drastically improve our ML capabilities."

First customer: one LoRA on GGUF `llama3.1:8b`, trained on Vertex's real play, kept only if it beats the base model on
`ToolChoiceTests` on a held-out split with decode speed not worse. The machinery is general; LoRA is its first user.

## Today

`Training/` = a hand-written MLP (`TrainableModel`: Linear + ReLU, per-layer Backward, `TrainingKernels`: ReLU / MaxPool2D /
SoftmaxCrossEntropy backward). Nothing backpropagates through attention, norms or quantized matmuls; no optimizer state,
no checkpointing, no LoRA for language models (`StableDiffusion/SdLora.cs` is load-and-patch only).

## Shape

1. **Graph-level reverse-mode autograd.** GGUF models already run as an op graph (`GGUFGraphBuilder` -> `GraphExecutor`).
   Each op type gets a gradient rule that appends backward nodes to the graph, so backward runs on the same executor,
   pool, planned reuse and kernels as inference, on every backend that runs the forward. llama's op set: `Gather`
   (embedding, frozen), `RMSNormalization` / `AddRMSNorm`, `MatMul` over quantized weights (FusedDequantMatMul),
   `FusedAttention` (+ RoPE), `SwiGLU`, `Add`, `Mul`, `Reshape` / `Transpose`, plus a cross-entropy loss.
2. **Frozen quantized base** (QLoRA-style). Input gradients through a frozen Q4_K / Q6_K weight need one new kernel family:
   the fused dequant matmul against the TRANSPOSED weight (dX = dY W^T). No weight gradients for the base.
3. **LoRA as a graph rewrite**: A (r x in), B (out x r), scale alpha / r, beside chosen projections (q/k/v/o first, MLP
   optional); B initialised to zero, A Kaiming-uniform (PEFT defaults). Saved as safetensors with PEFT names
   (`base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight`, ...) so it can be checked against Hugging Face PEFT.
4. **AdamW** (decoupled weight decay), gradient accumulation, per-layer **activation checkpointing** (keep each decoder
   layer's input; recompute the layer in backward), loss only on target tokens with logits computed only there, chunked.
5. **Inference: unmerged adapter** as a fused low-rank epilogue on the projection. Merging into Q4_K would requantize the
   delta (a rank-16 delta is likely below the Q4 step); merging into fp16 needs ~16 GB.

## Estimates (to be measured, not trusted)

- llama3.1:8b on a 12 GB RTX 2060: base Q4_K_M ~4.9 GB; checkpoints 32 x T x 4096 x 4 B = 0.5 GB at T = 1024; one
  layer's recompute working set a few hundred MB; LoRA + AdamW state < 0.3 GB. Fits at T = 1024-2048; T = 3,400 fits if
  attention backward does not materialise T^2 per head.
- ~50 GFLOP per trained token with recompute: a 3,400-token prompt x hundreds of examples is hours per epoch on the 2060.
  Training with a SHORT prompt (no tool definitions, the adapter carries them) is ~10x cheaper and is Seven's goal 3.

## Milestones

- [x] **M1** (2026-11-04: `Training_TinyLlama_LoraStepMatchesTorch` 7/7 lanes: loss, adapter grads ~1.5e-6 rel, AdamW 1.2e-7) autograd + LoRA on a tiny llama GGUF (synthetic f32 weights, `BuildTinyLfm2GGUF`'s pattern, plus a quantized
      variant). Gate: every gradient rule against finite differences and against PyTorch; full-model loss, input and
      adapter gradients against torch + PEFT on the same weights; one AdamW step equal. CUDA / OpenCL / WebGPU.
  - [x] M1.1 oracle: `tools/gen_tiny_llama_train_reference.py` (f32 tiny llama, GQA 4:2, LoRA q/v rank 4) -> `wwwroot/test-train/`; `Training_TinyLlama_ForwardMatchesTorch` 7/7 lanes, logits max |diff| ~3e-6 (2026-11-04).
  - [x] M1.2 autograd core: `Training/GraphAutograd.cs` (AddLora + AddBackward graph rewrite), `CreateFromGGUF(..., graphTransform)` hook.
  - [x] M1.3 rules + gradient kernels (75cdc277), CE loss op on device per ONNX.
  - [~] M1.4 LoRA rewrite + AdamW (`Training/GgufLoraTrainer.cs`) done; PEFT safetensors save / load TODO.
- [ ] **M2** train a small real llama GGUF (SmolLM2-135M class) on a toy task; loss curve within tolerance of a PEFT run
      with the same data, seed and hyperparameters.
- [ ] **M3** llama3.1:8b on the 2060: Q4_K / Q6_K transposed-dequant kernels, checkpointing, measured VRAM and tokens/s;
      fused unmerged LoRA at decode with a decode-speed gate.
- [ ] **M4** Seven's gate: a LoRA trained on Vertex's play beats the base on `ToolChoiceTests` (held-out, split by game
      session), decode not slower. If it does not, we learned something and do not ship it.

## Data (asked of Seven, 2026-11-04)

JSONL, one example per line: `id`, `split` (train | heldout, by game session), `messages` (exactly what Vertex sends),
`tools`, `target_turn`, `truth`, `source`, plus `rendered_prompt` / `rendered_target` from SpawnDev.AI's llama3.1 chat
template at inference, checked line by line against our renderer.

### Seven's update (2026-11-04, later)

- `ToolChoiceTests` has 20 cases now (the 16-17/18 figure was October); no baseline pinned until Seven's fresh run on
  llama3.1:8b (per-case results + exact command).
- Data: `turns-YYYY-MM-DD.jsonl` under Vertex's data folder. `rendered_prompt` / `rendered_target` are null until Data
  adds a hook in SpawnDev.AI's AiChatEngine (tool rounds run inside it); tool calls arrive as a `tool_runs` string list.
  Split: one session in five held out (hash of the session id). Lines without `rendered_target` are not trainable.
- PRIVACY: only game-event turns are logged; nothing a person said, nor facts about a person (Vertex's privacy page
  promises chat is not logged, and an adapter would carry it). Real tool-choice requests from people are excluded until
  TJ decides. Proposed: synthetic tool-choice requests, disjoint from the 20 test cases (checked automatically); the
  test cases stay held out as the gate and are never trained on.

## M1.2 / M1.3 design (2026-11-04, from the tiny llama's actual graph)

The llama graph GGUFGraphBuilder emits (dumped from tiny-llama-train.gguf): `Gather(token_embd, input_ids)`,
`RMSNormalization(x, w)`, `MatMul(x, W[K,N])` (weights transposed to [in, out] on upload), `Reshape`, `RoPE`,
`FusedAttention(q, k, v)` (causal, GQA inside), `AddRMSNorm(a, b, w) -> (res, norm)`, `SwiGLU(g, u)`, `Add`, final
`RMSNormalization` + `MatMul(.., output.weight) -> logits`.

Hook: `CreateFromGGUF` gets a graph transform (graph + cpu weights) applied after BuildGraph and before compile; the
LoRA rewrite adds f32 initializers, the autograd appends backward nodes and graph outputs (loss, parameter grads) and
inputs (targets, loss mask). Trainable parameters are the session's own weight tensors, updated in place by AdamW
after each RunAsync. Shape-recompile keeps working because the transformed graph is the one it re-compiles.

Backward rules (each a graph rewrite; gradients of a value with several consumers are summed with Add):
| forward | backward | op used |
|---|---|---|
| Add(a, b) | da = db = dy (same shapes in llama) | - |
| MatMul(x, W) | dx = dy W^T; dW = x^T dy (trainable W only) | `Gemm` transB / transA (+ Reshape(Shape(x))) |
| Reshape(x) | dx = Reshape(dy, Shape(x)) | `Shape` + `Reshape` |
| RMSNormalization(x, w) | dx = rstd * (w dy - xhat * mean(xhat w dy)) | NEW `RMSNormGrad` |
| AddRMSNorm(a, b, w) | dres = dres_out + RMSNormGrad(res, w, dnorm); da = db = dres | NEW `RMSNormGrad` + Add |
| SwiGLU(g, u) | dg = dy u silu'(g); du = dy silu(g) | NEW `SwiGLUGrad` (2 outputs) |
| RoPE(x) | rotate dy by -theta | NEW `RoPE` inverse (attribute) |
| FusedAttention(q, k, v) | dq, dk, dv (causal, GQA: sum dk/dv over the head group) | NEW `FusedAttentionGrad` (naive first, flash at M3) |
| Gather(table, ids) | stop (table frozen) | - |
| loss | masked mean softmax-CE over target positions, emits dlogits | NEW `SoftmaxCrossEntropyLoss` (2 outputs) |

Each NEW op: CPU-oracle kernel test + finite-difference check, all backends; then the full-model gate against
tiny-llama-train-ref.json (loss, LoRA grads, one AdamW step).

### M4 gate replaced (2026-11-04, Seven's baseline: ToolChoiceTests 20/20 on llama3.1:8b, one run, temp 0.7)

"Beats the base on ToolChoiceTests" has no room. New gate, in order:
1. PRIMARY, prompt shrink: adapter + SHORT prompt (no tool definitions) equals base + FULL prompt on ToolChoiceTests with a
   strict scorer (arguments too), 5 runs per configuration (per-case pass rate), and decodes faster on the 2060. Before
   training: measure (a) base + full prompt and (b) base + short prompt with the strict scorer; (b)'s gap is the room.
2. SECOND, honesty: beat the base on Seven's held-out real game-event set scored against `truth`.
3. Only if both are empty: a harder tool-choice set (instinct hint off, several runs).
Data: format 2 lines carry the engine's exact rendered_prompt / rendered_target (Data's PassObserver, SpawnDev.AI.Server
1.4.1-local.1); the target excludes the end-of-turn token, which training appends so the adapter learns to stop.
