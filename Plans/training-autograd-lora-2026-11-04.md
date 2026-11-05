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
  - [x] M1.4 LoRA rewrite + AdamW (`Training/GgufLoraTrainer.cs`) + PEFT files (`Training/LoraAdapterFile.cs`, `SaveAdaptersAsync`): reload exact; Python safetensors reads them (tools/TrainTinyLora + check_tiny_lora_peft_files.py, 6e-8 vs torch).
- [x] **M2** (2026-11-04: smollm2:360m F16 GGUF, 12 AdamW steps, per-step loss vs float64 torch <= 1.3e-5 rel on CUDA and OpenCL, loss 14.37 -> 4.10; tools/TrainLoraCurve + tools/gen_lora_curve_reference.py) train a small real llama GGUF (SmolLM2-135M class) on a toy task; loss curve within tolerance of a PEFT run
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

### Gate settled (2026-11-04)

TJ (via Seven): "Family chat out of training for now. Both in one adapter." Seven's reference measurements on the 2060
(`_DevComms/2026-11-04-1059-Seven-TO-Tuvok-reference-measurements.md`, scorer Vertex 630672c `Vertex.Tests/ToolChoiceScoring.cs`):
strict tool choice, 5 runs: full prompt 93/100, names-only prompt (descriptions + schemas emptied) 59/100. Decode barely
moves (24.4 -> 26.7 tok/s), but turn time-to-first-token 1,175 -> 682 ms median (-42%) and cold prefill 4,238 -> 2,055 ms
(-52%). So the M4 gate: ONE adapter (synthetic tool-choice + real honesty turns) that (1) scores >= 93/100 strict on the
names-only prompt with TTFT/prefill down, and (2) beats the base on the held-out honesty set. Data (Seven): ~30 synthetic
requests per tool + ~15% no-tool lines, 20% synthetic dev split, targets DISTILLED from base + full prompt and kept only when
the strict scorer passes; honesty targets are the corrected sentence, not the live one.

## M3 log (llama3.1:8b Q4_K_M, RTX 4070, tools/TrainLoraProbe)

- 2026-11-04 11:11-11:16, naive transposed kernels, seq 128, rank 8 on q/v (64 adapters): load + compile 4.5 s; process VRAM
  +6.05 GB (weights + one step's activations); directional derivative grad . v 1.35489 vs central difference 1.35522
  (2.4e-4 relative: gradients correct through Q4_K / Q6_K, GQA 32/8, rope_freqs); 3 AdamW steps 12.66 -> 12.05 -> 11.48;
  39.5 s per step (the naive MatMulTransposedQuant decodes every weight once per output row).
- Large files: GgufLoraTrainer.CreateFromFileAsync streams (CreateFromGGUFStreamAsync / FileAsync gained graphTransform).
- Inference with adapters: LoraRuntime (unmerged, two small kernels per adapted projection) hooked into GgufBatchDecoder
  (the engine GgufEngine.Auto picks for llama on CUDA/OpenCL/WebGPU: Vertex's), including the fused q|k|v and gate|up outputs.
- Training pairs: GenerationResult.GeneratedTokenIds (both engines, stop token excluded) so Data's PassObserver can record
  the ids the model generated; prompt_ids already in Seven's format 3.
- 2026-11-04 11:18-11:20, tiled transposed-quant kernel: identical loss / gradient check / trajectory, 6.5 s per step (6x).
  GgufBatchDecoder + LoraRuntime vs the training forward with the same adapters: max |diff| 9.3e-3 (max |logit| 8.5),
  same argmax; the base decoder differs by 14.6 (the adapters' effect). Blocked (shared-memory) attention backward added
  for long prompts (the per-element one is O(seq^2 hd) per output value); FD-checked across block boundaries, 7/7 lanes.
- 2026-11-04 11:41-11:49, checkpointed (LoraOptions.CheckpointLayers) seq 512: directional derivative 4e-4, loss 12.69 -> 12.29,
  14.5 s/step; decoder + LoraRuntime vs training forward 6.8e-3. GraphCompiler's topological sort is now stable (it
  ran recompute nodes in the forward under FIFO). Long-sequence VRAM (1024 / 2048) pending a quiet card (Geordi).
- tools/TrainLoraJsonl: format-3 JSONL -> PEFT adapters with dev loss and a generation eval through GgufGenerator +
  gen.Lora. Synthetic smollm2 check: dev loss 13.87 -> 0.29, dev generation base 0/24 -> adapters 13/24 exact.
- Inference API for AI (Data): LoraRuntime.Load(accelerator, safetensors, adapter_config.json); generator.Lora = it.

### Next
- [ ] 8B probe at 1024 / 2048 tokens, checkpointed (quiet card), then a full default sweep (the compiler change) and local.10.
- [ ] M4 run on Seven's data once target_ids flow (AI 1.4.1-local.2 has them): ToolChoiceTests strict on the short prompt
      >= 93/100 with TTFT down, honesty above the base; on the 2060 (GATEGAMER) for the production-card timing.

## M4 runbook (the 2060 on GATEGAMER, Vertex stopped)

1. Data: Seven's format-3 lines with `prompt_ids` + `target_ids` (AI 1.4.1-local.2 passes GeneratedTokenIds; end-of-turn id from
   `end_of_turn_ids`), synthetic tool-choice (short prompt, distilled from base + full prompt) and real honesty turns.
2. Train: `dotnet run --project tools/TrainLoraJsonl -c Release -- <llama3.1:8b blob> <train.jsonl,...> --out <dir> --checkpoint
   --rank 16 --targets attn_q,attn_k,attn_v,attn_output --lr 2e-4 --accum 8 --epochs 3 --max-len 2560`; it keeps the best dev loss and
   reports greedy dev generation through GgufGenerator (base vs adapters).
3. Gate in Vertex (Seven): load the adapter (`GgufGenerator.CreateAsync(lora: LoraAdapterSet.Load(...))`, AI's SetAdapterAsync), run
   ToolChoiceTests strict x5 on the names-only prompt (>= 93/100) and measure TTFT / prefill; honesty set vs the base.
4. Decode speed: adapters cost two small dispatches per adapted projection per step; measure tok/s with and without on the 2060.

## Open: Wasm MoveNet regression from the stable topological order (2026-11-04 sweep, 7481/2)

`CreateFromFile_MoveNet_Inference` and `Pipeline_MoveNet_DetectsKeypoints` fail on the Wasm lane only:
`[GE node-192 sync] Arg_NullReferenceException` - an error from a QUEUED Wasm kernel surfacing at the drain after the
zero-copy Unsqueeze view at node 192. tools/DumpCompiledOrder: under the stable order `Cast(concat) -> GatherNd_3__439`
(190) is followed directly by its consumer `GatherND(Transpose__510, GatherNd_3__439)` (191) and the view chain; under FIFO
several nodes (and view drains) sat between them. GatherND reads its indices on the device (no host read). Next: the Wasm
lane with GraphExecutor.PerOpSync and GraphCompiler.StableTopologicalOrder = false/true A/B, to name the kernel that faults.
Held until Geordi's AubsCraft suite ends (a Wasm lane slowed his Paper servers 418 s before). Blocks local.10.

## Training speed (TJ 2026-11-05: "training takes a LONG time ... plans to improve?")

Measured baseline (llama3.1:8b Q4_K_M, RTX 4070, 1,600 tokens, rank 16 q/k/v/o, checkpointed, labeled-only): 21.7 s per step on an idle
GPU (re-measured 2026-11-05 in the real loop: 21 s per line); the v1 run averaged 42.8 s per line on a shared GPU. Per-op profile
(`TrainLoraProbe --profile`): MatMulGradInput (dx through the frozen quantized weights) 8.45 s, FusedAttentionGrad 9.24 s, the rest
(forward + checkpoint recompute + small ops) ~4 s. The inference engine runs the same forward at ~1,640 tok/s with tensor cores, so a
step is ~4 forwards of work: target 4-5 s per step (a 7 h run in ~1.5 h).

- [x] MatMulGradInput on tensor cores (2026-11-05): W decoded + transposed once per call to bf16 scratch, then Trellis2.MmaGemm bf16
      (`BackwardKernels.UseTensorCoreQuantT`, default on, CUDA sm_80+). bf16 because gradients sit below f16's range. Gate
      `Backward_MatMulTransposedQuant_TensorCores_MatchReference`: 2.5e-3 relative vs float64 (same with dy x 1e-7), red 1.29.
      Qwen2.5-0.5B Q4_K_M probe: directional derivative 7.4e-4 / 2.9e-4 (SIMT 1.2e-3 / 1.9e-3), same loss curve, step 1,253 vs 1,520 ms
      (shared GPU). 8B measurement pending (after the v2 run).
- [x] Attention backward as batched GEMMs (2026-11-05, `BackwardKernels.UseGemmAttentionBackward`, default on, not WebGL): the blocked
      kernels ran ~42 GFLOP at ~4.5 GFLOPS. Per chunk of query heads (score buffers within `GemmAttentionScoreBudgetBytes`, 640 MiB):
      S = Q K^T, lse and D from S, P, dP = dO V^T, dS, dQ = dS K, dK = dS^T Q, dV = P^T dO on the register-blocked batched matmul, fp32;
      GQA heads summed into their kv head. `Backward_Attention_MatchesFiniteDifference` covers GEMM / blocked / per-element (red: dS without
      the scale, dq off by 0.57); TinyLlama LoRA step vs torch green on CUDA + OpenCL. Qwen2.5-0.5B 512 tokens: FusedAttentionGrad
      954 -> 68 ms (14x, shared GPU). 8B measurement pending.
- [ ] Attention backward on tensor cores (FlashAttention-2 style) only if the GEMM path is still a large share after the 8B profile.
- [x] 8B step at 1,600 tokens: 21.7 -> 5.9 s with the two items above (profiled: attention backward 9.26 -> 0.94 s, quantized input
      gradient 8.48 -> 1.13 s; gradient check 2.5e-3 / 5.3e-4). Remaining: forward MatMul 1.6 s (already MmaKQuantGemm: it is two forwards
      with the checkpoint recompute), forward FusedAttention 1.5 s.
- [x] Training forward attention on MmaFlashPrefill (2026-11-05): GraphAutograd marks a training graph's FusedAttention nodes `train_tc`
      (`GraphAutograd.TensorCoreForwardAttention`, default on); the operator converts K / V to f16 and runs the batch decoder's prefill kernel
      (CUDA sm_80+, hd 128, causal, no window / sinks). Inference graphs are never marked. `AttnOp_FusedAttention_TrainTensorCore_MatchesCPU`:
      2.8e-4 vs float64 (generic 1.2e-7). PENDING: the 8B gradient check + step time with it (after v2.1).
- [ ] Then: packing several short sequences per step; skipping work that has no gradient.
- [x] Turing (sm_75, Seven's RTX 2060) paths (2026-11-05). 2060 baseline (Seven, 08b95dd9): 31.5 s per 1,600-token step; MatMulGradInput
      18.3 s (58%, SIMT there: bf16 needs sm_80), FusedAttention 5.4 s (17%). Now: the quantized input gradient as f16 MMA (m16n8k8 pairs,
      `Trellis2.MmaGemm.KernelF16Sm75Scaled`, generated) with a device-side power-of-two scale from max|dy| (no readback; exact to undo),
      and training attention on `MmaFlashPrefill.KernelSm75`. Gates: the tensor-core test runs both paths (Turing f16 ~3e-4 vs float64,
      also at dy x 1e-7; red: scale off -> 0.30), the attention op test runs both kernels (2.8e-4). 8B on the 4070 with
      GGUF_GEMM_MMA_SM75=1: 4.5 s per step, directional derivative 4.4e-4 / 3.1e-4. 2060 timing: Seven.
- [x] Attention backward GEMMs on tensor cores (2026-11-05, `BackwardKernels.UseTensorCoreAttentionBackward`, CUDA sm_75+, hd % 32 == 0):
      S = Q K^T and dP = dO V^T with K / V as the 16-bit MmaGemm operand; dQ / dK / dV with K^T / Q^T / dO^T; sequence axes zero-padded to
      32; bf16 on sm_80, f16 + device power-of-two scales (dO, dS) on sm_75. `Backward_Attention_TensorCores_MatchFp32`: bf16 <= 4.6e-3,
      f16 <= 5.2e-4 vs the fp32 path, also at dout x 1e-6 (red: dV not unscaled -> fails). 8B on the 4070: 4.5 -> 4.18 s per step
      (FusedAttentionGrad 937 -> 681 ms), grads 2.1e-3 / 7.4e-4 (forced sm_75 1.1e-3 / 1.9e-4); VRAM unchanged vs the fp32 path.
      The rest of FusedAttentionGrad is now the row stats (a serial loop per row), P / dS, two transposes and the packs: next.

