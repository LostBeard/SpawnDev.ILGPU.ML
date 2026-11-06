# Training: LoRA fine-tuning of GGUF language models

SpawnDev.ILGPU.ML can fine-tune a GGUF language model on the device that runs it. It trains LoRA adapters, the small low-rank matrices that
PEFT trains. The base weights stay frozen and quantized. The result is a standard PEFT adapter (`adapter_model.safetensors` +
`adapter_config.json`), which our inference engines load and which Python's `peft` also loads.

The smaller training API (`TrainableModel`: Conv2D, ReLU, max-pool, linear, SGD / Adam) is separate and not covered here.

## What it does

| Piece | Type | What it is |
|-------|------|------------|
| Graph autograd | `GraphAutograd` | Rewrites a model graph into a training graph: `AddLora` inserts `y = Wx + (alpha / r) B(Ax)` on chosen projections, `AddBackward` appends the backward pass. One `RunAsync` then computes the loss and every adapter gradient. |
| Trainer | `GgufLoraTrainer` | Loads a GGUF, builds the training graph, runs steps, AdamW (torch's order of operations), saves PEFT files. |
| Adapter files | `LoraAdapterFile`, `LoraAdapterSet` | Reads and writes PEFT safetensors + config. rsLoRA, DoRA and bias adapters are refused, not applied wrongly. |
| Inference with adapters | `LoraRuntime`, `GgufGenerator.CreateAsync(lora:)` | Applies adapters UNMERGED (merging a rank-r delta into Q4_K blocks would requantize it away). Works on both text engines. |

Supported graph ops in the backward pass: Gather (embedding), RMSNorm (and the fused add + norm), MatMul (quantized, low-precision and f32
weights), Reshape, RoPE (with frequency factors), causal grouped-query attention, SwiGLU, Add, Mul, softmax cross-entropy. These cover the
llama family as GGUFGraphBuilder builds it. Tested models: a tiny synthetic llama, SmolLM2-360M, llama3.1:8b (Q4_K_M). Other architectures
are untested; one using an op outside this list fails at graph build with the op's name.

Adapter targets: `attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down` (PEFT `q_proj` ... `down_proj`).

## Training

```csharp
using SpawnDev.ILGPU.ML.Training;

var options = new LoraOptions
{
    Targets = new[] { "attn_q", "attn_k", "attn_v", "attn_output" },
    Rank = 16, Alpha = 32, LearningRate = 2e-4f,
    CheckpointLayers = true,            // recompute activations in the backward: long sequences on big models
    LossOnLabeledPositionsOnly = true,  // LM head + loss only where a label is set (the reply), never the full-vocab logits of the prompt
};
using var trainer = await GgufLoraTrainer.CreateFromFileAsync(accelerator, "llama3.1-8b.gguf", options);   // streams; files over 2 GB are fine

// labels[t] = the token that should FOLLOW position t, or -100 (no loss: the prompt, padding)
foreach (var (ids, labels) in batch) await trainer.AccumulateAsync(ids, labels);   // sequences of any lengths, one at a time
await trainer.ApplyAsync();                                                        // one AdamW step on the mean gradient

var (safetensors, configJson) = await trainer.SaveAdaptersAsync("llama3.1:8b");
File.WriteAllBytes("adapter_model.safetensors", safetensors);
File.WriteAllText("adapter_config.json", configJson);
```

`StepAsync(ids, labels)` is one sequence, one step. `StepAsync(ids, labels, update: false)` gives the loss without changing anything (a dev
loss). Token ids outside the vocabulary and labels that are neither in range nor `IgnoreIndex` throw before the GPU sees them.

Memory: `CheckpointLayers` is what makes long sequences on large models fit. Without it, llama3.1:8b at 2,048 tokens keeps about 20 GB of
activations. With it and labeled positions only, a training run of llama3.1:8b on sequences up to 2,560 tokens used about 8.9 GB of a 12 GB
RTX 4070 (device total, measured with nvidia-smi).

## Inference with an adapter

```csharp
var set = LoraAdapterSet.Load(File.ReadAllBytes("adapter_model.safetensors"), File.ReadAllText("adapter_config.json"));
using var gen = await GgufGenerator.CreateAsync(accelerator, ggufStream, 4096, lora: set);
var result = await gen.GenerateAsync(promptIds, config, stopTokenIds: new[] { eot });
// result.GeneratedTokenIds (stop token excluded), result.StopTokenId (the id that ended it, or null)
```

On the batch decoder (CUDA, OpenCL, WebGPU) the adapters run as `LoraRuntime`, a down and an up kernel after each adapted projection. The
decoder's fused q|k|v and gate|up outputs take their adapters in one packed launch (`LoraRuntime.FuseAdjacent`). From 16 rows (a prefill)
the down projection is a tiled GEMM (`LoraRuntime.PrefillGemm`). `GgufBatchDecoder.Lora` can be swapped on a loaded model; setting it
forgets the cached prefix. On the session engine the adapters are inserted into the graph (`AddLora`) at load.

## How it is checked

| Gate | Test / tool | Result |
|------|-------------|--------|
| One LoRA step equals float64 PyTorch (loss, every gradient, updated adapters) | `Training_TinyLlama_LoraStepMatchesTorch` (+ `_Checkpointed`, `_LabeledOnly`), fixture from `tools/gen_tiny_llama_train_reference.py` | all 7 lanes (labeled-only: not WebGL) |
| Accumulating two sequences equals one step on both | `Training_TinyLlama_AccumulateTwiceEqualsOneStep` | green |
| Every backward kernel against finite differences or an exact reference | `Backward_*` in `MLTestBase.BackwardKernelTests.cs` | green |
| 12 steps on SmolLM2-360M track a PyTorch run | `tools/TrainLoraCurve`, `tools/gen_lora_curve_reference.py` | per-step loss within 1.3e-5 (CUDA, OpenCL) |
| Gradients at real lengths on llama3.1:8b | `tools/TrainLoraProbe` (directional derivative) | ~1e-3 relative at 128 / 512 / 1,024 / 2,048 tokens |
| The PEFT files load in Python | `tools/check_tiny_lora_peft_files.py` | green |
| Decoder + adapters reproduce the training forward | `Training_DecoderLora_MatchesTrainingForward` | CUDA, OpenCL, WebGPU |
| Both engines generate the same tokens with the same adapter | `Training_GeneratorLora_BothEnginesAgree` | green |
| Steps do not grow device memory | `Training_StepsDoNotGrowThePool` | +0 over 5 steps |

## Speed

llama3.1:8b Q4_K_M, one sequence of 1,600 tokens, rank 16 on q/k/v/o, checkpointed (forward, recompute, backward):

| GPU | Step | Before (2026-11-05 morning) |
|---|---|---|
| RTX 4070 (sm_89, bf16 / f16 tensor cores) | **1.78 s** | 21.7 s |
| RTX 2060 (sm_75, f16 tensor cores) | **4.53 s** | 31.5 s |

Every speedup is exact: bit-identical to the path it replaced, or gradient-checked (the fused attention backward and LoRA kernels change
summation order only). 2,560 tokens with a 200-token answer: 2.65 s on the RTX 4070, 8.7 s on the RTX 2060, both within 12 GB. A step's
memory grows with the number of LABELLED positions (the LM head's logits and their gradient over the vocabulary): label the answer, not the
prompt (`LossOnLabeledPositionsOnly`).

Where the time goes: the 16-bit tensor-core GEMMs, ~70% of a step on the RTX 4070 (~52 TFLOPS of its ~58 fp32-accumulate peak) and ~80% on
the RTX 2060, whose GEMMs are DRAM-bound (336 GB/s, 3 MB L2). Profile with `tools/TrainLoraProbe --profile` (per op); add
`GGUF_KERNEL_TIMING=1` for the pieces inside the GEMM ops (dequantize, pack, scale, GEMM).

Each fast path has a switch for A/B:
- attention: `BackwardKernels.UseFlashAttentionBackward`, `UseTensorCoreAttentionBackward`, `UseGemmAttentionBackward`,
  `GraphAutograd.TensorCoreForwardAttention`;
- graph: `GraphAutograd.FusedLora`, `FuseGradientSums`, `GroupSiblingProjections` (q | k | v and gate | up as one GEMM, SwiGLU in the gate | up
  epilogue);
- GEMMs: `FusedDequantMatMul.EnableMmaScratch`, `EnableMmaV5` / `EnableMmaV6` (sm_80+), `EnableMmaSm75V5` / `EnableMmaSm75V6` /
  `Sm75V7Mode` (Turing; env `GGUF_GEMM_SM75_V5` / `_V6` / `_V7`), `BackwardKernels.UseTensorCoreQuantT`, `UseDequantTranspose64`.

## Tools

| Tool | What it does |
|------|--------------|
| `tools/TrainLoraJsonl` | Trains on JSONL lines (`prompt_ids`, `target_ids`, `end_of_turn_ids` or `--eot`, `split` train / dev, optional `id` and `expected_tool`), saves the adapter, then generates on the dev split base vs adapter (exact match, and tool choice when lines carry `expected_tool`). `--rank --alpha --lr --accum --epochs --targets --max-len --checkpoint --no-dev-loss --eval-only --engine --exclude-source --progress --save-every --gpu-share --dry-run` |
| `tools/TrainLoraProbe` | One step at a chosen length: time, memory, gradient check, `--profile` per op (with `GGUF_KERNEL_TIMING=1`, per piece inside the GEMM ops), `--seq --steps --rank --targets --checkpoint --labeled N --live --opencl`; for the A/B `--no-tc` / `--no-gemm-attn` / `--no-tc-attn` / `--no-flash-attn` / `--no-group` / `--no-fused-lora` / `--no-sums` / `--no-scratch` |
| `tools/LoraDecodeBench` | Decode speed and time to first token, base vs adapter (`--unfused`, `--no-prefill-gemm`) |
| `tools/TrainTinyLora`, `tools/TrainLoraCurve` | The small reference runs behind the gates above |

## Limits

- LoRA only: no full fine-tuning, no QLoRA-style quantized adapters, no rsLoRA / DoRA / bias.
- Labeled-positions-only training uses a scatter kernel that WebGL cannot run; WebGL trains with full logits.
- One sequence at a time (batches by gradient accumulation).
