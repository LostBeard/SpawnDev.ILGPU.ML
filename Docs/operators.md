# ONNX operator support

**This file is the human single source of truth** for what "100% ONNX support" means, what we advertise today, and every remaining op TODO.

**Code single source of truth (advertised set):** [`OperatorRegistry.BuiltinOpTypes`](../SpawnDev.ILGPU.ML/Operators/OperatorRegistry.cs)

That set is the only authoritative list of op-types this engine **advertises** as supported right now. Everything that answers "do we support X?" must point at it — never maintain a second advertised list.

| Consumer | How it reads support |
|----------|----------------------|
| Live registry (`RegisterBuiltins`) | Locked to `BuiltinOpTypes` by `MLTestBase.Op_BuiltinOpTypes_MatchesLiveRegistry` |
| Model Inspector (demo `/inspector` + `ModelInspectorHelper`) | `KnownSupportedOps` → `BuiltinOpTypes` |
| Home page operator count | `BuiltinOpTypes.Count` (live; never hardcode) |
| `tools/audit-operator-support.cs` | Audits advertise vs `OpType =>` impl + silent-fallback shapes |

**Do not paste the advertised op list into README or UI copy.** Counts go stale the moment someone adds a `Register(...)`. Link here, or read `BuiltinOpTypes.Count` from code. **Do** keep the TODO / gap sections in *this* file current when policy or remaining work changes.

## Goal: 100% ONNX

"100% ONNX support" for this library means:

1. **ONNX standard opset** — every op-type models use from the empty / `ai.onnx` domain (real semantics, not stubs).
2. **`com.microsoft` contrib** — every op-type in ONNX Runtime's Microsoft contrib domain. Optimum / ORT / onnxruntime-genai fused exports (BERT blocks, GQA, MatMulNBits, etc.) are first-class, not optional.

The Model Inspector matches on **op-type name only** (domain is ignored). A model that reports `Missing: Attention, BiasGelu, SkipLayerNormalization` is correctly failing: those names are not in `BuiltinOpTypes`.

Inventory baseline: ORT [`ContribOperators.md`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md) TOC (audited 2026-09-28) plus `SimplifiedLayerNormalization` (used by Florence / Phi vision encoders; not always listed in that TOC).

| Bucket | Count |
|--------|------:|
| `com.microsoft` names in baseline | 122 |
| Already covered by a same-named entry in `BuiltinOpTypes` | 12 |
| **Still TODO (not advertised)** | **110** |

Already covered by name overlap (standard or prior alias — not the same as full contrib-schema parity):  
`DequantizeLinear`, `GatherND`, `Gelu`, `GridSample`, `MoE`, `Pad`, `QLinearConv`, `QuantizeLinear`, `Range`, `SimplifiedLayerNormalization`, `Trilu`, `Unique`.

## What "supported" means

Being in `BuiltinOpTypes` means the op-type is **registered** and `IsSupported` / the Model Inspector will say yes. Execute must implement real semantics (or throw naming what is missing) — never Fill-zeros / CopyFrom-identity / Scale(×1) as a silent stand-in.

For whether a **demo / model** actually works end-to-end, see [DEMO_AND_MODEL_STATUS.md](DEMO_AND_MODEL_STATUS.md).

## Value types

Float activations stay on GPU as [`Tensor`](../SpawnDev.ILGPU.ML/Tensors/Tensor.cs). Non-float ONNX types use host [`OnnxValue`](../SpawnDev.ILGPU.ML/Tensors/OnnxValue.cs) (`Tensor | Sequence | Optional | String | Bytes`) via `OnnxOpContext.HostValues`.

## Implementation notes (code-verified)

### Correct inference no-ops

| Op | Behavior |
|----|----------|
| `Dropout` | Inference pass-through (`CopyFrom`). Correct per ONNX. |
| `Identity` | `CopyFrom` input → output. |
| `SequenceEmpty` | Writes an empty host Sequence. |

### Host `OnnxValue` ops

| Group | Ops | Notes |
|-------|-----|--------|
| Sequence | Construct, Empty, At, Insert, Erase, Length, Map, ConcatFromSequence, SplitToSequence | Real sequence semantics on `HostValues`. `SequenceMap` runs the `body` subgraph via `SubgraphRunner` (one pass per element; throws if `body` is missing). |
| Optional | Optional, OptionalGetElement, OptionalHasElement | Real optional wrap/unwrap. |
| String | StringConcat, StringNormalizer, StringSplit | Host strings (not float Scale). |
| ImageDecoder | ImageDecoder | PNG bytes (`OnnxValue.Bytes`) → float RGB via `PngDecoder`. JPEG/BMP: throw (pre-decode outside or extend). |

### GPU-wired detection / ROI

| Op | Notes |
|----|--------|
| `MaxRoiPool` | GPU kernel (`ElementWise.MaxRoiPool`), 4- or 5-element rois. |
| `RoiAlign` | GPU kernel; optional `batch_indices` staged into fparams. |
| `AffineGrid` | GPU; size from constant or inferred output shape (never silent zeros). |
| `MeanVarianceNormalization` | GPU for NCHW keep-channel (axes `{0,2,3}`) and contiguous suffix reductions; otherwise host via `Require`/`RequireAsync`. |

### CPU math with honest host read

These use `OperatorInputReader.Require` (sync readback on desktop; throws if unreadable — never silent zeros): `Det`, `NonMaxSuppression`, `NegativeLogLikelihoodLoss`, `SoftmaxCrossEntropyLoss`, `Unique`. `MeanVarianceNormalization` uses the same for non-GPU axes only.

### Aliases / contrib already landed

| Op-type | Notes |
|---------|--------|
| `RMSNormalization` | True RMS (no mean subtraction). |
| `SimplifiedLayerNormalization` | ORT contrib ≈ RMSNorm + optional bias. |
| `LayerNormalization` | Mean-centered — not interchangeable with the above. |

Internal (not ONNX / not ORT contrib): `FusedAttention`, `FusedLinear`, `FusedScaledMatMul`, `RoPE`, `ShortConv`, `GatedDeltaNet`, `AddRMSNorm`, `SiLU`, `SwiGLU`, etc. These do **not** satisfy inspector checks for ORT names like `Attention` or `RotaryEmbedding`.

---

## TODO — `com.microsoft` contrib gaps (110)

When an op below is implemented: add `Register(...)` + `BuiltinOpTypes` entry, cover with a CPU-oracle test, then **strike it from this section** (or move to "Aliases / contrib already landed"). Do not claim support only in prose.

Priority within packages follows what real Downloads / Optimum / genai models hit; order is guidance, not a gate that blocks later packages.

### P0 — Optimum BERT / encoder fused block (inspector hit)

These are the three that produced `82% compatible … Missing: Attention, BiasGelu, SkipLayerNormalization`.

- [ ] `Attention`
- [ ] `BiasGelu`
- [ ] `SkipLayerNormalization`
- [ ] `FastGelu`
- [ ] `EmbedLayerNormalization`
- [ ] `SkipSimplifiedLayerNormalization`

### P1 — Attention family

- [ ] `MultiHeadAttention`
- [ ] `GroupQueryAttention`
- [ ] `PackedAttention`
- [ ] `PackedMultiHeadAttention`
- [ ] `PagedAttention`
- [ ] `QAttention`
- [ ] `QOrderedAttention`
- [ ] `SparseAttention`
- [ ] `LongformerAttention`
- [ ] `LinearAttention`
- [ ] `LinearAttentionGate`
- [ ] `DecoderAttention`
- [ ] `DecoderMaskedMultiHeadAttention`
- [ ] `DecoderMaskedSelfAttention`
- [ ] `QOrderedLongformerAttention`

### P2 — Norms / fused activations / dropout

- [ ] `BiasSoftmax`
- [ ] `BiasDropout`
- [ ] `BitmaskDropout`
- [ ] `BitmaskBiasDropout`
- [ ] `BiasSplitGelu`
- [ ] `BiasAdd`
- [ ] `QuickGelu`
- [ ] `GemmFastGelu`
- [ ] `GroupNorm`
- [ ] `SkipGroupNorm`
- [ ] `GatedRMSNorm`
- [ ] `GatedAdd`
- [ ] `QOrderedGelu`
- [ ] `QOrderedLayerNormalization`

### P3 — RoPE / position / padding helpers

- [ ] `RotaryEmbedding`
- [ ] `GemmaRotaryEmbedding`
- [ ] `MRotaryEmbedding`
- [ ] `RelativePositionBias`
- [ ] `GatedRelativePositionBias`
- [ ] `RemovePadding`
- [ ] `RestorePadding`

### P4 — Quantized / blocked MatMul (ORT-GenAI / int4 path)

- [ ] `MatMulNBits`
- [ ] `MatMulNBitsMlp`
- [ ] `MatMulNBitsQkv`
- [ ] `MatMulBnb4`
- [ ] `MatMulFpQ4`
- [ ] `MatMulBlockQuantizedFp4Weight`
- [ ] `MatMulBlockQuantizedFp8Weight`
- [ ] `DynamicQuantizeMatMul`
- [ ] `MatMulInteger16`
- [ ] `MatMulIntegerToFloat`
- [ ] `MulInteger`
- [ ] `ReduceSumInteger`
- [ ] `QGemm`
- [ ] `QOrderedMatMul`
- [ ] `GatherBlockQuantized`
- [ ] `DequantizeBFP`
- [ ] `DequantizeWithOrder`
- [ ] `QuantizeBFP`
- [ ] `QuantizeWithOrder`
- [ ] `GemmFloat8`

### P5 — Fused MatMul / Conv / NHWC

- [ ] `FusedMatMul`
- [ ] `FusedMatMulActivation`
- [ ] `FusedGemm`
- [ ] `FusedConv`
- [ ] `TransposeMatMul`
- [ ] `NhwcConv`
- [ ] `NhwcFusedConv`
- [ ] `NhwcMaxPool`
- [ ] `MaxpoolWithMask`
- [ ] `ConvTransposeWithDynamicPads`
- [ ] `CausalConvWithState`

### P6 — QLinear\* elementwise / pool / concat

- [ ] `QLinearAdd`
- [ ] `QLinearMul`
- [ ] `QLinearConcat`
- [ ] `QLinearAveragePool`
- [ ] `QLinearGlobalAveragePool`
- [ ] `QLinearLeakyRelu`
- [ ] `QLinearReduceMean`
- [ ] `QLinearSigmoid`
- [ ] `QLinearSoftmax`
- [ ] `QLinearWhere`

### P7 — MoE / search / sampling

- [ ] `QMoE`
- [ ] `BeamSearch`
- [ ] `GreedySearch`
- [ ] `WhisperBeamSearch`
- [ ] `Sampling`
- [ ] `SampleOp`
- [ ] `NGramRepeatBlock`

### P8 — Remainder

- [ ] `AttnLSTM`
- [ ] `DynamicQuantizeLSTM`
- [ ] `BifurcationDetector`
- [ ] `CDist`
- [ ] `ComplexMul`
- [ ] `ComplexMulConj`
- [ ] `CropAndResize`
- [ ] `DynamicTimeWarping`
- [ ] `EPContext`
- [ ] `ExpandDims`
- [ ] `Inverse`
- [ ] `Irfft`
- [ ] `Rfft`
- [ ] `MurmurHash3`
- [ ] `Snpe`
- [ ] `SparseToDenseMatMul`
- [ ] `Tokenizer`
- [ ] `TorchEmbedding`
- [ ] `UnfoldTensor`
- [ ] `WordConvEmbedding`

---

## How to check a model

1. Demo **Model Inspector** (`/inspector`) — advertise-list coverage against `BuiltinOpTypes`.
2. `dotnet run --file tools/audit-operator-support.cs` — silent-fallback sites on advertised ops.
3. ORT / CPU oracle for numerical truth (`tools/README.md`).
4. This file — remaining contrib TODOs above.

## Audit

```bash
dotnet run --file tools/audit-operator-support.cs
```

When changing the **advertised** set: update **only** `BuiltinOpTypes` + matching `Register(...)`, then update the TODO checkboxes / counts in this doc. When ORT adds new `com.microsoft` ops, append them here as unchecked TODOs until implemented.
