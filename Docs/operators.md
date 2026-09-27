# ONNX operator support

**Single source of truth (code):** [`OperatorRegistry.BuiltinOpTypes`](../SpawnDev.ILGPU.ML/Operators/OperatorRegistry.cs)

That set is the only authoritative list of op-types this engine **advertises** as supported. Everything else that answers "do we support X?" must point at it — never maintain a second list.

| Consumer | How it reads support |
|----------|----------------------|
| Live registry (`RegisterBuiltins`) | Locked to `BuiltinOpTypes` by `MLTestBase.Op_BuiltinOpTypes_MatchesLiveRegistry` |
| Model Inspector (demo `/inspector` + `ModelInspectorHelper`) | `KnownSupportedOps` → `BuiltinOpTypes` |
| Home page operator count | `BuiltinOpTypes.Count` (live; never hardcode) |
| `tools/audit-operator-support.cs` | Audits advertise vs `OpType =>` impl + silent-fallback shapes |

**Do not paste the op list into docs, README, or UI copy.** Counts go stale the moment someone adds a `Register(...)`. Link here, or read `BuiltinOpTypes.Count` from code.

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

### Aliases / contrib

| Op-type | Notes |
|---------|--------|
| `RMSNormalization` | True RMS (no mean subtraction). |
| `SimplifiedLayerNormalization` | ORT contrib ≈ RMSNorm + optional bias. |
| `LayerNormalization` | Mean-centered — not interchangeable with the above. |

## How to check a model

1. Demo **Model Inspector** (`/inspector`) — advertise-list coverage.
2. `dotnet run --file tools/audit-operator-support.cs` — silent-fallback sites.
3. ORT / CPU oracle for numerical truth (`tools/README.md`).

## Audit

```bash
dotnet run --file tools/audit-operator-support.cs
```

When changing the registered set: update **only** `BuiltinOpTypes` + matching `Register(...)`. Update this doc when policy or value-type contracts change.
