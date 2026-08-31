# ZipVoice text-encoder ORT reference dumps

Captured with `tools/zipvoice/ort_intermediates.py` so the ground truth for a live bug does not have to be
re-derived (each capture needs the model downloaded and ORT installed).

## `ort-encoder-nonlin_attention-int8.txt`

`text_encoder_int8.onnx` (HuggingFace `k2-fsa/ZipVoice`, `zipvoice_distill/`, 5,570,211 B) on the pinned
`paint-the-sockets.json` fixture, truncated to 6 tokens / 57 frames:

```
ZIPVOICE_ENCODER=<path>\text_encoder_int8.onnx python tools/zipvoice/ort_intermediates.py nonlin_attention
```

### Why it was captured — the open bug it documents

`Pipeline_ZipVoice_SpeaksInTheBrowser` fails on **every backend** (CPU, CUDA, OpenCL, WebGL, WebGPU) with:

```
Node 222/2182 'Mul' failed: Shapes [106,432] and [106,432,1] are not broadcastable at dim 1
  Inputs: [.../nonlin_attention/Slice_1_output_0, .../nonlin_attention/Reshape_output_0]
```

What ORT actually produces for those tensors:

| tensor | ORT | ours |
|---|---|---|
| `in_proj/Add_output_0` | `[13, 1, 432]` | - |
| `Mul_output_0` (a slice bound) | `[1]` = **144** | - |
| `Slice_1_output_0` | `[13, 1, 144]` | `[106, 432]` |
| `Reshape_output_0` | `[13, 1, 144]` | `[106, 432, 1]` |
| `Mul_3_output_0` | `[13, 1, 144]` | fails |

⚠️ In ORT both `Mul_3` operands are the SAME shape - the node does no broadcasting at all, so any
broadcast error here means an operand is already wrong, not that broadcasting is mis-implemented.

Two facts locate it:

1. **432 = 3 x 144.** `in_proj` is projected to 432 and sliced into three 144-wide parts. Our
   `Slice_1_output_0` is **432 wide** - the WHOLE tensor - so the slice did not slice. `SliceOperator`
   documents a "full copy fallback" for when starts/ends cannot be resolved, and these starts/ends are
   RUNTIME tensors (`Mul_output_0`, `Mul_1_output_0`), not constants.
2. **Our rank is 2 where ORT's is 3** - the size-1 batch axis is missing, and in `Reshape_output_0` a
   size-1 axis appears at the END instead of the MIDDLE. Element counts match exactly (45,792), so nothing
   is lost, only mis-shaped.

Both follow from ONE upstream cause: a size-1 axis lost before `nonlin_attention` corrupts the runtime
shape arithmetic (`Shape` -> `Div` -> `Mul` -> `Concat`) that produces *both* the slice bounds and the
Reshape target. **The first divergence is upstream of node 222** - fixing node 222 would be treating a
symptom.

### Next step

Dump our side for the same fixture and run `tools/zipvoice/first_divergence.py` against this file.
⚠️ `zipvoice-harness runonnx` does NOT currently work for this model - it exits with
`Tensor 'prompt_features_len' not found (needed by Cast)`, because the fixture does not carry all four
encoder inputs. Either extend the fixture or dump via the `compare` command.
