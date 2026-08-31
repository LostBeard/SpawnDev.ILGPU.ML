"""Ground truth for whisper-tiny's decoder causal-mask subgraph, from ONNX Runtime.

Our executor produces a mask with only row 0 set. Rather than reverse-engineer what the exported graph
INTENDS node by node, expose the same intermediates as graph outputs in ORT and read what they should be.
Whichever node first disagrees with ours is the bug.

    python tools/whisper-harness/ort_mask_reference.py
"""
import sys
import numpy as np
import onnx
import onnxruntime as ort

MODEL = r"D:\users\tj\Projects\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML.Demo\wwwroot\models\whisper-tiny\decoder_model.onnx"

# The tensors this session has been dumping on the C# side, in graph order.
WANT = [
    "/model/decoder/Range_1_output_0",
    "/model/decoder/Tile_output_0",
    "/model/decoder/Reshape_1_output_0",
    "/model/decoder/Reshape_2_output_0",
    "/model/decoder/Reshape_4_output_0",
    "/model/decoder/Equal_output_0",
    "/model/decoder/ConstantOfShape_2_output_0",
    "/model/decoder/LessOrEqual_output_0",
    "/model/decoder/Cast_4_output_0",
    "/model/decoder/And_output_0",
    "/model/decoder/And_1_output_0",
    "/model/decoder/Expand_1_output_0",
    "/model/decoder/layers.0/self_attn/Slice_output_0",
    "/model/decoder/layers.0/self_attn/Where_output_0",
    "/model/decoder/layers.0/self_attn/Softmax_output_0",
]

model = onnx.load(MODEL)
produced = {o for n in model.graph.node for o in n.output}
existing = {o.name for o in model.graph.output}
added = 0
for name in WANT:
    if name in produced and name not in existing:
        model.graph.output.append(onnx.ValueInfoProto(name=name))
        added += 1
print(f"exposed {added} intermediate tensors as graph outputs")

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
feeds = {
    "input_ids": np.array([[50258, 50259, 50359, 50363]], dtype=np.int64),
    # The mask subgraph depends only on input_ids' shape, so random encoder states are fine here.
    "encoder_hidden_states": np.random.RandomState(0).randn(1, 1500, 384).astype(np.float32) * 0.1,
}
names = [o.name for o in sess.get_outputs()]
outs = dict(zip(names, sess.run(None, feeds)))

def describe(a):
    a = np.asarray(a)
    flat = a.reshape(-1).astype(np.float64)
    finite = np.isfinite(flat)
    head = " ".join(f"{v:.4g}" for v in flat[:8])
    return (f"shape={list(a.shape)} dtype={a.dtype} "
            f"min={flat[finite].min() if finite.any() else float('nan'):.4g} "
            f"max={flat[finite].max() if finite.any() else float('nan'):.4g} "
            f"mean={flat[finite].mean() if finite.any() else float('nan'):.4g} "
            f"nonfinite={int((~finite).sum())} head=[{head}]")

for name in WANT:
    if name in outs:
        print(f"[ort] {name:<52} {describe(outs[name])}")
    else:
        print(f"[ort] {name:<52} (not in graph)")

# The mask is the whole question: print it as a matrix so its shape is unmistakable.
for key in ("/model/decoder/layers.0/self_attn/Where_output_0", "/model/decoder/Expand_1_output_0"):
    if key in outs:
        m = np.asarray(outs[key]).reshape(-1)
        n = int(round(len(m) ** 0.5))
        if n * n == len(m):
            print(f"\n{key} as {n}x{n}:")
            for r in range(n):
                print("   " + " ".join(f"{v:>8.3g}" for v in m[r * n:(r + 1) * n]))
