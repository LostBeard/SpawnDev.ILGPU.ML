# Build a tiny LSTM/GRU/RNN model and its onnxruntime reference outputs.
#
#   python tools/gen_lstm_reference.py
#
# WHY: LSTM, GRU and RNN were advertised in BuiltinOpTypes and produced NOTHING for any real model - each
# read its inputs with TryGetInputValues, which returns compile-time constants only, so X (the runtime
# input) was always null and the operator returned leaving its output buffer untouched. Fixing that needs a
# gate that feeds X as a GRAPH INPUT, because a fixture with constant-folded inputs would take the branch
# that always worked and prove nothing.
#
# The reference comes from onnxruntime rather than from our own maths, so it is an independent target.
# Writes to SpawnDev.ILGPU.ML.Demo/wwwroot/references/recurrent/ where the test suite fetches fixtures.
import json, os
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

OUT = os.path.join(os.path.dirname(__file__), "..", "SpawnDev.ILGPU.ML.Demo", "wwwroot", "references", "recurrent")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

SEQ, BATCH, INPUT, HIDDEN = 5, 1, 3, 4
rng = np.random.default_rng(20260830)


def build(kind: str):
    """One-layer forward recurrent model with X as a real graph input."""
    gates = {"LSTM": 4, "GRU": 3, "RNN": 1}[kind]
    W = rng.standard_normal((1, gates * HIDDEN, INPUT)).astype(np.float32) * 0.5
    R = rng.standard_normal((1, gates * HIDDEN, HIDDEN)).astype(np.float32) * 0.5
    B = rng.standard_normal((1, 2 * gates * HIDDEN)).astype(np.float32) * 0.1

    outputs = ["Y", "Y_h"] + (["Y_c"] if kind == "LSTM" else [])
    node = helper.make_node(
        kind, inputs=["X", "W", "R", "B"], outputs=outputs,
        hidden_size=HIDDEN, direction="forward",
    )

    graph = helper.make_graph(
        [node], f"tiny_{kind.lower()}",
        # X is a graph INPUT on purpose: that is the case the old code could not handle.
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [SEQ, BATCH, INPUT])],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [SEQ, 1, BATCH, HIDDEN]),
            helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [1, BATCH, HIDDEN]),
        ] + ([helper.make_tensor_value_info("Y_c", TensorProto.FLOAT, [1, BATCH, HIDDEN])] if kind == "LSTM" else []),
        initializer=[
            helper.make_tensor("W", TensorProto.FLOAT, W.shape, W.ravel()),
            helper.make_tensor("R", TensorProto.FLOAT, R.shape, R.ravel()),
            helper.make_tensor("B", TensorProto.FLOAT, B.shape, B.ravel()),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


summary = {}
for kind in ("LSTM", "GRU", "RNN"):
    model = build(kind)
    path = os.path.join(OUT, f"tiny_{kind.lower()}.onnx")
    onnx.save(model, path)

    X = (rng.standard_normal((SEQ, BATCH, INPUT)).astype(np.float32) * 0.8)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    vals = sess.run(names, {"X": X})

    ref = {
        "kind": kind,
        "seq": SEQ, "batch": BATCH, "input_size": INPUT, "hidden_size": HIDDEN,
        "X": X.ravel().tolist(),
        "outputs": {n: np.asarray(v).ravel().tolist() for n, v in zip(names, vals)},
    }
    with open(os.path.join(OUT, f"tiny_{kind.lower()}.json"), "w", encoding="utf-8") as f:
        json.dump(ref, f)

    yh = np.asarray(vals[names.index("Y_h")]).ravel()
    summary[kind] = dict(model=os.path.basename(path),
                         y_h_first=float(yh[0]), y_h_absmax=float(np.abs(yh).max()))
    print(f"{kind:5s} -> {os.path.basename(path)}  Y_h[0]={yh[0]:+.6f}  |Y_h|max={np.abs(yh).max():.6f}")

print()
print(f"wrote {len(summary) * 2} files to {OUT}")
print("A correct engine reproduces Y_h to float32 tolerance; the old one left the buffer untouched.")
