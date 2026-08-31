# Build tiny ScatterElements / ScatterND models and their onnxruntime reference outputs.
#
#   python tools/gen_scatter_reference.py
#
# WHY: both operators read their inputs with TryGetInputValues (compile-time constants only) and, when that
# returned null, copied `data` to the output and RETURNED - discarding every update. ScatterND's own comment
# called it "fall back to identity". A real model computes indices and updates at runtime, so that was the
# only path ever taken and both ops were no-ops producing a correctly shaped, plausible tensor.
#
# ⚠️ INDICES AND UPDATES ARE GRAPH INPUTS HERE, deliberately. Making them initializers would constant-fold
# them into the branch that always worked, and the fixture would pass against the bug - the same trap as
# testing a resampler with audio already at the target rate.
#
# The scatters also change values the copy of `data` would leave alone, so "did nothing" is distinguishable
# from "did the right thing".
import json, os
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                   "SpawnDev.ILGPU.ML.Demo", "wwwroot", "references", "scatter"))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260901)


def save(name, model, feeds):
    onnx.checker.check_model(model)
    onnx.save(model, os.path.join(OUT, f"{name}.onnx"))
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    vals = sess.run(names, feeds)
    ref = {
        "inputs": {k: dict(shape=list(np.asarray(v).shape), data=np.asarray(v).ravel().tolist())
                   for k, v in feeds.items()},
        "outputs": {n: dict(shape=list(np.asarray(v).shape), data=np.asarray(v).ravel().tolist())
                    for n, v in zip(names, vals)},
    }
    with open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(ref, f)

    data = np.asarray(feeds["data"])
    out = np.asarray(vals[0])
    changed = int((data.ravel() != out.ravel()).sum())
    print(f"{name:20s} -> {list(out.shape)}, {changed} of {out.size} elements differ from `data`")
    if changed == 0:
        raise SystemExit(f"{name}: the scatter changed nothing, so a no-op would pass - fixture is useless")


# ── ScatterElements along axis 1 ─────────────────────────────────────────────
DATA = rng.standard_normal((3, 5)).astype(np.float32)
# ⚠️ No DUPLICATE index within a row. ONNX leaves the order undefined for duplicates with
# reduction='none', ORT resolves them last-wins, and a GPU kernel races - a fixture with duplicates
# would be legitimately nondeterministic and the test would flake.
IDX = np.array([[0, 3], [4, 1], [2, 4]], dtype=np.int64)      # [3,2] - a SUBSET scatter
UPD = (rng.standard_normal((3, 2)).astype(np.float32) + 10)   # far from data, so changes are obvious

node = helper.make_node("ScatterElements", ["data", "indices", "updates"], ["Y"], axis=1)
graph = helper.make_graph(
    [node], "tiny_scatter_elements",
    [helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 5]),
     helper.make_tensor_value_info("indices", TensorProto.INT64, [3, 2]),
     helper.make_tensor_value_info("updates", TensorProto.FLOAT, [3, 2])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
model.ir_version = 9
save("tiny_scatter_elements", model, {"data": DATA, "indices": IDX, "updates": UPD})

# ── ScatterND: index depth 2 into a [4,3,2] tensor, so each update is a 2-vector ──
ND_DATA = rng.standard_normal((4, 3, 2)).astype(np.float32)
ND_IDX = np.array([[0, 0], [1, 2], [3, 1]], dtype=np.int64)    # [3,2] tuples
ND_UPD = (rng.standard_normal((3, 2)).astype(np.float32) - 10)

nd_node = helper.make_node("ScatterND", ["data", "indices", "updates"], ["Y"])
nd_graph = helper.make_graph(
    [nd_node], "tiny_scatter_nd",
    [helper.make_tensor_value_info("data", TensorProto.FLOAT, [4, 3, 2]),
     helper.make_tensor_value_info("indices", TensorProto.INT64, [3, 2]),
     helper.make_tensor_value_info("updates", TensorProto.FLOAT, [3, 2])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 3, 2])])
nd_model = helper.make_model(nd_graph, opset_imports=[helper.make_opsetid("", 16)])
nd_model.ir_version = 9
save("tiny_scatter_nd", nd_model, {"data": ND_DATA, "indices": ND_IDX, "updates": ND_UPD})

print()
print(f"wrote fixtures to {OUT}")
