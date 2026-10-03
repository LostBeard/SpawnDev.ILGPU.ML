# Build tiny If / Loop / Scan models and their onnxruntime reference outputs.
#
#   python tools/gen_controlflow_reference.py
#
# WHY: all three inferred their output shapes from inputs[0] - the CONDITION for If, the SCALAR trip count
# for Loop, the first STATE for Scan - so every one allocated a buffer of the wrong size and the branch or
# body result was silently truncated into it. If was found the hard way, through a 95%-of-peak divergence in
# ZipVoice's decoder; Loop and Scan were found by tools/audit-operator-support.cs.
#
# ⚠️ Each fixture's real output is DELIBERATELY LARGER than inputs[0], because that is the only way the old
# behaviour fails: a fixture whose output happened to match the condition's shape would pass against the
# broken code and prove nothing.
import json, os
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                   "SpawnDev.ILGPU.ML.Demo", "wwwroot", "references", "controlflow"))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260831)


def save(name, model, feeds):
    onnx.checker.check_model(model)
    path = os.path.join(OUT, f"{name}.onnx")
    onnx.save(model, path)
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
    shapes = ", ".join(f"{n}{list(np.asarray(v).shape)}" for n, v in zip(names, vals))
    print(f"{name:12s} -> {shapes}")
    return ref


# ── If: the taken branch returns a [4,3] table, far bigger than the bool condition ────────────────
TABLE = (rng.standard_normal((4, 3)).astype(np.float32))
then_const = helper.make_node("Constant", [], ["then_out"],
                              value=helper.make_tensor("t", TensorProto.FLOAT, [4, 3], TABLE.ravel()))
else_const = helper.make_node("Constant", [], ["else_out"],
                              value=helper.make_tensor("e", TensorProto.FLOAT, [4, 3], (TABLE * -1).ravel()))
then_g = helper.make_graph([then_const], "then", [],
                           [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [4, 3])])
else_g = helper.make_graph([else_const], "else", [],
                           [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [4, 3])])
if_node = helper.make_node("If", ["cond"], ["Y"], then_branch=then_g, else_branch=else_g)
if_graph = helper.make_graph(
    [if_node], "tiny_if",
    [helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 3])])
if_model = helper.make_model(if_graph, opset_imports=[helper.make_opsetid("", 14)])
if_model.ir_version = 9
save("tiny_if", if_model, {"cond": np.array(True)})

# ── Loop: 3 iterations over loop-carried state [5], no scan outputs ───────────────────────────────
# body(iter, cond, carried) -> (cond, carried + delta)
delta = helper.make_tensor("delta", TensorProto.FLOAT, [5], np.arange(1, 6, dtype=np.float32))
body_add = helper.make_node("Add", ["carried_in", "delta"], ["carried_out"])
body_cond = helper.make_node("Identity", ["cond_in"], ["cond_out"])
loop_body = helper.make_graph(
    [body_cond, body_add], "loop_body",
    [helper.make_tensor_value_info("iter", TensorProto.INT64, []),
     helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
     helper.make_tensor_value_info("carried_in", TensorProto.FLOAT, [5])],
    [helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
     helper.make_tensor_value_info("carried_out", TensorProto.FLOAT, [5])],
    initializer=[delta])
loop_node = helper.make_node("Loop", ["trips", "cond", "carried_init"], ["carried_final"], body=loop_body)
loop_graph = helper.make_graph(
    [loop_node], "tiny_loop",
    [helper.make_tensor_value_info("trips", TensorProto.INT64, []),
     helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
     helper.make_tensor_value_info("carried_init", TensorProto.FLOAT, [5])],
    [helper.make_tensor_value_info("carried_final", TensorProto.FLOAT, [5])])
loop_model = helper.make_model(loop_graph, opset_imports=[helper.make_opsetid("", 14)])
loop_model.ir_version = 9
save("tiny_loop", loop_model, {
    "trips": np.array(3, dtype=np.int64),
    "cond": np.array(True),
    "carried_init": np.zeros(5, dtype=np.float32),
})

# ── Scan: state [3] plus a stacked scan output over a 4-step sequence ─────────────────────────────
sb_add = helper.make_node("Add", ["state_in", "elem"], ["state_out"])
sb_id = helper.make_node("Identity", ["state_out"], ["scan_out"])
scan_body = helper.make_graph(
    [sb_add, sb_id], "scan_body",
    [helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [3]),
     helper.make_tensor_value_info("elem", TensorProto.FLOAT, [3])],
    [helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [3]),
     helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [3])])
scan_node = helper.make_node("Scan", ["state_init", "seq"], ["state_final", "stacked"],
                             body=scan_body, num_scan_inputs=1)
scan_graph = helper.make_graph(
    [scan_node], "tiny_scan",
    [helper.make_tensor_value_info("state_init", TensorProto.FLOAT, [3]),
     helper.make_tensor_value_info("seq", TensorProto.FLOAT, [4, 3])],
    [helper.make_tensor_value_info("state_final", TensorProto.FLOAT, [3]),
     helper.make_tensor_value_info("stacked", TensorProto.FLOAT, [4, 3])])
scan_model = helper.make_model(scan_graph, opset_imports=[helper.make_opsetid("", 14)])
scan_model.ir_version = 9
save("tiny_scan", scan_model, {
    "state_init": np.zeros(3, dtype=np.float32),
    "seq": rng.standard_normal((4, 3)).astype(np.float32),
})

# ── If as torch exports squeeze(dim): If(shape[dim] == 1, Squeeze, Identity), consumed twice ────────
# Branch outputs carry NO static shape (torch declares them symbolic), so the compiler cannot size the If and
# the executed branch's buffer is ADOPTED at run time. One If feeds a Reshape (the executor's zero-copy rename
# of its input's pool record), the other an elementwise op. Gates the 2026-10-03 per-forward leak: the
# adopted buffer was rented under a private name, so the Reshape returned the WRONG buffer, and the branch
# executor's own outputs were never returned - Video Depth Anything leaked a buffer per If per frame. The C#
# test runs this repeatedly and requires ZERO fresh device allocations once warm.
def squeeze_if(prefix, src):
    sq_axes = helper.make_node("Constant", [], [f"{prefix}_axes"],
                               value=helper.make_tensor("a", TensorProto.INT64, [1], [1]))
    then_g = helper.make_graph(
        [sq_axes, helper.make_node("Squeeze", [src, f"{prefix}_axes"], [f"{prefix}_sq"])], f"{prefix}_then", [],
        [helper.make_tensor_value_info(f"{prefix}_sq", TensorProto.FLOAT, None)])
    else_g = helper.make_graph(
        [helper.make_node("Identity", [src], [f"{prefix}_id"])], f"{prefix}_else", [],
        [helper.make_tensor_value_info(f"{prefix}_id", TensorProto.FLOAT, None)])
    return helper.make_node("If", [f"{prefix}_cond"], [f"{prefix}_out"], then_branch=then_g, else_branch=else_g)


sq_nodes = [
    helper.make_node("Relu", ["X"], ["xr"]),
    helper.make_node("Shape", ["xr"], ["xs"]),
    helper.make_node("Constant", [], ["one_i"], value=helper.make_tensor("i1", TensorProto.INT64, [], [1])),
    helper.make_node("Gather", ["xs", "one_i"], ["dim1"], axis=0),
    helper.make_node("Equal", ["dim1", "one_i"], ["s1_cond"]),
    helper.make_node("Equal", ["dim1", "one_i"], ["s2_cond"]),
    squeeze_if("s1", "xr"),
    squeeze_if("s2", "xr"),
    helper.make_node("Constant", [], ["flat_shape"], value=helper.make_tensor("fs", TensorProto.INT64, [2], [4, 6])),
    helper.make_node("Reshape", ["s1_out", "flat_shape"], ["Y"]),
    helper.make_node("Neg", ["s2_out"], ["Z"]),
]
sq_graph = helper.make_graph(
    sq_nodes, "if_squeeze_reshape",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 6])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 6]),
     helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 4, 6])])
sq_model = helper.make_model(sq_graph, opset_imports=[helper.make_opsetid("", 14)])
sq_model.ir_version = 9
save("if_squeeze_reshape", sq_model, {"X": rng.standard_normal((1, 1, 4, 6)).astype(np.float32)})

print()
print(f"wrote fixtures to {OUT}")
print("Each output is larger than inputs[0], so the old inputs[0] inference cannot pass these.")
