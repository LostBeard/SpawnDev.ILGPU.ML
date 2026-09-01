# Build tiny models whose SHAPES are only knowable at RUNTIME, plus their onnxruntime references.
#
#   python tools/gen_branch_shape_reference.py
#
# WHY (both found 2026-09-01, in ZipVoice's fm_decoder, one hiding the other):
#
#   branch_slice - a Slice whose input length depends on which If BRANCH ran. The compiler can only see one
#     branch, so it resolved the Slice window at COMPILE time (`_resolved_starts`/`_resolved_ends`) and the
#     runtime cascade preferred those stale values. The window collapsed, the output came out EMPTY, and a
#     zero-element output SKIPS the operator entirely - so everything downstream read a pooled buffer nobody
#     had written. It held the previous tensor's plausible numbers, so nothing threw and the answer was
#     confidently wrong.
#
#   branch_unary - `Sign` was missing from the executor's allowlist of unary ops permitted to adopt their
#     input's RUNTIME shape, while `Abs` was on it. Both read the SAME [N,1] tensor; Abs resolved correctly
#     and Sign kept a compile-time [1], collapsing a whole vector to one scalar. A rank-1 one-element tensor
#     is legal everywhere downstream, so nothing errored.
#
# ⚠️ THE FIXTURE MUST TAKE THE BRANCH THE COMPILER CANNOT SEE. Both models take the LONGER branch at
# runtime, and its length differs from the branch a compiler would fold. A fixture that happened to take the
# short branch, or whose two branches were the same length, passes against the broken code and proves
# nothing - the same discipline as the control-flow fixtures next door.
import json, os
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                   "SpawnDev.ILGPU.ML.Demo", "wwwroot", "references", "controlflow"))
os.makedirs(OUT, exist_ok=True)

SHORT, LONG, C = 4, 11, 2      # LONG != SHORT is the whole point


def const(name, arr, dtype=TensorProto.FLOAT):
    return helper.make_node("Constant", [], [name], value=helper.make_tensor(
        name + "_v", dtype, list(np.asarray(arr).shape), np.asarray(arr).ravel().tolist()))


def save(name, graph, feeds):
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    path = os.path.join(OUT, name + ".onnx")
    onnx.save(m, path)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    outs = s.run(None, feeds)
    fixture = {
        "inputs": {k: {"shape": list(np.asarray(v).shape), "data": np.asarray(v).ravel().tolist()}
                   for k, v in feeds.items()},
        "outputs": {o.name: {"shape": list(np.asarray(a).shape), "data": np.asarray(a).ravel().tolist()}
                    for o, a in zip(s.get_outputs(), outs)},
    }
    with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(fixture, f)
    for o, a in zip(s.get_outputs(), outs):
        print(f"  {name}: {o.name} shape={list(np.asarray(a).shape)}")


# ── branch_slice ────────────────────────────────────────────────────────────────────────────────────
# cond=False takes the else branch -> a [LONG, C] table. A Shape/Gather reads its length AT RUNTIME and
# feeds a Slice covering the whole thing. Compile-time resolution sees only the [SHORT, C] then branch.
short_tbl = np.arange(SHORT * C, dtype=np.float32).reshape(SHORT, C)
long_tbl = (np.arange(LONG * C, dtype=np.float32).reshape(LONG, C) + 100.0)

then_g = helper.make_graph([const("t_short", short_tbl)], "then", [],
                           [helper.make_tensor_value_info("t_short", TensorProto.FLOAT, [SHORT, C])])
else_g = helper.make_graph([const("t_long", long_tbl)], "else", [],
                           [helper.make_tensor_value_info("t_long", TensorProto.FLOAT, [LONG, C])])

nodes = [
    helper.make_node("If", ["cond"], ["table"], then_branch=then_g, else_branch=else_g),
    helper.make_node("Shape", ["table"], ["tbl_shape"]),
    const("zero_i", np.array([0], dtype=np.int64), TensorProto.INT64),
    const("one_i", np.array([1], dtype=np.int64), TensorProto.INT64),
    const("axis0", np.array([0], dtype=np.int64), TensorProto.INT64),
    helper.make_node("Gather", ["tbl_shape", "zero_i"], ["tbl_len"], axis=0),
    helper.make_node("Slice", ["table", "zero_i", "tbl_len", "axis0", "one_i"], ["sliced"]),
]
g = helper.make_graph(nodes, "branch_slice",
                      [helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
                      [helper.make_tensor_value_info("sliced", TensorProto.FLOAT, [None, C])])
save("branch_slice", g, {"cond": np.array(False)})

# ── branch_unary ────────────────────────────────────────────────────────────────────────────────────
# A [LONG,1] vector built at RUNTIME (Range over the branch length), then Sign and Abs of the SAME tensor.
# With Sign missing from the runtime shape-adoption allowlist it collapses to one element while Abs does not,
# so the two outputs disagree in LENGTH - which is exactly how it presented.
nodes2 = [
    helper.make_node("If", ["cond"], ["table"], then_branch=then_g, else_branch=else_g),
    helper.make_node("Shape", ["table"], ["tbl_shape"]),
    const("zero_i", np.array([0], dtype=np.int64), TensorProto.INT64),
    const("one_i", np.array([1], dtype=np.int64), TensorProto.INT64),
    const("start0", np.array(0, dtype=np.float32), TensorProto.FLOAT),
    const("step1", np.array(1, dtype=np.float32), TensorProto.FLOAT),
    const("half", np.array(LONG // 2, dtype=np.float32), TensorProto.FLOAT),
    helper.make_node("Gather", ["tbl_shape", "zero_i"], ["tbl_len"], axis=0),
    helper.make_node("Squeeze", ["tbl_len", "zero_i"], ["len_s"]),
    helper.make_node("Cast", ["len_s"], ["len_f"], to=TensorProto.FLOAT),
    helper.make_node("Range", ["start0", "len_f", "step1"], ["rng"]),
    # centre it so the vector genuinely carries negative, zero and positive values - a Sign whose input is
    # all one polarity would pass even when collapsed to a single element.
    helper.make_node("Sub", ["rng", "half"], ["centred"]),
    helper.make_node("Unsqueeze", ["centred", "one_i"], ["col"]),
    helper.make_node("Sign", ["col"], ["sign_out"]),
    helper.make_node("Abs", ["col"], ["abs_out"]),
]
g2 = helper.make_graph(nodes2, "branch_unary",
                       [helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
                       [helper.make_tensor_value_info("sign_out", TensorProto.FLOAT, [None, 1]),
                        helper.make_tensor_value_info("abs_out", TensorProto.FLOAT, [None, 1])])
save("branch_unary", g2, {"cond": np.array(False)})

print("done ->", OUT)
