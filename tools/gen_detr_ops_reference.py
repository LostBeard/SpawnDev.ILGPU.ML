# Fixtures for the three defects that kept the DETR family (D-FINE, RT-DETRv2) from running or from matching onnxruntime
# (Riker, AssistZ, 2026-11-06; fixed 2026-11-08). Each is the smallest graph that has the defect's exact shape.
#
#   python -P tools/gen_detr_ops_reference.py
#
# - add_zero_rank_promote: `Add(x, zeros[1,1,1])` RAISES x's rank (values unchanged). GraphOptimizer.StrengthReduce
#   rewrote it to Identity, so the Shape after it lost a dim (D-FINE's anchor grid: "Shapes [3] and [4] are not
#   broadcastable"). The Shape output is compared, so the dropped dim is a different element count.
# - matmul_vector: ONNX rank-1 MatMul operands ([..,K] x [K] -> [..], [K] x [K,N] -> [N]), with a constant and an
#   activation vector, and as FusedScaledMatMul (MatMul of two activations times a scalar) incl. a shared 2-D B under a
#   batched A. D-FINE's box integral is MatMul(softmax [1200,33], project [33]); [1200,33] came out and became a batch of 33.
# - linear_clip: MatMul -> Add(bias) -> Clip(-0.5, 0.5). FusedLinear ran a fused Clip as ReLU; D-FINE's query_pos_head
#   Clip(-10, 10) lost every negative value.
import json, os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                   "SpawnDev.ILGPU.ML.Demo", "wwwroot", "references", "fusion"))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261108)
F = TensorProto.FLOAT


def save(name, model, feeds):
    onnx.checker.check_model(model)
    onnx.save(model, os.path.join(OUT, f"{name}.onnx"))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    vals = sess.run(names, feeds)
    ref = {"inputs": {k: dict(shape=list(v.shape), data=v.ravel().tolist()) for k, v in feeds.items()},
           "outputs": {n: dict(shape=list(np.asarray(v).shape), data=np.asarray(v).astype(np.float32).ravel().tolist()) for n, v in zip(names, vals)}}
    json.dump(ref, open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8"))
    print(f"{name:22s} -> {[(n, np.asarray(v).shape) for n, v in zip(names, vals)]}  ops={sorted(set(n.op_type for n in model.graph.node))}")


def init(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name)


def model(nodes, inputs, outputs, inits):
    g = helper.make_graph(nodes, "g", inputs, outputs, inits)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])


# ── add_zero_rank_promote ──
x = rng.standard_normal((3, 4)).astype(np.float32)
m = model([
    helper.make_node("Mul", ["X", "two"], ["xm"]),                    # an intermediate (rank unknown to the optimizer)
    helper.make_node("Add", ["xm", "zero4"], ["Y"]),                  # [3,4] + [1,1,1,1] -> [1,1,3,4]
    helper.make_node("Shape", ["Y"], ["shp"]),
    helper.make_node("Cast", ["shp"], ["S"], to=F),
    helper.make_node("Add", ["X", "zero3"], ["Y2"]),                  # graph input (rank known): [3,4] + [1,1,1] -> [1,3,4]
    helper.make_node("Shape", ["Y2"], ["shp2"]),
    helper.make_node("Cast", ["shp2"], ["S2"], to=F),
], [helper.make_tensor_value_info("X", F, [3, 4])],
   [helper.make_tensor_value_info("Y", F, [1, 1, 3, 4]), helper.make_tensor_value_info("S", F, [4]),
    helper.make_tensor_value_info("S2", F, [3])],
   [init("two", np.array(2.0, np.float32)), init("zero4", np.zeros((1, 1, 1, 1), np.float32)),
    init("zero3", np.zeros((1, 1, 1), np.float32))])
save("add_zero_rank_promote", m, {"X": x})

# ── matmul_vector ──
xa = rng.standard_normal((2, 5, 7)).astype(np.float32)
wv = rng.standard_normal((7,)).astype(np.float32)
v = rng.standard_normal((7,)).astype(np.float32)
bm = rng.standard_normal((7, 3)).astype(np.float32)
m = model([
    helper.make_node("MatMul", ["A", "w"], ["Y1"]),                   # [2,5,7] x const [7] -> [2,5]
    helper.make_node("MatMul", ["A", "V"], ["Y2"]),                   # [2,5,7] x input [7] -> [2,5]
    helper.make_node("MatMul", ["V", "Bm"], ["Y3"]),                  # [7] x [7,3] -> [3]
    helper.make_node("MatMul", ["A", "V"], ["mv"]),                   # FusedScaledMatMul with a rank-1 B
    helper.make_node("Mul", ["mv", "half"], ["Y4"]),
    helper.make_node("MatMul", ["A", "Bm"], ["mb"]),                  # FusedScaledMatMul: shared 2-D B under batched A
    helper.make_node("Mul", ["mb", "half"], ["Y5"]),
    helper.make_node("Reshape", ["Y1", "r"], ["Y6"]),                 # D-FINE's form: the vector product's rank decides the reshape
], [helper.make_tensor_value_info("A", F, [2, 5, 7]), helper.make_tensor_value_info("V", F, [7]),
    helper.make_tensor_value_info("Bm", F, [7, 3])],
   [helper.make_tensor_value_info(n, F, s) for n, s in [("Y1", [2, 5]), ("Y2", [2, 5]), ("Y3", [3]), ("Y4", [2, 5]), ("Y5", [2, 5, 3]), ("Y6", [2, 5])]],
   [init("w", wv), init("half", np.array(0.5, np.float32)), init("r", np.array([-1, 5], np.int64))])
save("matmul_vector", m, {"A": xa, "V": v, "Bm": bm})

# ── linear_clip ──
xl = rng.standard_normal((4, 8)).astype(np.float32)
m = model([
    helper.make_node("MatMul", ["X", "W"], ["mm"]),
    helper.make_node("Add", ["mm", "b"], ["lin"]),
    helper.make_node("Clip", ["lin", "lo", "hi"], ["Y"]),
], [helper.make_tensor_value_info("X", F, [4, 8])], [helper.make_tensor_value_info("Y", F, [4, 6])],
   [init("W", rng.standard_normal((8, 6)).astype(np.float32)), init("b", rng.standard_normal((6,)).astype(np.float32)),
    init("lo", np.array(-0.5, np.float32)), init("hi", np.array(0.5, np.float32))])
save("linear_clip", m, {"X": xl})
