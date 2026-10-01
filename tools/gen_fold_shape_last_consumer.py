# Generates SpawnDev.ILGPU.ML.Demo/wwwroot/models/tests/fold_shape_last_consumer.onnx for
# MLTestBase.FoldWarmSkip_ReleasesAndOutputsUnchanged.
#
# The graph exists to put a FOLDED node in the position where its release matters:
#   v = Relu(x)          - input-dependent (varying), consumed ONLY by s
#   s = Shape(v)         - Shape folds even on a varying input (it reads the fixed shape) and is v's LAST consumer,
#                          so v is released when s is visited - a warm forward that skipped s would never free v
#   y = Reshape(x, s)    - a non-folded consumer of s (makes s a fold-frontier tensor), and the graph output
#   z = Mul(x, x)        - a second plain output
# Run: python tools/gen_fold_shape_last_consumer.py
import os
import onnx
from onnx import TensorProto, helper

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8, 8])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])
z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4, 8, 8])
nodes = [
    helper.make_node("Relu", ["x"], ["v"], name="relu"),
    helper.make_node("Shape", ["v"], ["s"], name="shape_of_v"),
    helper.make_node("Reshape", ["x", "s"], ["y"], name="reshape"),
    helper.make_node("Mul", ["x", "x"], ["z"], name="square"),
]
graph = helper.make_graph(nodes, "fold_shape_last_consumer", [x], [y, z])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="SpawnDev.ILGPU.ML tests")
model.ir_version = 8
onnx.checker.check_model(model)
out = os.path.join(os.path.dirname(__file__), "..", "SpawnDev.ILGPU.ML.Demo", "wwwroot", "models", "tests",
                   "fold_shape_last_consumer.onnx")
onnx.save(model, out)
print("wrote", os.path.abspath(out))
