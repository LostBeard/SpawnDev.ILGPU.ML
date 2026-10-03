# Build the session-lifecycle fixture: a tiny model whose operators keep device scratch in their kernels.
#
#   python tools/gen_lifecycle_reference.py
#
# WHY: SoftmaxKernel, ReductionKernels, TopK/Sign/DepthToSpace's own kernels and GatherND's params buffer were not
# IDisposable, so `(x as IDisposable)?.Dispose()` in OperatorRegistry was a silent no-op and their buffers outlived
# every session. The desktop GC eventually finalized them; a browser did not - SpawnScene's RaCo-ALIKED extractor
# left ~10 MB of WebGPU storage per create/run/dispose cycle (2026-10-03). The C# test creates, runs and disposes
# a session on this model several times and requires the accelerator's live buffers NOT to grow.
import json, os
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                   "SpawnDev.ILGPU.ML.Demo", "wwwroot", "references", "lifecycle"))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261003)

nodes = [
    helper.make_node("Softmax", ["X"], ["sm"], axis=-1),
    helper.make_node("Constant", [], ["k"], value=helper.make_tensor("k", TensorProto.INT64, [1], [8])),
    helper.make_node("TopK", ["sm", "k"], ["topv", "topi"], axis=-1),
    helper.make_node("ReduceMax", ["X"], ["mx"], keepdims=0),
    helper.make_node("Sign", ["X"], ["sg"]),
]
graph = helper.make_graph(
    nodes, "lifecycle_scratch",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 64])],
    [helper.make_tensor_value_info("topv", TensorProto.FLOAT, [4, 8]),
     helper.make_tensor_value_info("mx", TensorProto.FLOAT, []),
     helper.make_tensor_value_info("sg", TensorProto.FLOAT, [4, 64])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, os.path.join(OUT, "kernel_scratch.onnx"))

x = rng.standard_normal((4, 64)).astype(np.float32)
sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
names = ["topv", "mx", "sg"]
vals = sess.run(names, {"X": x})
ref = {
    "inputs": {"X": dict(shape=list(x.shape), data=x.ravel().tolist())},
    "outputs": {n: dict(shape=list(np.asarray(v).shape), data=np.asarray(v).ravel().tolist()) for n, v in zip(names, vals)},
}
with open(os.path.join(OUT, "kernel_scratch.json"), "w", encoding="utf-8") as f:
    json.dump(ref, f)
print("wrote", OUT, {n: np.asarray(v).shape for n, v in zip(names, vals)})
