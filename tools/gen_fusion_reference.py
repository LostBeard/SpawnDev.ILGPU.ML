# Fixtures for three graph optimizations (+ half_linear, for WeightStorage.Half's browser load) that change what the engine executes but must not change results.
#
#   python tools/gen_fusion_reference.py
#
# - erf_gelu: torch's EXACT GELU export (Div, Erf, Add, Mul, Mul) after a Linear, plus the two other association
#   orders torch/other exporters produce. GraphOptimizer.FuseErfGelu must turn each into one Gelu (and the first
#   then folds into the Linear's FusedLinear epilogue).
# - group_norm: a real torch nn.GroupNorm export (Reshape, InstanceNormalization, Shape, Reshape, Mul, Add).
#   GraphOptimizer.FuseGroupNorm must turn it into one GroupNormalization.
# - transpose_view: Transposes that only move size-1 axes (a reshape): one single-consumer intermediate (the
#   executor's zero-copy handoff) and one of a graph input with two consumers (TransposeOperator's native copy).
#
# MEASURED (2026-10-03, Video Depth Anything streaming at 98x168): the three took a forward from 547 to 463
# executed nodes - each node at least one WebGPU dispatch (~25-30 us in a browser).
import json, os
import numpy as np
import onnx
import torch
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                   "SpawnDev.ILGPU.ML.Demo", "wwwroot", "references", "fusion"))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261003)


def save(name, model, feeds):
    onnx.checker.check_model(model)
    onnx.save(model, os.path.join(OUT, f"{name}.onnx"))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    vals = sess.run(names, feeds)
    ref = {"inputs": {k: dict(shape=list(v.shape), data=v.ravel().tolist()) for k, v in feeds.items()},
           "outputs": {n: dict(shape=list(np.asarray(v).shape), data=np.asarray(v).ravel().tolist()) for n, v in zip(names, vals)}}
    json.dump(ref, open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8"))
    print(f"{name:16s} -> {[ (n, np.asarray(v).shape) for n, v in zip(names, vals)]}  ops={sorted(set(n.op_type for n in model.graph.node))}")


def torch_export(module, x, name):
    path = os.path.join(OUT, f"_{name}_tmp.onnx")
    torch.onnx.export(module, (torch.from_numpy(x),), path, input_names=["X"], output_names=["Y"], opset_version=17, dynamo=False)
    m = onnx.load(path)
    os.remove(path)
    return m


# ── erf_gelu: torch's own GELU after a Linear (form A), + forms B and C built by hand on the same input ──
class LinGelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 32)
        self.act = torch.nn.GELU()

    def forward(self, x):
        return self.act(self.fc(x))


torch.manual_seed(0)
x = rng.standard_normal((8, 16)).astype(np.float32) * 2
m = torch_export(LinGelu().eval(), x, "erf_gelu")
g = m.graph
c = lambda n, v: g.initializer.append(numpy_helper.from_array(np.array(v, np.float32), n))
c("k_sqrt2", 1.4142135623730951); c("k_rsqrt2", 0.7071067811865476); c("k_one", 1.0); c("k_half", 0.5)
# form B: (x * 0.5) * (1 + erf(x * (1/sqrt2)))  on X directly
g.node.extend([
    helper.make_node("Mul", ["X", "k_rsqrt2"], ["b_d"]), helper.make_node("Erf", ["b_d"], ["b_e"]),
    helper.make_node("Add", ["k_one", "b_e"], ["b_a"]), helper.make_node("Mul", ["X", "k_half"], ["b_h"]),
    helper.make_node("Mul", ["b_h", "b_a"], ["YB"]),
    # form C: x * ((1 + erf(x / sqrt2)) * 0.5)
    helper.make_node("Div", ["X", "k_sqrt2"], ["c_d"]), helper.make_node("Erf", ["c_d"], ["c_e"]),
    helper.make_node("Add", ["c_e", "k_one"], ["c_a"]), helper.make_node("Mul", ["c_a", "k_half"], ["c_m"]),
    helper.make_node("Mul", ["X", "c_m"], ["YC"]),
])
g.output.extend([helper.make_tensor_value_info("YB", TensorProto.FLOAT, [8, 16]),
                 helper.make_tensor_value_info("YC", TensorProto.FLOAT, [8, 16])])
save("erf_gelu", m, {"X": x})

# ── group_norm: a real torch export ──
gn = torch.nn.GroupNorm(8, 32, eps=1e-6)
with torch.no_grad():
    gn.weight.copy_(torch.randn(32)); gn.bias.copy_(torch.randn(32))
xg = rng.standard_normal((1, 32, 5, 7)).astype(np.float32)
save("group_norm", torch_export(gn.eval(), xg, "group_norm"), {"X": xg})

# ── transpose_view: Transposes that move only size-1 axes ──
nodes = [
    helper.make_node("Relu", ["X"], ["xr"]),
    helper.make_node("Transpose", ["xr"], ["t1"], perm=[0, 2, 1, 3]),      # [1,4,1,6] -> [1,1,4,6]: single consumer
    helper.make_node("Neg", ["t1"], ["Y1"]),
    helper.make_node("Transpose", ["X"], ["t2"], perm=[2, 0, 1, 3]),       # graph input, two consumers below
    helper.make_node("Relu", ["t2"], ["Y2"]),
    helper.make_node("Neg", ["t2"], ["Y3"]),
]
tg = helper.make_graph(nodes, "transpose_view",
                       [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 1, 6])],
                       [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 1, 4, 6]) for n in ("Y1", "Y2", "Y3")])
tm = helper.make_model(tg, opset_imports=[helper.make_opsetid("", 13)])
tm.ir_version = 8
save("transpose_view", tm, {"X": rng.standard_normal((1, 4, 1, 6)).astype(np.float32)})
# ── half_linear: a Linear whose weight (256x512 fp32 = 512 KB) is far over the browser's 64 KB host-copy guard.
# WeightStorage.Half must stream it JS->GPU and downcast on the GPU; a managed downcast trips the guard.
class LinGeluBig(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(256, 512)
        self.act = torch.nn.GELU()

    def forward(self, x):
        return self.act(self.fc(x))


torch.manual_seed(1)
xh = rng.standard_normal((4, 256)).astype(np.float32)
hl = torch_export(LinGeluBig().eval(), xh, "half_linear")
save("half_linear", hl, {"X": xh})

# ── half_linear_fp16w: the same model with its weights STORED as FP16 (W__fp16 -> Cast(to=FLOAT) -> W; compute stays
# FP32) by tools/onnx-weights-fp16.py - the engine must fold the Cast at load (no per-run Cast) and match onnxruntime.
import importlib.util
_spec = importlib.util.spec_from_file_location("onnx_weights_fp16", os.path.join(os.path.dirname(__file__), "onnx-weights-fp16.py"))
_fp16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fp16)
_fp16.to_fp16_weights(hl, min_elements=1024)
save("half_linear_fp16w", hl, {"X": xh})
print("wrote", OUT)
