"""Reference outputs for Op_GridSample_PaddingModes_MatchOnnxRuntime (MLTestBase.OperatorCoverageTests.cs): ONNX GridSample (opset 16, mode linear)
for padding_mode zeros / border / reflection x align_corners 0 / 1, run by onnxruntime. Input and grid are closed-form (the test builds the same
values), the grid reaches +-1.6 so samples fall outside the image and the padding is what is tested. Prints the C# array literals.

    py -3.13 tools/gen_gridsample_ref.py
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

N, C, H, W, HO, WO = 1, 2, 5, 7, 4, 6


def inputs():
    x = np.array([np.sin(i * 0.37) + i * 0.01 for i in range(N * C * H * W)], dtype=np.float32).reshape(N, C, H, W)
    g = np.array([1.6 * np.sin(i * 0.53 + 0.2) for i in range(N * HO * WO * 2)], dtype=np.float32).reshape(N, HO, WO, 2)
    return x, g


def run(padding, align):
    node = helper.make_node("GridSample", ["X", "grid"], ["Y"], mode="linear", padding_mode=padding, align_corners=align)
    graph = helper.make_graph([node], "gs",
                              [helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, C, H, W]),
                               helper.make_tensor_value_info("grid", TensorProto.FLOAT, [N, HO, WO, 2])],
                              [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [N, C, HO, WO])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    s = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    x, g = inputs()
    return s.run(None, {"X": x, "grid": g})[0].reshape(-1)


print(f"// onnxruntime {ort.__version__}, onnx {onnx.__version__}")
for padding in ("zeros", "border", "reflection"):
    for align in (0, 1):
        y = run(padding, align)
        body = ", ".join(f"{v:.7g}f" for v in y)
        print(f'        ["{padding}/{align}"] = new float[] {{ {body} }},')
