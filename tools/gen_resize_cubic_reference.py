"""onnxruntime reference for ResizeOperator mode="cubic" (2026-11-03): every coordinate_transformation_mode the operator implements, with
exclude_outside and a non-default cubic_coeff_a, on a 5x7 input resized to 8x4 (H up, W down) through the `sizes` input. Prints C# rows for
MLTestBase.ResizeTests.cs (Resize_CubicModes_MatchOnnxRuntime). Usage: py -3.13 tools/gen_resize_cubic_reference.py
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

IN_H, IN_W, OUT_H, OUT_W = 5, 7, 8, 4
x = np.array([np.sin(i * 0.7) * 3 + (i % 3) for i in range(IN_H * IN_W)], dtype=np.float32).reshape(1, 1, IN_H, IN_W)
CASES = [("half_pixel", -0.75, 0), ("align_corners", -0.75, 0), ("asymmetric", -0.5, 0), ("pytorch_half_pixel", -0.75, 1), ("half_pixel", -0.5, 1)]

print("        // input: x[i] = sin(0.7 i) * 3 + i %% 3, [1,1,%d,%d] -> [1,1,%d,%d]" % (IN_H, IN_W, OUT_H, OUT_W))
for mode, a, excl in CASES:
    node = helper.make_node("Resize", ["x", "", "", "sizes"], ["y"], mode="cubic", coordinate_transformation_mode=mode,
                            cubic_coeff_a=a, exclude_outside=excl)
    g = helper.make_graph([node], "r", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, IN_H, IN_W])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
                          [helper.make_tensor("sizes", TensorProto.INT64, [4], [1, 1, OUT_H, OUT_W])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    y = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0].ravel()
    vals = ", ".join(f"{v:.6g}f" for v in y)
    print(f'        ("{mode}", {a}f, {excl}, new float[] {{ {vals} }}),')
