# Build Slice reference cases that a REVERSED slice cannot pass by accident.
#
#   python tools/gen_slice_reference.py
#
# WHY: `SliceOperator.Execute` clamped `ends` into [0, dim] whatever the sign of the step. For a negative
# step ONNX clamps it into [-1, dim-1] instead, because a reversed slice legitimately ends one BEFORE
# index 0 - which is what the INT64_MIN sentinel in `x[..., ::-1]` means. Clamped to 0, the copy loops ran
# `for (i = 2; i < 0; i += -1)`: zero iterations, output buffer never written, ALL ZEROS, no error.
# GraphCompiler's shape resolution already handled negative steps, so the output had the RIGHT SHAPE full
# of zeros and nothing downstream complained. Found via Silero VAD, whose adaptive_normalization reverses
# a [1,1,3] axis twice.
#
# ⚠️ Fixture design (feedback-choose-a-fixture-that-can-violate-the-property): every case here is built so
# the OLD code cannot pass it.
#   - DATA is a graph INPUT, never an initializer - a constant folds into the shape interpreter, which was
#     always correct, and would prove nothing about the execution path.
#   - Values are all NON-ZERO, so the pre-fix all-zeros output is a guaranteed mismatch rather than
#     something that might coincide.
#   - Cases are ASYMMETRIC along the reversed axis, so a reversal that runs but does not actually reverse
#     (returning the input unchanged) also fails.
#   - Forward cases are included alongside, so the fix cannot regress ordinary slicing.
import json, os
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                   "SpawnDev.ILGPU.ML.Demo", "wwwroot", "references", "slice"))
os.makedirs(OUT, exist_ok=True)

INT64_MIN_SENTINEL = -9223372036854775807   # what torch emits for "to the beginning"

# name, input shape, starts, ends, axes, steps
CASES = [
    # The exact shape Silero VAD reverses, and the exact sentinel it uses.
    ("reverse_last_axis",      [1, 1, 3],  [-1], [INT64_MIN_SENTINEL], [-1], [-1]),
    # A longer axis, so an off-by-one in the reversed bound cannot hide inside 3 elements.
    ("reverse_long_axis",      [1, 2, 8],  [-1], [INT64_MIN_SENTINEL], [-1], [-1]),
    # Reversal on a NON-last axis exercises the recursive descent rather than the last-axis run.
    ("reverse_middle_axis",    [2, 4, 3],  [-1], [INT64_MIN_SENTINEL], [1],  [-1]),
    # Step -2 proves the stride is honoured, not just the direction.
    ("reverse_step2",          [1, 1, 7],  [-1], [INT64_MIN_SENTINEL], [-1], [-2]),
    # A reversed slice that stops EARLY - the end bound is doing real work here, not just sentinel handling.
    ("reverse_partial",        [1, 1, 6],  [4],  [1],                  [-1], [-1]),
    # Two axes reversed in one node.
    ("reverse_two_axes",       [3, 4, 2],  [-1, -1], [INT64_MIN_SENTINEL, INT64_MIN_SENTINEL], [0, 1], [-1, -1]),
    # Forward cases: the fix must not disturb ordinary slicing.
    ("forward_basic",          [1, 2, 8],  [2],  [6],                  [-1], [1]),
    ("forward_step2",          [1, 1, 9],  [0],  [9],                  [-1], [2]),
    ("forward_to_end",         [2, 5],     [1],  [9223372036854775807], [1], [1]),
]

rng = np.random.default_rng(20260831)
summary = {}

for name, shape, starts, ends, axes, steps in CASES:
    n = int(np.prod(shape))
    # Non-zero everywhere: the pre-fix failure mode is an untouched all-zero buffer, and a fixture
    # containing zeros would let part of that pass.
    data = (rng.standard_normal(n).astype(np.float32) * 2.0)
    data[np.abs(data) < 0.25] += 0.5
    data = data.reshape(shape)

    def const(vals, tag):
        return helper.make_tensor(f"{tag}", TensorProto.INT64, [len(vals)], vals)

    node = helper.make_node("Slice", ["data", "starts", "ends", "axes", "steps"], ["out"])
    graph = helper.make_graph(
        [node], f"slice_{name}",
        # data is a graph INPUT on purpose - as an initializer it would be constant-folded by the shape
        # interpreter, which never had this bug.
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)],
        # Symbolic dims: Slice keeps the rank, and the checker requires a shape even when the sizes are
        # exactly what is under test. Naming them rather than fixing them keeps the fixture honest.
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT,
                                               [f"d{i}" for i in range(len(shape))])],
        initializer=[const(starts, "starts"), const(ends, "ends"), const(axes, "axes"), const(steps, "steps")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    got = np.asarray(sess.run(["out"], {"data": data})[0])

    if got.size == 0:
        raise SystemExit(f"REFUSING {name}: onnxruntime produced an EMPTY slice - the case is malformed.")
    if np.abs(got).min() == 0:
        raise SystemExit(f"REFUSING {name}: expected output contains a zero, so an all-zeros result "
                         f"could partially match. Reroll the data.")
    if any(st < 0 for st in steps):
        flat_in, flat_out = data.ravel(), got.ravel()
        if flat_out.shape == flat_in.shape and np.allclose(flat_out, flat_in):
            raise SystemExit(f"REFUSING {name}: reversed output equals the input, so an engine that "
                             f"ignores the reversal would pass.")

    onnx.save(model, os.path.join(OUT, f"slice_{name}.onnx"))
    with open(os.path.join(OUT, f"slice_{name}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "name": name, "shape": shape, "starts": starts, "ends": ends, "axes": axes, "steps": steps,
            "data": data.ravel().tolist(),
            "out_shape": list(got.shape),
            "out": got.ravel().tolist(),
        }, f)
    summary[name] = list(got.shape)
    print(f"  {name:20s} {str(shape):12s} -> {str(list(got.shape)):12s} "
          f"steps={steps}  |min| {np.abs(got).min():.4f}")

with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
    json.dump({"cases": list(summary)}, f)
print(f"wrote {len(summary)} cases to {OUT}")
