# Stores an ONNX model's large FP32 weights as FP16, keeping the model's COMPUTE in FP32: each weight W becomes a FP16
# initializer W__fp16 and a Cast(to=FLOAT) node that produces W, so every consumer still sees FP32 and the file stays
# valid ONNX (onnxruntime runs it as-is). Half the bytes of the weights; the graph is otherwise untouched.
#
#   py -3.13 tools/onnx-weights-fp16.py <in.onnx> <out.onnx> [--min-elements 1024]
#
# Small tensors (biases, norms, constants below --min-elements) stay FP32: they cost nothing and some are read as
# exact values (shapes, axes). External data (model.onnx + model.onnx_data) is read; the output is one file.
# SpawnDev.ILGPU.ML folds the Cast at load (InferenceSession.FoldWeightUpcastCasts: the weight is upcast once on the
# GPU), so there is no per-run Cast. tools/gen_fusion_reference.py imports to_fp16_weights for its test fixture.
import argparse
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def to_fp16_weights(m, min_elements=1024):
    """Rewrites model m in place; returns (weights converted, initializer bytes before, after)."""
    g = m.graph
    casts, kept, converted, before, after = [], [], 0, 0, 0
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        before += arr.nbytes
        if init.data_type != TensorProto.FLOAT or arr.size < min_elements:
            kept.append(init)
            after += arr.nbytes
            continue
        half = arr.astype(np.float16)
        if not np.all(np.isfinite(half[np.isfinite(arr)])):
            raise SystemExit(f'{init.name}: values overflow FP16 (max |w| {np.abs(arr).max():.3g})')
        t = numpy_helper.from_array(half, init.name + '__fp16')
        kept.append(t)
        after += half.nbytes
        casts.append(helper.make_node('Cast', [t.name], [init.name], to=TensorProto.FLOAT, name=init.name + '__upcast'))
        converted += 1
    del g.initializer[:]
    g.initializer.extend(kept)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)   # Casts first: topological order holds (they only read initializers)
    onnx.checker.check_model(m)
    return converted, before, after


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--min-elements', type=int, default=1024)
    a = ap.parse_args()
    model = onnx.load(a.src, load_external_data=True)
    n, b0, b1 = to_fp16_weights(model, a.min_elements)
    onnx.save(model, a.dst)
    print(f'{n} weights -> FP16: {b0 / 1048576:.1f} MB -> {b1 / 1048576:.1f} MB of initializers; wrote {a.dst}')
