# onnxruntime per-node reference for RMBG-1.4, on the EXACT input our tests feed.
#
#   python tools/gen_rmbg_node_reference.py <model.onnx> [side] [substring]
#
#   python tools/gen_rmbg_node_reference.py model.onnx 256 rebnconv2
#   python tools/gen_rmbg_node_reference.py model.onnx 1024 stage1
#
# WHY: Pipeline_BackgroundRemoval_PerOpDiagnostic reports
#   "FIRST SATURATED #13: 019_Relu_/stage1/rebnconv2/relu_s1/Relu_output_0  min=max=mean=0.0000"
# identically on all six backends. Identical everywhere rules out a BACKEND bug but cannot separate
# "the model really produces no positive activation there" from "something upstream is wrong in a
# reproducible way". Only an onnxruntime reference on the SAME input separates those.
#
# ⚠️ TWO THINGS THE DIAGNOSTIC'S VERDICT CANNOT SEE, both of which this script prints:
#
#   1. GraphExecutor.CaptureMaxElements is 1024, so the diagnostic's min/max/mean/variance are computed
#      over the first 1024 CONTIGUOUS elements. For a [1,32,64,64] NCHW tensor that is rows 0..15 of
#      CHANNEL 0 - one corner of one channel out of 131,072 values. A zero prefix says nothing about the
#      tensor. So this prints FULL-tensor stats AND the same 1024-prefix the diagnostic sees, and the
#      two disagreeing is itself the finding.
#   2. The model's declared input is [batch, 3, 1024, 1024] - H and W are FIXED in the ONNX, not
#      symbolic. The diagnostic overrides them to 256 via inputShapes. Pass side=256 and side=1024 and
#      compare: if ORT itself produces a dead channel at 256 but not at 1024, the override is feeding
#      the model out of distribution and the "saturation" is our TEST's doing, not the engine's.
#
# Technique is the same as tools/ort_node_reference.py: promote intermediates to graph outputs (ORT
# will not fold or fuse a tensor it must return) and disable all optimisation, so values line up
# one-for-one with the nodes our executor runs.
#
# Python rather than C# for the documented exception: `onnx` rewrites the graph and we have no ONNX writer.
import sys
import numpy as np
import onnx
import onnxruntime as ort

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(2)

model_path = sys.argv[1]
side = int(sys.argv[2]) if len(sys.argv) > 2 else 256
substr = sys.argv[3] if len(sys.argv) > 3 else "rebnconv2"

# ── The EXACT input the tests build ────────────────────────────────────────────────────────────────
# MLTestBase: pixels[y*side+x] = x < side/2 ? white(255,255,255) : dark(30,30,30), alpha 255.
# ImagePreprocessKernel: v/255f, then (v - mean) * invStd, with mean=0.5 std=1.0 for RMBG.
# src dims == dst dims so the bilinear sample is the identity (no resize), and R==G==B in both halves
# so the channel order cannot matter.
WHITE = 255.0 / 255.0 - 0.5          #  0.5
DARK = 30.0 / 255.0 - 0.5            # -0.3823529411764706
x = np.arange(side, dtype=np.float32)
row = np.where(x < side // 2, np.float32(WHITE), np.float32(DARK))
plane = np.tile(row, (side, 1))                      # [side, side]
inp = np.repeat(plane[None, None, :, :], 3, axis=1)   # [1, 3, side, side]
inp = np.ascontiguousarray(inp, dtype=np.float32)
print(f"input side={side} shape={list(inp.shape)} white={WHITE:.6f} dark={DARK:.6f} "
      f"(cols 0..{side//2-1} white, {side//2}..{side-1} dark)")

model = onnx.load(model_path)
graph = model.graph

produced, seen = [], set()
for node in graph.node:
    for out in node.output:
        if out and out not in seen:
            seen.add(out)
            produced.append((node.op_type, node.name, out))

wanted = [t for t in produced
          if substr.lower() in t[2].lower() or substr.lower() in (t[1] or "").lower()]
if not wanted:
    print(f"no tensors match '{substr}' out of {len(produced)} produced")
    sys.exit(2)
print(f"promoting {len(wanted)} of {len(produced)} intermediates to outputs (filter '{substr}')")

existing = {o.name for o in graph.output}
for _op, _name, out in wanted:
    if out not in existing:
        graph.output.append(onnx.ValueInfoProto(name=out))

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])

feeds = {sess.get_inputs()[0].name: inp}
names = [o.name for o in sess.get_outputs()]
vals = sess.run(names, feeds)
by_name = dict(zip(names, vals))

PREFIX = 1024   # GraphExecutor.CaptureMaxElements


def stats(a):
    a = np.asarray(a, dtype=np.float64).ravel()
    return a.min(), a.max(), a.mean(), a.var()


print()
print(f"{'tensor':62} {'shape':20} {'FULL min/max/mean/var':46} {'first-1024 (what the diagnostic sees)'}")
for _op, _name, out in wanted:
    if out not in by_name:
        continue
    t = np.asarray(by_name[out])
    fmn, fmx, fmu, fvar = stats(t)
    p = t.ravel()[:PREFIX]
    pmn, pmx, pmu, pvar = stats(p)
    flag = ""
    if abs(pmu) < 0.05 and pvar < 0.001 and not (abs(fmu) < 0.05 and fvar < 0.001):
        flag = "  <== PREFIX LOOKS DEAD, TENSOR IS NOT"
    elif abs(fmu) < 0.05 and fvar < 0.001:
        flag = "  <== TENSOR REALLY IS ~ZERO"
    print(f"{out[:62]:62} {str(list(t.shape))[:20]:20} "
          f"{fmn:+.5f}/{fmx:+.5f}/{fmu:+.5f}/{fvar:.7f}   "
          f"{pmn:+.5f}/{pmx:+.5f}/{pmu:+.5f}/{pvar:.7f}{flag}")

# Per-channel view of the flagged node: a Relu whose CHANNEL 0 is dead while other channels are alive
# is completely normal (one filter that does not respond to this input), and is invisible to a
# contiguous prefix that only ever sees channel 0.
target = [o for _op, _n, o in wanted if o in by_name and "relu_s1" in o.lower()]
for out in target:
    t = np.asarray(by_name[out])
    if t.ndim == 4 and t.shape[1] > 1:
        ch = t[0]
        alive = [(c, float(ch[c].max())) for c in range(ch.shape[0])]
        dead = [c for c, m in alive if m <= 0.0]
        print()
        print(f"{out}  per-channel max over {ch.shape[0]} channels:")
        print(f"   dead channels (max <= 0): {len(dead)} of {ch.shape[0]} -> {dead[:16]}")
        print(f"   channel 0 max = {alive[0][1]:.6f}   overall max = {float(ch.max()):.6f}")
