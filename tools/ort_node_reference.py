# Per-node reference values for ANY onnx model, from onnxruntime.
#
#   python tools/ort_node_reference.py <model.onnx> <fixture.json> [substring] > ort.txt
#   ML_DUMP_TENSORS="<substring>" dotnet run --project tools/zipvoice-harness -c Release -- \
#       runonnx <model.onnx> <fixture.json> > ours.txt
#   python tools/zipvoice/first_divergence.py ort.txt ours.txt
#
# WHY: tools/zipvoice/ort_intermediates.py does this for the ZipVoice encoder alone - its inputs are
# hardwired to that model's tokens/prompt_tokens/speed. The technique is not ZipVoice-specific, so this is
# the same idea driven by the runonnx fixture format instead:
#
#   { "inputs": { "<name>": { "shape": [...], "data": [...] } }, "outputs": { ... } }
#
# ORT's C# API only returns DECLARED graph outputs, so intermediates are invisible until you promote them
# to outputs. ORT will not fold or fuse a tensor it has to return, which is exactly what we want: the
# values then line up one-for-one with the nodes our executor runs. Optimisation is disabled for the same
# reason - a fusion rewrites the very node under investigation.
#
# Output format is deliberately identical to ort_intermediates.py because first_divergence.py parses it.
#
# Python rather than C# for the documented exception: `onnx` can rewrite a graph and we have no ONNX writer.
import sys, os, json
import numpy as np
import onnx
import onnxruntime as ort

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(2)

model_path, fixture_path = sys.argv[1], sys.argv[2]
substr = sys.argv[3] if len(sys.argv) > 3 else ""

ELEM = {
    onnx.TensorProto.FLOAT: np.float32, onnx.TensorProto.DOUBLE: np.float64,
    onnx.TensorProto.INT64: np.int64, onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.BOOL: np.bool_, onnx.TensorProto.FLOAT16: np.float16,
}

with open(fixture_path, encoding="utf-8") as f:
    fx = json.load(f)
fixture_inputs = fx.get("inputs", {})

model = onnx.load(model_path)
graph = model.graph

# Every tensor a node produces, in node order. Initialisers are excluded - they are inputs, not results.
produced, seen = [], set()
for node in graph.node:
    for out in node.output:
        if out and out not in seen:
            seen.add(out)
            produced.append((node.op_type, node.name, out))

wanted = [t for t in produced if not substr
          or substr.lower() in t[2].lower() or substr.lower() in (t[1] or "").lower()]
if not wanted:
    print(f"no tensors match '{substr}' out of {len(produced)} produced")
    sys.exit(2)

existing = {o.name for o in graph.output}
for _, _, out in wanted:
    if out not in existing:
        graph.output.append(onnx.helper.make_empty_tensor_value_info(out))

# ORT rejects a graph whose declared outputs it cannot type; shape inference fills those in.
try:
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
except Exception as e:
    print(f"# (shape inference warning: {type(e).__name__}: {e})")

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])

feed = {}
for i in sess.get_inputs():
    if i.name not in fixture_inputs:
        raise SystemExit(f"fixture has no input '{i.name}' (has: {', '.join(fixture_inputs)})")
    spec = fixture_inputs[i.name]
    # The fixture stores every value as a JSON number; the model says what type it really is.
    vi = next((v for v in graph.input if v.name == i.name), None)
    dt = ELEM.get(vi.type.tensor_type.elem_type, np.float32) if vi is not None else np.float32
    feed[i.name] = np.asarray(spec["data"], dtype=dt).reshape(spec["shape"])

names = [out for _, _, out in wanted]
vals = sess.run(names, feed)

print(f"# ORT reference, {len(names)} of {len(produced)} tensors" + (f" matching '{substr}'" if substr else ""))
print(f"# model={os.path.basename(model_path)} fixture={os.path.basename(fixture_path)}")
print(f"# {'op':18s} {'shape':18s} {'min':>12s} {'max':>12s} {'mean':>12s}  name")
for (op, nm, out), v in zip(wanted, vals):
    a = np.asarray(v)
    if a.size == 0:
        print(f"  {op:18s} {str(list(a.shape)):18s} {'-':>12s} {'-':>12s} {'-':>12s}  {out}")
        continue
    f = a.astype(np.float64).ravel()
    print(f"  {op:18s} {str(list(a.shape)):18s} {f.min():12.6g} {f.max():12.6g} {f.mean():12.6g}  {out}")
