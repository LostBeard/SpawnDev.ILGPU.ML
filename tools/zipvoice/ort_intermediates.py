# Per-node reference values for the ZipVoice text encoder, from onnxruntime.
#
#   python tools/zipvoice/ort_intermediates.py [name-substring] [--tokens 6] [--frames 57]
#
# WHY THIS EXISTS: our engine runs the encoder but its output differs from onnxruntime by ~18.6% of peak
# at worst, and the divergence is a LOCAL computation rather than an accumulation (measured: shrinking to
# 6 tokens and a 0.6 s reference keeps the same error). Localising it needs ORT's INTERMEDIATE values, and
# ORT's C# API only returns declared graph outputs. The project notes recorded this as blocked on "an ONNX
# serialiser or the python onnx package, NOT currently installed" - onnx 1.21.0 and onnxruntime 1.26.0 are
# both installed now, so it is not blocked at all.
#
# Promoting every intermediate to a graph output is what makes them readable. ORT will not constant-fold
# or fuse a tensor it has to return, which is exactly what we want here - the values then correspond
# one-for-one with the nodes our executor runs.
#
# Pair it with our side:
#     ML_DUMP_TENSORS=<same substring> dotnet run --project tools/zipvoice-harness -c Release -- compare
# and compare in node order. The FIRST tensor that disagrees is the bug; everything after it is downstream.
#
# Python is used here rather than C# for one reason: `onnx` can rewrite the graph and we have no ONNX
# writer. That is the documented exception to the C#-first preference.
import sys, os, json
import numpy as np
import onnx
import onnxruntime as ort

MODEL = os.environ.get(
    "ZIPVOICE_ENCODER",
    r"D:\users\tj\Projects\SpawnDev.Reachy\SpawnDev.Reachy\models\sherpa-onnx-zipvoice-distill-zh-en-emilia\text_encoder.onnx")

# The harness's pinned fixture, truncated exactly as Compare.cs truncates it: tokens[..n], promptTokens[..n].
FIXTURE = r"D:\users\tj\Projects\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML\tools\zipvoice-harness\fixtures\paint-the-sockets.json"

args = sys.argv[1:]
substr = next((a for a in args if not a.startswith("--")), "layers.0")
def opt(name, default):
    if name in args:
        return int(args[args.index(name) + 1])
    return default
n_tokens = opt("--tokens", 6)
n_frames = opt("--frames", 57)

with open(FIXTURE, encoding="utf-8") as f:
    fx = json.load(f)
tokens = np.array([fx["tokens"][:n_tokens]], dtype=np.int64)
prompt = np.array([fx["promptTokens"][:n_tokens]], dtype=np.int64)

model = onnx.load(MODEL)
graph = model.graph

# Every tensor produced by a node, in node order. Initialisers are excluded - they are inputs, not results.
produced = []
seen = set()
for node in graph.node:
    for out in node.output:
        if out and out not in seen:
            seen.add(out)
            produced.append((node.op_type, node.name, out))

wanted = [(op, nm, out) for (op, nm, out) in produced if substr.lower() in out.lower() or substr.lower() in nm.lower()]
if not wanted:
    print(f"no tensors match '{substr}' - try a broader substring (e.g. layers.0, self_attn, Softmax)")
    sys.exit(2)

existing = {o.name for o in graph.output}
for _, _, out in wanted:
    if out not in existing:
        graph.output.append(onnx.helper.make_empty_tensor_value_info(out))

# ORT rejects a graph whose declared outputs it cannot type; shape inference fills those in.
try:
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
except Exception as e:
    print(f"  (shape inference warning: {type(e).__name__}: {e})")

so = ort.SessionOptions()
# Fusions rewrite the very nodes we are trying to observe, so turn them off.
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])

feed = {}
for i in sess.get_inputs():
    if i.name == "tokens":
        feed[i.name] = tokens
    elif i.name == "prompt_tokens":
        feed[i.name] = prompt
    elif i.name == "prompt_features_len":
        feed[i.name] = np.array(n_frames, dtype=np.int64)
    elif i.name == "speed":
        feed[i.name] = np.array(1.0, dtype=np.float32)
    else:
        raise SystemExit(f"unexpected encoder input '{i.name}' - the fixture does not cover it")

names = [out for _, _, out in wanted]
vals = sess.run(names, feed)

print(f"# ORT reference, {len(names)} tensors matching '{substr}'")
print(f"# tokens={tokens.tolist()} prompt={prompt.tolist()} frames={n_frames} speed=1.0")
print(f"# {'op':18s} {'shape':18s} {'min':>12s} {'max':>12s} {'mean':>12s}  name")
for (op, nm, out), v in zip(wanted, vals):
    a = np.asarray(v)
    if a.size == 0:
        print(f"  {op:18s} {str(list(a.shape)):18s} {'-':>12s} {'-':>12s} {'-':>12s}  {out}")
        continue
    f = a.astype(np.float64).ravel()
    print(f"  {op:18s} {str(list(a.shape)):18s} {f.min():12.6g} {f.max():12.6g} {f.mean():12.6g}  {out}")
