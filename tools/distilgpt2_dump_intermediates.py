# Dump EVERY intermediate tensor's first-K flat values from DistilGPT-2 decoder_model.onnx
# via onnxruntime with graph optimizations DISABLED (matches our engine's enableOptimization:false
# and preserves ONNX node-output names). Writes JSON keyed by tensor name → for node-by-node
# bisection against our engine's GraphExecutor.CapturedOutputs.
import sys, json, numpy as np, onnx, onnxruntime as ort

src = sys.argv[1]
out_json = sys.argv[2]
K = 4096  # cover full [1,5,768] and [1,12,5,64] block-0 tensors so per-token divergence is caught at its true source

ids = [464, 3797, 3332, 319, 262]
seq = len(ids)

m = onnx.load(src)
# Shape-inference populates value_info with concrete element types so ORT accepts the
# added outputs (UNDEFINED type 0 is rejected). Use the inferred ValueInfoProto directly.
m = onnx.shape_inference.infer_shapes(m)
vi_by_name = {vi.name: vi for vi in m.graph.value_info}
existing_outputs = {o.name for o in m.graph.output}
added = 0
for node in m.graph.node:
    for o in node.output:
        if o and o not in existing_outputs and o in vi_by_name:
            m.graph.output.extend([vi_by_name[o]])
            existing_outputs.add(o)
            added += 1
print(f"added {added} intermediate outputs via shape inference")

# Disable ALL optimizations so node names + execution match our engine's unoptimized path.
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])

feed = {}
for inp in sess.get_inputs():
    if inp.name == "input_ids": feed[inp.name] = np.array([ids], dtype=np.int64)
    elif inp.name == "attention_mask": feed[inp.name] = np.ones((1, seq), dtype=np.int64)
    elif inp.name == "position_ids": feed[inp.name] = np.array([list(range(seq))], dtype=np.int64)

out_names = [o.name for o in sess.get_outputs()]
results = sess.run(out_names, feed)

dump = {}
for name, arr in zip(out_names, results):
    a = np.asarray(arr).astype(np.float64).ravel()
    dump[name] = {
        "shape": list(np.asarray(arr).shape),
        "first": [round(float(v), 5) for v in a[:K].tolist()],
        "absmax": round(float(np.max(np.abs(a))) if a.size else 0.0, 5),
        "count": int(a.size),
    }

with open(out_json, "w") as f:
    json.dump(dump, f)
print(f"dumped {len(dump)} tensors to {out_json}")
# Sanity: print the final logits argmax for the last position
lg = np.asarray(results[out_names.index("logits")])
last = lg[0, -1, :]
print("logits argmax:", int(np.argmax(last)), "shape", lg.shape)
