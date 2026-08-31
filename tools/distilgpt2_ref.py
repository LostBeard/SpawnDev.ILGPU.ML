# Ground-truth next-token for DistilGPT-2 decoder_model.onnx via onnxruntime.
# Rule 4b: measure the reference, don't guess it. Prints argmax + top5 of the
# last-position logits for "The cat sat on the" = [464,3797,3332,319,262].
import sys, numpy as np, onnxruntime as ort

model = sys.argv[1]
ids = [464, 3797, 3332, 319, 262]
sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
print("INPUTS:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print("OUTPUTS:", [(o.name, o.shape) for o in sess.get_outputs()])

seq = len(ids)
feed = {}
for inp in sess.get_inputs():
    n = inp.name
    if n == "input_ids":
        feed[n] = np.array([ids], dtype=np.int64)
    elif n == "attention_mask":
        feed[n] = np.ones((1, seq), dtype=np.int64)
    elif n == "position_ids":
        feed[n] = np.array([list(range(seq))], dtype=np.int64)
    elif "past_key_values" in n or "past" in n:
        # empty past for non-merged prefill: shape [1, n_head, 0, head_dim]
        sh = [d if isinstance(d, int) else (1 if i == 0 else (12 if i == 1 else 0)) for i, d in enumerate(inp.shape)]
        feed[n] = np.zeros(sh, dtype=np.float32)
    else:
        print("UNHANDLED INPUT", n, inp.shape)

out = sess.run(None, feed)
logits = out[0]  # [1, seq, vocab]
print("logits shape:", logits.shape)
last = logits[0, -1, :]
top5 = np.argsort(last)[::-1][:5]
print("ARGMAX:", int(top5[0]))
print("TOP5:", [(int(t), round(float(last[t]), 4)) for t in top5])
