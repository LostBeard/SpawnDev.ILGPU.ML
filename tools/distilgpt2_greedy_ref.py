"""
Generate a REAL greedy-decoding reference for DistilGPT-2 (Xenova/distilgpt2 ONNX export,
non-merged decoder_model.onnx) using ONNX Runtime. This is the ground truth the engine's
multi-token generation is asserted against (Rule 5: real reference, not self-comparison).

The engine re-feeds the FULL sequence each step (no KV cache yet), so we mirror that exactly:
each step runs the decoder on the whole running sequence and takes the last-position argmax.

Output: Demo/wwwroot/references/gpt2/distilgpt2_greedy.json
  { prompt, input_ids, num_new_tokens, generated_ids (prompt+new), per_step:[{seq, argmax, top1_logit}] }
"""
import json, os, sys, urllib.request
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_URL = "https://huggingface.co/Xenova/distilgpt2/resolve/main/onnx/decoder_model.onnx"
MODEL_PATH = os.path.join(HERE, "distilgpt2_decoder_model.onnx")
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "SpawnDev.ILGPU.ML", "SpawnDev.ILGPU.ML.Demo", "wwwroot", "references", "gpt2"))
OUT_PATH = os.path.join(OUT_DIR, "distilgpt2_greedy.json")

PROMPT = "The cat sat on the"
PROMPT_IDS = [464, 3797, 3332, 319, 262]   # GPT-2 BPE for "The cat sat on the"
NUM_NEW = 12
EOS = 50256

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"downloading {MODEL_URL} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    print("inputs:", in_names)

    ids = list(PROMPT_IDS)
    per_step = []
    for step in range(NUM_NEW):
        seq = len(ids)
        feed = {"input_ids": np.array([ids], dtype=np.int64)}
        if "attention_mask" in in_names:
            feed["attention_mask"] = np.ones((1, seq), dtype=np.int64)
        if "position_ids" in in_names:
            feed["position_ids"] = np.arange(seq, dtype=np.int64).reshape(1, seq)
        logits = sess.run(None, feed)[0]      # [1, seq, vocab]
        last = logits[0, -1, :]
        nxt = int(np.argmax(last))
        per_step.append({"seq": seq, "argmax": nxt, "top1_logit": float(last[nxt])})
        print(f"step {step}: seq={seq} -> {nxt} (logit={last[nxt]:.4f})")
        if nxt == EOS:
            break
        ids.append(nxt)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {
        "model": "Xenova/distilgpt2 onnx/decoder_model.onnx (non-merged)",
        "prompt": PROMPT,
        "input_ids": PROMPT_IDS,
        "num_new_tokens": len(ids) - len(PROMPT_IDS),
        "generated_ids": ids,
        "per_step": per_step,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", OUT_PATH)
    print("generated_ids:", ids)

if __name__ == "__main__":
    main()
