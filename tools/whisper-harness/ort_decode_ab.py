"""A/B: is the residual repetition loop OUR mel, or OUR decode loop?

Feeds the mel our C# preprocessing produced into ONNX Runtime's encoder+decoder and greedy-decodes with the
same prompt and the same argmax rule. Same mel, same greedy policy, different runtime:

  * clean text here  -> our mel is fine and the loop is in our decode/graph
  * loops here too   -> our mel is the problem, not the decoder

    python tools/whisper-harness/ort_decode_ab.py
"""
import os
import numpy as np
import onnxruntime as ort
import json

MODEL_DIR = r"D:\users\tj\Projects\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML.Demo\wwwroot\models\whisper-tiny"
MEL = os.path.join(os.environ.get("TEMP", "/tmp"), "our_mel.bin")
if not os.path.exists(MEL):
    MEL = "/tmp/our_mel.bin"

mel = np.fromfile(MEL, dtype=np.float32)
print(f"mel      : {mel.size} floats, min {mel.min():.3f} max {mel.max():.3f} mean {mel.mean():.3f}")
mel = mel.reshape(1, 80, 3000)

enc = ort.InferenceSession(os.path.join(MODEL_DIR, "encoder_model.onnx"), providers=["CPUExecutionProvider"])
dec = ort.InferenceSession(os.path.join(MODEL_DIR, "decoder_model.onnx"), providers=["CPUExecutionProvider"])
hidden = enc.run(None, {enc.get_inputs()[0].name: mel})[0]
print(f"encoder  : {hidden.shape}")

SOT, EN, TRANSCRIBE, NOTS, EOT = 50258, 50259, 50359, 50363, 50257
tokens = [SOT, EN, TRANSCRIBE, NOTS]
for _ in range(64):
    logits = dec.run(None, {
        dec.get_inputs()[0].name: np.array([tokens], dtype=np.int64),
        dec.get_inputs()[1].name: hidden,
    })[0]
    nxt = int(np.argmax(logits[0, -1]))
    if nxt == EOT:
        break
    tokens.append(nxt)

print(f"tokens   : {tokens[4:16]}")
# Decode with the raw BPE vocab from tokenizer.json - no extra dependency needed.
vocab = json.load(open(os.path.join(MODEL_DIR, "tokenizer.json"), encoding="utf-8"))["model"]["vocab"]
inv = {v: k for k, v in vocab.items()}
byte_decoder = {chr(b if b > 32 else 256 + b): b for b in range(256)}
print("ORT TRANSCRIPT:")
raw = "".join(inv.get(t, "") for t in tokens[4:])
print("  \"" + raw.replace("0120", " ") + "\"")
