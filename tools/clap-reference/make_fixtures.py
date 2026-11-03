"""CLAP reference fixtures for SpawnDev.ILGPU.ML's CLAP parity gates (2026-11-03).

Runs transformers' ClapProcessor + ClapModel (laion/clap-htsat-unfused, the checkpoint Xenova's ONNX export was made from) on deterministic
clips and fixed prompts, and writes one JSON with everything each ML stage is gated against:
  - per clip: the 48 kHz waveform's generator (so ML regenerates the identical samples), input_features [1001, 64] (the log-mel after
    repeatpad), audio_embeds (512, L2-normalised by the model's projection), and the clip's logits_per_audio against the prompts;
  - per prompt: input_ids, attention_mask, text_embeds;
  - logit_scale (exp of the model's audio logit scale).
Usage (WSL, an env with torch + transformers):  python make_fixtures.py out.json
"""
import json, math, sys
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

SR = 48000
REPO = "laion/clap-htsat-unfused"


def clip(kind: str, seconds: float) -> np.ndarray:
    n = int(SR * seconds)
    t = np.arange(n, dtype=np.float64) / SR
    # deterministic noise C# reproduces bit for bit: an LCG (glibc constants), uniform in [-1, 1)
    state = 1234
    u = np.empty(n, dtype=np.float64)
    for i in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        u[i] = state / 1073741824.0 - 1.0
    if kind == "tone":          # 440 Hz + 2 kHz
        x = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 2000 * t)
    elif kind == "clicks":      # a click every 250 ms (gunshot-like transients) over faint noise
        x = 0.01 * u
        for c in range(0, n, SR // 4):
            m = min(n - c, 480)
            x[c:c + m] += np.exp(-np.arange(m) / 60.0) * 0.9
    elif kind == "noise":       # white noise
        x = 0.3 * u
    elif kind == "chirp":       # 200 Hz -> 8 kHz sweep
        f0, f1 = 200.0, 8000.0
        x = 0.4 * np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * seconds) * t * t))
    else:
        raise ValueError(kind)
    return x.astype(np.float32)


CLIPS = [("tone", 10.0), ("clicks", 10.0), ("noise", 10.0), ("chirp", 3.3)]   # chirp is short: exercises repeatpad
PROMPTS = ["a gunshot", "footsteps on gravel", "a car engine", "a pure musical tone", "white noise", "a zombie groaning"]


def main(out_path):
    proc = ClapProcessor.from_pretrained(REPO)
    model = ClapModel.from_pretrained(REPO).eval()
    out = {"repo": REPO, "sample_rate": SR, "prompts": PROMPTS, "clips": []}
    with torch.no_grad():
        # each prompt tokenised ALONE (no padding): Xenova's text_model.onnx has no attention_mask input, so a padded batch attends to its
        # pad tokens (MEASURED 2026-11-03 with onnxruntime: padded prompts at cosine 0.990-0.999 to transformers, unpadded at 1.0)
        texts = [proc(text=[p], return_tensors="pt") for p in PROMPTS]
        out["input_ids"] = [t["input_ids"][0].tolist() for t in texts]
        out["logit_scale"] = model.logit_scale_a.exp().item()
        for kind, seconds in CLIPS:
            wav = clip(kind, seconds)
            feats = proc(audio=[wav], sampling_rate=SR, return_tensors="pt")
            # the model's own forward, one prompt at a time: normalised text / audio embeddings and logits_per_audio
            os_ = [model(input_ids=t["input_ids"], attention_mask=t["attention_mask"], input_features=feats["input_features"]) for t in texts]
            if "text_embeds" not in out: out["text_embeds"] = [o.text_embeds[0].tolist() for o in os_]
            audio_embeds = os_[0].audio_embeds
            logits = torch.stack([o.logits_per_audio[0, 0] for o in os_])[None]
            out["clips"].append({
                "kind": kind, "seconds": seconds, "samples": int(wav.shape[0]),
                "input_features_shape": list(feats["input_features"].shape),
                "input_features": feats["input_features"][0, 0].tolist(),
                "audio_embeds": audio_embeds[0].tolist(),
                "logits_per_audio": logits[0].tolist(),
                "probs": torch.softmax(logits[0], dim=-1).tolist(),
            })
            print(kind, "features", tuple(feats["input_features"].shape), "top prompt", PROMPTS[int(logits[0].argmax())])
    json.dump(out, open(out_path, "w"))
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1])
