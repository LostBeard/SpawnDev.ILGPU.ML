# Build the Silero VAD reference: per-frame speech probabilities over real speech, from onnxruntime.
#
#   python tools/gen_silero_vad_reference.py
#
# WHY: Silero VAD is the endpointer for the hands-free loop, and it is STATEFUL in the one way that is
# easy to get silently wrong - its LSTM state arrives as GRAPH INPUTS h/c and comes back out as new_h/new_c
# every 512-sample frame. An engine that caches graph inputs, or that runs the LSTM against frozen state,
# still produces a plausible-looking probability for every frame. So the fixture has to be able to tell
# THREADED state from FROZEN state, not merely "did a number come out".
#
# It does that two ways:
#   1. Real speech (librivox, 16 kHz mono) rather than a tone, so the probabilities genuinely swing between
#      speech and silence and a constant output cannot pass.
#   2. A NEGATIVE CONTROL run that re-feeds zero state on every frame. Its probabilities are written into
#      the fixture, and the generator REFUSES to emit unless the two runs differ materially. That is the
#      assertion the consuming test makes: match the threaded reference, and be far from the frozen one.
#
# See feedback-choose-a-fixture-that-can-violate-the-property: "is there a test" is not the question;
# "can the fixture and the assertions together express the failure" is.
import json, os, shutil, wave
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
WWW = os.path.join(ROOT, "SpawnDev.ILGPU.ML.Demo", "wwwroot")
OUT = os.path.join(WWW, "references", "vad")
os.makedirs(OUT, exist_ok=True)

SRC_MODEL = r"D:\users\tj\Projects\SpawnDev.Reachy\SpawnDev.Reachy\models\silero_vad.onnx"
WAV = os.path.join(WWW, "test-audio", "librivox-public-domain.wav")

WINDOW = 512      # Silero's native frame at 16 kHz. The model's x input is fixed [1, 512].
RATE = 16000

model_path = os.path.join(OUT, "silero_vad.onnx")
if not os.path.exists(model_path):
    shutil.copyfile(SRC_MODEL, model_path)
print(f"model  : {model_path} ({os.path.getsize(model_path)} bytes)")

# Decode exactly as WavDecoder.DecodeWavFile does: int16 little-endian / 32768.
w = wave.open(WAV)
assert w.getnchannels() == 1 and w.getframerate() == RATE and w.getsampwidth() == 2, "fixture wav must be 16k mono s16"
pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
frames = len(pcm) // WINDOW
print(f"audio  : {len(pcm)} samples, {len(pcm)/RATE:.2f}s -> {frames} frames of {WINDOW}")

sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
out_names = [o.name for o in sess.get_outputs()]


def run(thread_state: bool):
    """Frame-by-frame. thread_state=False is the negative control: state never advances."""
    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)
    probs = []
    for i in range(frames):
        x = pcm[i * WINDOW:(i + 1) * WINDOW].reshape(1, WINDOW)
        prob, new_h, new_c = sess.run(out_names, {"x": x, "h": h, "c": c})
        probs.append(float(np.asarray(prob).ravel()[0]))
        if thread_state:
            h, c = np.asarray(new_h), np.asarray(new_c)
    return np.asarray(probs, dtype=np.float32), h, c


probs, final_h, final_c = run(True)
frozen, _, _ = run(False)

# ---- the fixture must be able to fail -------------------------------------------------------------
lo, hi = float(probs.min()), float(probs.max())
if hi < 0.85 or lo > 0.15:
    raise SystemExit(f"REFUSING: probabilities span {lo:.3f}..{hi:.3f} - this clip does not contain both "
                     f"clear speech and clear silence, so a constant output could pass it.")

drift = float(np.abs(probs - frozen).max())
if drift < 0.10:
    raise SystemExit(f"REFUSING: threaded and frozen-state runs differ by only {drift:.4f} - this fixture "
                     f"cannot detect a frozen LSTM state, which is the failure it exists to catch.")

speech = int((probs >= 0.5).sum())
print(f"probs  : min {lo:.4f} max {hi:.4f}, {speech}/{frames} frames >= 0.5")
print(f"control: frozen-state run differs by up to {drift:.4f} - fixture CAN detect frozen state")
print(f"state  : |final_h| max {float(np.abs(final_h).max()):.4f}, |final_c| max {float(np.abs(final_c).max()):.4f}")

ref = {
    "source": "onnxruntime " + ort.__version__,
    "model": "silero_vad.onnx",
    "audio": "test-audio/librivox-public-domain.wav",
    "sample_rate": RATE,
    "window": WINDOW,
    "frames": frames,
    "probs": [round(float(p), 7) for p in probs],
    "frozen_state_probs": [round(float(p), 7) for p in frozen],
    "max_threaded_vs_frozen": drift,
    "speech_frames": speech,
    "final_h": [round(float(v), 7) for v in np.asarray(final_h).ravel()],
    "final_c": [round(float(v), 7) for v in np.asarray(final_c).ravel()],
}
path = os.path.join(OUT, "silero_vad_librivox.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(ref, f)
print(f"wrote  : {path}")
