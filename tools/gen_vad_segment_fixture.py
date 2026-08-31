# Build a multi-utterance audio fixture for the VAD endpointer, then take its reference from sherpa-onnx.
#
#   python tools/gen_vad_segment_fixture.py
#   dotnet run --project tools/vad-oracle -c Release -- \
#       SpawnDev.ILGPU.ML.Demo/wwwroot/references/vad/silero_vad.onnx \
#       SpawnDev.ILGPU.ML.Demo/wwwroot/references/vad/vad_three_utterances.wav \
#       SpawnDev.ILGPU.ML.Demo/wwwroot/references/vad/vad_three_utterances_segments.json
#
# WHY NOT JUST USE THE LIBRIVOX CLIP: it is 4 s of near-continuous speech, and sherpa finds ONE segment
# spanning almost all of it. As a reference that is nearly worthless - a detector that simply declared
# everything to be speech would match it. The fixture has to be able to express the failure, so it is built
# to REQUIRE segmentation: three separate utterances of real speech with real gaps between them. A
# detector that never closes a segment returns one; a detector that closes on every dip returns many.
#
# The gaps are low-level room tone rather than digital silence. Pure zeros are a case a microphone never
# produces, and a detector tuned on them would fall apart on the first real recording - the same mistake as
# testing a resampler with audio already at the target rate.
import os
import wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
WWW = os.path.join(ROOT, "SpawnDev.ILGPU.ML.Demo", "wwwroot")
SRC = os.path.join(WWW, "test-audio", "librivox-public-domain.wav")
OUT = os.path.join(WWW, "references", "vad", "vad_three_utterances.wav")

RATE = 16000
GAP_SECONDS = 1.2          # comfortably longer than the 0.5 s min-silence, so each gap MUST close a turn
ROOM_TONE_DBFS = -60.0     # audible-floor room tone, not digital silence

w = wave.open(SRC)
assert w.getnchannels() == 1 and w.getframerate() == RATE and w.getsampwidth() == 2
speech = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0

rng = np.random.default_rng(20260831)


def room_tone(seconds: float) -> np.ndarray:
    n = int(seconds * RATE)
    amp = 10.0 ** (ROOM_TONE_DBFS / 20.0)
    return (rng.standard_normal(n).astype(np.float32) * amp)


# Three utterances cut from the real clip. Different lengths so a fixed-size assumption cannot pass, and
# each is well over the 0.25 s min-speech so none is discarded as noise.
utterances = [
    speech[:int(2.2 * RATE)],
    speech[int(1.0 * RATE):int(2.6 * RATE)],
    speech[int(0.6 * RATE):],
]

parts = [room_tone(0.8)]
for u in utterances:
    parts.append(u)
    parts.append(room_tone(GAP_SECONDS))

audio = np.concatenate(parts)
audio = np.clip(audio, -1.0, 1.0)

pcm = (audio * 32768.0).clip(-32768, 32767).astype("<i2")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with wave.open(OUT, "wb") as o:
    o.setnchannels(1)
    o.setsampwidth(2)
    o.setframerate(RATE)
    o.writeframes(pcm.tobytes())

print(f"utterances : {[round(len(u) / RATE, 2) for u in utterances]} s")
print(f"gaps       : {GAP_SECONDS}s room tone at {ROOM_TONE_DBFS} dBFS")
print(f"total      : {len(audio) / RATE:.2f}s, {len(audio)} samples")
print(f"wrote      : {OUT}")
print()
print("Now take the reference from the independent implementation:")
print("  dotnet run --project tools/vad-oracle -c Release -- \\")
print("      SpawnDev.ILGPU.ML.Demo/wwwroot/references/vad/silero_vad.onnx \\")
print("      SpawnDev.ILGPU.ML.Demo/wwwroot/references/vad/vad_three_utterances.wav \\")
print("      SpawnDev.ILGPU.ML.Demo/wwwroot/references/vad/vad_three_utterances_segments.json")
