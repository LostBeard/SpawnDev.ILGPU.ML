# CLAP zero-shot audio classifier (for AssistZ, Riker) - plan, 2026-11-03

Goal: `Classify(clip, string[] labels) -> probabilities` on the device, labels defined by text ("gunshot near", "footsteps on gravel",
"zombie groan"), features from `SpectralFrontend` so PCM never round-trips. Parity first: every stage against transformers' reference.

## Checkpoint
- `laion/clap-htsat-unfused` - HF card `license: apache-2.0`; LAION-AI/CLAP code is CC0 1.0 Universal. Only `pytorch_model.bin` there.
- Runtime: `Xenova/clap-htsat-unfused` ONNX conversion (`onnx/audio_model.onnx`, `onnx/text_model.onnx`, fp16 + quantized variants,
  `preprocessor_config.json`, `tokenizer.json`); no licence field of its own -> the base's apache-2.0.
- Mirrored to models.spawndev.com under TJ's standing go (2026-11-03).

## Feature extraction (preprocessor_config.json, ClapFeatureExtractor)
- 48,000 Hz; n_fft 1024; hop 480; 64 mel bins; fmin 50 Hz; fmax 14,000 Hz.
- max / chunk length 10 s = 480,000 samples (~1,001 frames); truncation `rand_trunc` (random 10 s window of longer audio; for
  determinism we take the centre or the latest window), padding `repeatpad` (repeat the clip to fill 10 s), padding value 0.
- top_db null; no attention mask.
- VERIFIED in transformers' feature_extraction_clap.py (2026-11-03): for `rand_trunc` (our path) `_np_extract_fbank_features(waveform,
  self.mel_filters_slaney)` = `mel_filter_bank(norm="slaney", mel_scale="slaney", min 50, max 14000)` (the HTK `mel_filters` are only
  for "fusion"); `spectrogram(window_function(1024, "hann") [periodic], frame_length 1024, hop 480, power 2.0, log_mel="dB")` with
  audio_utils' defaults: center=True, pad_mode "reflect", mel_floor 1e-10, reference 1.0, no db_range (top_db null) ->
  10 log10(max(p, 1e-10)); frames = 480000 / 480 + 1 = 1001; output transposed to [time, mel] = [1001, 64] per clip, [1, 1, 1001, 64].
  Short clip: `repeatpad` = np.tile(waveform, int(480000 / len)) then zero-pad to 480000. Long clip: a random 480000 window
  (we pick a deterministic one and document it).
- Device plan: reflect-pad (512 each side) into a power-of-two buffer, SpectralFrontend N = 1024 (FFT), frames end at 1024 + 480 f,
  f = 0..1000, `SetBandFilters` with the Slaney mel weights (port librosa's formula; gate the weights against transformers'), PoolBands
  with logFloor 1e-10 -> [1001][1][64] = CLAP's layout.

## Steps
1. Reference fixtures (WSL venv): ClapProcessor + ClapModel on fixed clips (synthetic tone + real sound effects) and fixed prompts:
   dump input_features, audio_embeds, text_embeds, logits_per_audio.
2. `SpectralFrontend.SetBandFilters(weights)` (custom filter matrix) + whatever log / layout CLAP needs; gate the features against the
   fixture input_features.
3. Audio tower (audio_model.onnx) and text tower (text_model.onnx) on InferenceSession; gate embeddings (cosine >= 0.9999).
4. Tokenizer (RoBERTa BPE, tokenizer.json) - reuse ML's BPE tokenizer if it covers RoBERTa; gate ids against the reference.
5. `ClapAudioClassifier`: cached text embeddings per label set, logit scale, softmax; gate logits; a real-clip check
   (gunshot / footsteps / engine) for sanity.
6. Speed on CUDA / WebGPU; tell Riker.

## Status (2026-11-03, 341fcbd7) - all six steps DONE
1. Fixtures: tools/clap-reference/make_fixtures.py, each prompt tokenised alone (text_model.onnx has no attention_mask).
2. Features: `Clap_Features_MatchTransformers` green (signal within 0.0014 dB).
3. Towers: `Clap_Towers_MatchTransformers` green on CUDA / OpenCL / WebGPU after the cubic Resize align_corners fix (the operator fell back
   to bilinear; found with ort_node_reference.py + first_divergence.py).
4. Tokenizer: `BPETokenizer.Encode` now exact GPT-2 byte-level; `EncodeRoberta`; `Clap_Tokenizer_MatchesTransformers` green.
5. `ClapAudioClassifier` + `Clap_Classifier_EndToEnd` green. Real clip: bare labels score speech as "music" in transformers itself; the
   "This is a sound of X" template fixes it (speech 0.929).
6. Warm classify per clip (RTX 4070): CUDA 11-18 ms, OpenCL 11-12 ms, WebGPU 50-76 ms. Riker told (API, timings, template).

Later: GPU-side logits (skip the 512-float readback when only the top label is wanted); batch several clips per audio run.
