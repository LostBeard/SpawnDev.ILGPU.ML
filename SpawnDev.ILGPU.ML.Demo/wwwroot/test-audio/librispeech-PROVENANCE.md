# librispeech-*.wav

Three utterances from LibriSpeech `test-clean` (Panayotov et al., OpenSLR 12, https://www.openslr.org/12), CC BY 4.0, read
speech from public-domain LibriVox recordings. 16 kHz mono 16-bit, unmodified except for the FLAC -> WAV conversion.

| file | speaker | text |
|---|---|---|
| librispeech-1089_1089-134686-0003.wav | 1089 | HELLO BERTIE ANY GOOD IN YOUR MIND |
| librispeech-1089_1089-134691-0018.wav | 1089 | AGAIN AGAIN |
| librispeech-4077_4077-13751-0019.wav | 4077 | WHO BEGAN THE QUARREL WAS IT THE MORMONS |

`librispeech-speaker-fixture.json` holds the onnxruntime (CPU, fp32) ReDimNet2-B6 embedding of each clip, the reference for
`Pipeline_SpeakerEmbedding_MatchesOnnxRuntimeAndSeparatesVoices`. Same-speaker cosine 0.565, different-speaker 0.194 / 0.111 (measured).
