# ZipVoice gates

Voice-CLONING TTS. Three ONNX graphs (text encoder, flow-matching decoder, vocos vocoder) plus the
code around them: mel features, the Euler sampler, and an inverse STFT.

```
dotnet run --project tools/zipvoice-harness -c Release -- roundtrip [wav]
dotnet run --project tools/zipvoice-harness -c Release -- synth [fixture.json] [out.wav]
dotnet run --project tools/zipvoice-oracle  -c Release -- "text" out.wav 2> tokens.log
```

`ZIPVOICE_MODEL_DIR` points at a sherpa-onnx ZipVoice package; `ZIPVOICE_INT8=1` selects the quantized
graphs; `ZIPVOICE_NO_PAD=1` drops the reference tail padding so a run matches the reference
implementation exactly.

## What each one proves

**roundtrip** - audio to mel to the REAL vocoder to audio. A correct mel is a fixed point of that loop,
so a wrong window, mel scale, normalisation, or magnitude-vs-power choice shows up as drift. Needs no
tokenizer, so it was the first thing that could run. Grade it further by transcribing the output:

```
dotnet run --project ../tools/whisper-harness -c Release -- <the .roundtrip.wav>
```

**synth** - the whole cloning path from ground-truth token ids, with onnxruntime running the graphs.
The orchestration is the SHIPPING code, so a bad result here is the algorithm and never the engine.

**oracle** - sherpa-onnx, the independent implementation. It answers the two questions our own code
cannot answer about itself: which token ids espeak-ng really produces (English has no lexicon entry -
`lexicon.txt` is Chinese-only), and what a correct clone of a given reference sounds like.

## Two things that will mislead you

- Output is NOT reproducible. Flow matching draws fresh noise per call, so two renders differ. The
  harness pins a seed; production does not.
- sherpa shrinks silences to 20% (`silence_scale = 0.2`) AFTER generating, so its output is shorter
  than ours for the same input even when the generated frame counts match exactly.
