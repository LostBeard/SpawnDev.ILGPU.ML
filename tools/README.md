# tools

Diagnostics and gates for the ML engine. Most are single-file `dotnet run` scripts or small python
generators; the `*-harness` / `*-oracle` folders are projects.

Run C# ones from the repo root (`SpawnDev.ILGPU.ML/`):

```
dotnet run tools/<name>.cs -- [args]
dotnet run --project tools/<harness> -c Release -- <command>
```

## Is the engine telling the truth?

| tool | answers |
|---|---|
| `audit-operator-support.cs` | **Is what we ADVERTISE real?** Checks `OperatorRegistry.BuiltinOpTypes` against actual implementations, and flags the three shapes that make an operator a no-op: a silent fallback when an input is not a compile-time constant, an output shape taken from `inputs[0]`, and listed-but-unimplemented. Found 8 operators doing nothing on real data (2026-08-30). It prints file:line and does not judge - a fallback is fine when what it falls back TO is correct. |
| `zipvoice/ort_intermediates.py` | Promotes EVERY intermediate tensor of a model to a graph output, runs onnxruntime, prints per-node shape/min/max/mean. This is how you get a reference for the inside of a graph; ORT's C# API only returns declared outputs. |
| `zipvoice/first_divergence.py` | Diffs that against our `ML_DUMP_TENSORS=<substr>` output and names the **first** node where we disagree. Everything after it is downstream. These two together localised a months-old ZipVoice divergence in four bisections. |
| `zipvoice-harness -- runonnx <model.onnx> [fixture.json]` | Runs ANY onnx on our engine and prints each output beside its ORT reference, plus the compiled node list. Use when a unit test says a number is wrong but not what the engine did - PMT buffers the operator-level diagnostics that would tell you. |

⚠️ **python `onnx`, `onnxruntime` and `numpy` are installed.** A project note once recorded ORT
localisation as blocked on them and turned a four-step bisection into an open-ended hunt. Reach for a
reference runtime early - that is why we keep three of them.

## Reference-fixture generators (onnx + onnxruntime)

`gen_lstm_reference.py` · `gen_controlflow_reference.py` · `gen_scatter_reference.py` ·
`gen_operator_tests.py` · `gen_nlp_audio_references.py`

They write into `SpawnDev.ILGPU.ML.Demo/wwwroot/references/<group>/` where the PMT tests fetch them.

⚠️ **A fixture must be able to express the failure.** Every one of these is built so the OLD code cannot
pass it:
- X as a **graph input**, never an initializer - a constant folds onto the branch that already worked.
- An output **larger than `inputs[0]`** - control flow sized its buffer from the condition.
- A scatter that provably **changes values**; the generator refuses to emit a fixture whose output equals
  its input.
- **No duplicate indices** - ONNX leaves their order undefined, so such a fixture would legitimately flake.

The same discipline in one sentence: *a resampler tested with audio already at the target rate proves
nothing.*

## Browser gates

| tool | answers |
|---|---|
| `drive-ml-pages.cs [url] [routes]` | Do the demo ROUTES actually load a model and render a result? PMT drives the unit-test harness, not the pages. Waits on the page's **result element**, never a console line (some pages only log on error). Only routes marked ✅ VERIFIED in `Docs/DEMO_AND_MODEL_STATUS.md` belong in its table. |
| `drive-mic-capture.cs [url] [--transcribe]` | Microphone capture, and the full mic→text loop, against `/whisper`. |
| `probe-fake-mic.cs [url] [--file-audio] [--raw]` | Is Chrome's fake microphone emitting anything? |

⚠️ **Chrome's fake audio device produces DIGITAL SILENCE on this machine** - 24 AnalyserNode readings of
0.0000 over 6 s, measured in plain browser JS with none of our code involved, with
`--use-file-for-fake-audio-capture`, with the default device, and with the audio processing module
disabled. Frames still arrive and sample counters still advance, so a gate can report "9 seconds captured"
with every sample zero, and Whisper turns silence into confident fluent text.
**The working approach:** replace `getUserMedia` before the app boots (Playwright `AddInitScriptAsync`)
with a looping `BufferSource` of a known WAV. `drive-mic-capture.cs` does this.

## ZipVoice (voice-cloning TTS)

`zipvoice-harness` (roundtrip / synth / compare / endtoend / verify / trimsweep / runonnx) ·
`zipvoice-oracle` (sherpa-onnx, the independent implementation) · `zipvoice-fixture` ·
`zipvoice-g2p-probe` · `zipvoice-listen` · `zipvoice/inspect_zipvoice.cs`

- `ZIPVOICE_MODEL_DIR` points at a sherpa-onnx ZipVoice package; `ZIPVOICE_INT8=1` selects quantized
  graphs; `ZIPVOICE_NO_PAD=1` matches the reference implementation exactly.
- `ZIPVOICE_MAX_TOKENS` / `ZIPVOICE_MAX_REF_MS` shrink a case until a divergence is small enough to read.
- ⚠️ `ZIPVOICE_ENGINE=ilgpu` makes `synth` render through **our** engine. Every other render mode uses
  onnxruntime, which is why our engine had never produced a finished clip until 2026-08-30.

## Other

`whisper-harness` (transcript regression gate - must stay word-identical) · `stft-oracle` ·
`extract_onnx.py` · `distilgpt2_*.py` (reference logits) · `bmpdiff.cs` · `lts-train` · `GptOssRun`

## Standing gates after ANY engine change

```
PMT_FILTER=Operator dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj   # 97/97
dotnet run --project tools/whisper-harness -c Release                            # PASS 57 words
```

Then the group you touched: `ControlFlow_`, `Recurrent_`, `Scatter_`, `Resample`, `Microphone_`.
A full six-backend sweep is the release gate, not routine.
