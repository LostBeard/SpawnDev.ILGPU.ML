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
| `ort_node_reference.py <model> <fixture> [substr]` | The same idea for **ANY** model, driven by the runonnx fixture format instead of ZipVoice's hardwired inputs. Reach for this one first; the ZipVoice script above only exists because it predates it. |
| `zipvoice/first_divergence.py` | Diffs either of those against our `ML_DUMP_TENSORS=<substr>` output and names the **first** node where we disagree. Everything after it is downstream. These together localised a months-old ZipVoice divergence in four bisections, and the Silero VAD reverse-slice bug in one. |
| `audit_reverse_slice.py [root ...]` | **Which models reverse a slice axis?** Written when a negative-step Slice was found writing NOTHING (5.2.4): the blast radius of an operator bug is every model that uses it, and the only way to know which is to look. MEASURED at the time: of 47 models in this repo only Silero VAD, and in the Reachy model tree also pyannote-segmentation's sincnet and ZipVoice's text encoder. ⚠️ A hit is not automatically a bug - ZipVoice's `/Slice` sat on the shape/interpreter lane, which never had the defect, and its encoder matched onnxruntime to 6.7e-8 while the GPU path was broken. |
| `zipvoice-harness -- runonnx <model.onnx> [fixture.json]` | Runs ANY onnx on our engine and prints each output beside its ORT reference, plus the compiled node list. Use when a unit test says a number is wrong but not what the engine did - PMT buffers the operator-level diagnostics that would tell you. |

⚠️ **`ML_DUMP_TENSORS=/` DOES NOT WORK FROM GIT BASH.** MSYS rewrites a value that looks like a POSIX
path before it reaches a native process, so the engine receives `C:/Program Files/Git/` and matches
nothing - silently, with zero dump lines and no error. Use a substring that cannot be read as a path:
`ML_DUMP_TENSORS=_output_0` matches every ONNX node output. PowerShell is unaffected.

⚠️ **A tensor can appear TWICE in our dump, and the two lines disagree.** Shape-lane nodes print an
`[interp]` line (the value the graph actually consumed) AND a `[dump]` line reading the GPU buffer
registered under that name - which for an interpreter-evaluated node is a stale, usually all-zero
allocation. `first_divergence.py` now prefers the `[interp]` value and excludes truncated ones, but if you
are reading the dump by eye: **the `[interp]` line is the real one.** This cost a full false lead on
Silero VAD, where a perfectly correct Slice appeared to be 100% wrong.

⚠️ **python `onnx`, `onnxruntime` and `numpy` are installed.** A project note once recorded ORT
localisation as blocked on them and turned a four-step bisection into an open-ended hunt. Reach for a
reference runtime early - that is why we keep three of them.

## Reference-fixture generators (onnx + onnxruntime)

`gen_lstm_reference.py` · `gen_controlflow_reference.py` · `gen_scatter_reference.py` ·
`gen_slice_reference.py` · `gen_silero_vad_reference.py` · `gen_vad_segment_fixture.py` ·
`gen_operator_tests.py` · `gen_nlp_audio_references.py`

They write into `SpawnDev.ILGPU.ML.Demo/wwwroot/references/<group>/` where the PMT tests fetch them.

⚠️ **A fixture must be able to express the failure.** Every one of these is built so the OLD code cannot
pass it:
- X as a **graph input**, never an initializer - a constant folds onto the branch that already worked.
- An output **larger than `inputs[0]`** - control flow sized its buffer from the condition.
- A scatter that provably **changes values**; the generator refuses to emit a fixture whose output equals
  its input.
- **No duplicate indices** - ONNX leaves their order undefined, so such a fixture would legitimately flake.
- A reversed slice whose expected output contains **no zeros**, and which is **asymmetric** along the
  reversed axis - the old failure was an untouched all-zero buffer, and an engine that runs but does not
  actually reverse must fail too.
- For the VAD, a **negative control**: the same clip run with the LSTM state re-zeroed every frame. The
  generator refuses to emit unless the threaded and frozen runs differ widely (MEASURED 0.978), because
  that gap is the only thing that makes "the state really threads" a testable claim.
- For VAD endpointing, **three utterances with real gaps** rather than the plain librivox clip: that clip
  is near-continuous speech and yields ONE segment, which a detector that declared everything to be speech
  would match perfectly.

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

## Voice activity detection

| tool | answers |
|---|---|
| `vad-harness -- bench [frames]` | **How fast is a frame, really?** Warms up, then times ONLY `ProcessFrameAsync`, and prints mean/p50/p99 against the 32 ms a frame of audio represents. ⚠️ Do NOT derive this from a PMT `durationMs` - that includes fetching the model and compiling 125 nodes, and dividing it by the frame count gave a per-frame figure ~2.5x too slow. MEASURED here: OpenCL 4.99 ms (6.4x realtime). |
| `vad-harness -- segments` | Our boundaries beside the oracle's, per segment, in ms. PMT buffers the test's console output, so a passing endpointing run tells you the boundaries were inside tolerance and not what they were. MEASURED: 3/3 segments, worst 14 ms. |
| `vad-oracle -- <silero_vad.onnx> <audio.wav> [out.json]` | Where does the INDEPENDENT implementation put the utterance boundaries? sherpa-onnx (C++), with RoseEars's exact parameters. Our `VoiceActivityDetector` is a PORT of that endpointing, so the claim under test is "same behaviour" - and a reference transcribed by hand from the same upstream source would only prove I read it the same way twice. Deliberately does NOT reference SpawnDev.ILGPU.ML, so it can be built during a PMT sweep. |
| `gen_silero_vad_reference.py` | Per-frame speech probabilities from onnxruntime, plus the frozen-state negative control. |
| `gen_vad_segment_fixture.py` | Builds the three-utterance wav the endpointing gate runs on. |

## Other

⚠️ **`whisper-harness` and `stft-oracle` are NOT in this repository.** They sit in the workspace folder
one level up (`SpawnDev.ILGPU.ML/tools/`, alongside a 314 MB `distilgpt2_decoder_model.onnx`), which is
outside the git root - so run them as `--project ../tools/<name>`, and be aware a standing release gate is
currently unversioned.

`whisper-harness` (transcript regression gate - must stay word-identical) · `stft-oracle` ·
`extract_onnx.py` · `distilgpt2_*.py` (reference logits) · `bmpdiff.cs` · `lts-train` · `GptOssRun`

⚠️ **The browser rate is measured by `Vad_Benchmark_FrameRate` in PMT, not by `vad-harness`** -
that harness is desktop-only, and the browser backends are where the problem is. The test prints through
the `[Benchmark]` tag, which is the ONLY console text PMT echoes (`ProjectRunner.cs` filters on it), and
the line lands in `PlaywrightMultiTest/WGSLDumps/browser_console.log` rather than stdout.

MEASURED 2026-08-31, per frame, against the 32 ms a frame of audio lasts:

| backend | 5.2.3 | GPU recurrence | + skip-set |
|---|---|---|---|
| WebGPU | 177.9 ms | 170.1 ms | **126.7 ms** |
| WebGL | 85.6 ms | 71.8 ms | **65.6 ms** |
| Wasm | 191.8 ms | 157.0 ms | **168.0 ms** |
| OpenCL | 4.99 mean / 47.4 p99 | **4.25 mean / 7.78 p99** | - |

⚠️ **NAME the readbacks, never infer them.** The test also prints
`GraphExecutor.LastRunReadbackNames`, which records the node behind each one. I inferred the owners twice
and was wrong both times - first assuming the 125 dispatches were the cost (it was readbacks), then
assuming the host LSTM owned the counted ones (that counter lives in the executor's SHAPE-LANE promotion
path, which is a different mechanism from an operator's own `OperatorInputReader` call). The list settles
it in one run.

⚠️ **Still 0.25x realtime on WebGPU.** The remaining ten readbacks all trace to `Concat`, whose
inputs are ALL data - so it needs a TRANSITIVE value-need analysis (a tensor needs a host value only if
something downstream truly reads one; Concat propagates the need rather than creating it), not another
per-position entry in `ReadbackValueNeedingInputs`. Not started.

⚠️ `SessionGraphCapture` is NOT a drop-in here: tried 2026-08-31, it left WebGPU finding 0
utterances and crashed CUDA with an access violation. `TryCaptureAsync` runs the graph six times to
discover patch points and Silero's `h`/`c` are genuine per-frame state, so they do not survive the probing.

⚠️ **Kernel work is not verified until it has run on WebGPU.** Two defects in the LSTM kernel
produced CORRECT results on CUDA and OpenCL and failed only in a browser: a placeholder view that aliased a
real binding (WebGPU rejects storage-buffer aliasing), and a third output buffer whose write silently
produced ZEROS on WebGL while the other two outputs were right. The desktop backends verify the MATH; the
browser backends verify the kernel is LEGAL.

## Standing gates after ANY engine change

```
PMT_FILTER=Operator dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj   # 97/97
dotnet run --project ../tools/whisper-harness -c Release                         # PASS 57 words
```

Then the group you touched: `ControlFlow_`, `Recurrent_`, `Scatter_`, `Slice_`, `Vad_`, `Resample`,
`Microphone_`. `PMT_FILTER=MatchesOnnxRuntime` runs the ORT-referenced operator gates in one pass.
A full six-backend sweep is the release gate, not routine.
