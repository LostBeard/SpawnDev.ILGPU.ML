# CPU-backend style-transfer "failure" investigation — 2026-06-01 (Tuvok)

Picking up the teed-up next target from `tuvok-session-handoff-2026-06-01`: root-cause the
CPU-backend style-transfer correctness bug (`Reference_Style{Pointilism,RainPrincess,Udnie}` +
`CreateFromFile_StyleTransfer_Mosaic`, reported as CPU-lane-only while all GPU backends match ORT).

## Methodology

Built a self-contained CPU-vs-CUDA per-node bisection diagnostic (NOT a PMT substitute — an
investigation tool, like the `_mldump/` ORT scripts used for the MoveNet fix):

- `SpawnDev.ILGPU.ML.DemoConsole/StyleBisect.cs` + a `STYLEBISECT` arg branch in `Program.cs`.
- Loads a style model + the ORT cat reference I/O directly from `Demo/wwwroot` (no HTTP server).
- Runs the SAME `InferenceSession.CreateFromFile` + `RunAsync` path the tests use, on CUDA
  (ground truth = matches ORT) and the ILGPU CPU backend, with `GraphExecutor.CapturedOutputs`
  enabled (`CaptureMaxElements=8192`). Diffs every node's fingerprint to find the first divergence.
- `STYLEBISECT CPUCHECK` sub-mode runs the EXACT production reference assertion
  (`CompareOnGpuAsync` vs ORT at the test's `tol meanErr<=5.0`) on CPU for all 5 style models.

Invoke: `dotnet run --project SpawnDev.ILGPU.ML.DemoConsole -c Release -- STYLEBISECT [model|CPUCHECK]`

## Findings (verified, isolated)

1. **CPU == CUDA == ORT, per-node, for the whole network.** On `style-pointilism` the per-node
   CPU-vs-CUDA max abs diff stays at float-rounding (~1e-5) across all 102 nodes. Final node vs ORT:
   meanErr=0.0002, maxErr=0.0022 — for BOTH CUDA and CPU. **No divergent node exists.**

2. **All 5 style models PASS the production CPU reference assertion in isolation:**
   ```
   style-pointilism      PASS  meanErr=0.0002 maxErr=0.0057
   style-rain-princess   PASS  meanErr=0.0001 maxErr=0.0013
   style-udnie           PASS  meanErr=0.0002 maxErr=0.0010
   style-mosaic          PASS  meanErr=0.0003 maxErr=0.0072
   style-candy           PASS  meanErr=0.0002 maxErr=0.0043
   ```
   (tolerance is 5.0; full 150,528-element output compared on-GPU via the same `CompareOnGpuAsync`
   the tests use.) **The CPU backend computes style transfer correctly.**

3. The handoff's premise (a CPU compute bug) is therefore wrong. The conv asymmetric-pad bug
   (42b8a4a) was deterministic + backend-agnostic, so it cannot have been the cause of a CPU-ONLY
   failure either. Whatever caused the earlier CPU-lane failures does NOT reproduce in isolation.

## Why not corruption-under-contention

- Every style-path kernel (`Conv2DKernel`, `ConvTranspose2DKernel`, `NormalizationKernels`
  InstanceNorm two-pass) uses `LoadAutoGroupedStreamKernel` — no explicit `Group.Barrier`, no
  shared memory. That is the safe class; the two InstanceNorm passes are separate ordered kernel
  launches (no inter-pass race). On x86 CPU (strong TSO memory model, no spin-barriers) there is no
  plausible wrong-output path under thread oversubscription — at worst it runs slow.

## Leading hypothesis: contention SLOWNESS / outer-timeout, not correctness

- Solo CPU runtime: **~57s per style model** (288.5s wall for all 5, single process, no contention).
- PMT CPU lane runs `PMT_CPU_PARALLELISM=4` (4 concurrent CPU-accelerator subprocesses, each of
  which already saturates all cores) PLUS the browser WebGPU lane competing for cores in Phase A.
  4x+ oversubscription inflates a 57s model dramatically.
- Desktop per-method timeout: `ConsoleRunner` does NOT enforce `[TestMethod(Timeout=120000)]`
  (no timeout/retry code in it); only PMT's outer 600s subprocess kill applies on desktop. So the
  earlier CPU "failures" were most likely the 600s outer kill (or a crash / cascade), under
  oversubscription — NOT a compute error.

## CONFIRMED: timeout under CPU-lane oversubscription (NOT a compute bug)

`PMT_FILTER=Style` full-lane run (TRX `style_cpu_check.trx`, 120 tests, 8 failed): **ALL failures are
CPU-lane style tests, and EVERY one is a timeout** — no compute error among them:
```
CPUTests.Reference_StyleUdnie_MatchesOnnxRuntime        Test exceeded timeout of 120000ms
CPUTests.Reference_StyleMosaic_MatchesOnnxRuntime       Test exceeded timeout of 120000ms
CPUTests.Reference_StylePointilism_MatchesOnnxRuntime   Test exceeded timeout of 120000ms
CPUTests.Reference_StyleRainPrincess_MatchesOnnxRuntime Test exceeded timeout of 120000ms
CPUTests.Reference_StyleCandy_MatchesOnnxRuntime        Test exceeded timeout of 120000ms
CPUTests.CreateFromFile_StyleTransfer_Mosaic            Test exceeded timeout of 60000ms
CPUTests.WebModel_StyleTransfer_Mosaic                  Test exceeded timeout of 60000ms
IntegrationTests.DirectOnnx_StyleTransfer_Mosaic        (style, desktop)
```
ALL GPU lanes (WebGPU/WebGL/Wasm/CUDA/OpenCL) PASS every style test. Note: in the harness ALL 5 CPU
style models time out — including Mosaic and Candy, which PASS in isolation at meanErr 0.0002. So the
discriminator is unambiguous: **isolated = pass (correct compute); 4x-oversubscribed CPU lane = timeout.**

The `[TestMethod(Timeout=...)]` IS enforced by the PMT scheduler (reports "Test exceeded timeout of
Nms" with the scheduler's report-time duration ~0.1s, not the real ~57s+ run time).

Root cause: ILGPU `CPUAccelerator` already saturates all cores per process; `PMT_CPU_PARALLELISM=4`
runs 4 such processes concurrently (+ the browser WebGPU lane in Phase A) → 4x+ core
oversubscription → a 57s-solo style model blows the 120s (and 60s) budget. Pure scheduling artifact.

## Confirmation: PASSES at cap=1 (oversubscription is the SOLE cause)
`PMT_FILTER=Reference_Style PMT_CPU_PARALLELISM=1` (TRX `style_cpu_cap1.trx`): **32/32 passed, 0
failed. All 5 CPU Reference_Style tests PASS.** The browser/CUDA/OpenCL lanes still ran concurrently
in Phase A, so cross-lane contention is NOT the cause — CPU-lane self-oversubscription (cap=4 running
4 all-core CPU-accelerator processes) is the sole cause. Validated fix direction: don't oversubscribe
the CPU lane on all-core compute-bound work.

Proven 3 independent ways that CPU compute is correct: (1) per-node bisection CPU==CUDA==ORT,
(2) isolated production assertion meanErr 0.0002, (3) cap=1 PMT pass. Timeout enforced at
`SpawnDev.UnitTesting/UnitTestRunner.cs:405` (wall-clock `Task.WhenAny(task, Task.Delay(timeoutMs))`).

## Decision (TJ, 2026-06-01): serialize heavy tests on the CPU lane (no-compromise scheduler fix)

TJ chose the no-compromise option: the PMT scheduler runs compute-heavy CPU tests at cap=1
(serialized) on the CPU lane while light tests stay at cap=4. Implemented:

### Scheduler (`PlaywrightMultiTest/ProjectRunner.cs`)
- New `HeavyCpu` category concept, DECOUPLED from exclusion. `HeavyModel` = skipped everywhere;
  `HeavyCpu` = runs normally on every lane (browser/WebGPU/WebGL/Wasm/CUDA/OpenCL — where these
  tests are fast) but is SERIALIZED on the CPU lane (where the ILGPU CPU accelerator saturates all
  cores). This is why HeavyModel-tagging was the WRONG tool: it would have stripped the style tests'
  WebGPU coverage (WebGPU is the actual shipping demo backend).
- CPU lane now splits: light tests at the lane cap (default 4), then `HeavyCpu` tests at cap=1
  (default `PMT_CPU_HEAVY_PARALLELISM`=1) run AFTER the light burst drains, so a heavy all-core CPU
  test never contends with sibling CPU tests. Only the cpu lane needs this (cuda/opencl already
  cap=1; browser is single-page-sequential). New helpers: `RunCpuLaneAsync`, `IsCpuHeavy`,
  `CpuHeavyCategories()` (override via `PMT_CPU_HEAVY_CATEGORIES`).

### Tagged `Category="HeavyCpu"` (the 8 proven CPU-timeout tests)
`Reference_Style{Mosaic,Candy,RainPrincess,Udnie,Pointilism}_MatchesOnnxRuntime`,
`CreateFromFile_StyleTransfer_Mosaic`, `WebModel_StyleTransfer_Mosaic`,
`IntegrationTests.DirectOnnx_StyleTransfer_Mosaic`. (NOT BlazeFace — its handoff "fail" had no
timeout signature; treat as a separate possible compute issue until proven a timeout.)

### Verification
Scheduler log confirms `Phase A desktop lane 'cpu': 14 light (cap=4) + 8 HeavyCpu (cap=1, serialized
after light)`. Verify run (`PMT_FILTER=Style` at DEFAULT `PMT_CPU_PARALLELISM=4`, the original
failing condition) in flight → TRX `style_cpu_serialized_verify.trx`. Expect: all CPU style tests
PASS (serialization keeps each within budget), zero GPU-lane regressions.

### Extension note
Any future CPU test that times out under cap=4 (other real-model reference/pipeline tests at full
sweep) just gets `Category="HeavyCpu"` — no scheduler change needed. The STYLEBISECT diagnostic
(`DemoConsole/StyleBisect.cs`, `STYLEBISECT [model|CPUCHECK|INSPECTORCHECK|STREAMCHECK]`) remains for
CPU-vs-CUDA correctness bisection + Model Inspector checks on demand.

---

# Model Inspector work (same session, WebGPU/demo focus per TJ)

## Bug fixed: CheckCompatibility threw for non-ONNX formats
The Inspector demo calls `ModelInspectorHelper.CheckCompatibility(bytes)` for EVERY dropped file, but
it ran `OnnxParser.Parse` unconditionally → `InvalidOperationException` on TFLite/GGUF/SafeTensors
(diagnostic INSPECTORCHECK confirmed all 4 non-ONNX threw). Fix: format-detect first; non-ONNX
returns a non-throwing `Applicable=false` result (`Format`/`Applicable` added to `CompatibilityResult`,
format-aware `Summary`). Also cleaned a malformed doc-comment. 8 `ModelInspector_*` [TestMethod]s added
covering Inspect()+CheckCompatibility() against real models (ONNX/TFLite/GGUF/SafeTensors), incl. a
regression guard for the non-ONNX throw.

## Stream-based inspection (no full-model-in-memory) — TJ directive
`ModelInspectorHelper.InspectAsync(Stream)`: universal entry, detects format from a 256-byte prefix
(prefix-tolerant SafeTensors probe — a real bug STREAMCHECK caught: DetectModelFormat's
`headerSize < len-8` check fails on a short prefix and would send a multi-GB SafeTensors to the
full-read fallback). Header-only paths:
- **SafeTensors**: async reads (browser-safe) — `[8-byte len][JSON header]` only. PROVEN header-only:
  a synthetic checkpoint claiming ~1 GB of weights inspects from a 362-byte header, `readPastHeader=0`.
- **GGUF**: `GGUFParser.ParseHeader(Stream)` (new) reads metadata + tensor infos, stops at the data
  boundary. Wired for SEEKABLE streams (FileStream/MemoryStream/WebTorrent-seekable, which support
  sync reads); non-seekable async-only (browser OpenReadStream) takes the fallback.
- **ONNX/TFLite + non-seekable**: full-read fallback (works today; same as Inspect(byte[])).
Sources: browser OpenReadStream, HttpClient (GetStreamAsync), desktop FileStream, WebTorrent.
4 stream [TestMethod]s (SafeTensors match + header-only proof, GGUF seekable match, ONNX non-seekable
fallback). All green via STREAMCHECK.

## Streaming FOLLOW-UPS (next increment)
- GGUF **async** ParseHeaderAsync → strict header-only for browser (non-seekable) GGUF too.
- ONNX streaming: skip each initializer's `raw_data` via its protobuf length prefix (seek when seekable).
- TFLite: read the FlatBuffers metadata region only.
- Demo wiring: `InspectorPage` pass the file/HTTP stream into `InspectAsync` instead of `ToArray()`;
  add `CheckCompatibilityAsync(Stream)` (ONNX needs full bytes; non-ONNX is Applicable=false w/o parse).
- HttpClient `GetStreamAsync` source path + a URL-inspect entry in the demo.

## Correct-fix candidates (pending the signature; Rule 1 = no compromise, no gating a fixable test)

- If contention/timeout: the real bug is the SCHEDULER oversubscribing a backend (CPU) where one
  instance already uses all cores. Options: CPU-lane cap awareness for all-core backends, or run the
  heavy 224x224 style models isolated on the CPU lane. Tagging them `HeavyModel` would only HIDE a
  correct test (a Rule-1 compromise) — avoid unless TJ chooses the deferred-CPU path
  (`project_ml_fast_backends_first_focus_2026_05_31` defers CPU).
