# WebGPU capture memory: make a capture cost what a direct forward costs (2026-11-04)

## Why

Riker (AssistZ, RX 590, Chrome 157) measured 5.8.2-local.7, which turned CLAP audio-tower capture on by default:
bit-identical, no speedup (GPU-bound card), and Chrome dedicated GPU memory +700 MB (storage +337 MB, buffers +4,437,
pool allocations 181 -> 2,885; one-time, flat). Reproduced on the RTX 4070 by `Bench_ClapAudioTower_CaptureVsDirect`,
which now reports a capture's ONE-TIME cost:

| WebGPU capture of CLAP audio tower | storage | buffers | pool allocs | replay vs direct | bit-identical |
|---|---|---|---|---|---|
| default (no drains) | +323 MiB | +4,937 | +2,727 | 1.3-1.8x | yes |
| `SessionGraphCapture.KeepDrains = true` | +132 MiB | +2,876 | +666 | 1.2-1.4x | yes |
| direct forward, planned reuse (reference) | ~20 MiB of slots | - | - | - | - |

local.8 made capture opt-in for CLAP (`ClapAudioClassifier.EnableGraphCapture`, default off) and uses KeepDrains when
opted in. CUDA capture costs +68 pool allocations and, with FusedLinear split-K, gains nothing for CLAP (11.3 vs 11.5 ms).

## Where the memory goes (from the code)

1. `GraphExecutor.ImmediateReleaseActive` is false under `SuppressDrains`, and on WebGPU the capture-immediate return is
   off (CUDA only, `captureImmediate`), so the recorded pass releases nothing until a drain. Without KeepDrains there is
   no drain: every intermediate of the graph is pinned for the plan's lifetime.
2. `PlannedReuseEligible` is false whenever `UseCaptureParamSlots` is set (both WebGPU captures set it), and
   `BeginPlannedRun(owner, eligible: false)` drops the executor's plan for good. The warm passes then fill size buckets
   (the sum of every size's peak, not the live set): the +666 / +2,727 pool allocations.
3. Bind-group caching is off for the window, so every recorded dispatch retains its own scalar-params buffer (the
   buffer count, ~3,000 for CLAP; small).

## What blocks the obvious fix (reuse inside the recorded pass)

Measured earlier on ZipVoice's fm_decoder (GraphExecutor `captureImmediate` notes, WebGPUDispatchPlan.HostWriteCount):
with buffer reuse inside the recorded pass, the CAPTURE pass itself computes the wrong answer (16,900 of 16,900 values,
"observe pass: 0 differ | CAPTURE pass: 16,900 differ"), and with KeepDrains it replays wrong too. Ruled out so far:
unreplayable host writes (all 8,141 host writes in that window are the retained per-dispatch scalar uploads), the
output buffer being returned, read-before-write at node inputs/outputs. Remaining suspect: operator-internal rents
under colliding names (201 ALIEN-RETURNs per capture: `_mmi_a/_mmi_b`, DynamicQuantize temps), BufferPool's
name-ownership leak. CLAP replays bit-identically with KeepDrains, so the hazard is per graph.

## Plan

- [ ] Find the ZipVoice capture-pass divergence under reuse (the colliding-name internal rents first): a fix there is
      what makes reuse inside any WebGPU capture safe.
- [ ] Then let a WebGPU capture record under planned reuse: a separate plan owner for capture-regime runs (so the
      direct plan survives), immediate release during the WebGPU recorded pass, warm passes until the capture-regime
      plan is ready (two matching runs), so the recorded pass binds the plan's slots. Expected cost: ~the slot bytes.
      CUDA stays as is (BuildPlan inside a capture window could cuMemAlloc).
- [ ] Gate: `Bench_ClapAudioTower_CaptureVsDirect` capture cost within a few MiB of the direct forward's, bit-identical
      on two clips; ZipVoice `Pipeline_ZipVoice_CaptureReplayFidelity` 0 differ; then reconsider CLAP's default.
