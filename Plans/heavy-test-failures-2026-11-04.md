# HeavyModel failures found by the 2026-11-03 desktop sweep - pre-existing, not planned reuse / ImmediateRelease

Found by the (overlong) desktop sweep with `PMT_EXCLUDE_CATEGORIES=__none__` and ML_PLANNED_REUSE=1 (_TestStats `desktop-heavy-sweep-planned-reuse`).
HeavyModel tests are outside the default gates, so nothing caught these. Each was rerun alone, A/B:

| Test | Lane | planned reuse off | ImmediateRelease off | Symptom |
|---|---|---|---|---|
| Pipeline_ZipVoice_CaptureReplayFidelity | CUDA | fails | fails | TryCapture null: accelerator child objects +10 CudaMem per warm pass (19412 -> 19422 over 12 passes) - a per-forward registration leak, so CUDA graph capture never goes live |
| Pipeline_ZipVoice_CaptureEligibilityProbe | CUDA | fails | (not run) | subprocess crash after ~10 s, exit -1073741819 (access violation) |
| DA3Small_Pipeline_5D_ElideDispatch | OpenCL | fails | fails | "[ELIDE] no shape ops were elided (elided=0)" |
| Trellis2_Flow20kTokens_OpTimes | CPU (and Velocity) | fails | (not run) | crash after ~3 s, exit -2146232797 (0x80131623, FailFast) on CPU; access violation on Velocity |

Next: in this order - the ZipVoice capture leak (a real per-forward device-object leak on CUDA, and the reason capture is off for ZipVoice), the
eligibility-probe access violation (likely the same capture path), DA3 elide (test or elision regression), Trellis2 flow crash on the CPU lanes.

Also from that sweep, CPU / Velocity lanes: 18 heavy LLM / SD tests hit the 600 s console cap. Those tests never run on CPU lanes in a default
sweep, and Data's release gates overlapped that stretch (CPU contention), so they say nothing yet; SD-Turbo on CUDA (600 s cap) and
Embeddings_RealTokenizer on CUDA (300 s) are being A/B'd for planned reuse.
