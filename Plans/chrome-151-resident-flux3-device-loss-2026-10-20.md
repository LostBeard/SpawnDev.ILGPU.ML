# Chrome 151 kills the GPU process on the third FLUX 3 picture when the transformer stays resident (2026-10-20)

**Status:** an upstream Chrome bug. NO workaround in this repo (TJ, 2026-10-20: update the browser, do not work around other people's bugs unless absolutely required). The demo keeps the transformer resident as designed.

## Symptom

Browser demo, `/generate` with "Pictures: 4" (or 8) and `/text-to-3d` with "Draw 4", model FLUX 3 Image Klein, on installed Chrome 151.0.7922.174: pictures 1 and 2 draw, the third dies on its first denoise step with
`A valid external Instance reference no longer exists` (the WebGPU device is lost). Chrome's own log (`chrome://gpu`) says `GpuProcessHost: The GPU process died due to out of memory.`; no crash dump is written.

## Measured (RTX 4070 12 GB, driver 596.21, Windows 11, Release interpreted demo build, one page; `tools/audit-demo-models.mjs`, scenarios `/generate#flux3x4`, `/text-to-3d#flux3`)

| Browser | FLUX 3 weights | Result |
|---|---|---|
| Chrome 151.0.7922.174 | resident | FAIL, 5 of 5 runs, on picture 3, 346-547 s in |
| Chrome 151.0.7922.174 | resident, plus a queue drain after the decoder's memory is freed | FAIL (no change) |
| Chrome 151.0.7922.174 | not resident (reloaded per picture; an experiment, not shipped) | PASS: `/generate` x4 442 s; `/text-to-3d` knight, caption + 4 pictures + 3D + rig + animate, 778 s |
| Chromium 153.0.8010.12 (Playwright 1243, what PMT runs) | resident | PASS, `/generate` x4 399 s |
| Chrome 151 | FLUX.2 klein (2.4 GB) resident | PASS, `/generate` x4 278 s |

- Not a climb to a limit: sampled every 4-5 s, VRAM sat at 5.4-5.9 GB of 12 GB, the GPU process's private memory at about 4.8 GB and free system RAM at 2.8-3.4 GB, flat, until the process died at the first denoise step of picture 3.
- Not the bind-group cache (opt-in, off, never enabled for FLUX 3), not `Flux2Weights` state.
- The earlier SD Turbo device loss on 151 (memory `webgpu-flush-only-drain-and-parked-pools`) also passed on Chromium 153.
- Desktop backends (CUDA, OpenCL) are not affected.

## Resolution

TJ updated Chrome (installed is now 156.0.8078.48). The resident build, on that Chrome, passes both routes: `/generate#flux3x4` 393 s and `/text-to-3d#flux3` (knight: caption, 4 pictures, 3D, rig, animate) 737 s, 0 hub / 0 huggingface requests (records `audit-generate-flux3x4-browser-chrome156-resident` and `audit-text-to-3d-flux3-browser-chrome156-resident` in `_TestStats/SpawnDev.ILGPU.ML.jsonl`). Chrome 152-155 untested; the bug is fixed somewhere between 151 and 153.

## Lesson

PMT drives Chromium, so a bug in the installed Chrome that users have is invisible to it. Run browser GPU features on BOTH the installed Chrome and the Playwright Chromium.
