# Handoff — /face + /detect load OK, live results BAD (2026-09-27 night)

**Owner next session:** Trip (or whoever picks up SpawnDev.ILGPU.ML).  
**Repo:** `SpawnDev.ILGPU.ML` @ `217ff30` (5.2.31 on `master`).  
**Live:** Deploy + pages-build-deployment both succeeded for 5.2.31 (`36296763751` / `36296909713`).

## Captain observation (authoritative)

TJ 2026-09-27 ~01:23 local: **`/face` and `/detect` both load and run, but both give bad results.**  
Stop treating live demos as VERIFIED until that is fixed and re-checked on GH Pages after part-2 deploy.

## What already shipped this session (do not redo)

| Ver | What |
|-----|------|
| 5.2.25–28 | Truncated download refusal, COI `/models/` bypass, arrayBuffer wrap, Face known-size 229746 |
| 5.2.29 | BlazeFace NHWC: DepthwiseConv fused-act **field 4** (not 3); **MaxPool2DNHWC**; classificator relRMS **4.6e-4** on CUDA BLAZEDIFF; PMT Reference+Portrait **6/6** Cuda+WebGPU |
| 5.2.30 | `/detect` known-size YOLOv8 **12823637** + `?v=` (truncated ONNX → protobuf "Sub-message extends past end of data") |
| 5.2.31 | COI `registerFresh`: capture `currentScript.src` **before** async unregister (`currentScript` was null) |

**Not the bug for tomorrow:** model load / FlatBuffer / truncated body. Those are fixed. Symptom now is **wrong detections**, not crash-on-load.

## Status table (updated)

- `/face` → **PARTIAL** — PMT green; **live results bad** (TJ).
- `/detect` → **PARTIAL** — loads; **live results bad** (TJ). PMT YOLOv8 gates were previously green; **re-run and compare to live** before trusting them.

## Tomorrow — start here

1. **Reproduce on live** (after hard refresh / SW unregister if needed): note exact badness (0 boxes, wrong boxes, garbage labels, Faces:N nonsense, etc.). Screenshot or console `[Face]`/`[Detect]` lines.
2. **Same inputs through PMT** on WebGPU: `Pipeline_BlazeFace_Portrait_DetectsFace`, `Pipeline_YOLOv8_DetectsObjects` / `Pipeline_YOLOv8_Reference_MatchesOnnxRuntime` with `PMT_LANES=WebGPUTests,CudaTests` `PMT_PARALLEL=off`. If PMT green and live bad → page preprocess/decode path, not engine. If PMT red → engine.
3. **Do not invent causes.** Live bad + unit green is a known trap (page path ≠ test path). Grep FaceDetectPage / DetectPage vs the pipeline tests for preprocess (letterbox, mean/std, NHWC transpose) and postprocess (NMS, score thresh, box order).
4. **Deploy discipline:** GH Pages = Deploy workflow THEN `pages-build-deployment`. Do not redeploy to chase CDN TTL.

## Useful paths

- `SpawnDev.ILGPU.ML/Pipelines/FaceDetectionPipeline.cs`, `ObjectDetectionPipeline.cs`
- `SpawnDev.ILGPU.ML.Demo/Pages/FaceDetectPage.razor`, `DetectPage.razor`
- `MLTestBase.DetectionPipelineTests.cs`
- DemoConsole: `BLAZEDIFF CUDA` (BlazeFace parity probe)
- Memory: `feedback-tflite-nhwc-maxpool-and-depthwise-fused-field.md`, `feedback-download-chunked-must-not-return-prefix.md`, `feedback-gh-pages-two-part-deploy.md`

## Board

Claim `SpawnDev.ILGPU.ML` when starting. Release when done.
