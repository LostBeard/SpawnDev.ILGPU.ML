# Handoff — /face + /detect wrong boxes (UPDATED 2026-09-27 morning)

**Repo:** `SpawnDev.ILGPU.ML` @ working tree **5.2.32** (not pushed yet).  
**Prior:** `Plans/handoff-face-detect-live-bad-2026-09-27.md` (load OK, live results bad).

## Root causes (proved)

### `/detect` — preprocess/postprocess mismatch (engine/pipeline)
`ObjectDetectionPipeline` commented "letterbox + [0,1]" but called `Forward()` with defaults:
**stretch-to-square + ImageNet mean/std**. `YoloPostProcessor` unmapped as if letterboxed.
Live Street: one mis-placed "traffic light". After fix: 7 boxes on people/cars (CUDA DETECTPROBE).

### `/face` — NMS / max-faces (decode postprocess)
Forward already green (BLAZEDIFF classificator/regressor relRMS ~3e-4). Hard NMS kept
near-duplicates + hair false-positive. MediaPipe Face Detector defaults `num_faces=1` +
weighted NMS. After fix: Faces=1, box covers portrait center (FACEDETECT CUDA).

## Fix in 5.2.32
- `ObjectDetectionPipeline`: `preserveAspect:true`, mean/std 0/1, pass `Letterbox()` ints to postprocess
- `YoloPostProcessor`: accept content/pad rect
- `FaceDetectionPipeline`: weighted NMS, `maxFaces` default 1, clamp boxes
- Gates: Portrait geometry + `Pipeline_YOLOv8_Street_DetectsPeopleOrCars` (+ `samples/street_rgba.bin`)
- PMT: **12/12** CudaTests+WebGPUTests for BlazeFace+YOLOv8 Street/Reference/Detects/Portrait

## Still needed
1. Commit + push 5.2.32 (ask Captain)
2. Deploy workflow THEN `pages-build-deployment`
3. Hard-refresh live `/face` Portrait + `/detect` Street — only then mark VERIFIED
