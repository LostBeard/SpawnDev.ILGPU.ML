# References for DetrDetectionPipeline (2026-11-08):
#
#   python -P tools/gen_detr_pipeline_reference.py <dfine_s_coco model.onnx>
#
# - references/pil_resize/: Pillow's own Image.resize(BILINEAR) of samples/street_rgba.bin (whole image -> 160x160, the antialiased
#   shrink) and of a crop (crop(box).resize -> 96x64, one axis shrinking, one growing), as raw uint8 HWC RGB. ImagePreprocessKernel.ForwardPil
#   must reproduce them exactly (0 differing bytes).
# - references/detr_pipeline/dfine_s_street.json: D-FINE-S (onnxruntime, CPU) on the street picture through the model's own preprocessing
#   (PIL stretch to 640, /255) and transformers' post_process_object_detection (sigmoid, top-Q pairs, threshold 0.5), boxes in pixels.
import json, os, sys
import numpy as np
from PIL import Image
import onnxruntime as ort

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "SpawnDev.ILGPU.ML.Demo", "wwwroot"))
raw = open(os.path.join(ROOT, "samples", "street_rgba.bin"), "rb").read()
W, H = np.frombuffer(raw[:8], np.int32)
rgba = np.frombuffer(raw[8:8 + W * H * 4], np.uint8).reshape(H, W, 4)
im = Image.fromarray(rgba[:, :, :3].copy(), "RGB")
print("street", W, H)

pr = os.path.join(ROOT, "references", "pil_resize")
os.makedirs(pr, exist_ok=True)
CROP = (37, 21, 37 + 150, 21 + 110)
cases = {"whole_160x160": im.resize((160, 160), Image.BILINEAR),
         "crop_96x64": im.crop(CROP).resize((96, 64), Image.BILINEAR)}
for name, r in cases.items():
    a = np.asarray(r, np.uint8)
    a.tofile(os.path.join(pr, name + ".bin"))
    print(name, a.shape)
json.dump({"crop": list(CROP)}, open(os.path.join(pr, "cases.json"), "w"))

model = sys.argv[1]
s = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
x = (np.asarray(im.resize((640, 640), Image.BILINEAR), np.float32) / 255.0).transpose(2, 0, 1)[None]
res = dict(zip([o.name for o in s.get_outputs()], s.run(None, {s.get_inputs()[0].name: x})))
logits, boxes = res["logits"][0], res["pred_boxes"][0]
q, c = logits.shape
scores = 1 / (1 + np.exp(-logits.astype(np.float64)))
flat = scores.ravel()
top = np.argsort(-flat, kind="stable")[:q]                 # post_process_object_detection: topk(num_top_queries = Q) over (query, class)
dets = []
for i in top:
    if flat[i] <= 0.5:
        continue
    qi, ci = divmod(int(i), c)
    cx, cy, bw, bh = boxes[qi].astype(np.float64) * [W, H, W, H]
    dets.append(dict(score=float(flat[i]), query=qi, cls=ci, x=cx - bw / 2, y=cy - bh / 2, w=bw, h=bh))
dp = os.path.join(ROOT, "references", "detr_pipeline")
os.makedirs(dp, exist_ok=True)
json.dump({"width": int(W), "height": int(H), "threshold": 0.5, "detections": dets}, open(os.path.join(dp, "dfine_s_street.json"), "w"), indent=1)
for d in dets:
    print(f"  q{d['query']:3d} c{d['cls']:2d} {d['score']:.3f}  ({d['x']:.1f}, {d['y']:.1f}, {d['w']:.1f} x {d['h']:.1f})")
