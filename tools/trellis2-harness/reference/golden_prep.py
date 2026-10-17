"""Golden for the image preparation: the REAL steps of Trellis2ImageTo3DPipeline.preprocess_image (alpha path) + DinoV3FeatureExtractor.__call__ resize/normalise,
with Pillow, on a synthetic RGBA image whose alpha is a soft ellipse (so the alpha > 0.8 bounding box, the half-to-even crop rounding and the
premultiplied-alpha truncation are all exercised).

  python golden_prep.py        writes TRELLIS2_DIR/golden/prep/*
"""
import os, sys, json
W = os.environ["TRELLIS2_DIR"]
OUT = os.path.join(W, "golden", "prep"); os.makedirs(OUT, exist_ok=True)
import numpy as np
from PIL import Image

cases = {"ellipse": (701, 523), "wide": (900, 400)}
man = {}
for name, (w, h) in cases.items():
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:h, 0:w]
    r = (xx / w * 255).astype(np.uint8); g = (yy / h * 255).astype(np.uint8)
    b = (((xx // 37 + yy // 41) % 2) * 160 + 40).astype(np.uint8)
    cx, cy, rx, ry = w * 0.52, h * 0.47, w * 0.28, h * 0.33
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    alpha = np.clip((1.15 - d) * 4.0, 0, 1)
    a = (alpha * 255).astype(np.uint8)
    rgba = np.stack([r, g, b, a], axis=-1)
    inp = Image.fromarray(rgba, "RGBA")
    # --- preprocess_image (has_alpha path) ---
    max_size = max(inp.size); scale = min(1, 1024 / max_size)
    if scale < 1: inp = inp.resize((int(inp.width * scale), int(inp.height * scale)), Image.Resampling.LANCZOS)
    output = inp
    output_np = np.array(output); alpha_c = output_np[:, :, 3]
    bbox = np.argwhere(alpha_c > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1]); size = int(size * 1)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    pre = Image.fromarray((output * 255).astype(np.uint8))
    # --- get_cond: resize + normalise ---
    im = pre.resize((512, 512), Image.LANCZOS)
    arr = np.array(im.convert("RGB")).astype(np.float32) / 255
    t = np.transpose(arr, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]; std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    t = ((t - mean) / std).astype(np.float32)
    packed = (rgba[..., 0].astype(np.uint32) | (rgba[..., 1].astype(np.uint32) << 8) | (rgba[..., 2].astype(np.uint32) << 16) | (rgba[..., 3].astype(np.uint32) << 24)).astype(np.uint32)
    packed.tofile(os.path.join(OUT, name + ".rgba.u32"))
    t.tofile(os.path.join(OUT, name + ".tensor.f32"))
    man[name] = {"width": w, "height": h, "crop": [float(v) for v in bbox], "crop_size": list(pre.size)}
    print(name, man[name])
json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"))
