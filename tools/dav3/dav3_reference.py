"""
DAv3 reference fixtures: native onnxruntime (CPU) on the exact model bytes our engine and
Transformers.js load, for every input shape SpawnScene and the official DA3 pipeline produce.

For each case this writes, under --out (default: the demo's test-refs/dav3, served to browser tests):
    <case>.input.f32        pixel_values, float32, shape in manifest
    <case>.<output>.f32     predicted_depth / confidence / extrinsics / intrinsics
and manifest.json describing every case (images, preprocessing, shapes, ORT wall time).

Two preprocessings, because they are the two questions:
  official  - a port of ByteDance-Seed/Depth-Anything-3 utils/io/input_processor.py
              (upper_bound_resize to process_res, cv2 INTER_AREA/INTER_CUBIC, each side rounded to
              a multiple of 14, ImageNet mean/std, NO padding). What the model was trained for.
  ours      - a bit-for-bit emulation of SpawnDev.ILGPU.ML ImagePreprocessKernel with
              preserveAspect=true (2x2 bilinear, half-pixel, letterbox into a square with the border
              replicated). What SpawnScene feeds today.
Engine parity (ours vs ORT on the SAME tensor) is independent of which one a case uses.

Usage (from the repo root, SpawnDev.ILGPU.ML/):
    python tools/dav3/dav3_reference.py                 # all cases
    python tools/dav3/dav3_reference.py --only s518_truck,mv2_drj_off
Needs: onnxruntime numpy opencv-python-headless pillow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_OUT = os.path.join(REPO, "SpawnDev.ILGPU.ML.Demo", "wwwroot", "test-refs", "dav3")
HUB_MODEL = r"W:\srv\spawndev_hub\hf-cache\onnx-community_depth-anything-v3-small\onnx"
# onnxruntime refuses external data reached through the hub's sshfs mount ("path escapes model
# directory" - it resolves W: to its UNC sshfs path), so the reference runs from a local copy.
DEFAULT_MODEL = os.path.abspath(os.path.join(REPO, "_mldump", "models", "depth-anything-v3-small"))
# The bytes our tests, the hub and HF main all serve (verified 2026-09-23). A different hash means
# the comparison is no longer controlled - see fb-two-onnx-exports-same-model.
MODEL_SHA256 = {"model.onnx": "396008798244a074297fd88e450433b1357fc687f534939375c804ded86e7b2a",
                "model.onnx_data": "802bb24741e67f5bb2b369fc64d40afe11439cc895d676d658d65cfb75c9860f"}
DATASETS = os.path.abspath(os.path.join(REPO, "..", "..", "SpawnScene", "SpawnScene", "SpawnScene", "wwwroot", "datasets"))
TANDT = r"C:\Users\TJ\Downloads\tandt_db"

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
PATCH = 14


def temple(i: int) -> str:
    return os.path.join(DATASETS, "TempleRing", f"templeR{i:04d}.png")


def truck(i: int) -> str:
    return os.path.join(TANDT, "tandt", "truck", "images", f"{i:06d}.jpg")


def drj(name: str) -> str:
    return os.path.join(TANDT, "db", "drjohnson", "images", name)


def bathroom(k: int) -> str:
    d = os.path.join(DATASETS, "Bathroom")
    return os.path.join(d, sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg"))[k])


# name -> (images, prep, size). size = process_res for official, square side for ours.
# batch>1 is expressed as a list of image lists.
CASES = {
    # --- single view -----------------------------------------------------------------------
    "s518_truck":     ([truck(1)], "ours", 518),                   # SpawnScene single-view default
    "off_truck":      ([truck(1)], "official", 504),               # the official non-square shape
    "s672_bath":      ([bathroom(0)], "ours", 672),                # 12 MP phone photo, joint-path grid
    "off_bath":       ([bathroom(0)], "official", 504),
    "s896_bath":      ([bathroom(0)], "ours", 896),                # SpawnScene's high-detail single view
    # --- multi view (batch 1, N images) ----------------------------------------------------
    "mv2_drj_off":    ([drj("IMG_6292.jpg"), drj("IMG_6299.jpg")], "official", 504),
    "mv4_temple_off": ([temple(i) for i in (1, 4, 7, 10)], "official", 504),
    "mv4_temple_672": ([temple(i) for i in (1, 4, 7, 10)], "ours", 672),   # joint production grid
    "mv6_temple_518": ([temple(i) for i in (1, 4, 7, 10, 13, 16)], "ours", 518),
    # --- batch 2 x 2 views: the batch_size axis the export declares --------------------------
    "b2n2_temple_off": ([[temple(1), temple(4)], [temple(7), temple(10)]], "official", 280),
}


# ---------------------------------------------------------------------------------------------
# Official DA3 preprocessing (input_processor.py, upper_bound_resize)
# ---------------------------------------------------------------------------------------------
def official_one(path: str, process_res: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest != process_res:
        s = process_res / float(longest)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        interp = cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA
        img = Image.fromarray(cv2.resize(np.asarray(img), (nw, nh), interpolation=interp))
    w, h = img.size

    def nearest(x: int) -> int:
        down = (x // PATCH) * PATCH
        up = down + PATCH
        return up if abs(up - x) <= abs(x - down) else down

    nw, nh = max(1, nearest(w)), max(1, nearest(h))
    if (nw, nh) != (w, h):
        interp = cv2.INTER_CUBIC if (nw > w or nh > h) else cv2.INTER_AREA
        img = Image.fromarray(cv2.resize(np.asarray(img), (nw, nh), interpolation=interp))
    x = np.asarray(img, np.float32) / 255.0                      # T.ToTensor
    x = (x - MEAN) / STD                                          # T.Normalize
    return np.ascontiguousarray(x.transpose(2, 0, 1))             # CHW


def official_views(paths: list[str], process_res: int) -> np.ndarray:
    views = [official_one(p, process_res) for p in paths]
    hs = {v.shape[1] for v in views}; ws = {v.shape[2] for v in views}
    if len(hs) > 1 or len(ws) > 1:                                # _unify_batch_shapes: center crop
        mh, mw = min(hs), min(ws)
        out = []
        for v in views:
            t = max(0, (v.shape[1] - mh) // 2); l = max(0, (v.shape[2] - mw) // 2)
            out.append(v[:, t:t + mh, l:l + mw])
        views = out
    return np.stack(views)                                        # N,3,H,W


# ---------------------------------------------------------------------------------------------
# Our ImagePreprocessKernel (preserveAspect=true), emulated in float32 exactly as the kernel reads
# ---------------------------------------------------------------------------------------------
def letterbox(src_w: int, src_h: int, dst_w: int, dst_h: int):
    s = np.float32(min(np.float32(dst_w) / np.float32(src_w), np.float32(dst_h) / np.float32(src_h)))
    cw = max(1, min(dst_w, int(np.round(np.float32(src_w) * s))))
    ch = max(1, min(dst_h, int(np.round(np.float32(src_h) * s))))
    return cw, ch, (dst_w - cw) // 2, (dst_h - ch) // 2


def ours_one(path: str, side: int) -> tuple[np.ndarray, list[int]]:
    rgb = np.asarray(Image.open(path).convert("RGB"), np.float32) / np.float32(255.0)
    src_h, src_w = rgb.shape[:2]
    cw, ch, px, py = letterbox(src_w, src_h, side, side)
    lx = np.clip(np.arange(side) - px, 0, cw - 1).astype(np.float32)
    ly = np.clip(np.arange(side) - py, 0, ch - 1).astype(np.float32)
    fx = ((lx + np.float32(0.5)) * np.float32(src_w) / np.float32(cw)) - np.float32(0.5)
    fy = ((ly + np.float32(0.5)) * np.float32(src_h) / np.float32(ch)) - np.float32(0.5)
    x0f, y0f = np.floor(fx), np.floor(fy)
    tx, ty = (fx - x0f)[None, :, None], (fy - y0f)[:, None, None]
    x0 = x0f.astype(np.int64); x1 = x0 + 1; y0 = y0f.astype(np.int64); y1 = y0 + 1
    x0 = np.maximum(x0, 0); x1 = np.minimum(x1, src_w - 1)
    y0 = np.maximum(y0, 0); y1 = np.minimum(y1, src_h - 1)
    v00 = rgb[y0][:, x0]; v01 = rgb[y0][:, x1]; v10 = rgb[y1][:, x0]; v11 = rgb[y1][:, x1]
    one = np.float32(1.0)
    pix = v00 * (one - ty) * (one - tx) + v01 * (one - ty) * tx + v10 * ty * (one - tx) + v11 * ty * tx
    inv_std = (one / STD).astype(np.float32)
    x = (pix - MEAN) * inv_std
    return np.ascontiguousarray(x.transpose(2, 0, 1).astype(np.float32)), [cw, ch, px, py]


def build_input(images, prep: str, size: int):
    batches = images if isinstance(images[0], list) else [images]
    rects = []
    out = []
    for paths in batches:
        if prep == "official":
            out.append(official_views(paths, size))
        else:
            vs = []
            for p in paths:
                v, r = ours_one(p, size)
                vs.append(v); rects.append(r)
            out.append(np.stack(vs))
    return np.ascontiguousarray(np.stack(out).astype(np.float32)), rects   # B,N,3,H,W


def short(path: str) -> str:
    """dataset/file.ext - enough to find the image again, without machine-specific roots."""
    return "/".join(os.path.normpath(path).split(os.sep)[-3:])


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--only", default="")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    for f, want in MODEL_SHA256.items():
        dst = os.path.join(a.model_dir, f)
        if not os.path.exists(dst):
            import shutil
            os.makedirs(a.model_dir, exist_ok=True)
            print(f"copying {f} from the hub mount...", flush=True)
            shutil.copyfile(os.path.join(HUB_MODEL, f), dst)
        got = sha256(dst)
        if got != want:
            raise SystemExit(f"{dst}: sha256 {got} != expected {want}")
    model = os.path.join(a.model_dir, "model.onnx")
    so = ort.SessionOptions()
    if a.threads: so.intra_op_num_threads = a.threads
    t = time.perf_counter()
    sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - t) * 1000
    out_names = [o.name for o in sess.get_outputs()]

    man_path = os.path.join(a.out, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {"cases": {}}
    manifest.update({
        "model": "onnx-community/depth-anything-v3-small",
        "model_sha256": MODEL_SHA256,
        "onnxruntime": ort.__version__,
        "ort_session_load_ms": round(load_ms),
    })

    only = [s for s in a.only.split(",") if s]
    for name, (images, prep, size) in CASES.items():
        if only and name not in only:
            continue
        x, rects = build_input(images, prep, size)
        x.tofile(os.path.join(a.out, f"{name}.input.f32"))
        t = time.perf_counter()
        outs = sess.run(None, {"pixel_values": x})
        ms = (time.perf_counter() - t) * 1000
        case = {
            "images": [short(p) if not isinstance(p, list) else [short(q) for q in p] for p in images],
            "prep": prep, "size": size,
            "input_shape": list(x.shape),
            "letterbox_rects": rects,           # [contentW, contentH, padX, padY] per view (ours only)
            "ort_cpu_ms": round(ms),
            "outputs": {},
        }
        for n, v in zip(out_names, outs):
            v = np.ascontiguousarray(v.astype(np.float32))
            v.tofile(os.path.join(a.out, f"{name}.{n}.f32"))
            case["outputs"][n] = {"shape": list(v.shape), "min": float(v.min()), "max": float(v.max()),
                                  "mean": float(v.mean()), "nan": int(np.isnan(v).sum())}
        # The decoded pixels of each "ours" view, so a test can push them through the REAL
        # ImagePreprocessKernel and prove this script's emulation is what SpawnScene feeds.
        # Skipped above 4 MP (a 12 MP phone photo is 50 MB of RGBA and proves nothing more).
        if prep == "ours":
            flat = [p for p in (images if not isinstance(images[0], list) else sum(images, []))]
            dims = [Image.open(p).size for p in flat]
            if all(w * h <= 4_000_000 for w, h in dims):
                case["rgba"] = []
                for i, p in enumerate(flat):
                    rgba = np.asarray(Image.open(p).convert("RGBA"), np.uint8)
                    rgba.tofile(os.path.join(a.out, f"{name}.v{i}.rgba"))
                    case["rgba"].append([rgba.shape[1], rgba.shape[0]])
        manifest["cases"][name] = case
        json.dump(manifest, open(man_path, "w"), indent=1)
        d = case["outputs"]["predicted_depth"]
        print(f"{name:18s} in={x.shape} ort={ms:7.0f}ms depth=[{d['min']:.4f},{d['max']:.4f}] "
              f"extr={case['outputs']['extrinsics']['shape']} intr={case['outputs']['intrinsics']['shape']}", flush=True)
    print(f"wrote {man_path}")


if __name__ == "__main__":
    main()
