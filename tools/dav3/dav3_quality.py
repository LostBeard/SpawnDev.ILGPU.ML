"""
Is the INPUT we feed DAv3 costing quality? Engine-independent: runs native onnxruntime only, so the
answer holds for our engine wherever DA3_OrtParity_* is green (ours == ORT on the same tensor).

Ground truth comes from COLMAP (tandt_db Truck + DrJohnson): each image's own 2D observations of
triangulated points give exact sparse depth at exact pixels, and the posed cameras give real
extrinsics/intrinsics. Nothing here is a model-vs-model comparison.

Per preprocessing variant it reports:
  single view  AbsRel and delta<1.25 of median-scaled depth at the SfM observations
  single view  focal-length error of the predicted intrinsics vs COLMAP (scaled to the model input)
  joint N=4    AbsRel as above, per view, from ONE joint forward
  joint N=4    camera-centre error after a similarity (Umeyama) fit, % of GT camera spread,
               and mean relative-rotation error in degrees

Variants (what differs is the only thing that changes):
  ours_lb_bilinear_S   SpawnScene today: letterbox into SxS, 2x2 bilinear (ImagePreprocessKernel)
  ours_lb_area_S       same geometry, INTER_AREA instead of bilinear  -> isolates the resize FILTER
  official_S           ByteDance DA3 input_processor: long side S, no padding, INTER_AREA, x14 rounding
  sq_stretch_S         squash to SxS, INTER_AREA                      -> what a non-letterboxing app does

    python tools/dav3/dav3_quality.py [--views 8] [--sizes 518,672]
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dav3_reference as R  # noqa: E402  (model path/hash, official + our preprocessing)

TANDT = R.TANDT
SCENES = {"truck": os.path.join(TANDT, "tandt", "truck"), "drjohnson": os.path.join(TANDT, "db", "drjohnson")}
OUT = os.path.abspath(os.path.join(R.REPO, "_mldump", "test-out", "dav3", "quality"))


# --- COLMAP binary readers (same format as SpawnScene tools/colmap_to_dataset.py) -------------------
def _r(f, n, fmt):
    return struct.unpack("<" + fmt, f.read(n))


def read_cameras(p):
    n_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}
    cams = {}
    with open(p, "rb") as f:
        for _ in range(_r(f, 8, "Q")[0]):
            cid, model, w, h = _r(f, 24, "iiQQ")
            k = n_params[model]
            prm = _r(f, 8 * k, "d" * k)
            if model not in (0, 1):
                raise SystemExit(f"camera model {model} has distortion")
            fx, fy, cx, cy = (prm[0], prm[0], prm[1], prm[2]) if model == 0 else prm
            cams[cid] = dict(w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy)
    return cams


def read_images(p):
    out = []
    with open(p, "rb") as f:
        for _ in range(_r(f, 8, "Q")[0]):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = _r(f, 64, "idddddddi")
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            n = _r(f, 8, "Q")[0]
            obs = np.frombuffer(f.read(24 * n), dtype=[("x", "<f8"), ("y", "<f8"), ("p", "<i8")])
            obs = obs[obs["p"] != -1]
            w, x, y, z = qw, qx, qy, qz
            Rm = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                           [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                           [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            out.append(dict(name=name.decode(), R=Rm, t=np.array([tx, ty, tz]), cam=cid, obs=obs))
    return out


def read_points(p):
    pts = {}
    with open(p, "rb") as f:
        for _ in range(_r(f, 8, "Q")[0]):
            pid, x, y, z, _, _, _, err = _r(f, 43, "QdddBBBd")
            nt = _r(f, 8, "Q")[0]
            f.read(8 * nt)
            if err <= 2.0 and nt >= 3:            # well-triangulated points only
                pts[pid] = (x, y, z)
    return pts


# --- preprocessing variants; each returns (CHW float32, map) where map(u, v) -> model-input pixel ------
def variant(name: str, path: str):
    kind, size = name.rsplit("_", 1)
    size = int(size)
    img = np.asarray(Image.open(path).convert("RGB"))
    H0, W0 = img.shape[:2]
    if kind == "official":
        x = R.official_one(path, size)
        h, w = x.shape[1:]
        return x, (lambda u, v: ((u + 0.5) * w / W0 - 0.5, (v + 0.5) * h / H0 - 0.5))
    if kind == "ours_lb_bilinear":
        x, (cw, ch, px, py) = R.ours_one(path, size)
    elif kind == "ours_lb_area":
        cw, ch, px, py = R.letterbox(W0, H0, size, size)
        small = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        yy = np.clip(np.arange(size) - py, 0, ch - 1); xx = np.clip(np.arange(size) - px, 0, cw - 1)
        full = small[yy][:, xx]                                   # border replicated, as the kernel does
        x = np.ascontiguousarray((((full - R.MEAN) / R.STD).transpose(2, 0, 1)).astype(np.float32))
    elif kind == "sq_stretch":
        full = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = np.ascontiguousarray((((full - R.MEAN) / R.STD).transpose(2, 0, 1)).astype(np.float32))
        return x, (lambda u, v: ((u + 0.5) * size / W0 - 0.5, (v + 0.5) * size / H0 - 0.5))
    else:
        raise ValueError(name)
    return x, (lambda u, v: (px + (u + 0.5) * cw / W0 - 0.5, py + (v + 0.5) * ch / H0 - 0.5))


def bilinear(img, x, y):
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001); y = np.clip(y, 0, h - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); tx = x - x0; ty = y - y0
    return (img[y0, x0] * (1 - tx) * (1 - ty) + img[y0, x0 + 1] * tx * (1 - ty)
            + img[y0 + 1, x0] * (1 - tx) * ty + img[y0 + 1, x0 + 1] * tx * ty)


def depth_err(pred, gt):
    s = np.median(gt / pred)                                     # monocular: scale-free, per view
    p = pred * s
    absrel = float(np.mean(np.abs(p - gt) / gt))
    d1 = float(np.mean(np.maximum(p / gt, gt / p) < 1.25))
    return absrel, d1


def umeyama(src, dst):
    """Similarity dst ~ s R src + t (least squares)."""
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (xs ** 2).sum(1).mean()
    return s, Rm, md - s * Rm @ ms


def rot_angle(Rm):
    return np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--sizes", default="518,672")
    ap.add_argument("--joint", type=int, default=4)
    ap.add_argument("--joint-stride", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sess = ort.InferenceSession(os.path.join(R.DEFAULT_MODEL, "model.onnx"), providers=["CPUExecutionProvider"])
    sizes = [int(s) for s in a.sizes.split(",")]
    variants = [f"{k}_{s}" for s in sizes for k in ("ours_lb_bilinear", "ours_lb_area", "official", "sq_stretch")]

    results = {}
    for scene, root in SCENES.items():
        cams = read_cameras(os.path.join(root, "sparse", "0", "cameras.bin"))
        ims = sorted(read_images(os.path.join(root, "sparse", "0", "images.bin")), key=lambda i: i["name"])
        pts = read_points(os.path.join(root, "sparse", "0", "points3D.bin"))
        step = max(1, len(ims) // a.views)
        pick = ims[::step][: a.views]

        def gt_obs(im):
            path = os.path.join(root, "images", im["name"])
            W0, H0 = Image.open(path).size
            cam = cams[im["cam"]]
            sx, sy = W0 / cam["w"], H0 / cam["h"]                  # COLMAP camera may be a different resolution
            keep = [i for i, p in enumerate(im["obs"]["p"]) if int(p) in pts]
            X = np.array([pts[int(im["obs"]["p"][i])] for i in keep])
            z = (X @ im["R"].T + im["t"])[:, 2]
            u = (im["obs"]["x"][keep] + 0.5) * sx - 0.5; v = (im["obs"]["y"][keep] + 0.5) * sy - 0.5
            ok = z > 1e-6
            return path, u[ok], v[ok], z[ok], cam, (W0, H0)

        for vname in variants:
            key = f"{scene}/{vname}"
            singles = []
            for im in pick:
                path, u, v, z, cam, (W0, H0) = gt_obs(im)
                x, m = variant(vname, path)
                out = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {"pixel_values": x[None, None]})))
                d = out["predicted_depth"][0, 0]
                mx, my = m(u, v)
                absrel, d1 = depth_err(bilinear(d, mx, my), z)
                # focal: GT fx in model-input pixels = fx_file * (input px per file px)
                x1, _ = m(np.array([0.0]), np.array([0.0])); x2, _ = m(np.array([100.0]), np.array([0.0]))
                gt_f = cam["fx"] * (W0 / cam["w"]) * float(x2[0] - x1[0]) / 100.0
                ferr = abs(float(out["intrinsics"][0, 0, 0, 0]) - gt_f) / gt_f
                singles.append((absrel, d1, ferr))
            sa = np.array(singles)
            r = dict(absrel=float(sa[:, 0].mean()), d1=float(sa[:, 1].mean()), focal_err=float(np.median(sa[:, 2])))

            # joint N views: one forward; views must share a shape, so only size-deterministic variants
            # A CONTIGUOUS window of overlapping frames: DAv3 can only relate views that share
            # content. Spreading 4 views across a walk-through room (DrJohnson) gave ~90 deg rotation
            # error for EVERY variant - it measured the view choice, not the preprocessing.
            j0 = len(ims) // 2
            joint = ims[j0: j0 + a.joint * a.joint_stride: a.joint_stride]
            xs, maps, gts = [], [], []
            for im in joint:
                path, u, v, z, cam, _ = gt_obs(im)
                x, m = variant(vname, path)
                xs.append(x); maps.append((m, u, v, z))
            if len({x.shape for x in xs}) == 1:
                out = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {"pixel_values": np.stack(xs)[None]})))
                jd = []
                for k, (m, u, v, z) in enumerate(maps):
                    mx, my = m(u, v)
                    jd.append(depth_err(bilinear(out["predicted_depth"][0, k], mx, my), z)[0])
                E = out["extrinsics"][0]                          # [N,3,4] world-to-camera (OpenCV)
                c_pred = np.array([-E[k, :, :3].T @ E[k, :, 3] for k in range(len(joint))])
                c_gt = np.array([-im["R"].T @ im["t"] for im in joint])
                s, Rm, t = umeyama(c_pred, c_gt)
                resid = np.linalg.norm((s * (Rm @ c_pred.T)).T + t - c_gt, axis=1)
                spread = np.linalg.norm(c_gt - c_gt.mean(0), axis=1).mean()
                rerr = []
                for i in range(len(joint)):
                    for j in range(i + 1, len(joint)):
                        Rp = E[j, :, :3] @ E[i, :, :3].T
                        Rg = joint[j]["R"] @ joint[i]["R"].T
                        rerr.append(rot_angle(Rp @ Rg.T))
                r.update(joint_absrel=float(np.mean(jd)), center_err_pct=float(100 * resid.mean() / spread),
                         rel_rot_err_deg=float(np.mean(rerr)))
            results[key] = r
            print(f"{key:34s} AbsRel={r['absrel']:.4f} d1={r['d1']:.3f} focalErr={r['focal_err']*100:5.1f}%"
                  + (f" | joint AbsRel={r['joint_absrel']:.4f} centre={r['center_err_pct']:5.1f}% rot={r['rel_rot_err_deg']:5.2f}deg"
                     if "joint_absrel" in r else ""), flush=True)
            json.dump(results, open(os.path.join(OUT, "quality.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
