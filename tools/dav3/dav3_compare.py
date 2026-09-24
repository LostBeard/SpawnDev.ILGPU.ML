"""
Side-by-side DAv3: native onnxruntime (the reference) vs every engine that dumped outputs under
_mldump/test-out/dav3/<engine>/ - our PMT lanes (WebGPU, Cuda, ...) and Transformers.js (tjs-webgpu,
tjs-wasm).

    python tools/dav3/dav3_compare.py            # after dav3_reference.py, PMT DA3_OrtParity_*, dav3-tjs.mjs

Writes _mldump/test-out/dav3/compare/:
    <case>.png      one row per view: input | ORT depth | each engine's depth | |engine - ORT| x DIFF_GAIN
    summary.md      per case x engine x output: relRMS / correlation per view, camera max|diff|, timings

Statistics follow tools/README.md: relRMS (||a-b|| / ||b||) per VIEW, never a whole-tensor max. A view
indexing defect leaves view 0 exact and corrupts the rest.
"""
from __future__ import annotations

import json
import os
import re

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
REFS = os.path.join(REPO, "SpawnDev.ILGPU.ML.Demo", "wwwroot", "test-refs", "dav3")
OUT = os.path.abspath(os.path.join(REPO, "_mldump", "test-out", "dav3"))
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DIFF_GAIN = 100.0      # |diff| / depth-range x 100: a 1% error is full scale


def load(path, shape):
    return np.fromfile(path, np.float32).reshape(shape) if os.path.exists(path) else None


def stats(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    if np.isnan(a).any():
        return float("nan"), float("nan"), float("nan")
    rel = np.linalg.norm(a - b) / max(1e-300, np.linalg.norm(b))
    corr = np.corrcoef(a, b)[0, 1]
    return rel, corr, np.abs(a - b).max()


def turbo(x, lo, hi):
    t = np.clip((x - lo) / max(1e-12, hi - lo), 0, 1)
    return cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def ours_timings(engines):
    """{engine: {case: {create, cold, warm}}} from the timing-*.json each PMT parity test POSTs."""
    t = {}
    for e in engines:
        d = os.path.join(OUT, e)
        for f in os.listdir(d):
            if f.startswith("timing") and f.endswith(".json"):
                t.setdefault(e, {}).update(json.load(open(os.path.join(d, f))))
    return t


def main():
    man = json.load(open(os.path.join(REFS, "manifest.json")))
    engines = sorted(d for d in os.listdir(OUT) if os.path.isdir(os.path.join(OUT, d)) and d != "compare") if os.path.isdir(OUT) else []
    os.makedirs(os.path.join(OUT, "compare"), exist_ok=True)
    tjs = {}
    for f in os.listdir(OUT) if os.path.isdir(OUT) else []:
        if f.startswith("tjs-") and f.endswith(".json"):
            j = json.load(open(os.path.join(OUT, f)))
            tjs[f[:-5]] = j
    ours_t = ours_timings([e for e in engines if not e.startswith('tjs-')])

    md = ["# DAv3: engines vs native onnxruntime (CPU)", "",
          f"model sha256 {man.get('model_sha256')}, onnxruntime {man.get('onnxruntime')}", "",
          "engines found: " + ", ".join(engines), ""]
    for e, j in tjs.items():
        md.append(f"- {e}: transformers.js {j.get('tjs')} adapter `{j.get('adapter')}`" + (f" FATAL {j.get('error')}" if j.get("error") else ""))
    md.append("")

    for name, c in man["cases"].items():
        shp = c["input_shape"]; B, N, _, H, W = shp
        x = load(os.path.join(REFS, f"{name}.input.f32"), shp)
        ref = {o: load(os.path.join(REFS, f"{name}.{o}.f32"), v["shape"]) for o, v in c["outputs"].items()}
        md += [f"## {name}  `{'x'.join(map(str, shp))}`  prep={c['prep']}", ""]
        md.append("| engine | output | per-view relRMS | per-view corr | camera max abs diff |")
        md.append("|---|---|---|---|---|")
        rows = []
        dlo, dhi = float(ref["predicted_depth"].min()), float(ref["predicted_depth"].max())
        for b in range(B):
            for n in range(N):
                rgb = ((x[b, n].transpose(1, 2, 0) * STD + MEAN).clip(0, 1) * 255).astype(np.uint8)[:, :, ::-1]
                row = [label(rgb, f"b{b} v{n} input"), label(turbo(ref["predicted_depth"][b, n], dlo, dhi), "ORT-CPU")]
                for e in engines:
                    d = load(os.path.join(OUT, e, f"{name}.predicted_depth.f32"), ref["predicted_depth"].shape)
                    if d is None:
                        continue
                    rel, _, _ = stats(d[b, n], ref["predicted_depth"][b, n])
                    row.append(label(turbo(d[b, n], dlo, dhi), f"{e} rel={rel:.1e}"))
                    diff = np.abs(d[b, n] - ref["predicted_depth"][b, n]) / max(1e-12, dhi - dlo) * DIFF_GAIN
                    row.append(label(turbo(diff, 0, 1), f"|{e}-ORT| x{DIFF_GAIN:g}"))
                rows.append(np.concatenate(row, axis=1))
        width = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
        cv2.imwrite(os.path.join(OUT, "compare", f"{name}.png"), np.concatenate(rows, axis=0))

        for e in engines:
            for o, r in ref.items():
                d = load(os.path.join(OUT, e, f"{name}.{o}.f32"), r.shape)
                if d is None:
                    if os.path.exists(os.path.join(OUT, e)) and any(f.startswith(name + ".") for f in os.listdir(os.path.join(OUT, e))):
                        md.append(f"| {e} | {o} | MISSING or wrong size | | |")
                    continue
                if o in ("predicted_depth", "confidence"):
                    per = [stats(d[b, n], r[b, n]) for b in range(B) for n in range(N)]
                    md.append(f"| {e} | {o} | {' '.join(f'{p[0]:.1e}' for p in per)} | {' '.join(f'{p[1]:.6f}' for p in per)} | |")
                else:
                    md.append(f"| {e} | {o} | | | {np.abs(d - r).max():.2e} (ref max {np.abs(r).max():.2f}) |")
        md.append("")
        md.append(f"time: ORT-CPU {c['ort_cpu_ms']} ms")
        for be, t in ours_t.items():
            if name in t:
                tt = t[name]
                md.append(f"- ours {be}: create {tt['create']} ms, cold {tt['cold']} ms, **warm {tt['warm']} ms**"
                          + (f", capture {tt['capture']} ms, **replay {tt['replay']} ms**" if tt.get("replay", -1) >= 0 else ""))
        for e, j in tjs.items():
            for r in j.get("results", []):
                if r["name"] == name:
                    md.append(f"- {e}: load {r.get('loadMs', 0):.0f} ms, cold {r.get('coldMs', float('nan')):.0f} ms, **warm {r.get('warmMs', float('nan')):.0f} ms**"
                              + (f" ERROR {r['error']}" if r.get("error") else ""))
        md.append(f"\n![{name}]({name}.png)\n")

    open(os.path.join(OUT, "compare", "summary.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
