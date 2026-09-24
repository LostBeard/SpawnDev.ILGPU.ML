"""
Is ImagePreprocessKernel.NativeAspect's port of OpenCV exact, stage by stage?

Emulates the device kernel's AREA (exact integer overlaps) and CUBIC (11-bit fixed point, int
accumulation) in numpy, feeds each stage the SAME input cv2 got, and counts uint8 mismatches against
cv2 itself. A mismatch in a stage fed identical input is a porting error; mismatches that appear only
when stages are chained are rounding flips propagating (one stage-1 LSB moves up to 16 cubic taps).
Against IPP (cv2.ipp.setUseIPP(True)) the cubic differs on ~1.3% of pixels: that is IPP, not the port.

    python tools/dav3/check_cv2_port.py
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dav3_reference as R  # noqa: E402  (importing it turns Intel IPP OFF: see its header)

f32 = np.float32


def area(img, dw, dh):
    """The kernel's EXACT integer area average: overlaps in units of 1/dst pixel, round half even."""
    sh, sw = img.shape[:2]

    def overlaps(s, d):
        M = np.zeros((d, s), np.int64)
        for k in range(d):
            lo, hi = k * s, (k + 1) * s
            for i in range(lo // d, (hi - 1) // d + 1):
                M[k, i] = min(hi, (i + 1) * d) - max(lo, i * d)
        return M

    Wx, Wy, den = overlaps(sw, dw), overlaps(sh, dh), sw * sh
    out = np.empty((dh, dw, 3), np.int64)
    for c in range(3):
        num = Wy @ img[:, :, c].astype(np.int64) @ Wx.T
        q, r2 = num // den, 2 * (num % den)
        out[:, :, c] = q + ((r2 > den) | ((r2 == den) & (q % 2 == 1)))
    return np.clip(out, 0, 255).astype(np.uint8)


def cubic_coef(t):
    A = f32(-0.75); t = f32(t)
    c0 = ((A * (t + 1) - 5 * A) * (t + 1) + 8 * A) * (t + 1) - 4 * A
    c1 = ((A + 2) * t - (A + 3)) * t * t + 1
    c2 = ((A + 2) * (1 - t) - (A + 3)) * (1 - t) * (1 - t) + 1
    c3 = f32(1) - c0 - c1 - c2
    return [int(np.rint(f32(c) * f32(2048))) for c in (c0, c1, c2, c3)]


def cubic(img, dw, dh):
    sh, sw = img.shape[:2]
    src = img.astype(np.int64)
    out = np.zeros((dh, dw, 3), np.int64)
    for y in range(dh):
        fy = f32((2 * y + 1) * sh - dh) / f32(2 * dh); iy = int(np.floor(fy)); by = cubic_coef(fy - f32(iy))
        for x in range(dw):
            fx = f32((2 * x + 1) * sw - dw) / f32(2 * dw); ix = int(np.floor(fx)); ax = cubic_coef(fx - f32(ix))
            acc = np.zeros(3, np.int64)
            for j in range(4):
                syy = min(max(iy - 1 + j, 0), sh - 1)
                row = np.zeros(3, np.int64)
                for i in range(4):
                    sxx = min(max(ix - 1 + i, 0), sw - 1)
                    row += src[syy, sxx] * ax[i]
                acc += row * by[j]
            out[y, x] = acc
    return np.clip((out + (1 << 21)) >> 22, 0, 255).astype(np.uint8)


def report(tag, ours, ref):
    d = np.abs(ours.astype(int) - ref.astype(int))
    print(f"{tag:44s} max|diff|={d.max()} LSB, off={100.0 * (d > 0).mean():.3f}%")


if __name__ == "__main__":
    for path, res in [(R.drj("IMG_6292.jpg"), 504), (R.truck(1), 504), (R.temple(1), 896)]:
        img = np.asarray(Image.open(path).convert("RGB"))
        h, w = img.shape[:2]
        w1, h1 = int(round(w * res / max(w, h))), int(round(h * res / max(w, h)))
        cv1 = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_AREA if res < max(w, h) else cv2.INTER_CUBIC)
        ours1 = area(img, w1, h1) if res < max(w, h) else cubic(img, w1, h1)
        report(f"{os.path.basename(path)} stage1 {w}x{h}->{w1}x{h1}", ours1, cv1)
        n = lambda x: (x // 14 * 14 + 14) if abs(x // 14 * 14 + 14 - x) <= abs(x - x // 14 * 14) else x // 14 * 14
        w2, h2 = n(w1), n(h1)
        if (w2, h2) != (w1, h1):
            up = w2 > w1 or h2 > h1
            cv2s = cv2.resize(cv1, (w2, h2), interpolation=cv2.INTER_CUBIC if up else cv2.INTER_AREA)
            same_in = cubic(cv1, w2, h2) if up else area(cv1, w2, h2)
            report(f"  stage2 on cv2's stage1 ({'cubic' if up else 'area'})", same_in, cv2s)
            chained = cubic(ours1, w2, h2) if up else area(ours1, w2, h2)
            report("  stage2 chained on OUR stage1", chained, cv2s)
