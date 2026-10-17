"""Convert an image file to the harness' raw input: <out>.u32 (packed RGBA, R in the low byte, row-major) and <out>.size ("W H").
  python img2raw.py photo.jpg out        -> out.u32, out.size"""
import sys, numpy as np
from PIL import Image
im = Image.open(sys.argv[1]).convert("RGBA"); a = np.array(im)
packed = (a[..., 0].astype(np.uint32) | (a[..., 1].astype(np.uint32) << 8) | (a[..., 2].astype(np.uint32) << 16) | (a[..., 3].astype(np.uint32) << 24)).astype(np.uint32)
packed.tofile(sys.argv[2] + ".u32"); open(sys.argv[2] + ".size", "w").write(f"{im.width} {im.height}")
print(im.size, "alpha min", int(a[..., 3].min()))
