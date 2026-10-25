"""Export arrays of a ref_sd_dump.py fixture as raw little-endian float32 files for the browser stage tests (streamed by HTTP range requests onto the GPU, never parsed in the page):
  python export_browser_taps.py SD_DIR/ref/sd_<tag>.npz SD_DIR/browser/<name> name1 name2 ...
Each array keeps the reference layout (channel-first). shapes.json records the shapes.
"""
import json
import os
import sys

import numpy as np


def main():
    src, out, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out, exist_ok=True)
    z = np.load(src)
    shapes = {}
    for n in names:
        a = z[n].astype("<f4")
        a.tofile(os.path.join(out, n + ".f32"))
        shapes[n] = list(a.shape)
    with open(os.path.join(out, "shapes.json"), "w") as f:
        json.dump(shapes, f)
    print("wrote", out, shapes)


if __name__ == "__main__":
    main()
