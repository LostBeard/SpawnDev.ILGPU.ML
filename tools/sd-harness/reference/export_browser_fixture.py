"""Export one ref_sd_sample.py fixture as small files the browser test fetches from the local file server (no .npz parsing and no 600 MB archive in the page):
  python export_browser_fixture.py SD_DIR/ref/sample_<tag>.npz SD_DIR/browser/<tag>
Writes meta.json (ids, uncond ids, timesteps, alphas_cumprod, latent shape), noise.f32, final.f32 (the reference's last latent) and image.f32 (the decoded picture).
Serve SD_DIR with: node tools/unirig-harness/serve-files.mjs D:/Projects/sd 8769
"""
import json
import os
import sys

import numpy as np


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    z = np.load(src)
    meta = {k: z[k].astype(int).tolist() for k in ("ids", "uncond_ids", "ids2", "uncond_ids2") if k in z}
    meta["timesteps"] = z["timesteps"].astype(int).tolist()
    meta["alphas_cumprod"] = z["alphas_cumprod"].astype(float).tolist()
    meta["latent_shape"] = list(z["noise"].shape)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f)
    z["noise"].astype("<f4").tofile(os.path.join(out, "noise.f32"))
    z["latents"][-1].astype("<f4").tofile(os.path.join(out, "final.f32"))
    z["image"].astype("<f4").tofile(os.path.join(out, "image.f32"))
    print("wrote", out, os.listdir(out))


if __name__ == "__main__":
    main()
