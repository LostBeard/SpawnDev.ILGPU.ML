"""Wan 2.2 reference: how much of the umT5 encoder's difference to an fp32 port is the reference's own bf16 rounding?

Runs the reference encoder TWICE on the same token ids, layer by layer, with the same bf16 weights: once as the reference does (bf16 activations) and once with every layer
upcast to float32 (exact math on the same weights), and dumps both final contexts and the layer taps (embedding, layers 0, 12, 23) to OUT_DIR/t5x_<prompt>.npz.

  python ref_t5_noise.py OUT_DIR T5_DIR          (T5_DIR holds the t5_*.npz of ref_t5_dump.py: its token ids are reused)
"""
import os
import sys

import numpy as np
import torch

WAN = os.path.expanduser("~/wan/Wan2.2")
sys.path.insert(0, WAN)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_t5_dump import build_encoder  # noqa: E402


@torch.no_grad()
def run(enc, ids, fp32):
    taps = {}
    x = enc.token_embedding(ids[None]).to("cuda")
    x = x.float() if fp32 else x
    taps["embedding"] = x[0].float().cpu().numpy()
    for i, block in enumerate(enc.blocks):
        if fp32:
            block.float()
        x = block(x, None, pos_bias=None)
        if fp32:
            block.to(torch.bfloat16)           # bf16 -> fp32 -> bf16 is lossless for bf16 weights
        if i in (0, 12, 23):
            taps[f"layer_{i}"] = x[0].float().cpu().numpy()
    norm = enc.norm.float() if fp32 else enc.norm
    x = norm(x)
    if fp32:
        enc.norm.to(torch.bfloat16)
    taps["final"] = x[0].float().cpu().numpy()
    return taps


def main():
    out_dir, t5_dir = sys.argv[1], sys.argv[2]
    enc = build_encoder()
    for name in ("0", "1", "neg"):
        fx = np.load(os.path.join(t5_dir, f"t5_{name}.npz"))
        n = int(fx["mask"].sum())
        ids = torch.from_numpy(fx["ids"][:n]).long()
        bf = run(enc, ids, False)
        ex = run(enc, ids, True)
        out = {f"bf16_{k}": v for k, v in bf.items()}
        out.update({f"fp32_{k}": v for k, v in ex.items()})
        np.savez(os.path.join(out_dir, f"t5x_{name}.npz"), **out)
        d = bf["final"] - ex["final"]
        print(f"prompt {name} ({n} tokens): bf16 reference vs exact math on the same weights: relRMS {np.sqrt((d ** 2).sum() / (ex['final'] ** 2).sum()):.3e}, layer_23 {np.sqrt(((bf['layer_23'] - ex['layer_23']) ** 2).sum() / (ex['layer_23'] ** 2).sum()):.3e}, layer_0 {np.sqrt(((bf['layer_0'] - ex['layer_0']) ** 2).sum() / (ex['layer_0'] ** 2).sum()):.3e}")


if __name__ == "__main__":
    main()
