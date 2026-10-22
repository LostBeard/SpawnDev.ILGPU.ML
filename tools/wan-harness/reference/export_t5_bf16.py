"""Export the umT5-XXL encoder of Wan 2.2 (models_t5_umt5-xxl-enc-bf16.pth, 5.7B parameters, bf16) to one SafeTensors file for the C# `UmT5Encoder`.

Same tensors, same bf16 bits, with q / k / v fused per layer (one GEMM instead of three, the order of the output columns q | k | v):
  token_embedding.weight                  [256384, 4096]   (read ROW BY ROW from the file by the runner, never loaded onto the device)
  blocks.N.norm1.weight, blocks.N.norm2.weight, norm.weight   [4096]
  blocks.N.attn.qkv.weight                [12288, 4096]
  blocks.N.attn.o.weight                  [4096, 4096]
  blocks.N.ffn.gate.weight / ffn.fc1.weight   [10240, 4096],  blocks.N.ffn.fc2.weight  [4096, 10240]
  blocks.N.pos_embedding.embedding.weight [32, 64]         (the per-layer relative position bias table)
Layers are written in order so the runner can stream one layer's contiguous byte range at a time. Written tensor by tensor.

  python export_t5_bf16.py OUT.safetensors
"""
import json
import os
import struct
import sys

import numpy as np
import torch

CKPT = os.path.expanduser("~/wan/ckpt")


def main():
    out_path = sys.argv[1]
    sd = torch.load(os.path.join(CKPT, "models_t5_umt5-xxl-enc-bf16.pth"), map_location="cpu", mmap=True, weights_only=True)
    nblocks = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    entries = []

    def add(name, shape, fn):
        entries.append((name, list(shape), fn))

    add("token_embedding.weight", sd["token_embedding.weight"].shape, lambda: sd["token_embedding.weight"])
    for b in range(nblocks):
        p = f"blocks.{b}."
        add(p + "norm1.weight", sd[p + "norm1.weight"].shape, (lambda k=p + "norm1.weight": sd[k]))
        add(p + "attn.qkv.weight", [3 * 4096, 4096], (lambda p=p: torch.cat([sd[p + f"attn.{x}.weight"] for x in ("q", "k", "v")], 0)))
        add(p + "attn.o.weight", sd[p + "attn.o.weight"].shape, (lambda k=p + "attn.o.weight": sd[k]))
        add(p + "norm2.weight", sd[p + "norm2.weight"].shape, (lambda k=p + "norm2.weight": sd[k]))
        add(p + "ffn.gate.weight", sd[p + "ffn.gate.0.weight"].shape, (lambda k=p + "ffn.gate.0.weight": sd[k]))
        add(p + "ffn.fc1.weight", sd[p + "ffn.fc1.weight"].shape, (lambda k=p + "ffn.fc1.weight": sd[k]))
        add(p + "ffn.fc2.weight", sd[p + "ffn.fc2.weight"].shape, (lambda k=p + "ffn.fc2.weight": sd[k]))
        add(p + "pos_embedding.embedding.weight", sd[p + "pos_embedding.embedding.weight"].shape, (lambda k=p + "pos_embedding.embedding.weight": sd[k]))
    add("norm.weight", sd["norm.weight"].shape, lambda: sd["norm.weight"])

    header, off = {}, 0
    for name, shape, _ in entries:
        n = int(np.prod(shape)) * 2
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + n]}
        off += n
    header["__metadata__"] = {"format": "pt", "source": "Wan-AI/Wan2.2-TI2V-5B models_t5_umt5-xxl-enc-bf16.pth (q/k/v fused)", "dim": "4096", "dim_ffn": "10240",
                              "num_heads": "64", "num_layers": str(nblocks), "num_buckets": "32", "max_dist": "128", "vocab": "256384"}
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for i, (name, shape, fn) in enumerate(entries):
            t = fn().contiguous()
            assert t.dtype == torch.bfloat16 and list(t.shape) == shape, (name, t.dtype, list(t.shape), shape)
            f.write(t.view(torch.int16).numpy().tobytes())
            if i % 20 == 0:
                print(f"{i}/{len(entries)} {name} {shape}", flush=True)
    print("wrote", out_path, f"{os.path.getsize(out_path) / 1e9:.2f} GB, {len(entries)} tensors")


if __name__ == "__main__":
    main()
