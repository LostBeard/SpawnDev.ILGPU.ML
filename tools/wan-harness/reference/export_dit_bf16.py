"""Export the Wan 2.2 TI2V-5B DiT (20 GB fp32 in three shards) to ONE bf16 SafeTensors file in the layout the C# `WanDiT` reads.

bf16 is the dtype the reference itself runs in (`convert_model_dtype`: every parameter `.to(bfloat16)`, round to nearest even), so this export is
bit-identical to the reference's weights. Fused for the C# side (one GEMM instead of three, the TRELLIS.2 `to_qkv` layout):
  blocks.N.self_attn.qkv.{weight,bias}   = cat(q, k, v)  [3C, C] / [3C]
  blocks.N.cross_attn.kv.{weight,bias}   = cat(k, v)     [2C, C] / [2C]
and the patch embedding Conv3d (kernel = stride = (1,2,2)) is stored as the linear layer it is: patch_embedding.weight [C, 48*1*2*2] with the input
patch flattened (channel, kh, kw), plus `modulation` tables as [6, C] / [2, C]. Everything else keeps its reference name.
The output is written TENSOR BY TENSOR (the header is computed first): WSL has 7.8 GB of RAM and the export is 10 GB.

  python export_dit_bf16.py OUT.safetensors
"""
import json
import os
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open

CKPT = os.path.expanduser("~/wan/ckpt")


def main():
    out_path = sys.argv[1]
    index = json.load(open(os.path.join(CKPT, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
    handles = {s: safe_open(os.path.join(CKPT, s), framework="pt", device="cpu") for s in sorted(set(index.values()))}

    def get(name):
        return handles[index[name]].get_tensor(name)

    # (output name, shape, producer) in the order they are written
    entries = []

    def add(name, shape, fn):
        entries.append((name, list(shape), fn))

    def shape_of(name):
        return list(handles[index[name]].get_slice(name).get_shape())

    names = set(index.keys())
    nblocks = 1 + max(int(n.split(".")[1]) for n in names if n.startswith("blocks."))
    print("blocks", nblocks, "tensors", len(names))

    # embeddings
    pw = shape_of("patch_embedding.weight")                                   # [C, 48, 1, 2, 2]
    add("patch_embedding.weight", [pw[0], pw[1] * pw[2] * pw[3] * pw[4]], lambda: get("patch_embedding.weight").reshape(pw[0], -1))
    add("patch_embedding.bias", shape_of("patch_embedding.bias"), lambda: get("patch_embedding.bias"))
    for n in ("text_embedding.0", "text_embedding.2", "time_embedding.0", "time_embedding.2", "time_projection.1"):
        for k in ("weight", "bias"):
            add(f"{n}.{k}", shape_of(f"{n}.{k}"), (lambda nm=f"{n}.{k}": get(nm)))

    for b in range(nblocks):
        p = f"blocks.{b}."
        c = shape_of(p + "self_attn.q.weight")[0]
        mod = shape_of(p + "modulation")                                      # [1, 6, C]
        add(p + "modulation", [mod[1], mod[2]], (lambda nm=p + "modulation": get(nm).reshape(-1, get(nm).shape[-1])))
        for kind, parts in (("self_attn.qkv", ("q", "k", "v")),):
            add(p + f"{kind}.weight", [3 * c, c], (lambda p=p: torch.cat([get(p + f"self_attn.{x}.weight") for x in ("q", "k", "v")], 0)))
            add(p + f"{kind}.bias", [3 * c], (lambda p=p: torch.cat([get(p + f"self_attn.{x}.bias") for x in ("q", "k", "v")], 0)))
        for nm in ("self_attn.o.weight", "self_attn.o.bias", "self_attn.norm_q.weight", "self_attn.norm_k.weight", "norm3.weight", "norm3.bias",
                   "cross_attn.q.weight", "cross_attn.q.bias", "cross_attn.o.weight", "cross_attn.o.bias", "cross_attn.norm_q.weight", "cross_attn.norm_k.weight"):
            add(p + nm, shape_of(p + nm), (lambda nm=p + nm: get(nm)))
        add(p + "cross_attn.kv.weight", [2 * c, c], (lambda p=p: torch.cat([get(p + f"cross_attn.{x}.weight") for x in ("k", "v")], 0)))
        add(p + "cross_attn.kv.bias", [2 * c], (lambda p=p: torch.cat([get(p + f"cross_attn.{x}.bias") for x in ("k", "v")], 0)))
        for nm in ("ffn.0.weight", "ffn.0.bias", "ffn.2.weight", "ffn.2.bias"):
            add(p + nm, shape_of(p + nm), (lambda nm=p + nm: get(nm)))

    hm = shape_of("head.modulation")                                          # [1, 2, C]
    add("head.modulation", [hm[1], hm[2]], lambda: get("head.modulation").reshape(-1, get("head.modulation").shape[-1]))
    add("head.head.weight", shape_of("head.head.weight"), lambda: get("head.head.weight"))
    add("head.head.bias", shape_of("head.head.bias"), lambda: get("head.head.bias"))

    # header: all bf16, 2 bytes per element, offsets in order
    header, off = {}, 0
    for name, shape, _ in entries:
        n = int(np.prod(shape)) * 2
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + n]}
        off += n
    header["__metadata__"] = {"format": "pt", "source": "Wan-AI/Wan2.2-TI2V-5B (fp32 -> bf16 RNE, q/k/v and k/v fused, patch embedding as linear)",
                              "dim": "3072", "ffn_dim": "14336", "num_heads": "24", "num_layers": str(nblocks), "in_dim": "48", "out_dim": "48", "text_len": "512"}
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)                                      # pad the header to 8 bytes
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for i, (name, shape, fn) in enumerate(entries):
            t = fn().to(torch.bfloat16).contiguous()
            assert list(t.shape) == shape, (name, list(t.shape), shape)
            f.write(t.view(torch.int16).numpy().tobytes())                    # bf16 bits as int16
            if i % 100 == 0:
                print(f"{i}/{len(entries)} {name} {shape}", flush=True)
    print("wrote", out_path, f"{os.path.getsize(out_path) / 1e9:.2f} GB, {len(entries)} tensors")


if __name__ == "__main__":
    main()
