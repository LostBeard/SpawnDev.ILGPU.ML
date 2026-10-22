"""Export the LTX-Video 2B DiT from the single-file checkpoint (ltxv-2b-0.9.8-distilled.safetensors, the original LTX key names under `model.diffusion_model.`) to ONE bf16
SafeTensors file in the layout the C# `LtxDiT` reads. The names deliberately follow tools/wan-harness/reference/export_dit_bf16.py so the same 8-bit exporter (quantize_streaming.py) applies:

  patch_embedding.{weight,bias}                    <- patchify_proj                         [2048, 128]   (patch 1x1x1: a plain linear layer on the 128 latent channels)
  time_embedding.0 / .2                            <- adaln_single.emb.timestep_embedder.linear_1 / linear_2     [2048, 256] / [2048, 2048]
  time_projection.1                                <- adaln_single.linear                   [6 * 2048, 2048]
  text_embedding.0 / .2                            <- caption_projection.linear_1 / linear_2   [2048, 4096] / [2048, 2048]
  head.modulation [2, 2048]                        <- scale_shift_table;  head.head.{weight,bias} <- proj_out [128, 2048]
  blocks.N.modulation [6, 2048]                    <- transformer_blocks.N.scale_shift_table
  blocks.N.self_attn.qkv.{weight,bias}             = cat(attn1.to_q, to_k, to_v)             [3C, C] / [3C]
  blocks.N.self_attn.norm_q / norm_k .weight       <- attn1.q_norm / k_norm                  [C]   (RMS norm over the whole q / k vector, eps 1e-5)
  blocks.N.self_attn.o.{weight,bias}               <- attn1.to_out.0
  blocks.N.cross_attn.q.{weight,bias}, .kv.{weight,bias} = cat(attn2.to_k, to_v), .norm_q / .norm_k, .o   (no norm3: LTX cross-attends to the raw residual stream)
  blocks.N.ffn.0 / ffn.2                           <- ff.net.0.proj / ff.net.2

bf16 is the dtype of the source (and of the reference run), so this is a re-layout: every value is bit-identical. Written tensor by tensor (WSL has 7.8 GB of RAM).

  python export_dit_bf16.py SRC.safetensors OUT.safetensors
"""
import json
import os
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open

PREFIX = "model.diffusion_model."


def main():
    src, out_path = sys.argv[1], sys.argv[2]
    f = safe_open(src, framework="pt", device="cpu")
    keys = {k[len(PREFIX):] for k in f.keys() if k.startswith(PREFIX)}
    get = lambda name: f.get_tensor(PREFIX + name)
    shape_of = lambda name: list(f.get_slice(PREFIX + name).get_shape())
    nblocks = 1 + max(int(k.split(".")[1]) for k in keys if k.startswith("transformer_blocks."))
    dim = shape_of("patchify_proj.weight")[0]
    ffn = shape_of("transformer_blocks.0.ff.net.0.proj.weight")[0]
    print("blocks", nblocks, "dim", dim, "ffn", ffn, "tensors", len(keys))

    entries = []

    def add(name, shape, fn):
        entries.append((name, list(shape), fn))

    def copy(out, inp):
        add(out, shape_of(inp), (lambda inp=inp: get(inp)))

    for out, inp in (("patch_embedding", "patchify_proj"), ("time_embedding.0", "adaln_single.emb.timestep_embedder.linear_1"),
                     ("time_embedding.2", "adaln_single.emb.timestep_embedder.linear_2"), ("time_projection.1", "adaln_single.linear"),
                     ("text_embedding.0", "caption_projection.linear_1"), ("text_embedding.2", "caption_projection.linear_2"), ("head.head", "proj_out")):
        copy(out + ".weight", inp + ".weight")
        copy(out + ".bias", inp + ".bias")
    copy("head.modulation", "scale_shift_table")

    for b in range(nblocks):
        s = f"transformer_blocks.{b}."
        o = f"blocks.{b}."
        copy(o + "modulation", s + "scale_shift_table")
        for kind in ("weight", "bias"):
            add(o + f"self_attn.qkv.{kind}", [3 * dim, dim] if kind == "weight" else [3 * dim],
                (lambda s=s, kind=kind: torch.cat([get(s + f"attn1.to_{x}.{kind}") for x in ("q", "k", "v")], 0)))
            add(o + f"cross_attn.kv.{kind}", [2 * dim, dim] if kind == "weight" else [2 * dim],
                (lambda s=s, kind=kind: torch.cat([get(s + f"attn2.to_{x}.{kind}") for x in ("k", "v")], 0)))
        copy(o + "self_attn.norm_q.weight", s + "attn1.q_norm.weight")
        copy(o + "self_attn.norm_k.weight", s + "attn1.k_norm.weight")
        copy(o + "self_attn.o.weight", s + "attn1.to_out.0.weight")
        copy(o + "self_attn.o.bias", s + "attn1.to_out.0.bias")
        copy(o + "cross_attn.q.weight", s + "attn2.to_q.weight")
        copy(o + "cross_attn.q.bias", s + "attn2.to_q.bias")
        copy(o + "cross_attn.norm_q.weight", s + "attn2.q_norm.weight")
        copy(o + "cross_attn.norm_k.weight", s + "attn2.k_norm.weight")
        copy(o + "cross_attn.o.weight", s + "attn2.to_out.0.weight")
        copy(o + "cross_attn.o.bias", s + "attn2.to_out.0.bias")
        copy(o + "ffn.0.weight", s + "ff.net.0.proj.weight")
        copy(o + "ffn.0.bias", s + "ff.net.0.proj.bias")
        copy(o + "ffn.2.weight", s + "ff.net.2.weight")
        copy(o + "ffn.2.bias", s + "ff.net.2.bias")

    header, off = {}, 0
    for name, shape, _ in entries:
        n = int(np.prod(shape)) * 2
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + n]}
        off += n
    header["__metadata__"] = {"format": "pt", "source": "Lightricks/LTX-Video ltxv-2b-0.9.8-distilled (bf16 as stored, q/k/v and k/v fused, original names mapped to the Wan layout)",
                              "dim": str(dim), "ffn_dim": str(ffn), "num_heads": "32", "num_layers": str(nblocks), "in_dim": "128", "out_dim": "128", "text_len": "128"}
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(out_path, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)))
        fo.write(hb)
        for i, (name, shape, fn) in enumerate(entries):
            t = fn().to(torch.bfloat16).contiguous()
            assert list(t.shape) == shape, (name, list(t.shape), shape)
            fo.write(t.view(torch.int16).numpy().tobytes())
            if i % 100 == 0:
                print(f"{i}/{len(entries)} {name} {shape}", flush=True)
    print("wrote", out_path, f"{os.path.getsize(out_path) / 1e9:.2f} GB, {len(entries)} tensors")


if __name__ == "__main__":
    main()
