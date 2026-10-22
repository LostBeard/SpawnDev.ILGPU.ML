"""Export the T5-v1.1-XXL ENCODER that LTX-Video uses (HF `T5EncoderModel`, 4 fp32 shards, 4.8B parameters) to one bf16 SafeTensors file for the C# `UmT5Encoder`.

bf16 is what the LTX pipeline runs the encoder in (diffusers loads it with torch_dtype=bfloat16: fp32 -> bf16, round to nearest even), so this is bit-identical to the reference's weights.
Same tensor names and layout as tools/wan-harness/reference/export_t5_bf16.py (one reader for both encoders), q / k / v fused per layer:
  token_embedding.weight                  [32128, 4096]   (HF `shared.weight`; rows are read from the file by the runner)
  blocks.N.norm1.weight, blocks.N.norm2.weight, norm.weight   [4096]
  blocks.N.attn.qkv.weight                [4096 * 3, 4096]   (q | k | v;  HF 4096 = 64 heads x 64)
  blocks.N.attn.o.weight                  [4096, 4096]
  blocks.N.ffn.gate.weight (wi_0) / ffn.fc1.weight (wi_1)   [10240, 4096],  blocks.N.ffn.fc2.weight (wo)  [4096, 10240]
  blocks.0.pos_embedding.embedding.weight [32, 64]        T5 v1.1 has ONE relative position bias table (block 0), SHARED by all layers (umT5 has one per layer); metadata "shared_bias" = "1"
Written tensor by tensor (WSL has 7.8 GB of RAM).

  python export_t5_bf16.py SRC_DIR OUT.safetensors       (SRC_DIR holds model.safetensors.index.json and the shards)
"""
import json
import os
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open


def main():
    src, out_path = sys.argv[1], sys.argv[2]
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    handles = {s: safe_open(os.path.join(src, s), framework="pt", device="cpu") for s in sorted(set(index.values()))}
    get = lambda name: handles[index[name]].get_tensor(name)
    shape_of = lambda name: list(handles[index[name]].get_slice(name).get_shape())
    nblocks = 1 + max(int(k.split(".")[2]) for k in index if k.startswith("encoder.block."))
    cfg = json.load(open(os.path.join(src, "config.json")))
    d, dff, heads = cfg["d_model"], cfg["d_ff"], cfg["num_heads"]
    print("blocks", nblocks, "d_model", d, "d_ff", dff, "heads", heads, "feed_forward_proj", cfg.get("feed_forward_proj"))
    entries = []

    def add(name, shape, fn):
        entries.append((name, list(shape), fn))

    add("token_embedding.weight", shape_of("shared.weight"), lambda: get("shared.weight"))
    for b in range(nblocks):
        p = f"encoder.block.{b}."
        o = f"blocks.{b}."
        add(o + "norm1.weight", [d], (lambda k=p + "layer.0.layer_norm.weight": get(k)))
        add(o + "attn.qkv.weight", [3 * d, d], (lambda p=p: torch.cat([get(p + f"layer.0.SelfAttention.{x}.weight") for x in ("q", "k", "v")], 0)))
        add(o + "attn.o.weight", [d, d], (lambda k=p + "layer.0.SelfAttention.o.weight": get(k)))
        add(o + "norm2.weight", [d], (lambda k=p + "layer.1.layer_norm.weight": get(k)))
        add(o + "ffn.gate.weight", [dff, d], (lambda k=p + "layer.1.DenseReluDense.wi_0.weight": get(k)))
        add(o + "ffn.fc1.weight", [dff, d], (lambda k=p + "layer.1.DenseReluDense.wi_1.weight": get(k)))
        add(o + "ffn.fc2.weight", [d, dff], (lambda k=p + "layer.1.DenseReluDense.wo.weight": get(k)))
        if b == 0:
            k = p + "layer.0.SelfAttention.relative_attention_bias.weight"
            add(o + "pos_embedding.embedding.weight", shape_of(k), (lambda k=k: get(k)))
    add("norm.weight", [d], lambda: get("encoder.final_layer_norm.weight"))

    header, off = {}, 0
    for name, shape, _ in entries:
        n = int(np.prod(shape)) * 2
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + n]}
        off += n
    header["__metadata__"] = {"format": "pt", "source": "Lightricks/LTX-Video text_encoder (T5-v1.1-XXL encoder, fp32 -> bf16 RNE, q/k/v fused)", "dim": str(d), "dim_ffn": str(dff),
                              "num_heads": str(heads), "num_layers": str(nblocks), "num_buckets": "32", "max_dist": "128", "vocab": str(shape_of("shared.weight")[0]), "shared_bias": "1"}
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for i, (name, shape, fn) in enumerate(entries):
            t = fn().to(torch.bfloat16).contiguous()
            assert list(t.shape) == shape, (name, list(t.shape), shape)
            f.write(t.view(torch.int16).numpy().tobytes())
            if i % 20 == 0:
                print(f"{i}/{len(entries)} {name} {shape}", flush=True)
    print("wrote", out_path, f"{os.path.getsize(out_path) / 1e9:.2f} GB, {len(entries)} tensors")


if __name__ == "__main__":
    main()
