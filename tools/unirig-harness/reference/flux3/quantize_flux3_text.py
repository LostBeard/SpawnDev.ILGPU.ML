"""FLUX 3 Image Klein's text encoder (Qwen3-VL-4B-Instruct, text tower, first 32 layers) -> the layout Flux3TextEncoder reads (the same as FLUX.2's text encoder export), bf16 (16), 8-bit (8) or 4-bit (4).

  python -I quantize_flux3_text.py <text_encoder_dir> <out.safetensors> [bits=4]

Names: embed_tokens.weight; layers.<l>.self_attn.qkv_proj.weight (cat of q_proj, k_proj, v_proj rows), q_norm.weight, k_norm.weight, o_proj.weight, input_layernorm.weight, post_attention_layernorm.weight,
mlp.gate_up_proj.weight (cat of gate_proj, up_proj rows), mlp.down_proj.weight. The vision tower, layers 32 and up, the final norm and the lm head are left out (the pipeline never reads them).
Block formats: see quantize_flux3.py / quantize_flux2.py.
"""
import os, sys, json, struct
import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, dst = sys.argv[1], sys.argv[2]
BITS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
assert BITS in (4, 8, 16)
LAYERS = 32
LEVELS = 15 if BITS == 4 else 255
WORDS = 5 if BITS == 4 else 9
MIN_ELEMENTS = 1_000_000


def quantise(w):
    n, k = w.shape
    x = w.float().reshape(n, k // 32, 32)
    mn = x.amin(dim=2, keepdim=True); mx = x.amax(dim=2, keepdim=True)
    d = ((mx - mn) / LEVELS).to(torch.float16); m = mn.to(torch.float16)
    df = d.float(); mf = m.float()
    q = torch.where(df > 0, torch.round((x - mf) / df.clamp_min(1e-30)), torch.zeros_like(x)).clamp(0, LEVELS).to(torch.int64)
    head = (d.view(torch.int16).to(torch.int64) & 0xFFFF) | ((m.view(torch.int16).to(torch.int64) & 0xFFFF) << 16)
    words = [head.reshape(n, k // 32, 1)]
    per, bits = (8, 4) if BITS == 4 else (4, 8)
    for i in range(32 // per):
        wd = torch.zeros(n, k // 32, dtype=torch.int64)
        for j in range(per):
            wd |= q[:, :, per * i + j] << (bits * j)
        words.append(wd.reshape(n, k // 32, 1))
    allw = torch.cat(words, dim=2) & 0xFFFFFFFF
    allw = torch.where(allw >= 2 ** 31, allw - 2 ** 32, allw).to(torch.int32)
    return allw.reshape(n, k // 32 * WORDS).contiguous()


index = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
handles = {}
def get(name):
    shard = index[name]
    if shard not in handles: handles[shard] = safe_open(os.path.join(src, shard), framework="pt")
    return handles[shard].get_tensor(name)

P = "model.language_model."
entries = [("embed_tokens.weight", lambda: get(P + "embed_tokens.weight"))]
for l in range(LAYERS):
    b = f"{P}layers.{l}."; o = f"layers.{l}."
    entries += [
        (o + "self_attn.qkv_proj.weight", (lambda b=b: torch.cat([get(b + "self_attn.q_proj.weight"), get(b + "self_attn.k_proj.weight"), get(b + "self_attn.v_proj.weight")], dim=0))),
        (o + "self_attn.q_norm.weight", (lambda b=b: get(b + "self_attn.q_norm.weight"))), (o + "self_attn.k_norm.weight", (lambda b=b: get(b + "self_attn.k_norm.weight"))),
        (o + "self_attn.o_proj.weight", (lambda b=b: get(b + "self_attn.o_proj.weight"))),
        (o + "input_layernorm.weight", (lambda b=b: get(b + "input_layernorm.weight"))), (o + "post_attention_layernorm.weight", (lambda b=b: get(b + "post_attention_layernorm.weight"))),
        (o + "mlp.gate_up_proj.weight", (lambda b=b: torch.cat([get(b + "mlp.gate_proj.weight"), get(b + "mlp.up_proj.weight")], dim=0))),
        (o + "mlp.down_proj.weight", (lambda b=b: get(b + "mlp.down_proj.weight"))),
    ]
print(len(entries), "tensors", flush=True)

if BITS == 16:
    header, off = {}, 0
    for name, loader in entries:
        t = loader(); sh = tuple(t.shape); del t
        n = 2
        for x in sh: n *= x
        header[name] = {"dtype": "BF16", "shape": list(sh), "data_offsets": [off, off + n]}; off += n
    hb = json.dumps(header, separators=(",", ":")).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(dst, "wb") as out:
        out.write(struct.pack("<Q", len(hb))); out.write(hb)
        for name, loader in entries:
            t = loader().contiguous().view(torch.int16); out.write(t.numpy().tobytes()); del t
    print(dst, round(os.path.getsize(dst) / 1e9, 2), "GB")
    sys.exit(0)

out, worst = {}, 0.0
for name, loader in entries:
    t = loader()
    if t.ndim == 2 and t.shape[1] % 32 == 0 and t.numel() >= MIN_ELEMENTS:
        qw = quantise(t); out[name] = qw
        print(f"{name} {tuple(t.shape)}", flush=True)
    else:
        out[name] = t.contiguous()
    del t
meta = {"quant": f"q{BITS}_1 blocks of 32 along K: [d f16 | m f16] then " + ("4 words of 8 nibbles" if BITS == 4 else "8 words of 4 bytes") + f"; stored I32 [N, K/32*{WORDS}]"}
save_file(out, dst, metadata=meta)
print(dst, round(os.path.getsize(dst) / 1e9, 2), "GB")
