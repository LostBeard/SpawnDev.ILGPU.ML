"""FLUX 3 Image Klein DiT checkpoint -> the fused layout Flux3Transformer reads, in bf16 (16), 8-bit (8) or 4-bit (4) blocks. One fused tensor is built, quantised and kept at a time, so an 11 GB checkpoint fits a 16 GB box.

  python -I quantize_flux3.py <model.safetensors> <out.safetensors> [bits=4]      bits = 4 | 8 | 16 (bf16, written as a stream)

Layout: flux3_convert.py (qkv_mlp = cat(q, k, v, mlp_in) rows, out = cat(attn_out, mlp_out) columns, ...). Block format and the "every matrix [N, K] with K % 32 == 0 and >= 1M elements" rule: quantize_flux2.py
(Q4_1-style blocks of 32 along K; I32 tensors [N, K/32*5 or *9]; the metadata "quant" says which).
"""
import os, re, sys, json, struct
import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, dst = sys.argv[1], sys.argv[2]
BITS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
assert BITS in (4, 8, 16)
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


def dequantise(words, k):
    n = words.shape[0]
    u = words.reshape(n, k // 32, WORDS).to(torch.int64) & 0xFFFFFFFF
    d = (u[:, :, 0] & 0xFFFF).to(torch.int16).view(torch.float16).float(); m = ((u[:, :, 0] >> 16) & 0xFFFF).to(torch.int16).view(torch.float16).float()
    out = torch.empty(n, k // 32, 32)
    per, bits = (8, 4) if BITS == 4 else (4, 8)
    for i in range(32 // per):
        for j in range(per):
            out[:, :, per * i + j] = ((u[:, :, 1 + i] >> (bits * j)) & LEVELS).float() * d + m
    return out.reshape(n, k)


f = safe_open(src, framework="pt")
keys = set(f.keys())
entries = []        # (fused name, loader)
used = set()


def block(prefix_src, prefix_dst):
    parts = [prefix_src + s for s in (".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".mlp_in.weight", ".attn_out.weight", ".mlp_out.weight", ".norm.query_norm.scale", ".norm.key_norm.scale")]
    used.update(parts)
    entries.append((prefix_dst + ".qkv_mlp.weight", lambda: torch.cat([f.get_tensor(p) for p in parts[:4]], dim=0)))
    entries.append((prefix_dst + ".out.weight", lambda: torch.cat([f.get_tensor(parts[4]), f.get_tensor(parts[5])], dim=1)))
    entries.append((prefix_dst + ".norm_q.weight", lambda: f.get_tensor(parts[6])))
    entries.append((prefix_dst + ".norm_k.weight", lambda: f.get_tensor(parts[7])))


for k in sorted(keys):
    m = re.fullmatch(r"content_mode_blocks\.(image|image_cond)\.(\d+)\.q_proj\.weight", k)
    if m: block(k[:-len(".q_proj.weight")], f"early.{m.group(1)}.{m.group(2)}")
    m = re.fullmatch(r"txt_mode_blocks\.(\d+)\.q_proj\.weight", k)
    if m: block(k[:-len(".q_proj.weight")], f"early.text.{m.group(1)}")
    m = re.fullmatch(r"single_blocks\.(\d+)\.q_proj\.weight", k)
    if m: block(k[:-len(".q_proj.weight")], f"single.{m.group(1)}")
simple = {"txt_in.weight": "txt_in.weight", "time_in.in_layer.weight": "time_in.in_layer.weight", "time_in.out_layer.weight": "time_in.out_layer.weight",
          "ref_time_in.in_layer.weight": "ref_time_in.in_layer.weight", "ref_time_in.out_layer.weight": "ref_time_in.out_layer.weight",
          "emb_in.image.weight": "emb_in.image.weight", "emb_in.image_cond.weight": "emb_in.image_cond.weight",
          "final_layer.image.linear.weight": "final.linear.weight", "final_layer.image.adaLN_modulation.1.weight": "final.mod.weight"}
for k in ("image", "image_cond", "txt"):
    simple[f"early_stream_modulations.{k}.lin.weight"] = f"mod_early.{k}.weight"
    simple[f"single_stream_modulations.{k}.lin.weight"] = f"mod_single.{k}.weight"
for s, d in simple.items():
    if s in keys:
        used.add(s); entries.append((d, (lambda s=s: f.get_tensor(s))))
left = sorted(keys - used)
if left: raise SystemExit("unmapped tensors: " + ", ".join(left[:8]))
print(len(entries), "fused tensors", flush=True)

if BITS == 16:      # a streamed safetensors file: header first (shapes from the source slices), then each fused tensor's bytes in turn
    shapes = {}
    def shape_of(name, loader):
        # shapes without loading: derive from the loader by a cheap probe of the slices
        t = loader(); sh = tuple(t.shape); del t; return sh
    header, off = {}, 0
    for name, loader in entries:
        sh = shape_of(name, loader); n = 2
        for x in sh: n *= x
        header[name] = {"dtype": "BF16", "shape": list(sh), "data_offsets": [off, off + n]}; off += n
    hb = json.dumps(header, separators=(",", ":")).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(dst, "wb") as out:
        out.write(struct.pack("<Q", len(hb))); out.write(hb)
        for name, loader in entries:
            t = loader().contiguous().view(torch.int16); out.write(t.numpy().tobytes()); del t
            print("wrote", name, flush=True)
    print(dst, round(os.path.getsize(dst) / 1e9, 2), "GB")
    sys.exit(0)

out, worst = {}, 0.0
for name, loader in entries:
    t = loader()
    if t.ndim == 2 and t.shape[1] % 32 == 0 and t.numel() >= MIN_ELEMENTS:
        qw = quantise(t)
        err = float((dequantise(qw, t.shape[1]) - t.float()).pow(2).sum().sqrt() / t.float().pow(2).sum().sqrt()); worst = max(worst, err)
        out[name] = qw
        print(f"{name} {tuple(t.shape)} relRMS {err:.4f}", flush=True)
    else:
        out[name] = t.contiguous()
    del t
meta = {"quant": f"q{BITS}_1 blocks of 32 along K: [d f16 | m f16] then " + ("4 words of 8 nibbles" if BITS == 4 else "8 words of 4 bytes") + f"; stored I32 [N, K/32*{WORDS}]", "model": "flux-3-image-klein fused (flux3_convert.py layout)"}
save_file(out, dst, metadata=meta)
print(dst, round(os.path.getsize(dst) / 1e9, 2), "GB; worst per-matrix relRMS", round(worst, 4))
