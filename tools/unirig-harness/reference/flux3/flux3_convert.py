"""FLUX 3 DiT checkpoint -> the fused layout SpawnDev.ILGPU.ML.Flux3.Flux3Transformer reads.

The reference block (dit.py `Block`) keeps q_proj, k_proj, v_proj, mlp_in (gate | up), attn_out and mlp_out as separate matrices; the C# port runs the FLUX.2 single-stream block, which wants

  <prefix>.qkv_mlp.weight   [3*dim + 2*mlp, dim]   rows q | k | v | gate | up      (cat of q_proj, k_proj, v_proj, mlp_in)
  <prefix>.out.weight       [dim, dim + mlp]       columns attn_out | mlp_out      (cat of attn_out, mlp_out along the input axis)
  <prefix>.norm_q.weight, <prefix>.norm_k.weight   [128]                           (norm.query_norm.scale, norm.key_norm.scale)

with <prefix> = early.<mode>.<d> (content_mode_blocks.<mode>.<d>; txt_mode_blocks.<d> is mode "text") or single.<i> (single_blocks.<i>). The rest is renamed:

  emb_in.image.weight, emb_in.image_cond.weight, txt_in.weight, time_in.in_layer.weight, time_in.out_layer.weight, ref_time_in.* (flow map),
  mod_early.<image|image_cond|txt>.weight (early_stream_modulations.<k>.lin.weight), mod_single.<k>.weight (single_stream_modulations.<k>.lin.weight),
  final.linear.weight (final_layer.image.linear.weight), final.mod.weight (final_layer.image.adaLN_modulation.1.weight).
"""
import re
import torch


def fuse(sd):
    out, used = {}, set()

    def take(k):
        used.add(k)
        return sd[k]

    def block(src, dst):
        out[dst + ".qkv_mlp.weight"] = torch.cat([take(src + ".q_proj.weight"), take(src + ".k_proj.weight"), take(src + ".v_proj.weight"), take(src + ".mlp_in.weight")], dim=0).contiguous()
        out[dst + ".out.weight"] = torch.cat([take(src + ".attn_out.weight"), take(src + ".mlp_out.weight")], dim=1).contiguous()
        out[dst + ".norm_q.weight"] = take(src + ".norm.query_norm.scale").contiguous()
        out[dst + ".norm_k.weight"] = take(src + ".norm.key_norm.scale").contiguous()

    for k in list(sd):
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
    for src, dst in simple.items():
        if src in sd: out[dst] = take(src).contiguous()
    left = sorted(set(sd) - used - set(k for k in simple if k in sd))
    if left: raise KeyError("unmapped tensors: " + ", ".join(left[:8]))
    return out
