"""Export FLUX.2 klein 4B (black-forest-labs/FLUX.2-klein-4B, Apache-2.0) for the C# port.

  python -I export_flux2.py <klein4b_dir> <out_dir> [encoder|transformer|vae|all]

Writes bf16 safetensors, one file per stage so a browser loads (and frees) one at a time:

  text_encoder.safetensors   Qwen3-4B up to layer 27 (the pipeline only reads hidden states 9, 18 and 27; the last 9 layers and the final norm never run) + the embedding table
                             layers.{l}.self_attn.qkv_proj.weight   [4096 + 1024 + 1024, 2560]   rows q | k | v (head-major, as the checkpoint stores them)
                             layers.{l}.mlp.gate_up_proj.weight     [2 * 9728, 2560]             rows gate | up
                             the rest keep their names (input_layernorm, post_attention_layernorm, q_norm, k_norm, o_proj, down_proj); embed_tokens.weight
  transformer.safetensors    the 3.9B flow DiT: per double block attn.qkv [9216, 3072] (to_q|to_k|to_v) and attn.add_qkv (add_q|add_k|add_v) fused, everything else under its own name
  vae_decoder.onnx           (export_flux2_vae.py) the decoder + post_quant_conv as ONNX, so the library's existing ONNX path (as for SD Turbo's VAE) runs it

Fusing only concatenates rows, so the maths is the checkpoint's own. Metadata records the layout.
"""
import os, sys, json
import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, dst = sys.argv[1], sys.argv[2]
what = sys.argv[3] if len(sys.argv) > 3 else 'all'
os.makedirs(dst, exist_ok=True)
ENC_LAYERS = 27


def open_all(folder):
    import glob
    files = sorted(glob.glob(os.path.join(folder, '*.safetensors')))
    t = {}
    for f in files:
        h = safe_open(f, framework='pt')
        for k in h.keys():
            t[k] = (h, k)
    return t


def get(t, k):
    h, key = t[k]
    return h.get_tensor(key)


def export_encoder():
    t = open_all(os.path.join(src, 'text_encoder'))
    out = {}
    out['embed_tokens.weight'] = get(t, 'model.embed_tokens.weight').contiguous()
    for l in range(ENC_LAYERS):
        p = f'model.layers.{l}.'
        a = p + 'self_attn.'
        out[f'layers.{l}.self_attn.qkv_proj.weight'] = torch.cat([get(t, a + 'q_proj.weight'), get(t, a + 'k_proj.weight'), get(t, a + 'v_proj.weight')], 0).contiguous()
        for n in ('q_norm', 'k_norm'):
            out[f'layers.{l}.self_attn.{n}.weight'] = get(t, a + n + '.weight').contiguous()
        out[f'layers.{l}.self_attn.o_proj.weight'] = get(t, a + 'o_proj.weight').contiguous()
        out[f'layers.{l}.mlp.gate_up_proj.weight'] = torch.cat([get(t, p + 'mlp.gate_proj.weight'), get(t, p + 'mlp.up_proj.weight')], 0).contiguous()
        out[f'layers.{l}.mlp.down_proj.weight'] = get(t, p + 'mlp.down_proj.weight').contiguous()
        out[f'layers.{l}.input_layernorm.weight'] = get(t, p + 'input_layernorm.weight').contiguous()
        out[f'layers.{l}.post_attention_layernorm.weight'] = get(t, p + 'post_attention_layernorm.weight').contiguous()
    path = os.path.join(dst, 'text_encoder.safetensors')
    save_file(out, path, metadata={'source': 'black-forest-labs/FLUX.2-klein-4B text_encoder (Qwen3-4B)', 'dtype': 'bf16', 'layers': str(ENC_LAYERS),
                                   'qkv_rows': 'q|k|v', 'gate_up_rows': 'gate|up'})
    print(path, round(os.path.getsize(path) / 1e9, 2), 'GB', len(out), 'tensors')


def export_transformer():
    t = open_all(os.path.join(src, 'transformer'))
    out = {}
    cfg = json.load(open(os.path.join(src, 'transformer', 'config.json')))
    for l in range(cfg['num_layers']):
        p = f'transformer_blocks.{l}.'
        a = p + 'attn.'
        out[a + 'qkv.weight'] = torch.cat([get(t, a + 'to_q.weight'), get(t, a + 'to_k.weight'), get(t, a + 'to_v.weight')], 0).contiguous()
        out[a + 'add_qkv.weight'] = torch.cat([get(t, a + 'add_q_proj.weight'), get(t, a + 'add_k_proj.weight'), get(t, a + 'add_v_proj.weight')], 0).contiguous()
        for n in ('norm_q', 'norm_k', 'norm_added_q', 'norm_added_k'):
            out[a + n + '.weight'] = get(t, a + n + '.weight').contiguous()
        out[a + 'to_out.weight'] = get(t, a + 'to_out.0.weight').contiguous()
        out[a + 'to_add_out.weight'] = get(t, a + 'to_add_out.weight').contiguous()
        for ff in ('ff', 'ff_context'):
            out[p + ff + '.linear_in.weight'] = get(t, p + ff + '.linear_in.weight').contiguous()
            out[p + ff + '.linear_out.weight'] = get(t, p + ff + '.linear_out.weight').contiguous()
    for l in range(cfg['num_single_layers']):
        p = f'single_transformer_blocks.{l}.'
        for n in ('attn.to_qkv_mlp_proj.weight', 'attn.to_out.weight', 'attn.norm_q.weight', 'attn.norm_k.weight'):
            out[p + n] = get(t, p + n).contiguous()
    for n in ('context_embedder.weight', 'x_embedder.weight', 'proj_out.weight', 'norm_out.linear.weight',
              'double_stream_modulation_img.linear.weight', 'double_stream_modulation_txt.linear.weight', 'single_stream_modulation.linear.weight',
              'time_guidance_embed.timestep_embedder.linear_1.weight', 'time_guidance_embed.timestep_embedder.linear_2.weight'):
        out[n] = get(t, n).contiguous()
    path = os.path.join(dst, 'transformer.safetensors')
    save_file(out, path, metadata={'source': 'black-forest-labs/FLUX.2-klein-4B transformer', 'dtype': 'bf16', 'qkv_rows': 'q|k|v',
                                   'config': json.dumps(cfg)})
    print(path, round(os.path.getsize(path) / 1e9, 2), 'GB', len(out), 'tensors')


if what in ('encoder', 'all'): export_encoder()
if what in ('transformer', 'all'): export_transformer()
