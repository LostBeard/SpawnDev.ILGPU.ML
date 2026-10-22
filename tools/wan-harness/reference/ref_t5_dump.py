"""Wan 2.2 reference: the umT5-XXL text embeddings, dumped as fixtures for the C# port (stage 2 of Plans/wan22-ti2v-5b-2026-10-22.md).

Runs in WSL (venv ~/wan/venv layered on the trellis2 env), against the reference code in ~/wan/Wan2.2 and the checkpoint in ~/wan/ckpt.
The host has 15.7 GB of RAM and WSL gets 7.8 GB, the encoder is 11.4 GB in bf16, so it is built on the meta device and its weights are
memory-mapped (never read into RAM); the 2.1 GB token-embedding table stays on the CPU (a lookup of at most 512 rows), the 24 layers run on the GPU.

  python ref_t5_dump.py OUT_DIR "a prompt" ["another prompt" ...]     (the default negative prompt is always dumped as `neg`)
Writes OUT_DIR/t5_<i>.npz with ids [L] int64, mask [L], context [n, 4096] float32 (only the n real tokens, what WanModel receives) and the text.
"""
import os
import sys

import numpy as np
import torch

WAN = os.path.expanduser("~/wan/Wan2.2")
CKPT = os.path.expanduser("~/wan/ckpt")
sys.path.insert(0, WAN)

from wan.configs.shared_config import wan_shared_cfg  # noqa: E402
from wan.modules.t5 import T5Encoder  # noqa: E402
from wan.modules.tokenizers import HuggingfaceTokenizer  # noqa: E402


def build_encoder():
    """umT5-XXL encoder with mmap-backed weights; everything but the token embedding moved to the GPU in bf16."""
    cfg = dict(vocab=256384, dim=4096, dim_attn=4096, dim_ffn=10240, num_heads=64, num_layers=24, num_buckets=32, shared_pos=False, dropout=0.1)
    with torch.device("meta"):
        enc = T5Encoder(**cfg)
    sd = torch.load(os.path.join(CKPT, "models_t5_umt5-xxl-enc-bf16.pth"), map_location="cpu", mmap=True, weights_only=True)
    missing, unexpected = enc.load_state_dict(sd, strict=False, assign=True)
    print("load_state_dict missing", missing[:5], "unexpected", unexpected[:5])
    enc.eval().requires_grad_(False)
    # token embedding: stays on the CPU (mmap). The rest to the GPU one tensor at a time (bf16 already).
    for name, module in enc.named_children():
        if name != "token_embedding":
            module.to("cuda")
    return enc


@torch.no_grad()
def encode(enc, tok, texts):
    ids, mask = tok(texts, return_mask=True, add_special_tokens=True)
    out = []
    for i in range(ids.shape[0]):
        x = enc.token_embedding(ids[i:i + 1]).to("cuda")                 # CPU lookup, then the layers on the GPU
        m = mask[i:i + 1].to("cuda")
        for block in enc.blocks:
            x = block(x, m, pos_bias=None)
        x = enc.norm(x)
        n = int(mask[i].gt(0).sum())
        out.append((ids[i].numpy(), mask[i].numpy(), x[0, :n].float().cpu().numpy()))
    return out


def main():
    out_dir = sys.argv[1]
    prompts = sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    tok = HuggingfaceTokenizer(name=os.path.join(CKPT, "google", "umt5-xxl"), seq_len=wan_shared_cfg.text_len, clean="whitespace")
    enc = build_encoder()
    texts = list(prompts) + [wan_shared_cfg.sample_neg_prompt]
    names = [f"{i}" for i in range(len(prompts))] + ["neg"]
    for name, text, (ids, mask, ctx) in zip(names, texts, encode(enc, tok, texts)):
        np.savez(os.path.join(out_dir, f"t5_{name}.npz"), ids=ids, mask=mask, context=ctx, text=np.array(text))
        print(f"t5_{name}: {int(mask.sum())} tokens, context {ctx.shape}, mean {ctx.mean():.5f}, std {ctx.std():.5f}, absmax {np.abs(ctx).max():.3f}")


if __name__ == "__main__":
    main()
