"""Reference for the training milestone M1 (Plans/training-autograd-lora-2026-11-04.md).

Builds a tiny llama (f32 weights, GQA) in numpy, writes it as a GGUF in llama.cpp's layout, and computes in float64 torch:
logits at every position, a masked next-token cross-entropy loss, the gradients of LoRA adapters on q_proj and v_proj
(PEFT semantics: y = W x + (alpha / r) * B (A x)), the gradient of the loss w.r.t. the token embeddings' rows, and one
AdamW step of the adapters. The engine's training path must reproduce these from the GGUF alone.

llama.cpp layout notes that the reference mirrors (the engine's GGUFGraphBuilder reads the same):
- a GGUF tensor's ne is the numpy shape reversed: a Linear weight stored as numpy (out, in) has ne = [in, out];
- q/k weights are stored PERMUTED for llama.cpp's rope, which rotates ADJACENT pairs (x[2i], x[2i+1]), so the reference
  rotates adjacent pairs of the projections exactly as stored (no un-permute);
- RMSNorm: x * rsqrt(mean(x^2) + eps) * w; SwiGLU MLP: down(silu(gate x) * up x); causal softmax(q k^T / sqrt(hd)).

Usage: py -3.13 tools/gen_tiny_llama_train_reference.py   (writes into SpawnDev.ILGPU.ML.Demo/wwwroot/test-train/)
"""
import json
import math
import os
import sys

import numpy as np
import torch
import gguf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "SpawnDev.ILGPU.ML.Demo", "wwwroot", "test-train")

VOCAB, EMBD, LAYERS, HEADS, KV_HEADS, FFN, CTX = 64, 64, 2, 4, 2, 96, 32
HD = EMBD // HEADS
EPS, ROPE_BASE = 1e-5, 10000.0
RANK, ALPHA = 4, 8.0
SCALE = ALPHA / RANK
LR, BETA1, BETA2, ADAM_EPS, WD = 1e-2, 0.9, 0.999, 1e-8, 0.01
SEQ = 12
TOKENS = [5, 17, 3, 42, 8, 8, 61, 0, 23, 37, 11, 50]
LOSS_FROM = 6          # positions LOSS_FROM .. SEQ-2 predict tokens LOSS_FROM+1 .. SEQ-1 (the "completion")


def main():
    rng = np.random.default_rng(20261104)

    def w(*shape, scale=0.1, center=0.0):
        return (center + scale * rng.standard_normal(shape)).astype(np.float32)

    weights = {"token_embd.weight": w(VOCAB, EMBD, scale=0.5)}
    for L in range(LAYERS):
        p = f"blk.{L}"
        weights[f"{p}.attn_norm.weight"] = w(EMBD, scale=0.1, center=1.0)
        weights[f"{p}.attn_q.weight"] = w(HEADS * HD, EMBD, scale=0.15)
        weights[f"{p}.attn_k.weight"] = w(KV_HEADS * HD, EMBD, scale=0.15)
        weights[f"{p}.attn_v.weight"] = w(KV_HEADS * HD, EMBD, scale=0.15)
        weights[f"{p}.attn_output.weight"] = w(EMBD, HEADS * HD, scale=0.15)
        weights[f"{p}.ffn_norm.weight"] = w(EMBD, scale=0.1, center=1.0)
        weights[f"{p}.ffn_gate.weight"] = w(FFN, EMBD, scale=0.15)
        weights[f"{p}.ffn_up.weight"] = w(FFN, EMBD, scale=0.15)
        weights[f"{p}.ffn_down.weight"] = w(EMBD, FFN, scale=0.15)
    weights["output_norm.weight"] = w(EMBD, scale=0.1, center=1.0)
    weights["output.weight"] = w(VOCAB, EMBD, scale=0.2)

    # LoRA adapters (PEFT names). B is NONZERO here so every gradient is exercised (PEFT initialises B to zero).
    lora = {}
    for L in range(LAYERS):
        for proj, out_dim in (("q_proj", HEADS * HD), ("v_proj", KV_HEADS * HD)):
            base = f"base_model.model.model.layers.{L}.self_attn.{proj}"
            lora[f"{base}.lora_A.weight"] = w(RANK, EMBD, scale=0.2)
            lora[f"{base}.lora_B.weight"] = w(out_dim, RANK, scale=0.2)

    os.makedirs(OUT, exist_ok=True)
    gguf_path = os.path.join(OUT, "tiny-llama-train.gguf")
    wr = gguf.GGUFWriter(gguf_path, "llama")
    wr.add_name("tiny-llama-train")
    wr.add_context_length(CTX)
    wr.add_embedding_length(EMBD)
    wr.add_block_count(LAYERS)
    wr.add_feed_forward_length(FFN)
    wr.add_head_count(HEADS)
    wr.add_head_count_kv(KV_HEADS)
    wr.add_layer_norm_rms_eps(EPS)
    wr.add_rope_freq_base(ROPE_BASE)
    wr.add_vocab_size(VOCAB)
    wr.add_tokenizer_model("llama")
    wr.add_token_list([f"<t{i}>".encode() for i in range(VOCAB)])
    wr.add_token_scores([0.0] * VOCAB)
    wr.add_token_types([1] * VOCAB)
    for name, arr in weights.items():
        wr.add_tensor(name, arr)
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_tensors_to_file()
    wr.close()

    # ── float64 torch reference ──
    T = {k: torch.tensor(v, dtype=torch.float64) for k, v in weights.items()}
    P = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in lora.items()}
    ids = torch.tensor(TOKENS)
    emb = T["token_embd.weight"][ids].clone().requires_grad_(True)   # [SEQ, EMBD]

    def rms(x, wt):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * wt

    def rope(x):  # x [SEQ, H, HD], adjacent pairs
        pos = torch.arange(SEQ, dtype=torch.float64)[:, None]
        inv = ROPE_BASE ** (-torch.arange(0, HD, 2, dtype=torch.float64) / HD)
        ang = pos * inv[None, :]                       # [SEQ, HD/2]
        c, s = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
        x0, x1 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x0 * c - x1 * s
        out[..., 1::2] = x0 * s + x1 * c
        return out

    def lin(x, name, lora_base=None):
        y = x @ T[name].T
        if lora_base is not None:
            y = y + SCALE * ((x @ P[f"{lora_base}.lora_A.weight"].T) @ P[f"{lora_base}.lora_B.weight"].T)
        return y

    def forward(h, use_lora):
        mask = torch.full((SEQ, SEQ), float("-inf"), dtype=torch.float64).triu(1)
        for L in range(LAYERS):
            p, lb = f"blk.{L}", f"base_model.model.model.layers.{L}.self_attn"
            x = rms(h, T[f"{p}.attn_norm.weight"])
            q = rope(lin(x, f"{p}.attn_q.weight", f"{lb}.q_proj" if use_lora else None).view(SEQ, HEADS, HD))
            k = rope(lin(x, f"{p}.attn_k.weight").view(SEQ, KV_HEADS, HD))
            v = lin(x, f"{p}.attn_v.weight", f"{lb}.v_proj" if use_lora else None).view(SEQ, KV_HEADS, HD)
            rep = HEADS // KV_HEADS
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
            att = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(HD) + mask
            att = torch.softmax(att, dim=-1)
            o = torch.einsum("hqk,khd->qhd", att, v).reshape(SEQ, HEADS * HD)
            h = h + lin(o, f"{p}.attn_output.weight")
            x = rms(h, T[f"{p}.ffn_norm.weight"])
            g = lin(x, f"{p}.ffn_gate.weight")
            u = lin(x, f"{p}.ffn_up.weight")
            h = h + lin(torch.nn.functional.silu(g) * u, f"{p}.ffn_down.weight")
        return rms(h, T["output_norm.weight"]) @ T["output.weight"].T      # [SEQ, VOCAB]

    with torch.no_grad():
        logits_base = forward(T["token_embd.weight"][ids], use_lora=False)
    logits = forward(emb, use_lora=True)
    tgt = torch.tensor(TOKENS[LOSS_FROM + 1:])
    loss = torch.nn.functional.cross_entropy(logits[LOSS_FROM:SEQ - 1], tgt)   # mean over the completion tokens
    loss.backward()

    # one AdamW step (torch.optim.AdamW semantics, step 1)
    after = {}
    for k, prm in P.items():
        g = prm.grad
        m = (1 - BETA1) * g
        v2 = (1 - BETA2) * g * g
        mhat, vhat = m / (1 - BETA1), v2 / (1 - BETA2)
        upd = prm.detach() * (1 - LR * WD) - LR * mhat / (vhat.sqrt() + ADAM_EPS)
        after[k] = upd
    # cross-check against torch's own AdamW
    chk = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in lora.items()}
    opt = torch.optim.AdamW(chk.values(), lr=LR, betas=(BETA1, BETA2), eps=ADAM_EPS, weight_decay=WD)
    for k in chk:
        chk[k].grad = P[k].grad.clone()
    opt.step()
    worst = max((chk[k].detach() - after[k]).abs().max().item() for k in chk)
    assert worst < 1e-12, f"hand AdamW disagrees with torch.optim.AdamW: {worst}"

    def flat(t):
        return [float(x) for x in t.detach().reshape(-1).tolist()]

    ref = {
        "about": "tools/gen_tiny_llama_train_reference.py; float64 torch; LoRA PEFT semantics y = Wx + (alpha/r) B A x",
        "config": {"vocab": VOCAB, "embd": EMBD, "layers": LAYERS, "heads": HEADS, "kv_heads": KV_HEADS, "ffn": FFN,
                   "rms_eps": EPS, "rope_base": ROPE_BASE, "rank": RANK, "alpha": ALPHA,
                   "lr": LR, "beta1": BETA1, "beta2": BETA2, "adam_eps": ADAM_EPS, "weight_decay": WD},
        "tokens": TOKENS,
        "loss_from": LOSS_FROM,
        "targets": TOKENS[LOSS_FROM + 1:],
        "logits": flat(logits),
        "logits_base": flat(logits_base),
        "loss": loss.item(),
        "grad_embeddings": flat(emb.grad),
        "lora": {k: {"shape": list(v.shape), "value": [float(x) for x in v.reshape(-1)],
                     "grad": flat(P[k].grad), "after_adamw": flat(after[k])} for k, v in lora.items()},
    }
    with open(os.path.join(OUT, "tiny-llama-train-ref.json"), "w", encoding="utf-8") as f:
        json.dump(ref, f)
    print(f"wrote {gguf_path} ({os.path.getsize(gguf_path)} bytes); loss {loss.item():.6f}; "
          f"|grad A0q| {P['base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'].grad.abs().max().item():.3e}")


if __name__ == "__main__":
    sys.exit(main())
