"""Training milestone M2 reference (Plans/training-autograd-lora-2026-11-04.md): a LoRA fine-tune of a REAL llama-architecture GGUF
in float64 torch, for tools/TrainLoraCurve to reproduce step by step.

Reads the GGUF itself (gguf.GGUFReader; F32/F16/BF16 tensors), builds the llama.cpp-layout forward the engine's GGUFGraphBuilder
builds (adjacent-pair rope on the stored, permuted q/k; GQA; RMSNorm; SwiGLU; tied LM head when there is no output.weight), adds PEFT
LoRA adapters (A Kaiming-uniform bound 1/sqrt(in), B zero, scale alpha/r) on the target projections, and trains with
torch.optim.AdamW on a toy task: every sequence is a random prompt followed by the SAME answer tokens, loss on the answer only. Writes
the sequences, the initial adapters, the loss at every step and the final adapters.

Usage: py -3.13 tools/gen_lora_curve_reference.py <model.gguf> <out.json> [steps=12] [rank=8] [alpha=16] [lr=2e-3]
"""
import json
import math
import sys

import numpy as np
import torch
from gguf import GGUFReader


def field(r, key, default=None):
    f = r.fields.get(key)
    if f is None:
        return default
    v = f.parts[f.data[0]]
    return v[0] if hasattr(v, "__len__") and len(v) == 1 else v


def main(path, out, steps=12, rank=8, alpha=16.0, lr=2e-3):
    torch.manual_seed(1104)
    rng = np.random.default_rng(1104)
    r = GGUFReader(path)
    arch = bytes(r.fields["general.architecture"].parts[r.fields["general.architecture"].data[0]]).decode()
    assert arch == "llama", arch
    n_layers = int(field(r, "llama.block_count"))
    embd = int(field(r, "llama.embedding_length"))
    heads = int(field(r, "llama.attention.head_count"))
    kv_heads = int(field(r, "llama.attention.head_count_kv", heads))
    eps = float(field(r, "llama.attention.layer_norm_rms_epsilon", 1e-5))
    rope_base = float(field(r, "llama.rope.freq_base", 10000.0))
    hd = embd // heads
    W = {}
    for t in r.tensors:
        if t.tensor_type.name not in ("F32", "F16", "BF16"):
            raise SystemExit(f"{t.name}: {t.tensor_type.name} not supported by this reference (use an F16/F32 GGUF)")
        a = np.asarray(t.data)
        if t.tensor_type.name == "BF16":
            a = (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        W[t.name] = torch.tensor(np.array(a, dtype=np.float64).reshape([int(x) for x in reversed(t.shape)]))
    vocab = W["token_embd.weight"].shape[0]
    lm_head = W.get("output.weight", W["token_embd.weight"])

    targets = {"attn_q": "self_attn.q_proj", "attn_v": "self_attn.v_proj"}
    lora = {}
    for L in range(n_layers):
        for g, mod in targets.items():
            w = W[f"blk.{L}.{g}.weight"]
            out_dim, in_dim = w.shape
            base = f"base_model.model.model.layers.{L}.{mod}"
            bound = 1.0 / math.sqrt(in_dim)
            lora[f"{base}.lora_A.weight"] = torch.tensor(rng.uniform(-bound, bound, (rank, in_dim)).astype(np.float32), dtype=torch.float64, requires_grad=True)
            lora[f"{base}.lora_B.weight"] = torch.zeros((out_dim, rank), dtype=torch.float64, requires_grad=True)
    initial = {k: v.detach().clone() for k, v in lora.items()}
    scale = alpha / rank

    def rms(x, w):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w

    def rope(x, seq):
        pos = torch.arange(seq, dtype=torch.float64)[:, None]
        inv = rope_base ** (-torch.arange(0, hd, 2, dtype=torch.float64) / hd)
        ang = pos * inv[None, :]
        c, s = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
        x0, x1 = x[..., 0::2], x[..., 1::2]
        o = torch.empty_like(x)
        o[..., 0::2] = x0 * c - x1 * s
        o[..., 1::2] = x0 * s + x1 * c
        return o

    def lin(x, name, peft=None):
        y = x @ W[name].T
        if peft is not None:
            y = y + scale * ((x @ lora[f"{peft}.lora_A.weight"].T) @ lora[f"{peft}.lora_B.weight"].T)
        return y

    def forward(ids):
        seq = len(ids)
        h = W["token_embd.weight"][torch.tensor(ids)]
        mask = torch.full((seq, seq), float("-inf"), dtype=torch.float64).triu(1)
        rep = heads // kv_heads
        for L in range(n_layers):
            p = f"blk.{L}"
            pb = f"base_model.model.model.layers.{L}"
            x = rms(h, W[f"{p}.attn_norm.weight"])
            q = rope(lin(x, f"{p}.attn_q.weight", f"{pb}.self_attn.q_proj").view(seq, heads, hd), seq)
            k = rope(lin(x, f"{p}.attn_k.weight").view(seq, kv_heads, hd), seq).repeat_interleave(rep, dim=1)
            v = lin(x, f"{p}.attn_v.weight", f"{pb}.self_attn.v_proj").view(seq, kv_heads, hd).repeat_interleave(rep, dim=1)
            att = torch.softmax(torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(hd) + mask, dim=-1)
            h = h + lin(torch.einsum("hqk,khd->qhd", att, v).reshape(seq, heads * hd), f"{p}.attn_output.weight")
            x = rms(h, W[f"{p}.ffn_norm.weight"])
            h = h + lin(torch.nn.functional.silu(lin(x, f"{p}.ffn_gate.weight")) * lin(x, f"{p}.ffn_up.weight"), f"{p}.ffn_down.weight")
        return rms(h, W["output_norm.weight"]) @ lm_head.T

    answer = [int(t) for t in rng.integers(100, vocab - 100, 6)]
    seqs = []
    for _ in range(4):
        prompt = [int(t) for t in rng.integers(100, vocab - 100, 12)]
        ids = prompt + answer
        labels = [-100] * len(ids)
        for t in range(len(prompt) - 1, len(ids) - 1):
            labels[t] = ids[t + 1]
        seqs.append({"input_ids": ids, "labels": labels})

    opt = torch.optim.AdamW(lora.values(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    losses = []
    for step in range(steps):
        s = seqs[step % len(seqs)]
        logits = forward(s["input_ids"])
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor(s["labels"]), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        print(f"step {step + 1}: loss {loss.item():.6f}", flush=True)

    def flat(t):
        return [float(x) for x in t.detach().reshape(-1).tolist()]

    json.dump({
        "about": "tools/gen_lora_curve_reference.py; float64 torch, torch.optim.AdamW, PEFT LoRA on q/v",
        "model": path, "config": {"rank": rank, "alpha": alpha, "lr": lr, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.01,
                                  "targets": list(targets)},
        "sequences": seqs, "losses": losses,
        "initial": {k: flat(v) for k, v in initial.items()},
        "final": {k: flat(v) for k, v in lora.items()},
    }, open(out, "w", encoding="utf-8"))


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], *(int(x) for x in a[3:5]), *(float(x) for x in a[5:7])) if len(a) > 3 else main(a[1], a[2])
