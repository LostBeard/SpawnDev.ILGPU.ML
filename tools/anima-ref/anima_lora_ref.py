"""Anima LoRA (PEFT layout, e.g. circlestone-labs anima-turbo-lora-v0.2) as a weight-getter wrapper for anima_ref.py.

File layout: diffusion_model.<module>.lora_A.weight [r, in], .lora_B.weight [out, r], no alpha tensors (rank 32). <module> = the DiT's
blocks.N.{self_attn,cross_attn}.{q,k,v,output}_proj, blocks.N.mlp.layer{1,2}, blocks.N.adaln_modulation_*.{1,2}, and the LLM adapter's
llm_adapter.blocks.N.{self_attn,cross_attn}.{q,k,v,o}_proj, mlp.{0,2}; the reference names are net.<module>.weight.

Patched weight W' = W + strength * scale * (B @ A). Which scale and whether ComfyUI keeps W' exact or re-rounds it to the stored bf16 are what
check_lora.py decides against Lore's dump (modes: 'exact' = float W', 'bf16' = W' rounded to bf16 nearest-even).
"""
import numpy as np


def to_bf16(x):
    """Round float64 -> bf16 (nearest even, through float32) -> float64."""
    f = np.asarray(x, dtype=np.float32)
    u = f.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32).astype(np.float64)


class LoraWeights:
    """w(name) for anima_ref with the LoRA merged: base getter + LoRA SafeTensors; strength, scale, mode ('exact' | 'bf16')."""

    def __init__(self, base, lora_st, strength=1.0, scale=1.0, mode='exact', prefix='diffusion_model.'):
        self.base, self.st, self.strength, self.scale, self.mode = base, lora_st, strength, scale, mode
        self.mods = {}
        for k in lora_st.header:
            if k.endswith('.lora_A.weight'):
                m = k[:-len('.lora_A.weight')]
                net = 'net.' + (m[len(prefix):] if m.startswith(prefix) else m) + '.weight'
                self.mods[net] = m
        self.cache = {}

    def covers(self, name):
        return name in self.mods

    def __call__(self, name):
        v = self.cache.get(name)
        if v is not None:
            return v
        W = self.base(name)
        m = self.mods.get(name)
        if m is not None and self.strength != 0.0:
            A = self.st.get(m + '.lora_A.weight')
            B = self.st.get(m + '.lora_B.weight')
            assert A.shape[1] == W.shape[1] and B.shape[0] == W.shape[0] and A.shape[0] == B.shape[1], (name, W.shape, A.shape, B.shape)
            W = W + (self.strength * self.scale) * (B @ A)
            if self.mode == 'bf16':
                W = to_bf16(W)
        self.cache[name] = W
        return W

    def drop(self, prefix):
        for k in [k for k in self.cache if k.startswith(prefix)]:
            del self.cache[k]
        if hasattr(self.base, 'drop'):
            self.base.drop(prefix)
