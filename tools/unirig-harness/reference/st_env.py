"""Import environment for running the REAL SkinTokens / TokenRig classes (github.com/VAST-AI-Research/SkinTokens, MIT) as a black box for goldens.

  ST_DIR   working folder (default E:/spawndev-unirig/skintokens): ref/ = git clone, ckpt/experiments/... = the two Lightning checkpoints, Qwen3-0.6B/ = config only

Run with E:/spawndev-unirig/venv-cuda (torch + CUDA, transformers 5.x, diffusers) as `python -I`. What differs from the shipped code path, and why:
  - the reference is built with attn_implementation "sdpa" (it hard-codes flash_attention_2, which has no Windows wheel); the flash_attn modules it imports at the top
    are minimal stand-ins (softmax attention, no mask) so the imports resolve;
  - the VAE checkpoint is read with the allowlist unpickler of safe_ckpt.py instead of torch.load(weights_only=False);
  - bpy / open3d / fast_simplification (mesh IO the golden does not use) are mocks; a mock hides a call, so nothing here relies on them returning anything;
  - the golden runs the model in float32 (the checkpoint is bf16; bf16 -> fp32 is exact, so the weights are identical): the port keeps bf16 weights with fp32
    activations and must match THAT, not bf16 activation noise.
"""
import os, sys, types
from unittest.mock import MagicMock

ST_DIR = os.environ.get('ST_DIR', 'E:/spawndev-unirig/skintokens')
REF = os.path.join(ST_DIR, 'ref')
CKPT = os.path.join(ST_DIR, 'ckpt')
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('HF_HOME', 'E:/spawndev-unirig/hf')

import importlib.machinery
for name in ('bpy', 'open3d', 'fast_simplification', 'gradio', 'bottle', 'tornado'):
    m = MagicMock(); m.__spec__ = importlib.machinery.ModuleSpec(name, None)       # diffusers probes optional packages with importlib.util.find_spec
    sys.modules.setdefault(name, m)

import torch                                                    # noqa: E402
import torch.nn.functional as F                                  # noqa: E402


def _flash_attn_func(q, k, v, softmax_scale=None, causal=False, **kw):
    """[B, S, H, D] softmax attention (the flash_attn layout)."""
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=softmax_scale, is_causal=causal)
    return o.transpose(1, 2)


import diffusers, transformers                                # noqa: E402,F401  (they probe optional packages at import: let them see the truth BEFORE the stand-ins below exist)
_fa_if = types.ModuleType('flash_attn_interface'); _fa_if.flash_attn_func = lambda q, k, v, **kw: (_flash_attn_func(q, k, v, **kw), None)
_fa = types.ModuleType('flash_attn'); _fa_mod = types.ModuleType('flash_attn.flash_attn_interface'); _fa_mod.flash_attn_func = _flash_attn_func
_fa.flash_attn_interface = _fa_mod
for _m in (_fa_if, _fa, _fa_mod):
    _m.__spec__ = importlib.machinery.ModuleSpec(_m.__name__, None)
sys.modules.update({'flash_attn_interface': _fa_if, 'flash_attn': _fa, 'flash_attn.flash_attn_interface': _fa_mod})

if REF not in sys.path:
    sys.path.insert(0, REF)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.chdir(REF)                                                   # configs/skeleton/*.yaml and models/Qwen3-0.6B are relative to the repo root

# the Qwen3 config the reference looks for under models/ (config files only; the weights are in the TokenRig checkpoint)
_q = os.path.join(REF, 'models', 'Qwen3-0.6B')
if not os.path.isdir(_q):
    import shutil
    shutil.copytree(os.path.join(ST_DIR, 'Qwen3-0.6B'), _q)

from transformers import AutoModelForCausalLM                    # noqa: E402
_from_config = AutoModelForCausalLM.from_config.__func__
AutoModelForCausalLM.from_config = classmethod(lambda cls, config, **kw: _from_config(cls, config, **{**kw, 'attn_implementation': 'sdpa'}))

from safe_ckpt import load_checkpoint                            # noqa: E402
import src.model.spec as _spec                                   # noqa: E402


def _safe_load_from_system_checkpoint(cls, checkpoint_path, strict=True, **kwargs):
    path = checkpoint_path if os.path.isabs(checkpoint_path) else os.path.join(CKPT, checkpoint_path)
    ckpt = load_checkpoint(path)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['state_dict'].items()}
    sd = {(k[len('model.'):] if k.startswith('model.') else k): v for k, v in sd.items()}
    hp = ckpt['hyper_parameters']
    model = cls(model_config=kwargs.get('model_config') or hp['model_config'], transform_config=kwargs.get('transform_config') or hp['transform_config'],
                tokenizer_config=kwargs.get('tokenizer_config') or hp.get('tokenizer_config'))
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing or unexpected:
        print('[warning] VAE load: missing', missing, 'unexpected', unexpected)
    return model


_spec.ModelSpec.load_from_system_checkpoint = classmethod(_safe_load_from_system_checkpoint)


def build_tokenrig(device='cuda'):
    """The real TokenRig with the shipped 'grpo_1400' weights, in float32 on `device`, eval mode. Returns (model, hyper_parameters)."""
    from src.model.tokenrig import TokenRig
    ck = load_checkpoint(os.path.join(CKPT, 'experiments', 'articulation_xl_quantization_256_token_4', 'grpo_1400.ckpt'))
    hp = ck['hyper_parameters']
    mc = dict(hp['model_config']); mc['mesh_encoder'] = dict(mc['mesh_encoder']); mc['mesh_encoder']['flash'] = False
    model = TokenRig(model_config=mc, transform_config=hp['transform_config'], tokenizer_config=hp['tokenizer_config'])
    sd = {k.replace('_orig_mod.', ''): v for k, v in ck['state_dict'].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print('TokenRig state dict: missing', len(missing), missing[:4], 'unexpected', len(unexpected), unexpected[:4])
    model = model.float().to(device).eval()
    # nn.RMSNorm(eps=None) uses finfo(input dtype).eps. The shipped model is bf16 (eps 0.0078125); the float32 golden would silently use 1.19e-7 instead and the mesh prefix would
    # differ by a fraction of a percent. Pin the shipped value so the golden is "the shipped computation at higher activation precision".
    model.output_proj[1].eps = torch.finfo(torch.bfloat16).eps
    return model, hp
