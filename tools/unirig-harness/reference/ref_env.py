"""Import environment for running the REAL UniRig reference classes on CPU, as a black box for goldens.

  UNIRIG_DIR   working folder (default E:/spawndev-unirig: ckpt/, ref/ = git clone of VAST-AI-Research/UniRig, venv/, golden/, weights/)

The reference imports packages that only exist with CUDA (flash_attn, spconv, torch_scatter) or Blender (bpy). The skeleton stage never runs them, but
its modules import them at the top, so they are replaced by MagicMock modules. A mock hides a call to it, so every config handed to the model here
switches the CUDA attention paths OFF (flash False, attention implementation eager) - never rely on the mocks to "work".
Run with `python -I` (isolated): this file adds the paths it needs itself.
"""
import os, sys
from unittest.mock import MagicMock

UNIRIG_DIR = os.environ.get('UNIRIG_DIR', 'E:/spawndev-unirig')
REF = os.path.join(UNIRIG_DIR, 'ref')
os.environ.setdefault('HF_HOME', os.path.join(UNIRIG_DIR, 'hf'))             # keep every cache off C: and D:
os.environ.setdefault('TRANSFORMERS_CACHE', os.path.join(UNIRIG_DIR, 'hf'))

for name in ('spconv', 'spconv.pytorch', 'torch_scatter', 'torch_cluster', 'bpy', 'open3d', 'pyrender', 'fast_simplification', 'wandb', 'ocnn'):
    sys.modules.setdefault(name, MagicMock())
if REF not in sys.path:
    sys.path.insert(0, REF)
os.chdir(REF)        # the tokenizer config names its skeleton orders relative to the repo root (./configs/skeleton/*.yaml)

import yaml                                  # noqa: E402
from box import Box                          # noqa: E402


def load_yaml(rel):
    with open(os.path.join(REF, rel), encoding='utf-8') as f:
        return Box(yaml.safe_load(f))


FPS_CALLS = []     # (input points [N,3], chosen indices) of every fps() the encoder made, for the golden


def install_fps():
    """torch_cluster.fps has no CPU build. The perceiver encoder uses it (eval: random_start=False) to pick its query tokens from 4096 pre-sampled points.
    Replaced here by plain farthest-point sampling with torch_cluster's documented semantics: ratio -> ceil(ratio * N) samples, the first sample is
    point 0 when random_start is False, then repeatedly the point farthest from everything chosen so far. Inputs and outputs are recorded in FPS_CALLS.
    One batch only (the golden runs one mesh)."""
    import math, torch
    import src.model.michelangelo.models.tsal.sal_perceiver as sp

    def fps(pos, batch=None, ratio=0.5, random_start=True):
        assert not random_start, 'the eval path does not use a random start'
        assert batch is None or int(batch.max()) == 0, 'one batch only'
        n = pos.shape[0]; k = int(math.ceil(ratio * n))
        idx = torch.empty(k, dtype=torch.long); idx[0] = 0
        d = ((pos - pos[0]) ** 2).sum(1)
        for i in range(1, k):
            j = int(torch.argmax(d)); idx[i] = j
            d = torch.minimum(d, ((pos - pos[j]) ** 2).sum(1))
        FPS_CALLS.append((pos.detach().clone(), idx.clone()))
        return idx

    sp.fps = fps


def build_skeleton_model():
    """UniRigAR with the shipped config and tokenizer; CUDA-only attention paths off. Weights NOT loaded."""
    import torch
    from src.tokenizer.spec import TokenizerConfig
    from src.tokenizer.parse import get_tokenizer
    from src.model.unirig_ar import UniRigAR
    tokenizer = get_tokenizer(TokenizerConfig.parse(load_yaml('configs/tokenizer/tokenizer_parts_articulationxl_256.yaml')))
    cfg = load_yaml('configs/model/unirig_ar_350m_1024_81920_float32.yaml')
    cfg.pop('__target__')
    cfg.llm['_attn_implementation'] = 'eager'
    cfg.mesh_encoder.flash = False
    install_fps()
    model = UniRigAR(tokenizer=tokenizer, **cfg)
    return model, tokenizer, cfg


def load_skeleton_weights(model):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from safe_ckpt import load_checkpoint
    sd = load_checkpoint(os.path.join(UNIRIG_DIR, 'ckpt', 'skeleton', 'model.ckpt'))['state_dict']
    sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected
