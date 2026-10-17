"""Golden for the TRELLIS-image-large sparse-structure decoder: the REAL SparseStructureDecoder (dense conv3d), CPU torch fp32.
Input: the sparse-structure flow golden's x as a stand-in latent, scaled to unit variance noise; writes logits [64^3] and the
pooled 32^3 occupancy coordinates exactly as Trellis2ImageTo3DPipeline.sample_sparse_structure computes them.

  python golden_ssdec.py        writes TRELLIS2_DIR/golden/ssdec/*
"""
import os, sys, json, types
REF = os.environ["TRELLIS2_REF"]; W = os.environ["TRELLIS2_DIR"]
OUT = os.path.join(W, "golden", "ssdec"); os.makedirs(OUT, exist_ok=True)
pkg = types.ModuleType("trellis2"); pkg.__path__ = [os.path.join(REF, "trellis2")]; sys.modules["trellis2"] = pkg
import numpy as np, torch
from safetensors.torch import load_file
from trellis2.models.sparse_structure_vae import SparseStructureDecoder

ck = os.path.join(W, "TRELLIS-image-large", "ckpts", "ss_dec_conv3d_16l8_fp16")
args = json.load(open(ck + ".json"))["args"]; args["use_fp16"] = False
dec = SparseStructureDecoder(**args)
sd = {k: v.float() for k, v in load_file(ck + ".safetensors").items()}
print(dec.load_state_dict(sd, strict=False))
dec.eval()
torch.set_num_threads(os.cpu_count())
g = torch.Generator().manual_seed(7)
z = torch.randn(1, 8, 16, 16, 16, generator=g) * 1.5
with torch.no_grad():
    logits = dec(z)
occ = logits > 0
print("logits", tuple(logits.shape), "occupied", int(occ.sum()))
pooled = torch.nn.functional.max_pool3d(occ.float(), 2, 2, 0) > 0.5
coords = torch.argwhere(pooled)[:, [0, 2, 3, 4]].int()
z.numpy().astype(np.float32).tofile(os.path.join(OUT, "latent.f32"))
logits.numpy().astype(np.float32).tofile(os.path.join(OUT, "logits.f32"))
coords[:, 1:].numpy().astype(np.int32).tofile(os.path.join(OUT, "coords32.i32"))
print("pooled 32^3 occupied", coords.shape[0])
