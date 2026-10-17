"""Golden tensors for the TRELLIS.2 sparse-structure flow DiT, from the REAL reference classes + REAL weights.

Runs the actual trellis2.models.sparse_structure_flow.SparseStructureFlowModel on CPU torch with the sdpa
attention backend, in float32 (bf16 weights upcast exactly), so it is the mathematically exact answer our
fp32-accumulate GPU kernels are held to. Inputs are deterministic. Saves raw little-endian float32 .bin files
plus a manifest.json (shapes) to GOLDEN_DIR.
"""
import os, sys, json, types, time
os.environ["ATTN_BACKEND"] = "sdpa"
REF = os.environ["TRELLIS2_REF"]
W = os.environ["TRELLIS2_DIR"]
OUT = os.path.join(W, "golden", "ss_flow")
os.makedirs(OUT, exist_ok=True)

# import the reference sub-packages WITHOUT running trellis2/__init__ (it pulls renderers / nvdiffrast)
pkg = types.ModuleType("trellis2")
pkg.__path__ = [os.path.join(REF, "trellis2")]
sys.modules["trellis2"] = pkg

import torch
import numpy as np
from safetensors.torch import load_file
from trellis2.models.sparse_structure_flow import SparseStructureFlowModel

torch.set_num_threads(os.cpu_count())
cfg = json.load(open(os.path.join(W, "TRELLIS.2-4B", "ckpts", "ss_flow_img_dit_1_3B_64_bf16.json")))
args = dict(cfg["args"])
args["dtype"] = "float32"          # exact upcast of the bf16 weights; no activation rounding
model = SparseStructureFlowModel(**args)
sd = load_file(os.path.join(W, "TRELLIS.2-4B", "ckpts", "ss_flow_img_dit_1_3B_64_bf16.safetensors"))
sd = {k: v.float() for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing:", [m for m in missing if "pos_emb" not in m and "rope_phases" not in m], "unexpected:", unexpected)
model.eval()

g = torch.Generator().manual_seed(1234)
n_cond = int(os.environ.get("N_COND", "1029"))
x = torch.randn(1, 8, 16, 16, 16, generator=g)
cond = torch.randn(1, n_cond, 1024, generator=g)
t = torch.tensor([float(os.environ.get("T", "700"))])

taps = {}
def tap(name):
    def hook(_m, _i, o):
        taps[name] = o.detach().clone()
    return hook
model.blocks[0].register_forward_hook(tap("block0"))
model.blocks[29].register_forward_hook(tap("block29"))
model.input_layer.register_forward_hook(tap("input_layer"))
model.t_embedder.register_forward_hook(tap("t_emb"))

t0 = time.time()
with torch.no_grad():
    y = model(x, t, cond)
print("forward %.1fs" % (time.time() - t0), y.shape, float(y.abs().mean()))

manifest = {}
def save(name, tensor):
    a = tensor.float().contiguous().numpy()
    a.tofile(os.path.join(OUT, name + ".bin"))
    manifest[name] = list(a.shape)
save("x", x); save("cond", cond); save("t", t); save("y", y)
for k, v in taps.items():
    save(k, v)
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print("saved", OUT, manifest)
