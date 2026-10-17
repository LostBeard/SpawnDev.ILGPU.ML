"""Golden tensors for the TRELLIS.2 structured-latent flow DiTs (shape: in 32 ch, texture: in 64 ch = 32 noise + 32 shape slat).

Runs the REAL trellis2.models.structured_latent_flow.SLatFlowModel on CPU torch (fp32, real weights). The reference
needs flash_attn / xformers varlen for sparse attention, which have no CPU build, so the one function that calls
them is replaced by torch scaled_dot_product_attention (identical maths for a single sequence). Everything else,
including SparseRotaryPositionEmbedder, SparseMultiHeadRMSNorm and the sparse block, is the reference's own code.

  python golden_slat_flow.py shape|tex|shape1024|tex1024 [N_TOKENS] [N_COND]
"""
import os, sys, json, types, time
REF = os.environ["TRELLIS2_REF"]
W = os.environ["TRELLIS2_DIR"]
kind = sys.argv[1] if len(sys.argv) > 1 else "shape"
n_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
n_cond = int(sys.argv[3]) if len(sys.argv) > 3 else 1029
ckpt = {"shape": "slat_flow_img2shape_dit_1_3B_512_bf16", "tex": "slat_flow_imgshape2tex_dit_1_3B_512_bf16",
        "shape1024": "slat_flow_img2shape_dit_1_3B_1024_bf16", "tex1024": "slat_flow_imgshape2tex_dit_1_3B_1024_bf16"}[kind]
OUT = os.path.join(W, "golden", "slat_" + kind)
os.makedirs(OUT, exist_ok=True)

pkg = types.ModuleType("trellis2")
pkg.__path__ = [os.path.join(REF, "trellis2")]
sys.modules["trellis2"] = pkg

import torch
import torch.nn.functional as F
import numpy as np
from safetensors.torch import load_file
from trellis2.modules import sparse as sp
import trellis2.modules.sparse.attention.modules as attn_modules
from trellis2.models.structured_latent_flow import SLatFlowModel


def sdpa_patched(*args):
    def attend(q, k, v):   # [T,H,C] x [L,H,C] -> [T,H,C]
        o = F.scaled_dot_product_attention(q.permute(1, 0, 2)[None], k.permute(1, 0, 2)[None], v.permute(1, 0, 2)[None])
        return o[0].permute(1, 0, 2)
    if len(args) == 1:
        qkv = args[0]
        q, k, v = qkv.feats.unbind(dim=1)
        return qkv.replace(attend(q, k, v))
    if len(args) == 3:
        q, k, v = args
        assert isinstance(k, torch.Tensor) and k.shape[0] == 1, "cross attention context must be a dense batch of 1"
        return q.replace(attend(q.feats, k[0], v[0]))
    raise NotImplementedError(f"{len(args)}-arg sparse attention")

attn_modules.sparse_scaled_dot_product_attention = sdpa_patched

torch.set_num_threads(os.cpu_count())
cfg = json.load(open(os.path.join(W, "TRELLIS.2-4B", "ckpts", ckpt + ".json")))
args = dict(cfg["args"])
# DTYPE=bf16 runs the model the way the released pipeline does (bf16 blocks) and writes only y_bf16.bin: the reference's
# OWN deviation from the fp32 golden, which is the yardstick for how much bf16 noise our tensor-core path may show.
BF16 = os.environ.get("DTYPE", "fp32") == "bf16"
args["dtype"] = "bfloat16" if BF16 else "float32"
model = SLatFlowModel(**args)
sd = {k: v.float() for k, v in load_file(os.path.join(W, "TRELLIS.2-4B", "ckpts", ckpt + ".safetensors")).items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing:", missing, "unexpected:", unexpected)
model.eval()

g = torch.Generator().manual_seed(4321)
res = args["resolution"]               # 32: voxel coordinates live on a res^3 grid
flat = torch.randperm(res ** 3, generator=g)[:n_tok].sort().values   # unique, sorted like argwhere output
coords = torch.stack([torch.zeros_like(flat), flat // (res * res), (flat // res) % res, flat % res], dim=1).int()
in_ch = args["in_channels"]
feats = torch.randn(n_tok, in_ch, generator=g)
cond = torch.randn(1, n_cond, 1024, generator=g)
t = torch.tensor([float(os.environ.get("T", "700"))])
x = sp.SparseTensor(feats=feats, coords=coords)

taps = {}
def tap(name):
    def hook(_m, _i, o):
        taps[name] = (o.feats if hasattr(o, "feats") else o).detach().clone()
    return hook
model.input_layer.register_forward_hook(tap("input_layer"))
model.blocks[0].register_forward_hook(tap("block0"))
model.blocks[len(model.blocks) - 1].register_forward_hook(tap("block29"))

t0 = time.time()
with torch.no_grad():
    y = model(x, t, cond)
print("forward %.1fs" % (time.time() - t0), y.feats.shape, float(y.feats.abs().mean()))

manifest = {"n_tok": n_tok, "n_cond": n_cond, "in_ch": in_ch, "out_ch": args["out_channels"], "t": float(t[0])}
def save(name, tensor, dtype=np.float32):
    a = tensor.detach().cpu().numpy().astype(dtype)
    a.tofile(os.path.join(OUT, name + ".bin"))
    manifest[name] = list(a.shape)
if BF16:
    save("y_bf16", y.feats.float())
    json.dump(manifest, open(os.path.join(OUT, "manifest_bf16.json"), "w"), indent=1)
    print("saved bf16 y"); sys.exit(0)
save("feats", feats); save("coords", coords[:, 1:], np.int32); save("cond", cond); save("y", y.feats)
for k, v in taps.items():
    save(k, v)
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print("saved", OUT, manifest)
