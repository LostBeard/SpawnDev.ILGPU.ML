"""Rerun the WHOLE texture-latent sampling in the reference (real SLatFlowModel + real FlowEulerGuidanceIntervalSampler with the pipeline.json
parameters, CPU torch fp32) on the noise / shape latent / condition dumped by the C# pipeline (TRELLIS2_DUMP_TEX=<dir>), and compare the sampled
latent with ours (tex_slat.f32): std, per-channel mean, relRMS. Decides whether a systematic difference (for instance the 1024 texture latent's
lower spread) is the model or our sampling.

  python golden_tex_stage.py <dump dir> <512|1024>
"""
import os, sys, json, types, time
REF = os.environ["TRELLIS2_REF"]; W = os.environ["TRELLIS2_DIR"]
D, RES = sys.argv[1], sys.argv[2]
pkg = types.ModuleType("trellis2"); pkg.__path__ = [os.path.join(REF, "trellis2")]; sys.modules["trellis2"] = pkg
import numpy as np, torch
import torch.nn.functional as F
from safetensors.torch import load_file
from trellis2.modules import sparse as sp
import trellis2.modules.sparse.attention.modules as attn_modules
from trellis2.models.structured_latent_flow import SLatFlowModel
from trellis2.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler

def sdpa_patched(*args):
    def attend(q, k, v):
        o = F.scaled_dot_product_attention(q.permute(1, 0, 2)[None], k.permute(1, 0, 2)[None], v.permute(1, 0, 2)[None])
        return o[0].permute(1, 0, 2)
    if len(args) == 1:
        q, k, v = args[0].feats.unbind(dim=1)
        return args[0].replace(attend(q, k, v))
    q, k, v = args
    return q.replace(attend(q.feats, k[0], v[0]))
attn_modules.sparse_scaled_dot_product_attention = sdpa_patched
torch.set_num_threads(os.cpu_count())

ck = os.path.join(W, "TRELLIS.2-4B", "ckpts", f"slat_flow_imgshape2tex_dit_1_3B_{RES}_bf16")
args = json.load(open(ck + ".json"))["args"]; args["dtype"] = "float32"
model = SLatFlowModel(**args)
model.load_state_dict({k: v.float() for k, v in load_file(ck + ".safetensors").items()}, strict=False); model.eval()
pipe = json.load(open(os.path.join(W, "TRELLIS.2-4B", "pipeline.json")))["args"]
tp = pipe["tex_slat_sampler"]
sampler = FlowEulerGuidanceIntervalSampler(**tp["args"])

coords = np.fromfile(os.path.join(D, "coords.i32"), dtype=np.int32).reshape(-1, 3)
n = coords.shape[0]
coords4 = torch.from_numpy(np.concatenate([np.zeros((n, 1), np.int32), coords], axis=1))
noise = sp.SparseTensor(feats=torch.from_numpy(np.fromfile(os.path.join(D, "noise.f32"), dtype=np.float32).reshape(n, 32)), coords=coords4)
shape = sp.SparseTensor(feats=torch.from_numpy(np.fromfile(os.path.join(D, "shape_slat.f32"), dtype=np.float32).reshape(n, 32)), coords=coords4)
cond = torch.from_numpy(np.fromfile(os.path.join(D, "cond.f32"), dtype=np.float32).reshape(1, -1, 1024))
print("tokens", n, "cond tokens", cond.shape[1], "params", tp["params"])
t0 = time.time()
with torch.no_grad():
    out = sampler.sample(model, noise, concat_cond=shape, cond=cond, neg_cond=torch.zeros_like(cond), verbose=True, **tp["params"]).samples
ref = out.feats.numpy()
print("sampled in %.0fs" % (time.time() - t0))
ref.astype(np.float32).tofile(os.path.join(D, "ref_tex_slat.f32"))
got = np.fromfile(os.path.join(D, "tex_slat.f32"), dtype=np.float32).reshape(n, 32)
rel = lambda a, b: float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))
print("std ours %.3f reference %.3f | mean ours %.3f reference %.3f | relRMS %.3f | corr %.4f" % (
    got.std(), ref.std(), got.mean(), ref.mean(), rel(got, ref), np.corrcoef(got.ravel(), ref.ravel())[0, 1]))
