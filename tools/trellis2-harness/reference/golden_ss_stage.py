"""Rerun the WHOLE sparse-structure stage in the reference (real SparseStructureFlowModel + real FlowEulerGuidanceIntervalSampler with the
pipeline.json parameters + real SparseStructureDecoder), with the noise and condition dumped by the C# pipeline (TRELLIS2_DUMP_SS=<dir>), so
the two implementations can be compared on identical inputs. CPU torch, fp32: the 22 model evaluations take about 11 minutes.

  python golden_ss_stage.py <dump dir>      reads noise.f32 / cond.f32 (+ zs.f32), writes ref_zs.f32 and ref_coords32.i32 next to them
"""
import os, sys, json, types, time
REF = os.environ["TRELLIS2_REF"]; W = os.environ["TRELLIS2_DIR"]
D = sys.argv[1]
os.environ["ATTN_BACKEND"] = "sdpa"
pkg = types.ModuleType("trellis2"); pkg.__path__ = [os.path.join(REF, "trellis2")]; sys.modules["trellis2"] = pkg
import numpy as np, torch
from safetensors.torch import load_file
from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
from trellis2.models.sparse_structure_vae import SparseStructureDecoder
from trellis2.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
torch.set_num_threads(os.cpu_count())

ck = os.path.join(W, "TRELLIS.2-4B", "ckpts", "ss_flow_img_dit_1_3B_64_bf16")
args = json.load(open(ck + ".json"))["args"]; args["dtype"] = "float32"
flow = SparseStructureFlowModel(**args)
flow.load_state_dict({k: v.float() for k, v in load_file(ck + ".safetensors").items()}, strict=False); flow.eval()
pipe = json.load(open(os.path.join(W, "TRELLIS.2-4B", "pipeline.json")))["args"]
sp = pipe["sparse_structure_sampler"]
sampler = FlowEulerGuidanceIntervalSampler(**sp["args"])
noise = torch.from_numpy(np.fromfile(os.path.join(D, "noise.f32"), dtype=np.float32).reshape(1, 8, 16, 16, 16))
cond = torch.from_numpy(np.fromfile(os.path.join(D, "cond.f32"), dtype=np.float32).reshape(1, -1, 1024))
t0 = time.time()
zs = sampler.sample(flow, noise, cond=cond, neg_cond=torch.zeros_like(cond), verbose=False, **sp["params"]).samples
print("sampled in %.0fs" % (time.time() - t0), float(zs.std()))
zs.numpy().astype(np.float32).tofile(os.path.join(D, "ref_zs.f32"))
dck = os.path.join(W, "TRELLIS-image-large", "ckpts", "ss_dec_conv3d_16l8_fp16")
dargs = json.load(open(dck + ".json"))["args"]; dargs["use_fp16"] = False
dec = SparseStructureDecoder(**dargs)
dec.load_state_dict({k: v.float() for k, v in load_file(dck + ".safetensors").items()}, strict=False); dec.eval()
with torch.no_grad(): logits = dec(zs)
occ = torch.nn.functional.max_pool3d((logits > 0).float(), 2, 2, 0) > 0.5
coords = torch.argwhere(occ)[:, [0, 2, 3, 4]].int()
coords[:, 1:].numpy().astype(np.int32).tofile(os.path.join(D, "ref_coords32.i32"))
print("reference occupied cells:", coords.shape[0])
