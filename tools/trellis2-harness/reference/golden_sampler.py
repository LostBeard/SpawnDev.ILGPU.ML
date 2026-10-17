"""Golden for the flow sampler: the REAL FlowEulerGuidanceIntervalSampler with a cheap mock velocity model.

The model is already verified against the reference separately, so the sampler logic (schedule, CFG, guidance interval,
CFG rescale, Euler step) is tested with an elementwise model both sides can evaluate exactly:
    v(x, t, c) = 0.5 sin(x) + 0.0003 t + c * 0.2 x      (c = 1 for the condition, 0 for the negative condition)
"""
import os, sys, types, json
REF = os.environ["TRELLIS2_REF"]; W = os.environ["TRELLIS2_DIR"]
OUT = os.path.join(W, "golden", "sampler"); os.makedirs(OUT, exist_ok=True)
pkg = types.ModuleType("trellis2"); pkg.__path__ = [os.path.join(REF, "trellis2")]; sys.modules["trellis2"] = pkg
import numpy as np, torch
from trellis2.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler

def model(x, t, cond, **kw):
    return 0.5 * torch.sin(x) + 0.0003 * t.view(-1, *([1] * (x.ndim - 1))) + cond * 0.2 * x

g = torch.Generator().manual_seed(99)
noise = torch.randn(1, 8, 16, 16, 16, generator=g)
cases = {
  "ss":    dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7, guidance_interval=[0.6, 1.0], rescale_t=5.0),
  "shape": dict(steps=12, guidance_strength=7.5, guidance_rescale=0.5, guidance_interval=[0.6, 1.0], rescale_t=3.0),
  "tex":   dict(steps=12, guidance_strength=1.0, guidance_rescale=0.0, guidance_interval=[0.6, 0.9], rescale_t=3.0),
}
noise.numpy().astype(np.float32).tofile(os.path.join(OUT, "noise.bin"))
for name, p in cases.items():
    s = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    out = s.sample(model, noise.clone(), cond=torch.ones(1), neg_cond=torch.zeros(1), verbose=False, **p).samples
    out.numpy().astype(np.float32).tofile(os.path.join(OUT, name + ".bin"))
    print(name, float(out.std()))
json.dump(cases, open(os.path.join(OUT, "cases.json"), "w"), indent=1)
