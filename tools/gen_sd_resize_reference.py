# References for SdKernels.Resize modes Area and Bislerp (deep shrink's PatchModelAddDownscale methods, Lore's request), from ComfyUI's OWN code:
#
#   py -3.13 -P tools/gen_sd_resize_reference.py <ComfyUI checkout>/comfy/utils.py
#
# comfy.utils.bislerp is compiled from the given file's source (only that function, so none of ComfyUI's imports are needed), and "area" is
# torch.nn.functional.interpolate(mode="area"), exactly what common_upscale calls. Features are channel-last [H * W, C] like the UNet's, as raw
# float32: references/sd_resize/<case>_in.bin and <case>_<method>.bin, cases listed in cases.json. The input plants bislerp's edge cases: zero
# vectors, a pixel equal to its right neighbour (dot > 1 - 1e-5) and one opposite to it (dot < -1 + 1e-5).
import ast, json, os, sys
import numpy as np
import torch

src = open(sys.argv[1], encoding="utf-8").read()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "bislerp")
ns = {"torch": torch}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "comfy/utils.py:bislerp", "exec"), ns)
bislerp = ns["bislerp"]

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "SpawnDev.ILGPU.ML.Demo", "wwwroot", "references", "sd_resize"))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261108)
C = 48
cases = [("down_24x20", 24, 20, 12, 10), ("up_12x10", 12, 10, 24, 20), ("down_23x17", 23, 17, 12, 9), ("up_12x9", 12, 9, 23, 17)]
manifest = []
for name, h, w, oh, ow in cases:
    x = rng.standard_normal((1, C, h, w)).astype(np.float32)
    x[0, :, 1, 2] = 0.0                       # zero vector
    x[0, :, 2, 3] = x[0, :, 2, 4]             # identical neighbours
    x[0, :, 3, 5] = -x[0, :, 3, 6]            # opposite neighbours
    t = torch.from_numpy(x)
    outs = {"area": torch.nn.functional.interpolate(t, size=(oh, ow), mode="area"), "bislerp": bislerp(t, ow, oh)}
    x.transpose(0, 2, 3, 1).reshape(h * w, C).astype(np.float32).tofile(os.path.join(OUT, f"{name}_in.bin"))
    for m, o in outs.items():
        a = o.numpy().astype(np.float32)
        assert np.isfinite(a).all(), (name, m)
        a.transpose(0, 2, 3, 1).reshape(oh * ow, C).tofile(os.path.join(OUT, f"{name}_{m}.bin"))
    manifest.append(dict(name=name, h=h, w=w, oh=oh, ow=ow, c=C))
    print(name, {m: list(o.shape) for m, o in outs.items()})
json.dump({"torch": torch.__version__, "cases": manifest}, open(os.path.join(OUT, "cases.json"), "w"), indent=1)
