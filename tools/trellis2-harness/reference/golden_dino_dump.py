"""Compare the C# DINOv3 features at any size with timm's own implementation on the SAME input tensor.
The C# side writes image<S>.f32 / cond<S>.f32 with TRELLIS2_DUMP_COND=<dir> (Trellis2ImageTo3D, 1024 path).

  python golden_dino_dump.py <dir> <size>        e.g. 1024 -> reads image1024.f32, cond1024.f32; prints relRMS per token group
"""
import os, sys
W = os.environ["TRELLIS2_DIR"]
D, S = sys.argv[1], int(sys.argv[2])
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
from safetensors.torch import load_file

model = timm.create_model("vit_large_patch16_dinov3_qkvb", pretrained=False, num_classes=0, img_size=S)
print(model.load_state_dict(load_file(os.path.join(W, "dinov3-timm-qkvb", "model.safetensors")), strict=False))
model.norm = nn.Identity(); model.eval()
img = torch.from_numpy(np.fromfile(os.path.join(D, f"image{S}.f32"), dtype=np.float32).reshape(1, 3, S, S))
with torch.no_grad():
    x = model.forward_features(img)
    ref = F.layer_norm(x, x.shape[-1:])[0].numpy()
got = np.fromfile(os.path.join(D, f"cond{S}.f32"), dtype=np.float32).reshape(ref.shape)
def rel(a, b): return float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))
print("tokens", ref.shape, "relRMS all %.3e" % rel(got, ref), "| cls+reg %.3e" % rel(got[:5], ref[:5]), "| patches %.3e" % rel(got[5:], ref[5:]))
