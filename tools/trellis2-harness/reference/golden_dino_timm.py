"""Independent check of the DINOv3 extractor on the REAL weights: timm's own DINOv3 implementation (timm/vit_large_patch16_dinov3_qkvb.lvd1689m,
the layout with the q / v biases) on CPU torch fp32, with the TRELLIS.2 head (no final norm, F.layer_norm without parameters).
Input is the image-prep golden tensor of the 'ellipse' case (a real [3,512,512] normalised image).

  python golden_dino_timm.py        writes TRELLIS2_DIR/golden/dino_real/*
"""
import os, sys
W = os.environ["TRELLIS2_DIR"]
OUT = os.path.join(W, "golden", "dino_real"); os.makedirs(OUT, exist_ok=True)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
from safetensors.torch import load_file

model = timm.create_model("vit_large_patch16_dinov3_qkvb", pretrained=False, num_classes=0, img_size=512)
sd = load_file(os.path.join(W, "dinov3-timm-qkvb", "model.safetensors"))
print(model.load_state_dict(sd, strict=False))
model.norm = nn.Identity(); model.eval()
img = torch.from_numpy(np.fromfile(os.path.join(W, "golden", "prep", "ellipse.tensor.f32"), dtype=np.float32).reshape(1, 3, 512, 512))
with torch.no_grad():
    x = model.forward_features(img)
    out = F.layer_norm(x, x.shape[-1:])
print(out.shape, float(out.abs().mean()))
img.numpy().astype(np.float32).tofile(os.path.join(OUT, "image.f32"))
out.numpy().astype(np.float32).tofile(os.path.join(OUT, "features.f32"))
