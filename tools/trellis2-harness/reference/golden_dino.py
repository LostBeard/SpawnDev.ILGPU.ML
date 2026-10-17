"""Golden for the DINOv3 ViT-L/16 image-conditioning extractor with RANDOM weights of the real architecture.

The real weights (facebook/dinov3-vitl16-pretrain-lvd1689m) are gated, so this validates the ARCHITECTURE: a transformers DINOv3ViTModel
with the ViT-L config (1024 wide, 24 layers, 16 heads, 4 register tokens, rope_theta 100) is initialised with scaled random weights and
saved as safetensors (the key layout a real checkpoint has); the TRELLIS.2 extractor path (DinoV3FeatureExtractor.extract_features:
embeddings -> every layer -> F.layer_norm without parameters) is run on CPU torch fp32. Our GPU implementation loads the SAME file.
When the gated weights are available they load through the same code path.

  python golden_dino.py [image_size]       writes TRELLIS2_DIR/golden/dino/*
"""
import os, sys, json
W = os.environ["TRELLIS2_DIR"]
size = int(sys.argv[1]) if len(sys.argv) > 1 else 512
OUT = os.path.join(W, "golden", "dino"); os.makedirs(OUT, exist_ok=True)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import save_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

cfg = DINOv3ViTConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16, patch_size=16,
                      image_size=size, num_register_tokens=4, rope_theta=100.0, layerscale_value=0.5, query_bias=True, key_bias=False,
                      value_bias=True, proj_bias=True, mlp_bias=True, use_gated_mlp=False, hidden_act="gelu", layer_norm_eps=1e-5)
torch.manual_seed(11)
m = DINOv3ViTModel(cfg).float().eval()
with torch.no_grad():
    for mod in m.modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            fan = mod.weight[0].numel()
            mod.weight.normal_(0, 1.0 / np.sqrt(fan))
            if mod.bias is not None: mod.bias.normal_(0, 0.05)
        elif isinstance(mod, nn.LayerNorm):
            mod.weight.copy_(1 + 0.1 * torch.randn_like(mod.weight)); mod.bias.normal_(0, 0.05)
    m.embeddings.cls_token.normal_(0, 0.5); m.embeddings.register_tokens.normal_(0, 0.5)
    for name, p in m.named_parameters():
        if name.endswith("lambda1"): p.copy_(0.4 + 0.3 * torch.rand_like(p))

sd = {k: v.detach().contiguous() for k, v in m.state_dict().items()}
print("keys:", len(sd), [k for k in sd if "layer.0." in k][:12], [k for k in sd if "layer" not in k])
save_file(sd, os.path.join(OUT, "dino_random.safetensors"))

g = torch.Generator().manual_seed(5)
img = torch.randn(1, 3, size, size, generator=g)
taps = {}
with torch.no_grad():
    h = m.embeddings(img, bool_masked_pos=None); taps["embed"] = h.clone()
    pos = m.rope_embeddings(img)
    layers = m.model.layer
    for i, layer in enumerate(layers):
        h = layer(h, position_embeddings=pos)
        if i == 0: taps["layer0"] = h.clone()
    taps["layers"] = h.clone()
    out = F.layer_norm(h, h.shape[-1:])              # TRELLIS DinoV3FeatureExtractor.extract_features
man = {"image_size": size, "tokens": int(out.shape[1])}
img.numpy().astype(np.float32).tofile(os.path.join(OUT, "image.f32"))
out.numpy().astype(np.float32).tofile(os.path.join(OUT, "features.f32"))
for k, v in taps.items(): v.numpy().astype(np.float32).tofile(os.path.join(OUT, k + ".f32"))
json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"))
print(man, float(out.abs().mean()))
