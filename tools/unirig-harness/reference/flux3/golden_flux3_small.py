"""A SMALL FLUX 3 DiT with random weights, run through the reference code (flux3_dit.py) in float32 on the CPU: the architecture gate for Flux3Transformer that needs no gated checkpoint.

  python -I golden_flux3_small.py <out_dir> [seed=0]

Writes weights.safetensors (bf16, the fused layout of flux3_convert.py), config.json (the small sizes) and the float32 .npy inputs / outputs:
  image.npy [Ni,128]  cond.npy [Nc,128]  text.npy [Nt,TEXT]  image_ids.npy cond_ids.npy text_ids.npy [.,4]  t.npy t_next.npy  out.npy [Ni,128]
  tap_early_image2.npy, tap_single1.npy, tap_single_last.npy   block outputs (the single taps are the whole stream [image | image_cond | text])
The weights are rounded to bf16 BEFORE the reference runs, so the C# port (bf16 weights, fp32 activations) and this float32 run use identical numbers.
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flux3_dit as D
from flux3_convert import fuse
from safetensors.torch import save_file

out = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
os.makedirs(out, exist_ok=True)
CFG = dict(dim=256, heads=2, early=2, single=3, mlp=768, text=640)        # head_dim 128 as in the real model (the rotary axes are 4 x 32)
D.HIDDEN, D.HEADS, D.DEPTH, D.DEPTH_SINGLE, D.MLP_RATIO = CFG["dim"], CFG["heads"], CFG["early"], CFG["single"], CFG["mlp"] / CFG["dim"]
D.IN_CHANNELS["text"] = CFG["text"]
torch.manual_seed(seed)
dit = D.DiT(flow_map=True).float().eval()
with torch.no_grad():
    for p in dit.parameters():                                   # bigger weights than the default init so the gate has something to find, rounded to bf16
        p.copy_((p * 3.0).bfloat16().float())
    for name, p in dit.named_parameters():
        if name.endswith(".scale"): p.copy_((1.0 + 0.3 * torch.randn_like(p)).bfloat16().float())        # RMS norm scales around 1

# inputs: a 6 x 5 picture, two references (4 x 4 and 3 x 4), 24 text tokens; ids as flux3/pipeline.py builds them (rows and columns halved, l = -(1 + i) for references)
def grid_ids(h, w, l=0.0):
    r, c = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    ids = torch.stack([torch.zeros_like(r), r, c, torch.zeros_like(r)], -1).reshape(h * w, 4).float()
    ids[:, 1:3] *= 0.5
    ids[:, 3] = l
    return ids

g = torch.Generator().manual_seed(seed + 1)
image = torch.randn(30, 128, generator=g); cond = torch.randn(28, 128, generator=g); text = torch.randn(24, CFG["text"], generator=g)
image_ids = grid_ids(6, 5); cond_ids = torch.cat([grid_ids(4, 4, -1.0), grid_ids(3, 4, -2.0)]); text_ids = torch.zeros(24, 4); text_ids[:, 3] = torch.arange(24)
t, t_next = 0.7, 0.55
packed = D.Packed(data={"image": image, "image_cond": cond, "text": text}, ids={"image": image_ids, "image_cond": cond_ids, "text": text_ids},
                  timesteps={"image": torch.full((30,), t), "image_cond": torch.zeros(28), "text": torch.zeros(24)},
                  mode_indices={"image": [(0, 30)], "image_cond": [(0, 28)], "text": [(0, 24)]})
taps = {}
dit.content_mode_blocks["image"][1].register_forward_hook(lambda m, a, o: taps.__setitem__("early_image2", o.detach().clone()))
dit.single_blocks[0].register_forward_hook(lambda m, a, o: taps.__setitem__("single1", o.detach().clone()))
dit.single_blocks[-1].register_forward_hook(lambda m, a, o: taps.__setitem__("single_last", o.detach().clone()))
with torch.no_grad():
    result = dit(packed, timesteps_ref=torch.tensor([t_next]))
print("out", tuple(result.shape), float(result.abs().mean()))

sd = {k: v.detach().bfloat16().contiguous() for k, v in fuse(dict(dit.state_dict())).items()}
save_file(sd, os.path.join(out, "weights.safetensors"))
json.dump(CFG, open(os.path.join(out, "config.json"), "w"))
def save(name, x): np.save(os.path.join(out, name + ".npy"), np.ascontiguousarray(np.asarray(x.detach().float().numpy() if torch.is_tensor(x) else x, dtype=np.float32)))
for name, x in dict(image=image, cond=cond, text=text, image_ids=image_ids, cond_ids=cond_ids, text_ids=text_ids, out=result).items(): save(name, x)
save("t", np.array([t])); save("t_next", np.array([t_next]))
for k, v in taps.items(): save("tap_" + k, v)
print("saved", out, len(sd), "tensors")
