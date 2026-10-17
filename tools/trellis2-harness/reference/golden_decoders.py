"""Golden tensors for the TRELLIS.2 SC-VAE decoders (shape: pred_subdiv, 7 out ch; texture: guided by the shape subdivisions, 6 out ch).

Runs the REAL trellis2.models.sc_vaes.sparse_unet_vae.SparseUnetVaeDecoder on CPU torch (fp32, real fp16 weights upcast exactly)
on a small random latent. The reference's sparse convolution is flex_gemm (Triton, CUDA only); FlexGEMM ships its own pure-torch
EXPLICIT_GEMM path (neighbour map by sorted keys + im2col + addmm), reproduced here verbatim as a backend replacement, so the
convolution semantics (kernel index = a0*9 + a1*3 + a2 for offsets (a0-1, a1-1, a2-1) on coordinate columns x, y, z; weight
[Co, Kd, Kh, Kw, Ci]) are the reference's, not mine. Everything else is the reference's own code.

  python golden_decoders.py [N0]      writes TRELLIS2_DIR/golden/decoders/*
"""
import os, sys, json, types, time
REF = os.environ["TRELLIS2_REF"]
W = os.environ["TRELLIS2_DIR"]
n0 = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = os.path.join(W, "golden", "decoders")
os.makedirs(OUT, exist_ok=True)

pkg = types.ModuleType("trellis2")
pkg.__path__ = [os.path.join(REF, "trellis2")]
sys.modules["trellis2"] = pkg

import math
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from trellis2.modules import sparse as sp
from trellis2.modules.sparse.conv import conv as conv_mod


# ---------------------------------------------------------------- torch-only flex_gemm replacement (explicit gemm)
def neighbor_map_torch(coords, shape, kernel_size=(3, 3, 3), dilation=(1, 1, 1)):
    # verbatim from flex_gemm.ops.spconv.submanifold_conv3d.SubMConv3dFunction._compute_neighbor_cache_torch
    N, C, W, H, D = shape
    L = coords.shape[0]
    M = torch.tensor([W * H * D, H * D, D, 1]).int()
    keys = (coords * M[None]).sum(dim=-1)
    sorted_keys, indices = torch.sort(keys)
    offset = torch.meshgrid(
        torch.arange(-(kernel_size[0] // 2) * dilation[0], kernel_size[0] // 2 * dilation[0] + 1, dilation[0]),
        torch.arange(-(kernel_size[1] // 2) * dilation[1], kernel_size[1] // 2 * dilation[1] + 1, dilation[1]),
        torch.arange(-(kernel_size[2] // 2) * dilation[2], kernel_size[2] // 2 * dilation[2] + 1, dilation[2]),
        indexing='ij')
    offset = torch.stack(offset, dim=-1).reshape(-1, 3).int()
    neighbor_coords = coords.unsqueeze(1).repeat(1, kernel_size[0] * kernel_size[1] * kernel_size[2], 1)
    neighbor_coords[:, :, -3:] += offset.unsqueeze(0)
    neighbor_coords = neighbor_coords.reshape(-1, 4)
    neighbor_valid = (neighbor_coords[:, 1] >= 0) & (neighbor_coords[:, 1] < W) & \
                     (neighbor_coords[:, 2] >= 0) & (neighbor_coords[:, 2] < H) & \
                     (neighbor_coords[:, 3] >= 0) & (neighbor_coords[:, 3] < D)
    neighbor_keys = (neighbor_coords * M[None]).sum(dim=-1)
    idx = torch.searchsorted(sorted_keys, neighbor_keys)
    idx = torch.clamp(idx, 0, sorted_keys.shape[0] - 1)
    neighbor_valid &= sorted_keys[idx] == neighbor_keys
    out = torch.full((L * kernel_size[0] * kernel_size[1] * kernel_size[2],), -1, dtype=torch.long)
    out[neighbor_valid] = indices[idx[neighbor_valid]]
    return out.reshape(L, -1)


def backend_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    assert stride == 1 and padding is None
    self.in_channels, self.out_channels = in_channels, out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
    self.stride = (1, 1, 1)
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3
    self.weight = nn.Parameter(torch.empty((out_channels, *self.kernel_size, in_channels)))   # (Co, Kd, Kh, Kw, Ci) as flex_gemm stores it
    self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
    if not bias:
        self.register_parameter("bias", None)


def backend_forward(self, x):
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    key = f'SubMConv3d_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
    nb = x.get_spatial_cache(key)
    if nb is None:
        nb = neighbor_map_torch(x.coords, torch.Size([*x.shape, *x.spatial_shape]), (Kd, Kh, Kw), self.dilation)
        x.register_spatial_cache(key, nb)
    V = Kd * Kh * Kw
    feats = x.feats
    im2col = torch.zeros((feats.shape[0] * V, Ci), dtype=feats.dtype)
    flat = nb.reshape(-1)
    mask = flat >= 0
    im2col[mask] = feats[flat[mask]]
    im2col = im2col.view(feats.shape[0], V * Ci)
    wmat = self.weight.reshape(Co, V * Ci).transpose(0, 1)
    out = torch.addmm(self.bias, im2col, wmat) if self.bias is not None else torch.mm(im2col, wmat)
    return x.replace(out)


fake = types.ModuleType("conv_fake")
fake.sparse_conv3d_init = backend_init
fake.sparse_conv3d_forward = backend_forward
conv_mod._backends["flex_gemm"] = fake

from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder

torch.set_num_threads(os.cpu_count())
manifest = {"n0": n0}


def save(name, arr, dtype):
    a = np.ascontiguousarray(arr.detach().cpu().numpy().astype(dtype))
    a.tofile(os.path.join(OUT, name + (".i32" if dtype == np.int32 else ".f32")))
    manifest[name] = list(a.shape)


def build(ckpt, out_channels, pred_subdiv):
    cfg = json.load(open(os.path.join(W, "TRELLIS.2-4B", "ckpts", ckpt + ".json")))["args"]
    args = {k: v for k, v in cfg.items() if k not in ("resolution", "voxel_margin", "use_fp16", "pred_subdiv", "out_channels")}
    dec = SparseUnetVaeDecoder(out_channels=out_channels, use_fp16=False, pred_subdiv=pred_subdiv, **args)
    sd = {k: v.float() for k, v in load_file(os.path.join(W, "TRELLIS.2-4B", "ckpts", ckpt + ".safetensors")).items()}
    missing, unexpected = dec.load_state_dict(sd, strict=False)
    print(ckpt, "missing:", missing[:3], "unexpected:", unexpected[:3])
    return dec.eval()


pipe = json.load(open(os.path.join(W, "TRELLIS.2-4B", "pipeline.json")))["args"]
shape_mean = torch.tensor(pipe["shape_slat_normalization"]["mean"])[None]
shape_std = torch.tensor(pipe["shape_slat_normalization"]["std"])[None]
tex_mean = torch.tensor(pipe["tex_slat_normalization"]["mean"])[None]
tex_std = torch.tensor(pipe["tex_slat_normalization"]["std"])[None]

g = torch.Generator().manual_seed(2024)
res = 32
# a compact blob of occupied cells (a sphere-ish shell) so neighbours exist, like a real sparse structure
ax = torch.arange(res)
grid = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), dim=-1).reshape(-1, 3)
r = ((grid.float() - 15.5) ** 2).sum(-1).sqrt()
shell = grid[(r > 8.0) & (r < 9.6)]
sel = shell[torch.randperm(shell.shape[0], generator=g)[:n0]]
sel = sel[torch.argsort(sel[:, 0] * res * res + sel[:, 1] * res + sel[:, 2])]
coords = torch.cat([torch.zeros(sel.shape[0], 1, dtype=torch.int32), sel.int()], dim=1)
n0 = coords.shape[0]
shape_lat = torch.randn(n0, 32, generator=g) * shape_std + shape_mean          # what the pipeline hands the decoder (slat * std + mean)
tex_lat = torch.randn(n0, 32, generator=g) * tex_std + tex_mean
save("coords0", coords[:, 1:], np.int32)
save("shape_latent", shape_lat, np.float32)
save("tex_latent", tex_lat, np.float32)

# ------------------------------------------------------------------ shape decoder
dec = build("shape_dec_next_dc_f16c32_fp16", 7, True)
taps = {}
def tap(name):
    def hook(_m, _i, o):
        t = o[0] if isinstance(o, tuple) else o
        taps[name] = (t.feats.detach().clone(), t.coords[:, 1:].detach().clone().int())
    return hook
dec.from_latent.register_forward_hook(tap("shape_from_latent"))
for i, res_blocks in enumerate(dec.blocks):
    n = len(res_blocks)
    ups = i < len(dec.blocks) - 1
    last_conv = n - 2 if ups else n - 1          # the last ConvNeXt block of the level
    if last_conv >= 0:
        res_blocks[last_conv].register_forward_hook(tap(f"shape_level{i}_convnext"))
    if ups:
        res_blocks[n - 1].register_forward_hook(tap(f"shape_level{i}_up"))
t0 = time.time()
with torch.no_grad():
    x = sp.SparseTensor(feats=shape_lat, coords=coords)
    h, subs = dec(x, return_subs=True)
print("shape decoder %.1fs" % (time.time() - t0), h.feats.shape, [s.feats.shape for s in subs])
for k, (f, c) in taps.items():
    save(k, f, np.float32); save(k + "_coords", c, np.int32)
save("shape_out", h.feats, np.float32); save("shape_out_coords", h.coords[:, 1:], np.int32)
for i, s in enumerate(subs):
    save(f"shape_sub{i}", s.feats, np.float32); save(f"shape_sub{i}_coords", s.coords[:, 1:], np.int32)
del dec

# ------------------------------------------------------------------ texture decoder (guided)
taps.clear()
dec = build("tex_dec_next_dc_f16c32_fp16", 6, False)
dec.from_latent.register_forward_hook(tap("tex_from_latent"))
for i, res_blocks in enumerate(dec.blocks):
    n = len(res_blocks)
    ups = i < len(dec.blocks) - 1
    last_conv = n - 2 if ups else n - 1
    if last_conv >= 0:
        res_blocks[last_conv].register_forward_hook(tap(f"tex_level{i}_convnext"))
    if ups:
        res_blocks[n - 1].register_forward_hook(tap(f"tex_level{i}_up"))
t0 = time.time()
with torch.no_grad():
    x = sp.SparseTensor(feats=tex_lat, coords=coords)
    hs = dec(x, guide_subs=subs)
print("tex decoder %.1fs" % (time.time() - t0), hs.feats.shape)
for k, (f, c) in taps.items():
    save(k, f, np.float32); save(k + "_coords", c, np.int32)
save("tex_out", hs.feats, np.float32); save("tex_out_coords", hs.coords[:, 1:], np.int32)
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print("saved", OUT)
