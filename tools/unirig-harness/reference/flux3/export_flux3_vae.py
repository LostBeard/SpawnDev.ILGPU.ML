"""Export FLUX 3's autoencoder (ae.safetensors, flux3_ae.py) to ONNX for the C# pipeline (the same two graphs and vae_norm.json the FLUX.2 pipeline uses):

  python -I export_flux3_vae.py <ae.safetensors> <out_dir> [size=512]

  vae_decoder.onnx   input latents [1, 32, H/8, W/8] (after the batch-norm un-normalise and the 2x2 unpatchify, done by the C# side) -> image [1, 3, H, W] in [-1, 1]   (post_quant_conv + decoder)
  vae_encoder.onnx   input image [1, 3, H, W] in [-1, 1] -> latent mean [1, 32, H/8, W/8] (encoder + quant_conv, the first 32 of 64 channels)       (patchify + batch-norm are done by the C# side)
  vae_norm.json      the 128-channel batch-norm running mean / var and eps (1e-4) the autoencoder applies to the packed latents
fp32, spatial axes dynamic.
"""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flux3_ae as AE
from safetensors.torch import load_file

ae_path, out = sys.argv[1], sys.argv[2]
size = int(sys.argv[3]) if len(sys.argv) > 3 else 512
os.makedirs(out, exist_ok=True)
ae = AE.AutoEncoder()
ae.load_state_dict(load_file(ae_path), strict=True)
ae = ae.float().eval()


class Dec(torch.nn.Module):
    def __init__(self, d):
        super().__init__(); self.d = d
    def forward(self, z): return self.d(z)


class Enc(torch.nn.Module):
    def __init__(self, e):
        super().__init__(); self.e = e
    def forward(self, x): return self.e(x)[:, :AE.EMBED_DIM]          # mean (the second half is the log-variance)


with torch.no_grad():
    z = torch.randn(1, 32, size // 8, size // 8); ref = Dec(ae.decoder)(z)
    x = torch.randn(1, 3, size, size).clamp(-1, 1); refe = Enc(ae.encoder)(x)
torch.onnx.export(Dec(ae.decoder), (z,), os.path.join(out, "vae_decoder.onnx"), input_names=["latents"], output_names=["image"], opset_version=17, dynamo=False,
                  dynamic_axes={"latents": {2: "h", 3: "w"}, "image": {2: "H", 3: "W"}})
torch.onnx.export(Enc(ae.encoder), (x,), os.path.join(out, "vae_encoder.onnx"), input_names=["image"], output_names=["latents"], opset_version=17, dynamo=False,
                  dynamic_axes={"image": {2: "H", 3: "W"}, "latents": {2: "h", 3: "w"}})
json.dump({"mean": ae.bn.running_mean.float().tolist(), "var": ae.bn.running_var.float().tolist(), "eps": float(ae.bn_eps)}, open(os.path.join(out, "vae_norm.json"), "w"))
print("exported", tuple(ref.shape), tuple(refe.shape), "bn eps", ae.bn_eps)
