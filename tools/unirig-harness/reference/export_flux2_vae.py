"""Export FLUX.2's VAE decoder (post_quant_conv + decoder) to ONNX, so the library's ONNX session (the one that already runs SD Turbo's VAE) decodes the DiT's latents.

  python -I export_flux2_vae.py <klein4b_dir> <out.onnx> [size=512]

Input  latents  [1, 32, H/8, W/8]   after the pipeline's batch-norm un-normalise and unpatchify (golden vae_in.npy)
Output image    [1, 3, H, W]        in [-1, 1]
Also writes vae_norm.json beside it: the 128-channel batch-norm mean/var/eps the pipeline applies to the packed latents before the decoder.
fp32 weights (the checkpoint is bf16; fp32 is the decoder's own reference precision, force_upcast). Spatial axes are dynamic.
"""
import sys, torch
from diffusers import AutoencoderKLFlux2

src, out = sys.argv[1], sys.argv[2]
size = int(sys.argv[3]) if len(sys.argv) > 3 else 512
vae = AutoencoderKLFlux2.from_pretrained(src.rstrip('/\\') + '/vae', torch_dtype=torch.float32).eval()


class Decoder(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, latents):
        return self.v.decode(latents, return_dict=False)[0]


m = Decoder(vae)
x = torch.randn(1, 32, size // 8, size // 8)
with torch.no_grad():
    ref = m(x)
torch.onnx.export(m, (x,), out, input_names=['latents'], output_names=['image'], opset_version=17, dynamo=False,
                  dynamic_axes={'latents': {2: 'h', 3: 'w'}, 'image': {2: 'H', 3: 'W'}})
print('exported', out, tuple(ref.shape))
import json, os
norm = {'mean': vae.bn.running_mean.float().tolist(), 'var': vae.bn.running_var.float().tolist(), 'eps': float(vae.config.batch_norm_eps)}
json.dump(norm, open(os.path.join(os.path.dirname(os.path.abspath(out)), 'vae_norm.json'), 'w'))   # the batch-norm statistics the pipeline un-normalises the packed latents with
print('vae_norm.json', len(norm['mean']), 'channels, eps', norm['eps'])
