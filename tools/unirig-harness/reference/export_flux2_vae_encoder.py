"""Export FLUX.2's VAE ENCODER (encoder + quant_conv, the distribution's mode) to ONNX, for reference-image editing.

  python -I export_flux2_vae_encoder.py <klein4b_dir> <out.onnx> [size=512]

Input  image    [1, 3, H, W]        in [-1, 1] (the pipeline's preprocess output)
Output latents  [1, 32, H/8, W/8]   the mean of the latent distribution (diffusers' sample_mode="argmax"), BEFORE the 2x2 patchify and the batch-norm
The C# side does the patchify ([1, 128, H/16, W/16], channel = c*4 + dy*2 + dx), the normalisation with vae_norm.json and the token packing.
fp32 weights, spatial axes dynamic.
"""
import sys, torch
from diffusers import AutoencoderKLFlux2

src, out = sys.argv[1], sys.argv[2]
size = int(sys.argv[3]) if len(sys.argv) > 3 else 512
vae = AutoencoderKLFlux2.from_pretrained(src.rstrip('/\\') + '/vae', torch_dtype=torch.float32).eval()


class Encoder(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, image):
        return self.v.encode(image, return_dict=False)[0].mode()


m = Encoder(vae)
x = torch.randn(1, 3, size, size).clamp(-1, 1)
with torch.no_grad():
    ref = m(x)
torch.onnx.export(m, (x,), out, input_names=['image'], output_names=['latents'], opset_version=17, dynamo=False,
                  dynamic_axes={'image': {2: 'H', 3: 'W'}, 'latents': {2: 'h', 3: 'w'}})
print('exported', out, tuple(ref.shape))
