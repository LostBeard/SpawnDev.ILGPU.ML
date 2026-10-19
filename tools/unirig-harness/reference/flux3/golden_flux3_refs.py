"""Reference goldens for FLUX 3's reference-image conditioning only (the autoencoder's encoder, run on the CPU: no DiT, no text encoder), the part of golden_flux3.py the C# port gates reference pictures against:

  python -I golden_flux3_refs.py <ae.safetensors> <out_dir> ref1.png[,ref2.png,...]

Files (float32 .npy): ref_raw_<i>.npy [H, W, 3] the original picture in 0..255, ref_in_<i>.npy [3, H, W] the prepared picture in [-1, 1] (above 1024 x 1024 pixels shrunk with Lanczos, then centre-cropped to multiples of 16), cond.npy [Nc, 128] the encoder's tokens rounded to bf16 as
the reference feeds them, cond_ids.npy [Nc, 4] = (0, row / 2, column / 2, -(1 + i)).
"""
import sys, os, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flux3_ae as AE
from safetensors.torch import load_file
from PIL import Image

ae_path, out, refs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
os.makedirs(out, exist_ok=True)
ae = AE.AutoEncoder()
ae.load_state_dict(load_file(ae_path), strict=True)
ae = ae.float().eval()


def save(name, t):
    np.save(os.path.join(out, name + ".npy"), np.ascontiguousarray(t.detach().float().cpu().numpy()))


toks, idl = [], []
with torch.inference_mode():
    for i, path in enumerate(refs):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        save(f"ref_raw_{i}", torch.from_numpy(np.array(img)).float())          # the original pixels [H, W, 3] in 0..255, what the C# side is handed
        if w * h > 1024 * 1024:
            s = math.sqrt(1024 * 1024 / (w * h)); w, h = int(w * s), int(h * s); img = img.resize((w, h), Image.Resampling.LANCZOS)
        w16, h16 = w // 16 * 16, h // 16 * 16
        left, top = (w - w16) // 2, (h - h16) // 2
        img = img.crop((left, top, left + w16, top + h16))
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        save(f"ref_in_{i}", x)
        t, ids = ae.encode(x)
        ids = ids.float(); ids[:, 1:3] *= 0.5; ids[:, 3] = -(1 + i)
        toks.append(t.to(torch.bfloat16)); idl.append(ids)
        print("reference", i, (w16, h16), tuple(t.shape), flush=True)
save("cond", torch.cat(toks)); save("cond_ids", torch.cat(idl))
print("done")
