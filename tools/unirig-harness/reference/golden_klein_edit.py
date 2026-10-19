"""Reference goldens for FLUX.2 klein 4B REFERENCE-IMAGE EDITING (the C# port is gated against these, stage by stage).

  python -I golden_klein_edit.py <klein4b_dir> <out_dir> "<prompt>" <ref1.png[,ref2.png...]> [seed=0] [steps=4]

Like golden_klein.py (text to image), plus:
  ref_in_<i>.npy      [1, 3, H, W]    the preprocessed reference image i in [-1, 1] (what the VAE encoder gets)
  ref_z_<i>.npy       [1, 128, H/16, W/16]   its latents: VAE encode (distribution mode), 2x2 patchify, batch-norm normalise
  ref_latents.npy     [1, N*tokens, 128]     all references packed and concatenated
  ref_ids.npy         [1, N*tokens, 4]       (t = 10 + 10 i, h, w, 0)
The DiT runs on [noise tokens | reference tokens] (ids concatenated the same way); its output's first tokens are the prediction. The picture is the size of the first reference.
"""
import sys, os, gc, json, numpy as np, torch

src, out, prompt, refs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
steps = int(sys.argv[6]) if len(sys.argv) > 6 else 4
os.makedirs(out, exist_ok=True)

from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from transformers import AutoTokenizer, Qwen3ForCausalLM
from PIL import Image

dev = torch.device("cuda")
P = Flux2KleinPipeline
tokenizer = AutoTokenizer.from_pretrained(os.path.join(src, "tokenizer"))
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(src, "scheduler"))
VAE_SCALE = 8


def save(name, t):
    if torch.is_tensor(t):
        t = t.detach().cpu()
        a = t.numpy() if t.dtype in (torch.int32, torch.int64) else t.float().numpy()
        if a.dtype == np.int32: a = a.astype(np.int64)
    else:
        a = np.asarray(t)
    np.save(os.path.join(out, name + ".npy"), np.ascontiguousarray(a))


# ---- stage 1: text encoder -------------------------------------------------
te = Qwen3ForCausalLM.from_pretrained(os.path.join(src, "text_encoder"), dtype=torch.bfloat16).to(dev)
text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
tok = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
save("prompt_ids", tok["input_ids"].to(torch.int64)); save("prompt_mask", tok["attention_mask"].to(torch.int64))
with torch.no_grad():
    embeds = P._get_qwen3_prompt_embeds(text_encoder=te, tokenizer=tokenizer, prompt=prompt, device=dev)
    ids_txt = P._prepare_text_ids(embeds).to(dev)
save("prompt_embeds", embeds); save("txt_ids", ids_txt)
del te; gc.collect(); torch.cuda.empty_cache()

# ---- stage 1b: the reference images ----------------------------------------
ip = Flux2ImageProcessor(vae_scale_factor=VAE_SCALE * 2)
vae = AutoencoderKLFlux2.from_pretrained(os.path.join(src, "vae"), torch_dtype=torch.bfloat16).to(dev)
bnm = vae.bn.running_mean.view(1, -1, 1, 1).to(dev, torch.bfloat16)
bns = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(dev, torch.bfloat16)
zs, packed, size_h, size_w = [], [], None, None
for k, path in enumerate(refs):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w * h > 1024 * 1024:
        img = ip._resize_to_target_area(img, 1024 * 1024); w, h = img.size
    mo = VAE_SCALE * 2
    w, h = (w // mo) * mo, (h // mo) * mo
    x = ip.preprocess(img, height=h, width=w, resize_mode="crop")
    save(f"ref_in_{k}", x)
    if size_h is None: size_h, size_w = h, w
    with torch.no_grad():
        z = vae.encode(x.to(dev, torch.bfloat16), return_dict=False)[0].mode()
    z = (P._patchify_latents(z) - bnm) / bns
    save(f"ref_z_{k}", z)
    zs.append(z); packed.append(P._pack_latents(z).squeeze(0))
ref_ids = P._prepare_image_ids(zs).to(dev)
ref_lat = torch.cat(packed, dim=0).unsqueeze(0)
save("ref_latents", ref_lat); save("ref_ids", ref_ids)
del vae; gc.collect(); torch.cuda.empty_cache()

# ---- stage 2: noise + schedule (the picture has the first reference's size) -
tr = Flux2Transformer2DModel.from_pretrained(os.path.join(src, "transformer"), torch_dtype=torch.bfloat16)
ch = tr.config.in_channels // 4
g = torch.Generator("cpu").manual_seed(seed)
lat_h, lat_w = 2 * (size_h // (VAE_SCALE * 2)), 2 * (size_w // (VAE_SCALE * 2))
noise = torch.randn((1, ch * 4, lat_h // 2, lat_w // 2), generator=g, dtype=torch.bfloat16)
ids = P._prepare_latent_ids(noise).to(dev)
latents = P._pack_latents(noise.to(dev))
save("noise", latents); save("img_ids", ids)
sigmas = np.linspace(1.0, 1 / steps, steps)
mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=steps)
timesteps, _ = retrieve_timesteps(scheduler, steps, dev, sigmas=sigmas, mu=mu)
scheduler.set_begin_index(0)
save("sigmas", scheduler.sigmas); save("timesteps", timesteps)
json.dump({"mu": mu, "seq": int(latents.shape[1]), "ref_tokens": int(ref_lat.shape[1]), "steps": steps, "seed": seed, "h": size_h, "w": size_w, "prompt": prompt, "refs": refs},
          open(os.path.join(out, "meta.json"), "w"))

# ---- stage 3: the DiT on [noise | references] ------------------------------
tr = tr.to(dev)
cur = latents
n_img = latents.shape[1]
with torch.no_grad():
    for i, t in enumerate(timesteps):
        ts = t.expand(cur.shape[0]).to(cur.dtype)
        model_in = torch.cat([cur, ref_lat], dim=1).to(torch.bfloat16)
        all_ids = torch.cat([ids, ref_ids], dim=1)
        pred = tr(hidden_states=model_in, timestep=ts / 1000, guidance=None, encoder_hidden_states=embeds, txt_ids=ids_txt, img_ids=all_ids, return_dict=False)[0]
        pred = pred[:, :n_img]
        save(f"pred_{i}", pred)
        cur = scheduler.step(pred, t, cur, return_dict=False)[0]
        save(f"latents_{i}", cur)
        print("step", i, "t", float(t), flush=True)
del tr; gc.collect(); torch.cuda.empty_cache()

# ---- stage 4: un-normalise, unpatchify, VAE --------------------------------
vae = AutoencoderKLFlux2.from_pretrained(os.path.join(src, "vae"), torch_dtype=torch.bfloat16).to(dev)
lat = P._unpack_latents_with_ids(cur.to(torch.bfloat16), ids, lat_h // 2, lat_w // 2)
lat = lat * bns + bnm
lat = P._unpatchify_latents(lat)
save("vae_in", lat)
with torch.no_grad():
    img = vae.decode(lat, return_dict=False)[0]
save("image", img)
a = (img.float() / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
pil = Image.fromarray((a * 255).round().astype(np.uint8))
pil.save(os.path.join(out, "image.png"))
print("done", pil.size)
