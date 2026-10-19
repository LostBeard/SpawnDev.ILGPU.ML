"""Reference goldens for FLUX.2 klein 4B (the C# port is gated against these, stage by stage).

  python -I golden_klein.py <klein4b_dir> <out_dir> "<prompt>" [seed=0] [size=512] [steps=4]

Runs the diffusers pipeline one component at a time on the GPU (the 4B DiT and the Qwen3-4B text encoder do not fit
a 12 GB card together) and saves every intermediate the port needs as .npy float32:

  prompt_ids.npy, prompt_mask.npy          the tokenised chat-templated prompt, padded to 512
  prompt_embeds.npy                        [1, 512, 7680]  Qwen3 hidden states 9, 18, 27 concatenated (bf16 -> f32)
  noise.npy                                [1, tokens, 128] the packed start latents (seeded torch randn on the CPU)
  img_ids.npy, txt_ids.npy                 rotary position ids
  sigmas.npy, timesteps.npy                the schedule after the dynamic shift (mu)
  pred_<i>.npy, latents_<i>.npy            transformer output and the latents after scheduler step i
  vae_in.npy                               [1, 32, H/8, W/8] latents after the batch-norm un-normalise and unpatchify
  image.npy / image.png                    the decoded picture

dtype: bf16 (the shipped checkpoint's own dtype).  --fp32 as argv[7] = "fp32" runs the DiT stage in float32 on the CPU
(slow, ~16 GB RAM) so the bf16 and float32 results can be compared when a tolerance needs justifying.
"""
import sys, os, gc, json, numpy as np, torch

src, out, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
size = int(sys.argv[5]) if len(sys.argv) > 5 else 512
steps = int(sys.argv[6]) if len(sys.argv) > 6 else 4
fp32 = len(sys.argv) > 7 and sys.argv[7] == "fp32"
os.makedirs(out, exist_ok=True)

from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
from transformers import AutoTokenizer, Qwen3ForCausalLM

dev = torch.device("cuda")
P = Flux2KleinPipeline          # only its static helpers are used: the box has 16 GB of RAM, so the 15 GB of weights load one component at a time
tokenizer = AutoTokenizer.from_pretrained(os.path.join(src, "tokenizer"))
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(src, "scheduler"))
VAE_SCALE = 8


class _Pipe:                    # the few attributes the script reads
    pass
pipe = _Pipe(); pipe.tokenizer = tokenizer; pipe.scheduler = scheduler; pipe.vae_scale_factor = VAE_SCALE


def save(name, t):
    if torch.is_tensor(t):
        t = t.detach().cpu()
        a = t.numpy() if t.dtype in (torch.int32, torch.int64) else t.float().numpy()      # integer tensors (ids) stay integers; the C# reader wants int64
        if a.dtype == np.int32: a = a.astype(np.int64)
    else:
        a = np.asarray(t)
    a = np.ascontiguousarray(a)                    # permuted tensors would save as fortran_order, which the C# reader refuses
    np.save(os.path.join(out, name + ".npy"), a)


# ---- stage 1: text encoder -------------------------------------------------
te = Qwen3ForCausalLM.from_pretrained(os.path.join(src, "text_encoder"), dtype=torch.bfloat16).to(dev)
msgs = [{"role": "user", "content": prompt}]
text = pipe.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
open(os.path.join(out, "prompt_text.txt"), "w", encoding="utf8").write(text)
tok = pipe.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
save("prompt_ids", tok["input_ids"].to(torch.int64)); save("prompt_mask", tok["attention_mask"].to(torch.int64))
with torch.no_grad():
    embeds = P._get_qwen3_prompt_embeds(text_encoder=te, tokenizer=pipe.tokenizer, prompt=prompt, device=dev)
    ids_txt = P._prepare_text_ids(embeds).to(dev)
save("prompt_embeds", embeds); save("txt_ids", ids_txt)
del te; gc.collect(); torch.cuda.empty_cache()

# ---- stage 2: noise + schedule ---------------------------------------------
tr = Flux2Transformer2DModel.from_pretrained(os.path.join(src, "transformer"), torch_dtype=torch.float32 if fp32 else torch.bfloat16)
ch = tr.config.in_channels // 4
g = torch.Generator("cpu").manual_seed(seed)
lat_h = 2 * (size // (pipe.vae_scale_factor * 2)); lat_w = lat_h
noise = torch.randn((1, ch * 4, lat_h // 2, lat_w // 2), generator=g, dtype=torch.bfloat16)
ids = P._prepare_latent_ids(noise).to(dev)
latents = P._pack_latents(noise.to(dev))
save("noise", latents); save("img_ids", ids)
sigmas = np.linspace(1.0, 1 / steps, steps)
mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=steps)
timesteps, _ = retrieve_timesteps(pipe.scheduler, steps, dev, sigmas=sigmas, mu=mu)
pipe.scheduler.set_begin_index(0)
save("sigmas", pipe.scheduler.sigmas); save("timesteps", timesteps)
json.dump({"mu": mu, "seq": int(latents.shape[1]), "steps": steps, "seed": seed, "size": size, "prompt": prompt},
          open(os.path.join(out, "meta.json"), "w"))

# ---- stage 3: the DiT ------------------------------------------------------
tr = tr if fp32 else tr.to(dev)
cur = latents.to(torch.float32) if fp32 else latents
emb = embeds.float() if fp32 else embeds
tdev = torch.device("cpu") if fp32 else dev
with torch.no_grad():
    for i, t in enumerate(timesteps):
        ts = t.expand(cur.shape[0]).to(cur.dtype)
        hooks = []
        if i == 0:                  # per-block checkpoints of the first step, to localise a mismatch in the port (hooks must return None: a return value replaces the module output)
            def hook(fn):
                def h(m, a, o): fn(o)
                return h
            hooks.append(tr.time_guidance_embed.register_forward_hook(hook(lambda o: save("dbg_temb", o))))
            hooks.append(tr.x_embedder.register_forward_hook(hook(lambda o: save("dbg_x", o))))
            hooks.append(tr.context_embedder.register_forward_hook(hook(lambda o: save("dbg_ctx", o))))
            for k, blk in enumerate(tr.transformer_blocks):
                hooks.append(blk.register_forward_hook(hook(lambda o, k=k: (save(f"dbg_double{k + 1}_txt", o[0]), save(f"dbg_double{k + 1}_img", o[1])))))
            for k, blk in enumerate(tr.single_transformer_blocks):
                if k in (0, 4, 9, 14, 19):
                    hooks.append(blk.register_forward_hook(hook(lambda o, k=k: save(f"dbg_single{k + 1}", o))))
        pred = tr(hidden_states=cur.to(tdev), timestep=(ts / 1000).to(tdev), guidance=None, encoder_hidden_states=emb.to(tdev),
                  txt_ids=ids_txt.to(tdev), img_ids=ids.to(tdev), return_dict=False)[0]
        for h in hooks: h.remove()
        save(f"pred_{i}", pred)
        cur = pipe.scheduler.step(pred.to(dev), t, cur.to(dev), return_dict=False)[0]
        save(f"latents_{i}", cur)
        print("step", i, "t", float(t), flush=True)
del tr; gc.collect(); torch.cuda.empty_cache()

# ---- stage 4: un-normalise, unpatchify, VAE --------------------------------
vae = AutoencoderKLFlux2.from_pretrained(os.path.join(src, "vae"), torch_dtype=torch.bfloat16).to(dev)
lat = P._unpack_latents_with_ids(cur.to(torch.bfloat16), ids, lat_h // 2, lat_w // 2)
mean = vae.bn.running_mean.view(1, -1, 1, 1).to(dev, lat.dtype)
std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(dev, lat.dtype)
save("bn_mean", vae.bn.running_mean); save("bn_var", vae.bn.running_var)
lat = lat * std + mean
lat = P._unpatchify_latents(lat)
save("vae_in", lat)
with torch.no_grad():
    img = vae.decode(lat, return_dict=False)[0]
save("image", img)
from PIL import Image
a = (img.float() / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
pil = Image.fromarray((a * 255).round().astype(np.uint8))
pil.save(os.path.join(out, "image.png"))
print("done", pil.size)
