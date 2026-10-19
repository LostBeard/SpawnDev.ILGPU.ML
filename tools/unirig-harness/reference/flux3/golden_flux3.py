"""Reference goldens for FLUX 3 Image Klein (distilled, flow map): the C# port is gated against these, stage by stage.

  python -I golden_flux3.py <repo_dir> <out_dir> "<caption>" [seed=0] [size=512] [steps=16] [alpha=16] [ref1.png,ref2.png,...]

<repo_dir> holds model.safetensors, ae.safetensors and text_encoder/ (black-forest-labs/flux-3-image-klein). The reference classes run one component at a time on a 12 GB card: the text encoder first (8 GB), then the DiT
with every block streamed to the GPU from the checkpoint just for its forward (the 11 GB of weights never sit anywhere whole), then the autoencoder. Files (float32 .npy):
  text_ids.npy [L] token ids, text_tokens.npy [L, 20480] (8 stacked layers), text_pos.npy [L, 4]
  noise.npy [h*w, 128], image_ids.npy [h*w, 4] (rows / cols halved)      cond.npy [Nc, 128], cond_ids.npy [Nc, 4] when references are given (the autoencoder's tokens, l = -(1 + i))
  schedule.npy, pred_<i>.npy (the flow-map velocity), x_<i>.npy (the tokens after step i), vae_in.npy (final tokens), image.npy / image.png
"""
import sys, os, gc, math, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flux3_dit as D
import flux3_ae as AE
from safetensors import safe_open
from safetensors.torch import load_file

repo, out, caption = sys.argv[1], sys.argv[2], sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
size = int(sys.argv[5]) if len(sys.argv) > 5 else 512
steps = int(sys.argv[6]) if len(sys.argv) > 6 else 16
alpha = float(sys.argv[7]) if len(sys.argv) > 7 else 16.0
refs = sys.argv[8].split(",") if len(sys.argv) > 8 else []
os.makedirs(out, exist_ok=True)
dev = torch.device("cuda")


def save(name, t):
    a = t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32)
    np.save(os.path.join(out, name + ".npy"), np.ascontiguousarray(a))


# ---- the autoencoder (references need its encoder first) -----------------------------------------------------------------
ae = AE.AutoEncoder()
ae.load_state_dict(load_file(os.path.join(repo, "ae.safetensors")), strict=True)
ae = ae.float().to(dev).eval()
cond_tokens, cond_ids = None, None
if refs:
    from PIL import Image
    toks, idl = [], []
    for i, path in enumerate(refs):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w * h > 1024 * 1024:
            s = math.sqrt(1024 * 1024 / (w * h)); w, h = int(w * s), int(h * s); img = img.resize((w, h), Image.Resampling.LANCZOS)
        w16, h16 = w // 16 * 16, h // 16 * 16
        left, top = (w - w16) // 2, (h - h16) // 2
        img = img.crop((left, top, left + w16, top + h16))
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().to(dev) / 127.5 - 1.0
        save(f"ref_in_{i}", x)
        t, ids = ae.encode(x)
        ids = ids.float(); ids[:, 1:3] *= 0.5; ids[:, 3] = -(1 + i)
        toks.append(t.to(torch.bfloat16)); idl.append(ids)
    cond_tokens, cond_ids = torch.cat(toks), torch.cat(idl)
    save("cond", cond_tokens); save("cond_ids", cond_ids)

# ---- stage 1: the text encoder (flux3/text_encoder.py) ------------------------------------------------------------------------
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
te_path = os.path.join(repo, "text_encoder")
model = Qwen3VLForConditionalGeneration.from_pretrained(te_path, torch_dtype=torch.bfloat16).to(dev).eval()
processor = AutoProcessor.from_pretrained(te_path)
processor.tokenizer.padding_side = "right"
OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32); PAD_MULTIPLE = 80; MAX_LENGTH = 4096
text = processor.apply_chat_template([{"role": "user", "content": caption}], tokenize=False, add_generation_prompt=True)
open(os.path.join(out, "prompt_text.txt"), "w", encoding="utf8").write(text)
tk = processor.tokenizer
real = tk(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)["input_ids"].shape[1]
length = min(math.ceil(real / PAD_MULTIPLE) * PAD_MULTIPLE, MAX_LENGTH)
inputs = tk(text, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
mask = inputs["attention_mask"].to(dev)
positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
with torch.inference_mode():
    o = model(input_ids=inputs["input_ids"].to(dev), attention_mask=mask, position_ids=positions[None].expand(4, -1, -1), output_hidden_states=True, use_cache=False)
    stacked = torch.stack([o.hidden_states[k] for k in OUTPUT_LAYERS], dim=1)
    tokens = stacked.permute(0, 2, 1, 3).reshape(length, -1).to(torch.bfloat16)
pos = torch.arange(length, device=dev)
text_ids = torch.stack([torch.zeros_like(pos)] * 3 + [pos], dim=-1).float()
save("text_ids", inputs["input_ids"][0].float()); save("text_mask", mask[0]); save("text_tokens", tokens); save("text_pos", text_ids)
print("text", tuple(tokens.shape), "real tokens", real, flush=True)
del model, o, stacked; gc.collect(); torch.cuda.empty_cache()

# ---- stage 2: noise, ids, schedule ---------------------------------------------------------------------------------------------
h = w = size // 16
noise = torch.randn((h * w, 128), generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16).float().to(dev)
lat_ids = AE.latent_ids(h, w, dev)                      # the decoder wants the plain grid ids
ids = lat_ids.float().clone(); ids[:, 1:3] *= 0.5                   # the DiT sees rows and columns halved
save("noise", noise); save("image_ids", ids)
t = torch.linspace(1.0, 0.0, steps + 1); schedule = alpha * t / (1.0 + (alpha - 1.0) * t)
save("schedule", schedule)
json.dump({"caption": caption, "seed": seed, "size": size, "steps": steps, "alpha": alpha, "refs": refs, "text_tokens": int(length), "image_tokens": h * w}, open(os.path.join(out, "meta.json"), "w"))

# ---- stage 3: the DiT, its blocks streamed from the checkpoint ------------------------------------------------------------
with torch.device("meta"):
    dit = D.DiT(flow_map=True)
f = safe_open(os.path.join(repo, "model.safetensors"), framework="pt", device="cpu")


def load_params(module, prefix):
    sd = {n: f.get_tensor(prefix + n).to(dev) for n, _ in module.named_parameters()}
    module.load_state_dict(sd, strict=True, assign=True)


def stream(module, prefix):
    module.register_forward_pre_hook(lambda m, a: load_params(m, prefix))
    module.register_forward_hook(lambda m, a, o: (m.to_empty(device="meta"), None)[1])        # a hook must return None: a returned value replaces the output


for mode in ("image", "image_cond"):
    for d_, blk in enumerate(dit.content_mode_blocks[mode]): stream(blk, f"content_mode_blocks.{mode}.{d_}.")
for d_, blk in enumerate(dit.txt_mode_blocks): stream(blk, f"txt_mode_blocks.{d_}.")
for d_, blk in enumerate(dit.single_blocks): stream(blk, f"single_blocks.{d_}.")
for name in ("emb_in", "txt_in", "time_in", "ref_time_in", "early_stream_modulations", "single_stream_modulations", "final_layer"):
    mod = getattr(dit, name)
    load_params(mod, name + ".")


def pack(samples):
    data = {m: [] for m in D.MODES}; idl = {m: [] for m in D.MODES}; mi = {m: [] for m in D.MODES}; counts = dict.fromkeys(D.MODES, 0)
    for text_, cond_, image_ in samples:
        for mode, item in (("text", text_), ("image_cond", cond_), ("image", image_)):
            start = counts[mode]
            if item is not None:
                data[mode].append(item[0]); idl[mode].append(item[1]); counts[mode] += item[0].shape[0]
            mi[mode].append((start, counts[mode]))
    cat = lambda parts, width, dtype: torch.cat(parts) if parts else torch.zeros((0, width), device=dev, dtype=dtype)
    data = {m: cat(data[m], D.IN_CHANNELS[m], torch.bfloat16) for m in D.MODES}; idl = {m: cat(idl[m], 4, torch.float32) for m in D.MODES}
    return D.Packed(data=data, ids=idl, timesteps={m: torch.zeros(data[m].shape[0], device=dev, dtype=torch.bfloat16) for m in D.MODES}, mode_indices=mi)


cond = (cond_tokens, cond_ids) if cond_tokens is not None else None
packed = pack([[(tokens, text_ids), cond, (noise, ids)]])
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for step in range(len(schedule) - 1):
        tt, dt = schedule[step].item(), (schedule[step + 1] - schedule[step]).item()
        packed.timesteps["image"].fill_(tt)
        target = schedule[step + 1].reshape(1).to(dev)
        pred = dit(packed, timesteps_ref=target)
        packed.data["image"] = packed.data["image"] + dt * pred
        save(f"pred_{step}", pred); save(f"x_{step}", packed.data["image"])
        print("step", step, "t", tt, flush=True)
final = packed.data["image"]
save("vae_in", final)
del dit; gc.collect(); torch.cuda.empty_cache()

# ---- stage 4: decode ------------------------------------------------------------------------------------------------------------------
with torch.inference_mode():
    img = ae.decode(final.float(), lat_ids)
save("image", img)
from PIL import Image
a = ((img.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
Image.fromarray(a).save(os.path.join(out, "image.png"))
print("done", a.shape)
