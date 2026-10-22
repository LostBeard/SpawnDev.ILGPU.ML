"""LTX-Video reference: the T5-v1.1-XXL text embeddings, dumped as fixtures for the C# port (stage 2 of Plans/ltx-video-2b-2026-10-22.md).

Runs in WSL (venv ~/wan/venv: diffusers 0.41, transformers 5.x, torch 2.6) against the Hugging Face T5EncoderModel in bf16 on the GPU (what the LTX pipeline does; 9.5 GB, so nothing
else may be on the card), exactly as `LTXPipeline._get_t5_prompt_embeds` calls it: tokenizer padding "max_length" 128, truncation, attention mask, the encoder's last hidden state.

  python ref_t5_dump.py ENCODER_DIR TOKENIZER_DIR OUT_DIR "a prompt" ["another prompt" ...]
Writes OUT_DIR/t5_<i>.npz with ids [128] int64, mask [128] int64, context [128, 4096] float32 (the bf16 hidden states, padding rows included: the DiT masks the padding keys) and the text.
"""
import os
import sys

import numpy as np
import torch
from transformers import T5EncoderModel, T5TokenizerFast


def main():
    enc_dir, tok_dir, out_dir = sys.argv[1:4]
    prompts = sys.argv[4:]
    os.makedirs(out_dir, exist_ok=True)
    tok = T5TokenizerFast.from_pretrained(tok_dir)
    enc = T5EncoderModel.from_pretrained(enc_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda").eval()
    print("encoder on", next(enc.parameters()).device, next(enc.parameters()).dtype, "GPU MB", torch.cuda.memory_allocated() // 2**20)
    for i, text in enumerate(prompts):
        t = tok([text], padding="max_length", max_length=128, truncation=True, add_special_tokens=True, return_tensors="pt")
        with torch.no_grad():
            h = enc(t.input_ids.to("cuda"), attention_mask=t.attention_mask.to("cuda"))[0]
        np.savez(os.path.join(out_dir, f"t5_{i}.npz"), ids=t.input_ids[0].numpy().astype(np.int64), mask=t.attention_mask[0].numpy().astype(np.int64),
                 context=h[0].float().cpu().numpy(), text=np.array(text))
        print(i, repr(text[:60]), "tokens", int(t.attention_mask.sum()), "context rms", float(h[0].float().pow(2).mean().sqrt()))


if __name__ == "__main__":
    main()
