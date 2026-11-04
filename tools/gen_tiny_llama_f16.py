"""A tiny llama GGUF in F16 (matrices) / F32 (norms), the block types GgufBatchDecoder reads, for the unit test that the decoder's unmerged LoRA
(LoraRuntime) reproduces the training graph's forward with the same adapters (Training_DecoderLora_MatchesTrainingForward). Head dim 64 like
real models (the decoder's attention kernels are tuned for it); 2 layers, GQA 2:1. Random weights: the test compares two paths of the engine,
so no torch reference is needed.

Usage: py -3.13 tools/gen_tiny_llama_f16.py   (writes SpawnDev.ILGPU.ML.Demo/wwwroot/test-train/tiny-llama-f16.gguf)
"""
import os

import numpy as np
import gguf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "SpawnDev.ILGPU.ML.Demo", "wwwroot", "test-train", "tiny-llama-f16.gguf")
VOCAB, EMBD, LAYERS, HEADS, KV_HEADS, FFN, CTX = 256, 128, 2, 2, 1, 256, 128
HD = EMBD // HEADS


def main():
    rng = np.random.default_rng(1105)

    def m(*shape, scale=0.08):
        return (scale * rng.standard_normal(shape)).astype(np.float16)

    def norm(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    wr = gguf.GGUFWriter(OUT, "llama")
    wr.add_name("tiny-llama-f16")
    wr.add_context_length(CTX)
    wr.add_embedding_length(EMBD)
    wr.add_block_count(LAYERS)
    wr.add_feed_forward_length(FFN)
    wr.add_head_count(HEADS)
    wr.add_head_count_kv(KV_HEADS)
    wr.add_layer_norm_rms_eps(1e-5)
    wr.add_rope_freq_base(10000.0)
    wr.add_vocab_size(VOCAB)
    wr.add_tokenizer_model("llama")
    wr.add_token_list([f"<t{i}>".encode() for i in range(VOCAB)])
    wr.add_token_scores([0.0] * VOCAB)
    wr.add_token_types([1] * VOCAB)
    wr.add_tensor("token_embd.weight", m(VOCAB, EMBD, scale=0.5))
    for L in range(LAYERS):
        p = f"blk.{L}"
        wr.add_tensor(f"{p}.attn_norm.weight", norm(EMBD))
        wr.add_tensor(f"{p}.attn_q.weight", m(HEADS * HD, EMBD))
        wr.add_tensor(f"{p}.attn_k.weight", m(KV_HEADS * HD, EMBD))
        wr.add_tensor(f"{p}.attn_v.weight", m(KV_HEADS * HD, EMBD))
        wr.add_tensor(f"{p}.attn_output.weight", m(EMBD, HEADS * HD))
        wr.add_tensor(f"{p}.ffn_norm.weight", norm(EMBD))
        wr.add_tensor(f"{p}.ffn_gate.weight", m(FFN, EMBD))
        wr.add_tensor(f"{p}.ffn_up.weight", m(FFN, EMBD))
        wr.add_tensor(f"{p}.ffn_down.weight", m(EMBD, FFN))
    wr.add_tensor("output_norm.weight", norm(EMBD))
    wr.add_tensor("output.weight", m(VOCAB, EMBD, scale=0.2))
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_tensors_to_file()
    wr.close()
    print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
    main()
