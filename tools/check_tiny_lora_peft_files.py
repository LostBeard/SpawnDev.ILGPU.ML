"""Read the PEFT files tools/TrainTinyLora wrote with Python's own safetensors and compare them with the torch reference's adapters after
one AdamW step (wwwroot/test-train/tiny-llama-train-ref.json). Checks names, PEFT shapes ([rank, in] / [out, rank]), dtype, values, and
that adapter_config.json says what PEFT needs to load it.

Usage: py -3.13 tools/check_tiny_lora_peft_files.py <dir written by TrainTinyLora>
"""
import json
import os
import sys

import numpy as np
from safetensors.numpy import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "SpawnDev.ILGPU.ML.Demo", "wwwroot", "test-train", "tiny-llama-train-ref.json")


def main(out_dir):
    ref = json.load(open(REF, encoding="utf-8"))
    tensors = load_file(os.path.join(out_dir, "adapter_model.safetensors"))
    cfg = json.load(open(os.path.join(out_dir, "adapter_config.json"), encoding="utf-8"))
    assert cfg["peft_type"] == "LORA" and cfg["r"] == ref["config"]["rank"] and cfg["lora_alpha"] == ref["config"]["alpha"], cfg
    assert sorted(cfg["target_modules"]) == ["q_proj", "v_proj"], cfg["target_modules"]
    assert set(tensors) == set(ref["lora"]), (sorted(tensors), sorted(ref["lora"]))
    worst = 0.0
    for name, r in ref["lora"].items():
        t = tensors[name]
        assert t.dtype == np.float32, (name, t.dtype)
        assert list(t.shape) == r["shape"], (name, t.shape, r["shape"])
        worst = max(worst, float(np.abs(t.reshape(-1) - np.array(r["after_adamw"], dtype=np.float64)).max()))
    print(f"{len(tensors)} tensors, PEFT names and shapes match, max |file - torch after AdamW| = {worst:.3e}")
    assert worst < 1e-4, worst
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
