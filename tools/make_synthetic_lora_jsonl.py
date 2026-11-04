"""A synthetic format-3 training file for tools/TrainLoraJsonl (the driver's own test, not training data): each line is a random prompt of
token ids followed by one of a few fixed "answers" chosen by the prompt's first token, so an adapter has something learnable to pick up and the
dev split (unseen prompts, same rule) shows whether it generalises rather than memorises.

Usage: py -3.13 tools/make_synthetic_lora_jsonl.py <out.jsonl> <vocab> [lines=120]
"""
import json
import random
import sys


def main(out, vocab, n=120):
    rng = random.Random(1104)
    answers = [[rng.randrange(100, vocab - 100) for _ in range(5)] for _ in range(4)]
    with open(out, "w", encoding="utf-8") as f:
        for i in range(n):
            key = rng.randrange(4)
            prompt = [1000 + key] + [rng.randrange(100, vocab - 100) for _ in range(rng.randrange(10, 30))]
            f.write(json.dumps({"format": 3, "id": f"syn-{i}", "split": "heldout" if i % 5 == 0 else "train",
                                "prompt_ids": prompt, "target_ids": answers[key], "end_of_turn_ids": [2]}) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), *(int(x) for x in sys.argv[3:4]))
