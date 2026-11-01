# Test-time search for small LLMs (2026-10-31)

TJ's question: client-side search with a process reward model (PRM) or MCTS around a small reasoning model. Before building engine pieces, a pilot measured what each part buys.

## Pilot (`tools/prm-harness/pilot_search.py`, PyTorch on the RTX 4070, 20 min)

Qwen2.5-Math-1.5B-Instruct (Apache-2.0), its CoT prompt, 50 problems of MATH-500 (seeded draw), one greedy answer and 8 samples (temperature 0.7, top-p 0.8) each;
Skywork-o1-Open-PRM-Qwen-2.5-1.5B (Skywork Community License: commercial use allowed; no military / unlawful use; give users the source and the Agreement; mark modifications;
open-source derivatives labelled "Skywork Inside"; PRC law) scores every sample (Skywork's prepare_input, step token "\n", sigmoid of the value head); answers checked with math_verify.

| Method | Accuracy |
|---|---|
| greedy | 66.0 |
| majority of 8 | 80.0 |
| PRM best-of-8 (mean step reward) | 76.0 |
| PRM best-of-8 (min step reward) | 70.0 |
| PRM best-of-8 (last step reward) | 76.0 |
| PRM-weighted majority of 8 | 82.0 |
| pass@8 (any sample right) | 90.0 |

50 problems: about +-6 points of sampling noise. The finding: **sampling several answers and voting is the gain (+14)**; the PRM adds about 2 points on top (within the
noise), and as a best-of-N selector alone it is below plain voting. Consistent with the literature: search helps small policies most, and majority voting is a strong baseline
(Skywork's own table: Qwen2.5-7B on MATH greedy 75.2, majority@64 78.4, PRM best-of-64 79.8).

## The PRM on 200 problems (same seeded draw extended, same settings)

| Method | Accuracy |
|---|---|
| greedy | 73.0 |
| majority of 8 | 78.5 |
| PRM best-of-8 (mean / min / last step reward) | 78.5 / 75.5 / 79.0 |
| PRM-weighted majority of 8 | 80.5 |
| pass@8 | 86.0 |

Paired against plain majority (exact McNemar): the PRM-weighted vote is right on 5 problems the vote misses and wrong on 1 it gets, p = 0.22, not
significant; best-of-8 by the last step's reward 6 against 5, p = 1.0. All four net gains are level-5 problems (33 to 37 of 58; levels 1-4 identical),
a hint that the PRM helps on the hardest problems, not a result. Its price: a second 1.5B model (about 3 GB as the published bf16 weights, not a
GGUF) scoring every sample. Decision: no PRM in the library or the demo for now; revisit with a larger study restricted to hard problems, or a
smaller / quantised PRM.

## Our engine on the same 50 (`tools/prm-harness/sc_eval.cs`, `score_sc.py`)

`GgufBatchDecoder` on CUDA, Qwen2.5-Math-1.5B-Instruct Q8_0 (bartowski's GGUF), the same prompt, sampling and limits, the 8 samples decoded together;
scored with math_verify like the pilot.

| Method | PyTorch pilot | Our engine |
|---|---|---|
| greedy | 66.0 | 64.0 |
| majority of 8 | 80.0 | 78.0 (`SelfConsistency.Vote`; math_verify's parse as the key: also 78.0, differing on 2 problems) |
| pass@8 | 90.0 | 90.0 |

Within the sampling noise of the pilot (different random draws and a Q8_0 model). 44 minutes for the 50, 121 sampled tok/s on average: finished
sequences ride along to the end of the longest, so the batch is often mostly idle rows (next: drop finished rows from the step).

## Decision

1. Build **batched multi-sequence decode** first: N samples of one prompt decoded together (the prompt's KV shared, each sequence its own continuation), so N answers cost
   about one pass over the weights per token instead of N. That alone gives the +14 (self-consistency), and it is the primitive every search method needs (beam search, MCTS
   rollouts, speculative decoding, several chat candidates).
2. A browser demo of self-consistency: one problem, N answers streaming side by side, the vote.
3. The PRM later, if a larger study (or a better small PRM) shows a gain beyond the noise.
