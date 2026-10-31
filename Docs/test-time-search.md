# Many answers, one vote: batched decoding and self-consistency

A small language model gets a maths problem wrong fairly often, but it seldom gets it wrong the same way twice. Sampling several answers and taking the most
common final answer (self-consistency, Wang et al. 2022) is the cheapest accuracy gain there is for a small model, and on a GPU the several answers do not
cost several times one: decoded together, each step reads the weights once for all of them.

| Piece | What it is |
|---|---|
| `GgufBatchDecoder` | N continuations of one prompt decoded together (Qwen2-family GGUF) |
| `SelfConsistency` | the final answer of a reasoning chain (`\boxed{}`), a normaliser, the vote, an exact early exit |
| `/self-consistency` | the demo page: N answers streaming side by side, the vote updating as they finish |
| `tools/prm-harness/` | the pilot (PyTorch, with a process reward model) and the same evaluation on our engine |
| `tools/llm-bench/` | the GEMV micro-benchmark and the decoder's per-kernel profile |

## Why voting, not a reward model (yet)

Plans/test-time-search-2026-10-31.md: Qwen2.5-Math-1.5B-Instruct on 50 MATH-500 problems (seeded draw), 8 samples at temperature 0.7, top-p 0.8, answers
checked with math_verify. A process reward model (Skywork-o1-Open-PRM-Qwen-2.5-1.5B) scored every sample.

| Method | PyTorch pilot | Our engine (CUDA, Q8_0) |
|---|---|---|
| one greedy answer | 66 | 64 |
| majority of 8 | 80 | 78 |
| PRM best of 8 | 70 to 76 | |
| PRM-weighted majority of 8 | 82 | |
| any of the 8 right (pass@8) | 90 | 90 |

On 50 problems a point of difference is noise (about 6 points of sampling noise); the vote is the gain, the reward model adds about 2 points on top. So the
engine piece built first is batched decoding, which every search method (voting, best-of-N, beam search, tree search with a reward model) needs.

## GgufBatchDecoder

```csharp
using var dec = await GgufBatchDecoder.CreateAsync(accelerator, ggufStream, maxSeq: 1536, maxBatch: 8);
var tok = SentencePieceTokenizer.FromGGUF(dec.Model)!;
var (ids, stops) = ChatTemplates.BuildChatPrompt(dec.Model, tok, messages, thinking: false);
int[][] answers = await dec.GenerateAsync(ids, n: 8, maxNew: 768, temperature: 0.7f, topP: 0.8f, seed: 1, stopIds: stops,
    onToken: (seq, token) => { /* stream */ }, onFinished: seq => { /* this answer is done */ });
var vote = SelfConsistency.Vote(answers.Select(a => tok.Decode(a)).ToList());
```

The prompt is prefilled once and its keys and values copied into one cache region per sequence; each step feeds every live sequence's last token as one
batch. What keeps a step on the device:

- **Quantised matrices as the GGUF stores them**, read by `FusedDequantMatMul.ForwardRows`: a thread decodes one word of weights (block scale read once, the fp16
  exponent built in the float's bits) and multiplies it into all M rows. One weight read serves every sequence.
- **Embedding rows gathered and dequantised on the device.**
- **Exact top-64 and the softmax mass on the device.** The 64th-best chunk maximum is a floor at least 64 tokens reach; the tokens at or above it are gathered and
  sorted (ties to the lower id), and the row's softmax mass comes back as partial sums. A step reads back a few thousand floats instead of the vocabulary for
  every row, and `SampleAsync` makes the same draw as `SampleLogits` on the full row (it reads the row only when the nucleus is wider than 64 tokens).
- **Finished answers leave the batch.** `StepAsync(tokens, sequences)` advances only the live sequences; the cache kernels find each row's region through a
  sequence map, so nothing is copied.
- **Fused layers**: each residual add with the norm after it, q|k|v and gate|up joined into one matrix each where their block types agree, bias, rotary and the cache
  store in one kernel, and split-K attention (a group per row, head and chunk of 256 keys, then a merge). About 11 kernel launches a layer.

Measured, RTX 4070 (CUDA unless marked; `tools/llm-bench/batch_profile.cs`, the gate `GgufBatchDecoder_GreedyMatchesGenerator`):

| | before | now |
|---|---|---|
| Qwen2.5-Math-1.5B Q8_0, one sequence, short context | 38.8 tok/s | 85.8 tok/s |
| same, eight sequences | 226.9 tok/s | 366.2 tok/s |
| same at a 905-token context, one / eight sequences | 30.8 / 147.8 tok/s | 74.2 / 274.8 tok/s |
| Qwen2.5-0.5B Q8_0, four sequences: CUDA / OpenCL / WebGPU | 13.3 / 12.9 / 10.7 tok/s | 530.9 / 603.1 / 240.1 tok/s |
| 50 MATH problems, greedy + 8 samples each | 52.8 min | 22.3 min |

The word-wise GEMVs also serve the single-sequence engine (`Forward` at M = 1): Qwen2.5-0.5B greedy decode went from 94.6 to 140.3 tok/s on CUDA and from 85.7
to 114.5 tok/s on WebGPU with decode capture, token for token the same.

Options: `FusedAttention` (default on where a group can be a head wide), `JoinProjections` (static, env `GGUF_BATCH_SEPARATE=1` off), `ProfileKernels` /
`KernelTimes` (desktop: a sync after each kernel, time by kernel), `GGUF_GEMV_ROWS_GENERIC=1` (the per-element GEMVs).

## SelfConsistency

- `ExtractBoxed(text)`: the last `\boxed{...}` (braces balanced; `\boxed 5` too).
- `Normalize(answer)`: folds the spellings LaTeX allows for one value: `\dfrac` / `\frac`, `\left( \right)`, spacing, `\text{}` units, degrees, percent,
  thousands separators, `x = 5`, trailing zeros. It does not prove `1/2` equal to `0.5`. On the 50 problems its vote scored the same 78 as voting by math_verify's parser (they disagreed on 2 problems, one each way).
- `Vote(texts)`: groups by normalised answer, most votes first, ties to the answer seen first; chains without a boxed answer abstain.
- `IsDecided(vote, pending)`: true when the pending chains could not change the winner even if all joined the runner-up. Stopping then gives the same
  answer sooner; the page does it by default.

## The page

`/self-consistency` loads Qwen2.5-Math-1.5B-Instruct (Q4_K_M, about 1 GB) or Qwen2.5-0.5B-Instruct (Q8_0) and asks one problem N times. In Chrome on a 4070
(WebGPU), eight answers to "What is the sum of all positive divisors of 360?" take 21 s (181 tok/s across the answers) and vote 1170, decided after 5 of 8.
`tools/drive-self-consistency-page.cs` drives it the way a visitor does.

## Gates

| Test | What it proves |
|---|---|
| `GgufBatchDecoder_GreedyMatchesGenerator` | greedy = the single-sequence engine word for word; batched greedy rows identical; device draws = the full-row sampler (96 of 96); the shrinking batch = riding along; split-K attention = the three-kernel path (2-4e-6 at 489 tokens); joined projections = separate (bit-identical). CUDA, OpenCL, WebGPU, each check red-checked |
| `Gemv_Rows_AllTypes_MatchOracle` | the word-wise GEMVs against the CPU dequant oracle, Q8_0 / Q4_K / Q6_K / Q4_0 at M = 1, 3, 8 |
| `SelfConsistency_ExtractNormalizeVote` | extraction, the spellings that must and must not vote together, the vote order, `IsDecided` |

## Licences

Qwen2.5-Math-1.5B-Instruct and Qwen2.5-0.5B-Instruct: Apache-2.0. The GGUF of the maths model is bartowski's quantisation of it. The PRM used in the pilot,
Skywork-o1-Open-PRM-Qwen-2.5-1.5B, is under the Skywork Community License (commercial use allowed; no military or unlawful use; give users the source and the
Agreement; mark modifications; derivatives labelled "Skywork Inside"); it is not part of the library. MATH-500 (HuggingFaceH4, a subset of the MATH benchmark from
OpenAI's prm800k): its dataset card names no licence; only the evaluation reads it, nothing in the library or the demo ships it.
