# Fuse the eight-node tanh-GELU - 96 nodes to 12 in Kokoro's ALBERT alone

**Status:** specified, not implemented. Held deliberately until the worker-gap verdict lands, so two
optimizer changes are not stacked into one unverified sweep.

## Why

In a browser this engine costs per DISPATCH, not per FLOP. MEASURED on Kokoro/CUDA, `encoder/bert` is
**602 of 1,850 compiled nodes and 1.6% of the compute** - near-free on a GPU, ~900 ms of pure crossing in
a browser. So node COUNT in that subtree is the browser's bill, and the single largest removable block
there is the GELU activation, which the exporter writes out longhand.

MEASURED from the shipped export (`tools/onnx-nodes.mjs <model> albert_layers.0/activation/`), exactly
eight nodes, twelve times over (12 `Tanh` and 12 `Pow` in the bert subtree):

```
Mul   (x, 0.5)                -> half
Pow   (x, 3)                  -> x3
Mul_1 (x3, 0.044715)          -> c
Add   (x, c)                  -> inner
Mul_2 (inner, 0.7978845608)   -> scaled
Tanh  (scaled)                -> t
Add_1 (t, 1)                  -> t1
Mul_3 (half, t1)              -> y
```

96 nodes become 12. Every transformer export that uses tanh-GELU gets the same cut, so this is not a
Kokoro special case.

## 🔴 It must NOT fuse into the existing `Gelu` operator

`ElementWiseKernels.GELUImpl` is the **erf-based exact GELU** (Abramowitz-Stegun, `p/a1..a4`, plus a
clamp outside +-10). The chain above is the **tanh APPROXIMATION**. They are different functions - close,
but different - and quietly swapping one for the other is the same class of defect as the two this port
already paid for: the `Add(x, epsilon)` that strength reduction deleted (correlation 0.45), and the
longhand atan2 chain that is not atan2 at `y == +0` (correlation 0.95). Both produced full,
correctly-shaped, plausible output.

So the pass emits a NEW op - `FusedGeluTanh` - computing the identical formula, and the kernel is written
from the decomposition rather than from a textbook.

## Shape of the work

Follow `FuseAtan2` and `FuseInstanceNorm` (both added 2026-09-14) and `KokoroIstftTail.TryDetect`:

1. **Match structurally, read the constants OUT OF THE GRAPH.** `0.5`, `3`, `0.044715`, `0.7978845608`
   and `1` come from `FloatConstantData`, never from the source. A different export with a different
   constant must be DECLINED, not approximated - it loses the optimisation, it never runs the wrong
   function. (⚠️ `FloatConstantData`, never `ConstantData`: the latter is `int[]`, where `0.5` reads as
   `0` and `3` reads as `3`. That exact trap deleted every epsilon in this model.)
2. **Require a single consumer** on each intermediate (`half`, `x3`, `c`, `inner`, `scaled`, `t`, `t1`),
   or the fusion is deleting a value something else reads. `GraphOptimizer` already has the
   single-consumer helper the other passes use.
3. `LastGeluFused` / `LastGeluRejects` counters beside the existing ones, and the count printed by
   `DemoConsole -- KOKOROSPEAK` next to `fused-instancenorm` / `fused-atan2` - a fusion pass that silently
   stops matching after an exporter changes one attribute shows up there and nowhere else.

## Gate

- A dedicated test in the `MLTestBase.*FusionTests` family: build the eight-node chain, optimise, assert
  exactly one `FusedGeluTanh` and no `Tanh`/`Pow` left, then run it and compare against the CPU value of
  the same formula - and separately assert it does NOT equal the erf GELU, or the test cannot tell the
  two apart and is not testing the thing that matters.
- `Pipeline_Kokoro_MatchesOnnxRuntimeWaveform` must stay at correlation >= 0.9940 with the exact sample
  count. The tanh-GELU is on the text-encoder path, so a wrong one moves every phoneme duration.
- Full sweep: this is `GraphOptimizer`, so every model goes through it (`tools/run-full-gate.cmd`).

## Expected size of the win

84 fewer nodes of 1,850 (4.5%). At the browser's per-dispatch cost that is real but it is NOT the lever -
the open 2.2x worker gap is (3,889 ms in the demo worker vs 1,736 ms in PMT's page, same card, same
graph). Do that first; this compounds with it.
