# Liveness-planned activation arena (BufferPool fragmentation) - plan, 2026-11-03

## Problem (measured)
ReDimNet2-B6 (49 MiB weights) on CUDA after ImmediateRelease: the pool holds **682 MiB for 256 MiB peak live** (Data's Vertex report: the speech stack
shares a 6 GB RTX 2060). `Bench_SpeakerEmbedding_PoolPeak`, run directly (`SpawnDev.ILGPU.ML.DemoConsole.exe CudaTests.Bench_SpeakerEmbedding_PoolPeak`;
PMT keeps only pass/fail).

The pool reuses a free buffer only of the rent's exact power-of-two size, so it settles at the SUM of every size's own peak. Tweaks to that rule, A/B
2026-11-03 (pool total / peak live, MiB, after two embeddings; logged in _TestStats `pool-sizeclass-bestfit-ab`):

| rule | total | live |
|---|---|---|
| power-of-two, exact (current) | 682 | 256 |
| power-of-two, best fit up to 2x | 593 | 288 |
| power-of-two, best fit up to 4x | 618 | 336 |
| 4 classes per octave, exact | 848 | 200 |
| 4 per octave, best fit 2x | 750 | 200 |
| 8 per octave, exact | 826 | 194 |
| 8 per octave, best fit 2x | 733 | 194 |

Finer classes cut the rounding (live 256 -> 194) but add sizes, each with its own idle peak; best fit trades live for total. No rule of a
non-splitting pool gets near live. All reverted; the best (-13%) is not worth a full-sweep default flip when the structural fix supersedes it.

## Fix: plan the activations at compile time (what onnxruntime's memory pattern planner does)
A compiled executor's shapes are fixed, and the executor already knows each tensor's last use (ImmediateRelease's refcounts, including subgraph
outer-scope reads). So: per compiled executor, assign every intermediate an offset in ONE arena buffer by interval packing over its
[producer, last consumer] lifetime (greedy by size, first fit by offset; 256-byte aligned offsets); peak = the packing's high water, near live.

Constraints and steps:
1. **Backends**: CUDA / OpenCL / CPU / Velocity. NOT WebGPU (two views of one buffer bound read_write in one dispatch is a validation error, and the
   ML aliasing guard throws), NOT WebGL / Wasm (their own release rules). Those keep the pool.
2. **Planning pass** after GraphCompiler: dry-run the shape plan, collect (tensor, bytes, first, last) for every pooled intermediate; exclude graph
   outputs, captured (CUDA graph) runs at first, tensors an op keeps across runs (KV caches, `OutputsLiveUntilNextRun`), and anything a node returns
   as a view of an input (Reshape~view) - those share their source's interval.
3. **Rent hook**: `BufferPool.Rent(shape, name)` returns the planned sub-view when the executor has a plan for `name`; unnamed scratch keeps the pool
   (or gets planned too, keyed by node index + ordinal).
4. **Gates**: bit-identical outputs vs the pool on the full sweep (CUDA, OpenCL, CPU); `Bench_SpeakerEmbedding_PoolPeak` total near live; the
   two-sessions forced-reclaim test; a variable-shape model (re-plan per recompiled executor, like ImmediateRelease).
5. **Risks**: ops that read `t.Data.Length` instead of the element count (a sub-view is exact-length, unlike a rounded bucket: this surfaces real
   bugs); in-place ops whose input and output intervals touch; If/Loop bodies sharing the pool (plan per body, or leave bodies on the pool first).

Expected for ReDimNet2: pool ~256-300 MiB instead of 682 (about 400 MiB back on a 6 GB card).

## Built instead (2026-11-03): planned slot reuse from recorded runs
Offsets in one arena would break the pool's identity-keyed records (`_live`, `_namedBuffers`, `_rentable`) and WebGPU's read_write aliasing rule. So:
`BufferPool.PlannedReuse` (Tensors/BufferPool.PlannedReuse.cs) RECORDS each executor's runs (every fp32 rent's bucket and [rent, return] interval on one
clock) until two consecutive rent sequences match (run 1 is not steady state: measured), then packs every rent returned within the run into SLOTS,
greedy by size (largest first, first slot with no overlapping interval). A slot is a real buffer, so every pool record holds; a slot is handed out only
after the recorded Return that freed it (the pool's own safety); a runtime busy guard catches any plan/run mismatch and falls back to the buckets.
Slots are shared across all plans of a pool (runs never overlap), refcounted, so alternating input shapes cost the larger plan, not the sum.
Capture-regime runs drop their executor's plan (capture priming unchanged); a reclaim drops every plan first; a disposed executor's plan is forgotten.
Measured: ReDimNet2-B6 steady state 682 -> 372 MiB on CUDA, bit-identical. Remaining gap to live (256) is greedy-by-size's slot rounding.
Gates: PlannedReuse_RepeatedRuns_BitIdenticalAndPlanned (red with overlapping lifetimes sharing slots unguarded), PlannedReuse_TwoExecutorsOnePool_ShareSlots,
Bench_SpeakerEmbedding_PoolPeak; desktop sweeps with ML_PLANNED_REUSE=1.
