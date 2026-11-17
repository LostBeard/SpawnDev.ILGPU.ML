# LTX-2 fp8 LoRA rounding: black-box reproduction of ComfyUI's seeded stochastic fp8 rounding

TJ 2026-11-17: exact float W + B A stays the default; this is for an OPT-IN ComfyUI-compatible mode. Built from measurements only (never ComfyUI's GPL source).
State and findings: Plans/ltx-2-19b-2026-11-17.md ("LoRA on fp8 layers", "HANDOFF"). Taps: F:/spawndev-ltx2/comfyui-dumps/lora/{fp8_block1,fp8_block1_repeat,fp8_rounding}.

- fp8sr.py: per-element rounding direction vs fractional position (proves textbook stochastic rounding: P(up) = frac).
- uintervals.py: per-element u intervals from SpawnGen's 10 LoRA strengths (u in [lo, hi) for "round up iff u < frac"); writes u_intervals.npy.
- real.bin: the first 200 elements' intervals as float64 (lo[200] then hi[200]), the harness input.
- keyseed.py / seedsearch.py: hypotheses already ruled out (key hashes; seeds 0..20000), need a CPU torch venv (pip install torch --index-url https://download.pytorch.org/whl/cpu).
- SeedHarness (C#): torch's CPU MT19937 (self-test exact vs torch 2.15: f32 = low 24 bits, f64 = hi word first), all 2^32 seeds (torch truncates manual_seed to 32 bits),
  variants f32 / f64 x u / 1 - u, early exit. `dotnet run -c Release -- real.bin 0 4294967295 48` (about 9 min, all cores). `--selftest`; a planted-seed control passes.
