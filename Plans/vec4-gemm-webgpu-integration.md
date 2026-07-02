# Vec4 GEMM production integration plan (Seven, 2026-07-02)

## VERDICT (2026-07-02 PM, measured on SpawnDev.ILGPU 4.17.1-local.1 with the WGSL AsAligned16-trigger LIVE)

**NO production vec4 routing.** WebGPU 3-way (playwright-latest resultText, run 6):
scalar→struct = 1.12-1.14x (layout), struct→vec4 = ZERO (qkv 265.5→267.8ms; fc1 354.0→355.7; fc2
301.7→302.5 - noise). Same shape as CUDA (1.05-1.21x, struct≈vec4) and OpenCL (1.16-1.24x, struct≈vec4):
the access LAYOUT is the whole win on every backend; the 128-bit instruction adds nothing at DAv3 shapes -
drivers coalesce contiguous struct loads already. Additionally WebGPU absolute GEMM cost is per-dispatch-
overhead-dominated (~4-5 apparent GF vs 5000+ GF on the same card via CUDA), so even a real kernel-level
win would be invisible end-to-end until the executor lane (readbacks + dispatch + elide) is fixed.
The ~1.15x layout win is NOT promoted either: it requires F4 repacking gates + aligned-shape routing for
~40ms on a 300ms+ dispatch cycle - revisit ONLY after the executor work lands and kernels re-emerge as a
measurable fraction. `Vec4LoadMatMul` + `GemmVec4Tests` stay as (a) the measurement instrument and (b)
end-to-end consumer regression coverage of the ILGPU AsAligned16→vec4 WGSL trigger (`cc0495c`).

Historical planning content below (kept for the revisit case).

Status: INSTRUMENT phase. `Kernels/Vec4LoadMatMul.cs` + `MLTestBase.GemmVec4Tests.cs` are the 3-way A/B
(scalar-float / F4-struct-load / F4+AsAligned16) at DAv3 shapes. Promotion to production routing happens
ONLY if the post-trigger WebGPU number justifies it (decision data, not hope).

## Measured so far (RTX 4070, 2026-07-02, playwright-latest.json resultText)

| shape (M,K,N) | CUDA vec4/scalar | OpenCL vec4/scalar | struct vs vec4 |
|---|---|---|---|
| qkv 1344,384,1152 | 1.21x | 1.24x | equal (both) |
| fc1 1344,384,1536 | 1.05x | 1.16x | equal-ish (CUDA fc1 struct low - single-run noise, recheck) |
| fc2 1344,1536,384 | 1.09x | 1.23x | equal (both) |

KEY: struct-load ≈ vec4 on both desktop backends → the desktop win is the ACCESS LAYOUT (one contiguous
16-byte element per thread vs 4 strided floats), not the 128-bit instruction. WebGPU pending Geordi's
AsAligned16-trigger package; there the struct variant stays 4 scalar loads, so the same A/B isolates pure
load-width.

## Production routing design (when justified)

- **Zero repack.** `ArrayView<float>.Cast<Vec4LoadMatMul.F4>()` (ILGPU ArrayView.cs Cast) reinterprets the
  existing f32 weight/activation buffers in place; ML weight buffers are 256-byte aligned (allocator rule)
  so the AsAligned16 contract holds at the base. No data movement, zero-copy law preserved.
- **Routing site:** the large-matrix branch of `MatMulKernel.MatMul` / `RegisterBlockedMatMul.MatMul`
  (and `FusedLinear` which shares the core): when backend profits (decided per measurement) AND
  M%64==0 && N%64==0 && K%16==0 → dispatch `Vec4LoadImpl` with Cast views. Else existing scalar path.
  All DAv3 GEMM shapes (qkv/fc1/fc2, M=1344) qualify.
- **Batched + LowP variants:** same one-line load-site change as the f32 kernel if the f32 number
  justifies it; low-p weights keep the PrecisionConvert stage (B tile) - only A-side becomes F4.
- **WebGL:** never routed (no 128-bit loads + tracked GLSL struct-load bug
  `geordi-webgl-struct-of-4-load-glsl-bug-tracked-2026-07-01`).
- **Edge shapes (non-multiple-of-tile):** stay on the existing scalar kernel; do NOT add bounds checks to
  the vec4 kernel (the aligned gate is what makes one 128-bit load per thread per tile possible).

## Open items before promotion

1. Geordi's trigger package → WebGPU A/B numbers (THE gate).
2. Recheck CUDA fc1 struct-load outlier (4717 vs scalar 5601 GF) on an idle machine - single-run noise vs real.
3. If WebGPU wins: verify Cast<F4> binding behaves on WebGPU (binding element type comes from the view
   element - expect fine, prove with the existing GemmVec4 tests run through the routed path).
