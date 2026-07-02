using ILGPU;
using ILGPU.Runtime;
using System.Runtime.InteropServices;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Register-blocked GEMM whose global A/B tile loads are 128-bit F4 (4xf32) loads via
/// <c>AsAligned16()</c> - the vec4-load instrument for the DAv3 beat-ORT campaign.
///
/// Same 64x64-tile / 4x4-register core and identical MAC order as
/// <see cref="RegisterBlockedMatMul"/>; ONLY the global tile loads differ: each of the 256
/// threads loads exactly ONE 16-byte F4 per tile per matrix. On CUDA the construct emits
/// <c>ld.v4.b32</c> (measured 1.06-1.09x - Ada's L1/L2 absorb scalar loads). On WebGPU it
/// currently compiles to a scalar struct load; once the WGSL AsAligned16-trigger lands in
/// SpawnDev.ILGPU it becomes a single <c>vec4&lt;f32&gt;</c> load - the WebGPU-side win this
/// class exists to measure and then carry.
///
/// Aligned shapes only (M%64==0, N%64==0, K%16==0 - all DAv3 GEMM shapes qualify); the
/// launcher throws otherwise. Not yet routed by production MatMul dispatch - promotion
/// happens when the WebGPU A/B number justifies it.
///
/// WebGL: struct-of-4 element loads currently emit invalid GLSL (tracked:
/// geordi-webgl-struct-of-4-load-glsl-bug-tracked-2026-07-01); WebGL cannot issue 128-bit
/// loads regardless. Callers gate WebGL out.
/// </summary>
public class Vec4LoadMatMul
{
    /// <summary>4xf32, 16 bytes - the 128-bit load vehicle (PTX ld.v4.b32 / WGSL vec4&lt;f32&gt;).</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct F4 { public float A, B, C, D; }

    private const int BLOCK = 16;   // 16x16 = 256 threads (WebGPU max group size)
    private const int REG = 4;      // 4x4 outputs per thread
    private const int TILE = BLOCK * REG; // 64x64 output tile

    private readonly Accelerator _accelerator;
    private Action<KernelConfig, ArrayView<F4>, ArrayView<F4>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int>? _kernel;
    private Action<KernelConfig, ArrayView<F4>, ArrayView<F4>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int>? _structLoadKernel;

    public Vec4LoadMatMul(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// C[M,N] = A[M,K] x B[K,N] with A and B supplied as F4-packed row-major data
    /// (element [r,c] = float index r*cols+c; F4 index = float index / 4).
    /// Requires M%64==0, N%64==0, K%16==0 and 256-thread groups.
    /// </summary>
    public void MatMul(ArrayView<F4> A4, ArrayView<F4> B4,
        ArrayView1D<float, Stride1D.Dense> C, int M, int K, int N)
    {
        ValidateShape(M, K, N);
        _kernel ??= _accelerator.LoadStreamKernel<ArrayView<F4>, ArrayView<F4>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int, int>(Vec4LoadImpl);
        _kernel(MakeConfig(M, N), A4, B4, C, M, K, N, N / TILE);
    }

    /// <summary>
    /// Same GEMM, same F4 packing, but the tile loads OMIT <c>AsAligned16()</c> - so they stay
    /// scalar struct loads on every backend even after the WGSL vec4 trigger lands. This is the
    /// A/B middle point that isolates the 128-bit-load effect from the struct-packing effect:
    /// scalar-float GEMM vs THIS = packing; THIS vs the AsAligned16 kernel = load width.
    /// </summary>
    public void MatMulStructLoad(ArrayView<F4> A4, ArrayView<F4> B4,
        ArrayView1D<float, Stride1D.Dense> C, int M, int K, int N)
    {
        ValidateShape(M, K, N);
        _structLoadKernel ??= _accelerator.LoadStreamKernel<ArrayView<F4>, ArrayView<F4>,
            ArrayView1D<float, Stride1D.Dense>, int, int, int, int>(StructLoadImpl);
        _structLoadKernel(MakeConfig(M, N), A4, B4, C, M, K, N, N / TILE);
    }

    private void ValidateShape(int M, int K, int N)
    {
        if (M % TILE != 0 || N % TILE != 0 || K % BLOCK != 0)
            throw new ArgumentException($"Vec4LoadMatMul requires M%{TILE}==0, N%{TILE}==0, K%{BLOCK}==0 (got M={M}, K={K}, N={N})");
        if (_accelerator.MaxNumThreadsPerGroup < BLOCK * BLOCK)
            throw new NotSupportedException($"Vec4LoadMatMul requires {BLOCK * BLOCK}-thread groups (device max {_accelerator.MaxNumThreadsPerGroup})");
    }

    private static KernelConfig MakeConfig(int M, int N) =>
        new(new Index1D((M / TILE) * (N / TILE)), new Index1D(BLOCK * BLOCK));

    // Identical core to RegisterBlockedMatMul.RegBlockedImpl (same shared tiles, same MAC order);
    // only the global tile loads differ. AsAligned16 is applied to the RAW param with all offsets in
    // the index - the exact shape the WGSL AsAligned16-trigger recognizes (and the PTX-proven one).
    private static void Vec4LoadImpl(
        ArrayView<F4> A4, ArrayView<F4> B4,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N, int numTilesN)
    {
        var aTile = SharedMemory.Allocate<float>(TILE * BLOCK);
        var bTile = SharedMemory.Allocate<float>(BLOCK * TILE);

        int tileIdx = Grid.IdxX;
        int tileRow = tileIdx / numTilesN;
        int tileCol = tileIdx % numTilesN;
        int flat = Group.IdxX;             // 0..255
        int threadRow = flat / BLOCK;
        int threadCol = flat % BLOCK;

        int kF4 = K >> 2;                  // row stride of A4 in F4 units
        int nF4 = N >> 2;                  // row stride of B4 in F4 units

        float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
        float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
        float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
        float c30 = 0, c31 = 0, c32 = 0, c33 = 0;

        int numKTiles = K / BLOCK;         // K % 16 == 0 by launcher gate
        for (int t = 0; t < numKTiles; t++)
        {
            // A tile 64x16 floats = 256 F4s: thread f -> row f/4, F4-col f%4. One 128-bit load per thread.
            {
                int row = flat >> 2;
                int colF4 = flat & 3;
                var v = A4.AsAligned16()[(tileRow * TILE + row) * kF4 + t * 4 + colF4];
                int s = row * BLOCK + colF4 * 4;
                aTile[s + 0] = v.A; aTile[s + 1] = v.B; aTile[s + 2] = v.C; aTile[s + 3] = v.D;
            }
            // B tile 16x64 floats = 256 F4s: thread f -> row f/16, F4-col f%16. One 128-bit load per thread.
            {
                int row = flat >> 4;
                int colF4 = flat & 15;
                var v = B4.AsAligned16()[(t * BLOCK + row) * nF4 + tileCol * BLOCK + colF4];
                int s = row * TILE + colF4 * 4;
                bTile[s + 0] = v.A; bTile[s + 1] = v.B; bTile[s + 2] = v.C; bTile[s + 3] = v.D;
            }

            Group.Barrier();

            for (int k = 0; k < BLOCK; k++)
            {
                float a0 = aTile[(threadRow * REG + 0) * BLOCK + k];
                float a1 = aTile[(threadRow * REG + 1) * BLOCK + k];
                float a2 = aTile[(threadRow * REG + 2) * BLOCK + k];
                float a3 = aTile[(threadRow * REG + 3) * BLOCK + k];
                float b0 = bTile[k * TILE + threadCol * REG + 0];
                float b1 = bTile[k * TILE + threadCol * REG + 1];
                float b2 = bTile[k * TILE + threadCol * REG + 2];
                float b3 = bTile[k * TILE + threadCol * REG + 3];
                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
            }

            Group.Barrier();
        }

        int baseRow = tileRow * TILE + threadRow * REG;
        int baseCol = tileCol * TILE + threadCol * REG;
        C[(baseRow + 0) * N + baseCol + 0] = c00; C[(baseRow + 0) * N + baseCol + 1] = c01;
        C[(baseRow + 0) * N + baseCol + 2] = c02; C[(baseRow + 0) * N + baseCol + 3] = c03;
        C[(baseRow + 1) * N + baseCol + 0] = c10; C[(baseRow + 1) * N + baseCol + 1] = c11;
        C[(baseRow + 1) * N + baseCol + 2] = c12; C[(baseRow + 1) * N + baseCol + 3] = c13;
        C[(baseRow + 2) * N + baseCol + 0] = c20; C[(baseRow + 2) * N + baseCol + 1] = c21;
        C[(baseRow + 2) * N + baseCol + 2] = c22; C[(baseRow + 2) * N + baseCol + 3] = c23;
        C[(baseRow + 3) * N + baseCol + 0] = c30; C[(baseRow + 3) * N + baseCol + 1] = c31;
        C[(baseRow + 3) * N + baseCol + 2] = c32; C[(baseRow + 3) * N + baseCol + 3] = c33;
    }

    // Byte-for-byte the same GEMM as Vec4LoadImpl; the ONLY difference is the two tile loads index
    // the raw view without AsAligned16() - permanently scalar struct loads, the packing-only control.
    private static void StructLoadImpl(
        ArrayView<F4> A4, ArrayView<F4> B4,
        ArrayView1D<float, Stride1D.Dense> C,
        int M, int K, int N, int numTilesN)
    {
        var aTile = SharedMemory.Allocate<float>(TILE * BLOCK);
        var bTile = SharedMemory.Allocate<float>(BLOCK * TILE);

        int tileIdx = Grid.IdxX;
        int tileRow = tileIdx / numTilesN;
        int tileCol = tileIdx % numTilesN;
        int flat = Group.IdxX;
        int threadRow = flat / BLOCK;
        int threadCol = flat % BLOCK;

        int kF4 = K >> 2;
        int nF4 = N >> 2;

        float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
        float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
        float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
        float c30 = 0, c31 = 0, c32 = 0, c33 = 0;

        int numKTiles = K / BLOCK;
        for (int t = 0; t < numKTiles; t++)
        {
            {
                int row = flat >> 2;
                int colF4 = flat & 3;
                var v = A4[(tileRow * TILE + row) * kF4 + t * 4 + colF4];
                int s = row * BLOCK + colF4 * 4;
                aTile[s + 0] = v.A; aTile[s + 1] = v.B; aTile[s + 2] = v.C; aTile[s + 3] = v.D;
            }
            {
                int row = flat >> 4;
                int colF4 = flat & 15;
                var v = B4[(t * BLOCK + row) * nF4 + tileCol * BLOCK + colF4];
                int s = row * TILE + colF4 * 4;
                bTile[s + 0] = v.A; bTile[s + 1] = v.B; bTile[s + 2] = v.C; bTile[s + 3] = v.D;
            }

            Group.Barrier();

            for (int k = 0; k < BLOCK; k++)
            {
                float a0 = aTile[(threadRow * REG + 0) * BLOCK + k];
                float a1 = aTile[(threadRow * REG + 1) * BLOCK + k];
                float a2 = aTile[(threadRow * REG + 2) * BLOCK + k];
                float a3 = aTile[(threadRow * REG + 3) * BLOCK + k];
                float b0 = bTile[k * TILE + threadCol * REG + 0];
                float b1 = bTile[k * TILE + threadCol * REG + 1];
                float b2 = bTile[k * TILE + threadCol * REG + 2];
                float b3 = bTile[k * TILE + threadCol * REG + 3];
                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
            }

            Group.Barrier();
        }

        int baseRow = tileRow * TILE + threadRow * REG;
        int baseCol = tileCol * TILE + threadCol * REG;
        C[(baseRow + 0) * N + baseCol + 0] = c00; C[(baseRow + 0) * N + baseCol + 1] = c01;
        C[(baseRow + 0) * N + baseCol + 2] = c02; C[(baseRow + 0) * N + baseCol + 3] = c03;
        C[(baseRow + 1) * N + baseCol + 0] = c10; C[(baseRow + 1) * N + baseCol + 1] = c11;
        C[(baseRow + 1) * N + baseCol + 2] = c12; C[(baseRow + 1) * N + baseCol + 3] = c13;
        C[(baseRow + 2) * N + baseCol + 0] = c20; C[(baseRow + 2) * N + baseCol + 1] = c21;
        C[(baseRow + 2) * N + baseCol + 2] = c22; C[(baseRow + 2) * N + baseCol + 3] = c23;
        C[(baseRow + 3) * N + baseCol + 0] = c30; C[(baseRow + 3) * N + baseCol + 1] = c31;
        C[(baseRow + 3) * N + baseCol + 2] = c32; C[(baseRow + 3) * N + baseCol + 3] = c33;
    }
}
