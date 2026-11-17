using ILGPU;
using ILGPU.Runtime;
using UV = ILGPU.Runtime.ArrayView1D<uint, ILGPU.Stride1D.Dense>;
using BV = ILGPU.Runtime.ArrayView1D<ILGPU.BFloat16, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// NVIDIA's NVFP4 weights as ComfyUI's "mixed" checkpoints store them (comfy_quant {"format": "nvfp4"}; e.g. gemma_3_12B_it_fp4_mixed): <c>weight</c> U8 [N, K / 2],
/// <c>weight_scale</c> F8_E4M3 [N, K / 16], <c>weight_scale_2</c> F32 scalar. The layout, MEASURED bit-exact against ComfyUI's own dequantisation (SpawnGen's dump of a
/// Gemma 3 layer, all 15.7M values):
/// <list type="bullet">
/// <item>two 4-bit codes per byte, the HIGH nibble first (element 2 i = byte &gt;&gt; 4, element 2 i + 1 = byte &amp; 15);</item>
/// <item>a code is e2m1: bit 3 the sign, bits 0..2 index {0, 0.5, 1, 1.5, 2, 3, 4, 6};</item>
/// <item>one fp8 e4m3 scale per 16 consecutive inputs, stored in cuBLAS's block-scaled (swizzled) layout: the [N, K / 16] scales padded to
/// [ceil(N / 128) 128, ceil(K / 64) 4], cut in 128 x 4 tiles (row-tile major), each tile stored as 32 x 16 where row r of the 32 holds rows r, r + 32, r + 64, r + 96;
/// scale (n, j) is at tile (n / 128) C + j / 4 (C = ceil(K / 64)), offset (n % 32) 16 + ((n % 128) / 32) 4 + j % 4 within it;</item>
/// <item>W = e2m1 x block scale x weight_scale_2.</item>
/// </list>
/// An e2m1 value times an e4m3 scale has at most 5 significant bits, so <see cref="DecodeExact"/> writes it to bf16 EXACTLY; weight_scale_2 is left to the caller's input
/// (x W^T s2 = (s2 x) W^T), as for the fp8-scaled path (StreamedModelWeights.MatrixExact / ActivationScale). ComfyUI multiplies float32 activations by W in float32.
/// </summary>
public static class Nvfp4
{
    /// <summary>Index of the block scale of row <paramref name="n"/>, block <paramref name="j"/> (one per 16 inputs) in the swizzled storage, for K inputs.</summary>
    public static long ScaleIndex(int n, int j, int K)
    {
        int C = (K / 16 + 3) / 4;
        long tile = (long)(n / 128) * C + j / 4;
        return tile * 512 + (n % 32) * 16 + (n % 128) / 32 * 4 + j % 4;
    }

    /// <summary>Bytes the swizzled scale storage holds for N x K (padded to whole 128 x 4 tiles).</summary>
    public static long ScaleBytes(int N, int K) => (long)((N + 127) / 128) * 128 * ((K / 16 + 3) / 4) * 4;

    private static float E2M1(int code)
    {
        int m = code & 7;
        float v = m == 0 ? 0f : m == 1 ? 0.5f : m == 2 ? 1f : m == 3 ? 1.5f : m == 4 ? 2f : m == 5 ? 3f : m == 6 ? 4f : 6f;
        return (code & 8) != 0 ? -v : v;
    }

    private static float E4M3(int b)
    {
        int e = (b >> 3) & 15, m = b & 7;
        float v = e == 0 ? m * 0.001953125f : (1f + m * 0.125f) * Interop.IntAsFloat((uint)((e - 7 + 127) << 23));   // subnormal: m / 8 * 2^-6
        return (b & 0x80) != 0 ? -v : v;
    }

    private static int ByteAt(UV words, long i) => (int)((words[i >> 2] >> (int)((i & 3) * 8)) & 0xFF);

    private static void DecodeImpl(Index1D idx, UV packed, UV scales, BV dst, int K, int C, float scale)
    {
        long i = idx;
        int n = (int)(i / K), k = (int)(i % K);
        int b = ByteAt(packed, i >> 1);
        int code = (k & 1) == 0 ? b >> 4 : b & 15;
        int j = k >> 4;
        long tile = (long)(n / 128) * C + j / 4;
        long si = tile * 512 + (n % 32) * 16 + (n % 128) / 32 * 4 + j % 4;
        dst[i] = (BFloat16)(E2M1(code) * E4M3(ByteAt(scales, si)) * scale);
    }

    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Accelerator, Action<Index1D, UV, UV, BV, int, int, float>> Kernels = new();

    /// <summary>
    /// dst [N, K] bf16 = e2m1 x block scale (weight_scale_2 NOT applied: pass it as the input's scale). <paramref name="packed"/>: the weight bytes as little-endian words
    /// (N K / 2 bytes); <paramref name="scales"/>: the swizzled fp8 scale bytes as words (<see cref="ScaleBytes"/>).
    /// </summary>
    public static void DecodeExact(Accelerator accelerator, UV packed, UV scales, BV dst, int N, int K) => Decode(accelerator, packed, scales, dst, N, K, 1f);

    /// <summary>
    /// dst [N, K] bf16 = bf16(e2m1 x block scale x <paramref name="scale"/>): with weight_scale_2 the folded decode (each weight rounded to bf16 once); with 1 exactly
    /// <see cref="DecodeExact"/>.
    /// </summary>
    public static void Decode(Accelerator accelerator, UV packed, UV scales, BV dst, int N, int K, float scale)
    {
        if (K % 16 != 0) throw new ArgumentException($"K = {K}: NVFP4 blocks are 16 inputs");
        if ((long)N * K > int.MaxValue) throw new ArgumentException($"[{N}, {K}]: more than {int.MaxValue} elements");
        var kernel = Kernels.GetValue(accelerator, a => a.LoadAutoGroupedStreamKernel<Index1D, UV, UV, BV, int, int, float>(DecodeImpl));
        kernel((int)((long)N * K), packed, scales, dst, K, (K / 16 + 3) / 4, scale);
    }

    /// <summary>The same decode on the host (reference for tests): float values e2m1 x block scale x <paramref name="scale2"/>.</summary>
    public static float[] DecodeHost(byte[] packed, byte[] scales, float scale2, int N, int K)
    {
        var w = new float[(long)N * K];
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
            {
                long i = (long)n * K + k;
                int b = packed[i >> 1], code = (k & 1) == 0 ? b >> 4 : b & 15;
                w[i] = E2M1(code) * E4M3(scales[ScaleIndex(n, k >> 4, K)]) * scale2;
            }
        return w;
    }
}
