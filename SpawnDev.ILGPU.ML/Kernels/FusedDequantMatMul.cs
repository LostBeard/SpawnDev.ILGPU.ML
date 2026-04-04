using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Fused weight dequantization inside MatMul — dequantize Q4 weights
/// as they load into registers, never expanding the full weight matrix.
///
/// Standard path: dequantize ALL weights (2.3GB → 4.6GB) → MatMul
/// Fused path: load Q4 block → dequantize in register → accumulate → next block
///
/// Memory bandwidth saved: weights stay compressed in GPU memory (Q4 = 4 bits/param).
/// Only expand to FP32 in registers during the actual computation.
///
/// This enables Phi-4 Mini Q4 (2.3GB) to run without a separate dequant step
/// and without doubling GPU memory usage.
///
/// Q4_0 format: blocks of 32 values, each block has:
///   - 1 float16 scale factor (2 bytes)
///   - 32 × 4-bit quantized values (16 bytes)
///   - Total: 18 bytes per 32 values = 4.5 bits/value
/// </summary>
public class FusedDequantMatMul : IDisposable
{
    private readonly Accelerator _accelerator;

    // Byte-access kernel (CUDA/OpenCL/CPU — native byte access)
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<byte, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _kernel;

    // Int-packed kernel (WebGPU/WebGL/Wasm — bytes packed into int32 words)
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _kernelPacked;

    private MemoryBuffer1D<int, Stride1D.Dense>? _lastParamsBuf;

    public FusedDequantMatMul(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// MatMul with Q4_0 quantized weight matrix.
    /// input [M, K] (float) × weight [K, N] (Q4_0 packed) → output [M, N] (float)
    /// Weight is stored as Q4_0 blocks: each 32 values = 18 bytes (2 scale + 16 data).
    /// </summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<byte, Stride1D.Dense> weightQ4,
        ArrayView1D<float, Stride1D.Dense> output,
        int M, int K, int N)
    {
        var paramsData = new int[] { M, K, N };
        _lastParamsBuf?.Dispose();
        _lastParamsBuf = _accelerator.Allocate1D(paramsData);

        // Use int-packed kernel on browser backends where ArrayView<byte> transpilation
        // reads bytes from packed u32 words differently.
        bool usePacked = _accelerator.AcceleratorType == AcceleratorType.WebGPU
            || _accelerator.AcceleratorType == AcceleratorType.WebGL
            || _accelerator.AcceleratorType == AcceleratorType.Wasm;

        if (usePacked)
        {
            _kernelPacked ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(
                FusedDequantMatMulPackedImpl);
            // Reinterpret byte view as int view (4 bytes per int)
            var intView = weightQ4.Cast<byte, int>();
            _kernelPacked(M * N, input, intView, output, _lastParamsBuf.View);
        }
        else
        {
            _kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(
                FusedDequantMatMulImpl);
            _kernel(M * N, input, weightQ4, output, _lastParamsBuf.View);
        }
    }

    /// <summary>
    /// Each thread computes one output element [m, n] by iterating over K,
    /// dequantizing Q4_0 weight blocks on the fly.
    /// </summary>
    private static void FusedDequantMatMulImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<byte, Stride1D.Dense> weightQ4,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int M = p[0], K = p[1], N = p[2];
        int m = idx / N;
        int n = idx % N;
        if (m >= M) return;

        // Q4_0 block layout: 18 bytes per 32 values
        // [scale_fp16 (2 bytes)] [32 × 4-bit values (16 bytes)]
        int blocksPerRow = (K + 31) / 32;
        int bytesPerRow = blocksPerRow * 18;

        float sum = 0f;

        // Iterate over K in blocks of 32
        for (int blockIdx = 0; blockIdx < blocksPerRow; blockIdx++)
        {
            int blockStart = blockIdx * 32;
            int blockOffset = n * bytesPerRow + blockIdx * 18;

            // Read scale factor (stored as FP16 — read 2 bytes, convert)
            int scaleBits = weightQ4[blockOffset] | (weightQ4[blockOffset + 1] << 8);
            float scale = HalfToFloat(scaleBits);

            // Dequantize and accumulate 32 values
            for (int i = 0; i < 32 && blockStart + i < K; i++)
            {
                int byteIdx = blockOffset + 2 + (i / 2); // 2 values per byte
                int nibble = (i % 2 == 0)
                    ? (weightQ4[byteIdx] & 0xF)
                    : (weightQ4[byteIdx] >> 4);

                // Q4_0: value = (nibble - 8) * scale
                float dequantized = (nibble - 8) * scale;

                sum += input[m * K + blockStart + i] * dequantized;
            }
        }

        output[idx] = sum;
    }

    /// <summary>
    /// Packed-int version for WebGPU/WebGL/Wasm where ArrayView&lt;byte&gt; WGSL transpilation
    /// is broken. Reads bytes from packed int32 words with manual bit extraction.
    /// </summary>
    private static void FusedDequantMatMulPackedImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> weightQ4Packed,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int M = p[0], K = p[1], N = p[2];
        int m = idx / N;
        int n = idx % N;
        if (m >= M) return;

        int blocksPerRow = (K + 31) / 32;
        int bytesPerRow = blocksPerRow * 18;

        float sum = 0f;

        for (int blockIdx = 0; blockIdx < blocksPerRow; blockIdx++)
        {
            int blockStart = blockIdx * 32;
            int blockOffset = n * bytesPerRow + blockIdx * 18;

            // Read scale (2 bytes) via packed int extraction
            int scaleBits = ReadByte(weightQ4Packed, blockOffset) |
                           (ReadByte(weightQ4Packed, blockOffset + 1) << 8);
            float scale = HalfToFloat(scaleBits);

            for (int i = 0; i < 32 && blockStart + i < K; i++)
            {
                int byteIdx = blockOffset + 2 + (i / 2);
                int byteVal = ReadByte(weightQ4Packed, byteIdx);
                int nibble = (i % 2 == 0) ? (byteVal & 0xF) : (byteVal >> 4);
                float dequantized = (nibble - 8) * scale;
                sum += input[m * K + blockStart + i] * dequantized;
            }
        }

        output[idx] = sum;
    }

    /// <summary>Extract a single byte from a packed int32 array.</summary>
    private static int ReadByte(ArrayView1D<int, Stride1D.Dense> packed, int byteIndex)
    {
        int wordIndex = byteIndex / 4;
        int byteOffset = byteIndex % 4;
        int word = packed[wordIndex];
        return (word >> (byteOffset * 8)) & 0xFF;
    }

    /// <summary>Convert FP16 bits to float.</summary>
    private static float HalfToFloat(int h)
    {
        int sign = (h >> 15) & 1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;

        if (exp == 0)
        {
            if (mant == 0) return sign == 1 ? -0f : 0f;
            // Subnormal
            float val = mant / 1024f * (1f / 16384f);
            return sign == 1 ? -val : val;
        }
        if (exp == 31)
        {
            return mant == 0
                ? (sign == 1 ? float.NegativeInfinity : float.PositiveInfinity)
                : float.NaN;
        }

        float result = (1f + mant / 1024f) * MathF.Pow(2, exp - 15);
        return sign == 1 ? -result : result;
    }

    public void Dispose() => _lastParamsBuf?.Dispose();
}
