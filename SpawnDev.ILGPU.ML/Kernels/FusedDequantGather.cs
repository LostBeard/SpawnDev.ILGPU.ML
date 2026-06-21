using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.GGUF;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Fused dequantizing axis-0 Gather — embedding lookup directly from a GGUF-quantized
/// table that stays COMPRESSED in GPU memory. Each output element decodes exactly one
/// weight element from its quantization block in-register; the table is never expanded
/// to F32 (a 262k×3840 Q6_K gemma table is ~770MB compressed vs ~4GB as F32, and the
/// alternative CPU dequant pass is unacceptable in interpreted Blazor WASM).
///
/// Table layout: raw GGUF storage = [rows][rowLength contiguous] (row = vocab entry,
/// rowLength = n_embd), the same orientation contract as <see cref="FusedDequantMatMul"/> -
/// for tied embeddings ONE compressed buffer serves both this Gather and the LM-head
/// fused MatMul. Indices arrive as a float GPU tensor (token IDs), cast in-kernel,
/// mirroring GatherKernel.GatherAxis0Float.
///
/// Block layouts are the same verified ggml ports as FusedDequantMatMul; the per-element
/// random-access decode here must agree with the sequential decode there (asserted by the
/// unit tests via the shared CPU oracle). Decode helpers are SHARED (FusedDequantMatMul
/// internals) - one copy of every layout rule.
/// </summary>
public class FusedDequantGather : IDisposable
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _kernelQ4_0, _kernelQ8_0, _kernelQ4_K, _kernelQ6_K, _kernelMXFP4;

    // Per-shape params cache; never disposed mid-session (WebGPU pending-dispatch rule).
    private readonly Dictionary<(int numIdx, int rowLength, int rows), MemoryBuffer1D<int, Stride1D.Dense>> _paramsBufs = new();

    public FusedDequantGather(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>Same fused-supported set as <see cref="FusedDequantMatMul.Supports"/>.</summary>
    public static bool Supports(GGMLType type) => FusedDequantMatMul.Supports(type);

    /// <summary>
    /// output[i, :] = dequantize(table[indices[i], :]) for i in 0..numIdx-1.
    /// table = raw GGUF quantized bytes, [rows][rowLength]; indices = float GPU tensor
    /// (values cast to int in-kernel); rowLength must be a multiple of the block size.
    /// </summary>
    public void GatherAxis0(
        ArrayView1D<byte, Stride1D.Dense> tableQuant,
        ArrayView1D<float, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> output,
        int numIdx, int rowLength, int rows, GGMLType type)
    {
        if (!Supports(type))
            throw new NotSupportedException(
                $"FusedDequantGather does not support GGML type {type}.");
        int blockElems = type is GGMLType.Q4_K or GGMLType.Q6_K ? 256 : 32;
        if (rowLength % blockElems != 0)
            throw new ArgumentException(
                $"rowLength={rowLength} is not a multiple of the {type} block size {blockElems}.");

        if (!_paramsBufs.TryGetValue((numIdx, rowLength, rows), out var paramsBuf))
        {
            paramsBuf = _accelerator.Allocate1D(new[] { numIdx, rowLength, rows });
            _paramsBufs[(numIdx, rowLength, rows)] = paramsBuf;
        }

        var intView = tableQuant.Cast<byte, int>();
        int total = numIdx * rowLength;

        switch (type)
        {
            case GGMLType.Q4_0:
                _kernelQ4_0 ??= Load(GatherQ4_0Impl);
                _kernelQ4_0(total, intView, indices, output, paramsBuf.View);
                break;
            case GGMLType.Q8_0:
                _kernelQ8_0 ??= Load(GatherQ8_0Impl);
                _kernelQ8_0(total, intView, indices, output, paramsBuf.View);
                break;
            case GGMLType.Q4_K:
                _kernelQ4_K ??= Load(GatherQ4_KImpl);
                _kernelQ4_K(total, intView, indices, output, paramsBuf.View);
                break;
            case GGMLType.Q6_K:
                _kernelQ6_K ??= Load(GatherQ6_KImpl);
                _kernelQ6_K(total, intView, indices, output, paramsBuf.View);
                break;
            case GGMLType.MXFP4:
                _kernelMXFP4 ??= Load(GatherMXFP4Impl);
                _kernelMXFP4(total, intView, indices, output, paramsBuf.View);
                break;
            default:
                // Supports() admitted this type but no kernel case handles it - fail loud rather than leave
                // output unwritten (silent garbage). Any type added to Supports MUST get a case above.
                throw new NotSupportedException(
                    $"FusedDequantGather.Supports admits {type} but no gather kernel handles it - add the case.");
        }
    }

    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>> Load(
        Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>> impl) =>
        _accelerator.LoadAutoGroupedStreamKernel(impl);

    // Thread layout for all kernels: idx = gatherIdx * rowLength + col.
    // row = (int)indices[gatherIdx] (clamped); decode element col of table row.

    private static void GatherQ4_0Impl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int numIdx = p[0], rowLength = p[1], rows = p[2];
        int gatherIdx = idx / rowLength;
        int col = idx % rowLength;
        if (gatherIdx >= numIdx) return;
        int row = (int)indices[gatherIdx];
        if (row < 0) row += rows;
        if (row < 0 || row >= rows) { output[idx] = 0f; return; }

        int bytesPerRow = rowLength / 32 * 18;
        int bOff = row * bytesPerRow + (col / 32) * 18;
        int i = col % 32;
        float d = FusedDequantMatMul.HalfToFloat(
            FusedDequantMatMul.ReadByte(w, bOff) | (FusedDequantMatMul.ReadByte(w, bOff + 1) << 8));
        // ggml split order: element i<16 = low nibble of byte i; i>=16 = high nibble of byte i-16.
        int packed = FusedDequantMatMul.ReadByte(w, bOff + 2 + (i < 16 ? i : i - 16));
        int nibble = i < 16 ? (packed & 0xF) : (packed >> 4);
        output[idx] = (nibble - 8) * d;
    }

    // MXFP4 (17 B/block: [e:E8M0][16 nibble bytes]). Same ggml split order as Q4_0 (element i<16 = low nibble
    // of byte i, i>=16 = high nibble of byte i-16); value = E2M1[nibble] * 2^(e-127). Mirrors
    // FusedDequantMatMul's MXFP4 decode (Float8E8M0Extensions.RawBitsToFloat scale × Float4E2M1Extensions.RawBitsToFloat element).
    private static void GatherMXFP4Impl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int numIdx = p[0], rowLength = p[1], rows = p[2];
        int gatherIdx = idx / rowLength;
        int col = idx % rowLength;
        if (gatherIdx >= numIdx) return;
        int row = (int)indices[gatherIdx];
        if (row < 0) row += rows;
        if (row < 0 || row >= rows) { output[idx] = 0f; return; }

        int bytesPerRow = rowLength / 32 * 17;
        int bOff = row * bytesPerRow + (col / 32) * 17;
        int i = col % 32;
        float d = Float8E8M0Extensions.RawBitsToFloat(FusedDequantMatMul.ReadByte(w, bOff));
        int packed = FusedDequantMatMul.ReadByte(w, bOff + 1 + (i < 16 ? i : i - 16));
        int nibble = i < 16 ? (packed & 0xF) : (packed >> 4);
        output[idx] = Float4E2M1Extensions.RawBitsToFloat(nibble) * d;
    }

    private static void GatherQ8_0Impl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int numIdx = p[0], rowLength = p[1], rows = p[2];
        int gatherIdx = idx / rowLength;
        int col = idx % rowLength;
        if (gatherIdx >= numIdx) return;
        int row = (int)indices[gatherIdx];
        if (row < 0) row += rows;
        if (row < 0 || row >= rows) { output[idx] = 0f; return; }

        int bytesPerRow = rowLength / 32 * 34;
        int bOff = row * bytesPerRow + (col / 32) * 34;
        int i = col % 32;
        float d = FusedDequantMatMul.HalfToFloat(
            FusedDequantMatMul.ReadByte(w, bOff) | (FusedDequantMatMul.ReadByte(w, bOff + 1) << 8));
        int q = FusedDequantMatMul.SignExtend8(FusedDequantMatMul.ReadByte(w, bOff + 2 + i));
        output[idx] = q * d;
    }

    private static void GatherQ4_KImpl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int numIdx = p[0], rowLength = p[1], rows = p[2];
        int gatherIdx = idx / rowLength;
        int col = idx % rowLength;
        if (gatherIdx >= numIdx) return;
        int row = (int)indices[gatherIdx];
        if (row < 0) row += rows;
        if (row < 0 || row >= rows) { output[idx] = 0f; return; }

        int bytesPerRow = rowLength / 256 * 144;
        output[idx] = FusedDequantMatMul.DecodeQ4KElement(w, row * bytesPerRow, col);
    }

    private static void GatherQ6_KImpl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int numIdx = p[0], rowLength = p[1], rows = p[2];
        int gatherIdx = idx / rowLength;
        int col = idx % rowLength;
        if (gatherIdx >= numIdx) return;
        int row = (int)indices[gatherIdx];
        if (row < 0) row += rows;
        if (row < 0 || row >= rows) { output[idx] = 0f; return; }

        int bytesPerRow = rowLength / 256 * 210;
        int sbOff = row * bytesPerRow + (col / 256) * 210;
        int r = col % 256;
        // Inverse of the sequential Q6_K walk: half = r/128; within-half wPos = r%128;
        // quarter q = wPos/32; lane l = wPos%32. ql byte = 64*half + (q%2)*32 + l, low
        // nibble for q<2 else high; qh byte = 32*half + l, bits 2q..2q+1; scale index =
        // 8*half + l/16 + 2q (signed int8).
        int half = r / 128;
        int wPos = r % 128;
        int quarter = wPos / 32;
        int l = wPos % 32;

        float d = FusedDequantMatMul.HalfToFloat(
            FusedDequantMatMul.ReadByte(w, sbOff + 208) | (FusedDequantMatMul.ReadByte(w, sbOff + 209) << 8));
        int qlByte = FusedDequantMatMul.ReadByte(w, sbOff + 64 * half + (quarter % 2) * 32 + l);
        int lo4 = quarter < 2 ? (qlByte & 0xF) : (qlByte >> 4);
        int hb = FusedDequantMatMul.ReadByte(w, sbOff + 128 + 32 * half + l);
        int hi2 = (hb >> (2 * quarter)) & 3;
        int q = (lo4 | (hi2 << 4)) - 32;
        int sc = FusedDequantMatMul.SignExtend8(
            FusedDequantMatMul.ReadByte(w, sbOff + 192 + 8 * half + l / 16 + 2 * quarter));
        output[idx] = d * sc * q;
    }

    public void Dispose()
    {
        foreach (var buf in _paramsBufs.Values) buf.Dispose();
        _paramsBufs.Clear();
    }
}
