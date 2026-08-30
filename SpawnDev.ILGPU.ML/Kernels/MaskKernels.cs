using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Generates triangular (causal) attention masks directly on the GPU instead of uploading them.
/// </summary>
/// <remarks>
/// A decoder's causal mask is a pure function of its shape - row r admits column c iff c is on the allowed
/// side of the diagonal - so shipping one over the bus is pure waste. MEASURED on distilgpt2
/// (2026-08-29): the initializer <c>onnx::Slice_260</c> is a 1024x1024 BOOL mask, and because the streaming
/// weight loader handles only FLOAT32/FLOAT16 it could never stream; it was converted BOOL-&gt;float32
/// (turning 1 MiB of mask into 4 MiB) and host-copied, which is the single transfer that tripped
/// <c>BrowserBufferPolicy.StrictHostCopyMaxBytes</c> on every browser backend. Every OTHER tensor taking
/// that path in that model is a 4-byte scalar.
/// <para>
/// One store per thread at its own index, so this is WebGL Transform-Feedback safe like the other
/// element-wise kernels here.
/// </para>
/// </remarks>
public sealed class MaskKernels
{
    private readonly Accelerator _accelerator;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, int, int>? _fill;

    /// <summary>Which side of the diagonal is admitted, and whether the diagonal itself is.</summary>
    public enum TriangleMode
    {
        /// <summary>c &lt;= r (lower, diagonal admitted) - the usual causal mask.</summary>
        LowerInclusive = 0,
        /// <summary>c &lt; r (lower, diagonal excluded).</summary>
        LowerExclusive = 1,
        /// <summary>c &gt;= r (upper, diagonal admitted).</summary>
        UpperInclusive = 2,
        /// <summary>c &gt; r (upper, diagonal excluded).</summary>
        UpperExclusive = 3,
    }

    /// <summary>Creates a new instance bound to <paramref name="accelerator"/>.</summary>
    /// <param name="accelerator">Accelerator the kernel is compiled for.</param>
    public MaskKernels(Accelerator accelerator) => _accelerator = accelerator;

    private static void FillImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> dst, int cols, int mode)
    {
        int r = i / cols;
        int c = i - r * cols;
        bool admitted =
            mode == 0 ? c <= r :
            mode == 1 ? c < r :
            mode == 2 ? c >= r :
                        c > r;
        dst[i] = admitted ? 1f : 0f;
    }

    /// <summary>
    /// Fill <paramref name="dst"/> with a <paramref name="rows"/> x <paramref name="cols"/> triangular mask
    /// of 1f / 0f, row-major.
    /// </summary>
    /// <param name="dst">Destination view, at least rows*cols elements.</param>
    /// <param name="rows">Mask row count.</param>
    /// <param name="cols">Mask column count.</param>
    /// <param name="mode">Which triangle to admit.</param>
    public void FillTriangular(ArrayView1D<float, Stride1D.Dense> dst, int rows, int cols, TriangleMode mode)
    {
        _fill ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, int, int>(FillImpl);
        _fill(rows * cols, dst, cols, (int)mode);
    }

    /// <summary>
    /// Decide whether <paramref name="raw"/> (ONNX BOOL raw_data, one byte per element, row-major) is EXACTLY
    /// a triangular mask, and if so which one.
    /// </summary>
    /// <remarks>
    /// ⚠️ This verifies EVERY element rather than sampling. Substituting a generated mask for one that is
    /// only mostly triangular would silently change what the model computes - a far worse outcome than the
    /// 4 MiB copy this avoids - so anything that does not match exactly falls back to the upload path. The
    /// scan reads the bytes already in the proto and allocates nothing, unlike the float[] conversion it
    /// replaces.
    /// </remarks>
    /// <param name="raw">Raw BOOL bytes, length at least rows*cols.</param>
    /// <param name="rows">Mask row count.</param>
    /// <param name="cols">Mask column count.</param>
    /// <param name="mode">The matching triangle, when this returns true.</param>
    /// <returns>True when the bytes are exactly one of the four triangular forms.</returns>
    public static bool TryDetectTriangular(ReadOnlySpan<byte> raw, int rows, int cols, out TriangleMode mode)
    {
        mode = TriangleMode.LowerInclusive;
        long count = (long)rows * cols;
        if (rows <= 1 || cols <= 1 || raw.Length < count) return false;

        // Pick the candidate from two cells that differ between all four forms, then prove it holds
        // everywhere. Guessing from a corner alone would accept a mask that merely starts out triangular.
        bool d = raw[0] != 0;                       // (0,0) - on the diagonal
        bool lower = raw[cols] != 0;                // (1,0) - below the diagonal
        mode = (lower, d) switch
        {
            (true, true) => TriangleMode.LowerInclusive,
            (true, false) => TriangleMode.LowerExclusive,
            (false, true) => TriangleMode.UpperInclusive,
            _ => TriangleMode.UpperExclusive,
        };

        int m = (int)mode;
        for (int r = 0; r < rows; r++)
        {
            int rowBase = r * cols;
            for (int c = 0; c < cols; c++)
            {
                bool admitted = m == 0 ? c <= r : m == 1 ? c < r : m == 2 ? c >= r : c > r;
                if ((raw[rowBase + c] != 0) != admitted) return false;
            }
        }
        return true;
    }
}
