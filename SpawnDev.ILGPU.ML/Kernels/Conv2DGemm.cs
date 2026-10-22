using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Kernels;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;
using HV = ArrayView1D<global::ILGPU.Half, Stride1D.Dense>;

/// <summary>
/// The ONNX 2D Conv (NCHW, group 1, fp16 weights) as an implicit GEMM on the library's tiled GEMM (tensor cores on CUDA): the activation is moved to channel-last fp32, a neighbour table of the
/// convolution's taps (padding, stride and dilation resolved in it) lets the GEMM gather its A rows, the weight is permuted to [outC][tap][inC] (a scratch copy per call: a few MB against
/// seconds of work), and the result is written back to NCHW in the caller's precision. The direct per-output-element kernels this replaces ran at about 2.5 % of an RTX 4070's FP32 peak and about 4 % of an RX 590's.
/// Scratch buffers are owned here, sized with headroom and only grown (a buffer replaced while a dispatch is pending in a browser command batch would be a use-after-free, and a graph capture records
/// the buffers it ran on: the retired ones live until <see cref="Dispose"/>; growth is geometric so they never exceed the live ones). They are NOT rented from the graph's pool: a captured
/// dispatch plan must keep pointing at buffers the pool's reclaim cannot free (replay of a plan with a reclaimed scratch fails with "A valid external Instance reference no longer exists").
/// </summary>
public sealed class Conv2DGemm : IDisposable
{
    private readonly Accelerator _acc;
    private readonly SparseKernels _sk;
    private readonly PrecisionConvertKernels _cv;
    private readonly List<IDisposable> _retired = new();
    private MemoryBuffer1D<float, Stride1D.Dense>? _xNhwc, _oNhwc;
    private MemoryBuffer1D<int, Stride1D.Dense>? _table;
    private MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>? _wPerm, _biasHalf;
    private readonly Dictionary<Type, object> _toNhwc = new(), _fromNhwc = new();

    /// <summary>Off switch for comparing against the direct kernels (env <c>CONV_GEMM=0</c>).</summary>
    public static bool Enabled = Environment.GetEnvironmentVariable("CONV_GEMM") != "0";

    /// <summary>Above 0 while a graph capture is recording: the direct kernels run then (see SessionGraphCapture).</summary>
    public static int SuspendCount;

    public Conv2DGemm(Accelerator accelerator)
    {
        _acc = accelerator; _sk = new SparseKernels(accelerator); _cv = new PrecisionConvertKernels(accelerator);
    }

    private MemoryBuffer1D<T, Stride1D.Dense> Grow<T>(ref MemoryBuffer1D<T, Stride1D.Dense>? buf, long count) where T : unmanaged
    {
        if (buf != null && buf.Length >= count) return buf;
        long size = buf == null ? count : Math.Max(count, buf.Length + buf.Length / 2);          // 1.5x: the retired buffers stay below the live one
        if (buf != null) _retired.Add(buf);
        buf = _acc.Allocate1D<T>(size);
        return buf;
    }

    // NCHW (any element type) -> NHWC fp32: one thread per output element (channel fastest: coalesced writes)
    private static void ToNhwcImpl<T>(Index1D idx, ArrayView1D<T, Stride1D.Dense> src, FV dst, int C, int HW) where T : unmanaged, INumber<T>
    {
        int c = idx % C, rest = idx / C, p = rest % HW, b = rest / HW;
        dst[idx] = PrecisionConvert.ConvertToSingle(src[(b * C + c) * HW + p]);
    }
    // NHWC fp32 -> NCHW (any element type): one thread per output element (pixel fastest: coalesced writes)
    private static void FromNhwcImpl<T>(Index1D idx, FV src, ArrayView1D<T, Stride1D.Dense> dst, int C, int HW) where T : unmanaged, INumber<T>
    {
        int p = idx % HW, rest = idx / HW, c = rest % C, b = rest / C;
        dst[idx] = PrecisionConvert.ConvertFromSingle<T>(src[(b * HW + p) * C + c]);
    }

    // nbr[m * 27 + t] = row of the input pixel tap t of output pixel m reads (b * H * W + iy * W + ix), or -1 outside the picture (the row stride of 27 is what SparseKernels.ConvTaps takes)
    private Action<Index1D, IV, int, int, int, int, int, int, int, int, int, int, int>? _tableKernel2;
    private static void Table2Impl(Index1D idx, IV nbr, int inH, int inW, int outH, int outW, int kH, int kW, int strideH, int strideW, int padTop, int padLeft, int dil)
    {
        int taps = kH * kW;
        int t = idx % taps, m = idx / taps;
        int ox = m % outW, rest = m / outW, oy = rest % outH, b = rest / outH;
        int ky = t / kW, kx = t % kW;
        int dilH = dil >> 8, dilW = dil & 0xFF;
        int iy = oy * strideH - padTop + ky * dilH, ix = ox * strideW - padLeft + kx * dilW;
        nbr[m * 27 + t] = (iy >= 0 && iy < inH && ix >= 0 && ix < inW) ? (b * inH + iy) * inW + ix : -1;
    }

    /// <summary>True when this convolution takes a GEMM path in <see cref="SparseKernels.ConvTaps"/> (not its naive loop): 1 to 27 taps, the GEMM's K constraint, sizes inside 32-bit indices.</summary>
    public bool Supports(int batch, int inC, int inH, int inW, int outC, int kH, int kW, int outH, int outW)
    {
        if (!Enabled || SuspendCount > 0) return false;
        int taps = kH * kW;
        if (taps < 1 || taps > 27 || inC < 8 || outC < 8 || (outC & 1) != 0) return false;
        long M = (long)batch * outH * outW, rows = (long)batch * inH * inW;
        if (M * 27 >= int.MaxValue || rows * inC >= int.MaxValue || M * outC >= int.MaxValue || (long)outC * taps * inC >= int.MaxValue) return false;
        return _sk.ConvTapsIsGemm((int)M, taps, inC, outC, rows);
    }

    /// <summary>True when <see cref="RunF32"/> can take this convolution: the same shapes as <see cref="Supports"/> on the portable tiled GEMM (fp32 weights keep their precision, so no tensor cores).</summary>
    public bool SupportsExact(int batch, int inC, int inH, int inW, int outC, int kH, int kW, int outH, int outW)
    {
        if (!Enabled || SuspendCount > 0) return false;
        int taps = kH * kW;
        if (taps < 1 || taps > 27 || inC < 8 || outC < 8 || (outC & 1) != 0) return false;
        long M = (long)batch * outH * outW, rows = (long)batch * inH * inW;
        if (M * 27 >= int.MaxValue || rows * inC >= int.MaxValue || M * outC >= int.MaxValue || (long)outC * taps * inC >= int.MaxValue) return false;
        return _sk.ConvTapsExactOk((int)M, taps, inC, outC, rows);
    }

    // fp32 weight [co][ci][taps] -> two fp16 weights [co][taps][ci]: hi = half(w), lo = half((w - hi) * 2048); hi + lo / 2048 carries 22 bits of the mantissa
    private Action<Index1D, FV, HV, HV, int, int>? _splitKernel;
    private static void SplitImpl(Index1D idx, FV src, HV hi, HV lo, int cin, int taps)
    {
        int ci = idx % cin, t = idx / cin % taps, co = idx / (cin * taps);
        float v = src[((long)co * cin + ci) * taps + t];
        // hi = v rounded to 11 significant bits with integer arithmetic (a float -> half -> float round trip can be folded away by a compiler, and then lo is 0: seen on OpenCL),
        // lo = (v - hi) scaled by 2^11 so it stays out of the fp16 subnormal range; a weight below the fp16 normal range goes entirely into lo
        float hiF = 0f;
        if (MathF.Abs(v) >= 6.2e-5f)
        {
            uint bits = Interop.FloatAsInt(v);
            hiF = Interop.IntAsFloat((bits + 0x0FFFu + ((bits >> 13) & 1u)) & 0xFFFFE000u);
        }
        hi[idx] = PrecisionConvert.ConvertFromSingle<global::ILGPU.Half>(hiF);
        lo[idx] = PrecisionConvert.ConvertFromSingle<global::ILGPU.Half>((v - hiF) * 2048f);
    }
    private MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>? _wLo;
    private MemoryBuffer1D<float, Stride1D.Dense>? _oLo;

    /// <summary>
    /// <see cref="Run{T}"/> for an fp32 weight [outC, inC, kH, kW]: the weight is split into a high and a low fp16 half and the convolution is two GEMMs on the portable tiled kernel
    /// (fp32 activations, fp32 accumulation; the low half is scaled by 2^11 and combined with the high one in the layout change): float precision (about 1e-6 relative), twice the GEMM work, about 6x faster than the direct kernel.
    /// </summary>
    public void RunF32<T>(ArrayView1D<T, Stride1D.Dense> x, FV w, FV bias, ArrayView1D<T, Stride1D.Dense> y,
        int batch, int inC, int inH, int inW, int outC, int kH, int kW, int strideH, int strideW, int padTop, int padLeft, int dilH, int dilW, int outH, int outW)
        where T : unmanaged, INumber<T>
    {
        int taps = kH * kW, inHW = inH * inW, outHW = outH * outW;
        int M = batch * outHW;
        long nX = (long)batch * inHW * inC, nO = (long)M * outC, nW = (long)outC * taps * inC;
        var xn = Grow(ref _xNhwc, nX).View.SubView(0, nX); var on = Grow(ref _oNhwc, nO).View.SubView(0, nO);
        var table = Grow(ref _table, (long)M * 27).View.SubView(0, (long)M * 27);
        var hi = Grow(ref _wPerm, nW).View.SubView(0, nW); var lo = Grow(ref _wLo, nW).View.SubView(0, nW); var onLo = Grow(ref _oLo, nO).View.SubView(0, nO);

        ToNhwc<T>(x, xn, inC, inHW, batch * inHW * inC);
        _tableKernel2 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, int, int, int, int, int, int, int, int, int, int, int>(Table2Impl);
        _tableKernel2(M * taps, table, inH, inW, outH, outW, kH, kW, strideH, strideW, padTop, padLeft, (dilH << 8) | dilW);
        _splitKernel ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, HV, HV, int, int>(SplitImpl);
        _splitKernel((int)nW, w, hi, lo, inC, taps);
        _sk.ConvTapsExact(xn, table, hi, on, M, inC, outC, taps, false);
        _sk.ConvTapsExact(xn, table, lo, onLo, M, inC, outC, taps, false);
        FromNhwcBias2<T>(on, onLo, y, bias, outC, outHW, M * outC);          // y = hi product + lo product / 2048 + the fp32 bias, in the layout change: full precision
    }

    /// <summary>
    /// y = conv(x, w) + bias, NCHW in <typeparamref name="T"/> (fp32 or fp16) both ways; <paramref name="w"/> is the ONNX weight [outC, inC, kH, kW] in fp16, <paramref name="bias"/> fp32 [outC].
    /// </summary>
    public void Run<T>(ArrayView1D<T, Stride1D.Dense> x, HV w, FV bias, ArrayView1D<T, Stride1D.Dense> y,
        int batch, int inC, int inH, int inW, int outC, int kH, int kW, int strideH, int strideW, int padTop, int padLeft, int dilH, int dilW, int outH, int outW)
        where T : unmanaged, INumber<T>
    {
        int taps = kH * kW, inHW = inH * inW, outHW = outH * outW;
        int M = batch * outHW;
        long nX = (long)batch * inHW * inC, nO = (long)M * outC, nW = (long)outC * taps * inC;
        var xn = Grow(ref _xNhwc, nX).View.SubView(0, nX); var on = Grow(ref _oNhwc, nO).View.SubView(0, nO);
        var table = Grow(ref _table, (long)M * 27).View.SubView(0, (long)M * 27);
        var wp = Grow(ref _wPerm, nW).View.SubView(0, nW); var bh = Grow(ref _biasHalf, outC).View.SubView(0, outC);

        ToNhwc<T>(x, xn, inC, inHW, batch * inHW * inC);
        _tableKernel2 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, int, int, int, int, int, int, int, int, int, int, int>(Table2Impl);
        _tableKernel2(M * taps, table, inH, inW, outH, outW, kH, kW, strideH, strideW, padTop, padLeft, (dilH << 8) | dilW);
        _sk.PermuteConvWeight(w, wp, inC, outC, taps);
        _cv.FloatToHalf(bias, bh, outC);
        _sk.ConvTaps(xn, table, wp, bh, on, M, inC, outC, taps);
        FromNhwc<T>(on, y, outC, outHW, M * outC);
    }

    private static void FromNhwcBiasImpl<T>(Index1D idx, FV src, ArrayView1D<T, Stride1D.Dense> dst, FV bias, int C, int HW) where T : unmanaged, INumber<T>
    {
        int p = idx % HW, rest = idx / HW, c = rest % C, b = rest / C;
        dst[idx] = PrecisionConvert.ConvertFromSingle<T>(src[(b * HW + p) * C + c] + bias[c]);
    }
    private static void FromNhwcBias2Impl<T>(Index1D idx, FV hi, FV lo, ArrayView1D<T, Stride1D.Dense> dst, FV bias, int C, int HW) where T : unmanaged, INumber<T>
    {
        int p = idx % HW, rest = idx / HW, c = rest % C, b = rest / C;
        int s = (b * HW + p) * C + c;
        dst[idx] = PrecisionConvert.ConvertFromSingle<T>(hi[s] + lo[s] * (1f / 2048f) + bias[c]);
    }
    private readonly Dictionary<Type, object> _fromNhwcBias2 = new();
    private void FromNhwcBias2<T>(FV hi, FV lo, ArrayView1D<T, Stride1D.Dense> dst, FV bias, int C, int HW, int count) where T : unmanaged, INumber<T>
    {
        if (!_fromNhwcBias2.TryGetValue(typeof(T), out var k))
            _fromNhwcBias2[typeof(T)] = k = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, ArrayView1D<T, Stride1D.Dense>, FV, int, int>(FromNhwcBias2Impl<T>);
        ((Action<Index1D, FV, FV, ArrayView1D<T, Stride1D.Dense>, FV, int, int>)k)(count, hi, lo, dst, bias, C, HW);
    }
    private readonly Dictionary<Type, object> _fromNhwcBias = new();
    private void FromNhwcBias<T>(FV src, ArrayView1D<T, Stride1D.Dense> dst, FV bias, int C, int HW, int count) where T : unmanaged, INumber<T>
    {
        if (!_fromNhwcBias.TryGetValue(typeof(T), out var k))
            _fromNhwcBias[typeof(T)] = k = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, ArrayView1D<T, Stride1D.Dense>, FV, int, int>(FromNhwcBiasImpl<T>);
        ((Action<Index1D, FV, ArrayView1D<T, Stride1D.Dense>, FV, int, int>)k)(count, src, dst, bias, C, HW);
    }

    private void ToNhwc<T>(ArrayView1D<T, Stride1D.Dense> src, FV dst, int C, int HW, int count) where T : unmanaged, INumber<T>
    {
        if (!_toNhwc.TryGetValue(typeof(T), out var k))
            _toNhwc[typeof(T)] = k = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<T, Stride1D.Dense>, FV, int, int>(ToNhwcImpl<T>);
        ((Action<Index1D, ArrayView1D<T, Stride1D.Dense>, FV, int, int>)k)(count, src, dst, C, HW);
    }

    private void FromNhwc<T>(FV src, ArrayView1D<T, Stride1D.Dense> dst, int C, int HW, int count) where T : unmanaged, INumber<T>
    {
        if (!_fromNhwc.TryGetValue(typeof(T), out var k))
            _fromNhwc[typeof(T)] = k = _acc.LoadAutoGroupedStreamKernel<Index1D, FV, ArrayView1D<T, Stride1D.Dense>, int, int>(FromNhwcImpl<T>);
        ((Action<Index1D, FV, ArrayView1D<T, Stride1D.Dense>, int, int>)k)(count, src, dst, C, HW);
    }

    public void Dispose()
    {
        foreach (var r in _retired) r.Dispose();
        _retired.Clear();
        _xNhwc?.Dispose(); _oNhwc?.Dispose(); _table?.Dispose(); _wPerm?.Dispose(); _biasHalf?.Dispose(); _wLo?.Dispose(); _oLo?.Dispose();
        _xNhwc = _oNhwc = null; _table = null; _wPerm = _biasHalf = _wLo = null; _oLo = null;
    }
}
