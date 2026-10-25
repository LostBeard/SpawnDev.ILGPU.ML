using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;
using HV = ArrayView1D<global::ILGPU.Half, Stride1D.Dense>;
using UV = ArrayView1D<uint, Stride1D.Dense>;

/// <summary>
/// Sparse-voxel kernels of the TRELLIS.2 SC-VAE decoders, fp16 weights / fp32 activations:
///   neighbour tables (GPU hash map), submanifold 3x3x3 convolution as a gathered-row implicit GEMM (tensor cores on CUDA),
///   linear / layer norm with fp16 parameters, subdivision (prefix-sum compaction), channel-to-spatial, skip repeat.
/// The sparse-tensor semantics are the reference's (trellis2/modules/sparse): a voxel set is an int [N,3] coordinate array,
/// order = the order of creation (parent-major, child slot ascending); the kernel index of a 3x3x3 convolution is
/// a0*9 + a1*3 + a2 for the offset (a0-1, a1-1, a2-1) on the coordinate axes (x, y, z), weight layout [Co][27][Ci].
/// </summary>
public sealed class SparseKernels
{
    private readonly Accelerator _acc;
    private MemoryBuffer1D<int, Stride1D.Dense>? _noNbr;
    private MemoryBuffer1D<float, Stride1D.Dense>? _noRes;
    public SparseKernels(Accelerator accelerator) => _acc = accelerator;

    private bool Cuda => _acc.AcceleratorType == AcceleratorType.Cuda && DiTKernels.EnableTensorCores;

    // =====================================================================================================
    // Neighbour table: nbr[i*27 + kk] = index of the voxel at coords[i] + offset(kk), or -1.
    // =====================================================================================================
    private Action<Index1D, IV, IV, IV, int, int, int>? _hashInsert;
    private static void HashInsertImpl(Index1D i, IV coords, IV keys, IV vals, int grid, int mask, int shift)
    {
        int key = (coords[i * 3] * grid + coords[i * 3 + 1]) * grid + coords[i * 3 + 2];
        int slot = (int)(((uint)key * 2654435761u) >> shift);
        while (true)
        {
            int old = Atomic.CompareExchange(ref keys[slot], -1, key);
            if (old == -1 || old == key) { vals[slot] = i; return; }
            slot = (slot + 1) & mask;
        }
    }

    private Action<Index1D, IV, IV, IV, IV, int, int, int>? _hashNeighbors;
    private static void HashNeighborsImpl(Index1D idx, IV coords, IV keys, IV vals, IV nbr, int grid, int mask, int shift)
    {
        int i = idx / 27, kk = idx % 27;
        int x = coords[i * 3] + kk / 9 - 1, y = coords[i * 3 + 1] + (kk / 3) % 3 - 1, z = coords[i * 3 + 2] + kk % 3 - 1;
        int found = -1;
        if (x >= 0 && x < grid && y >= 0 && y < grid && z >= 0 && z < grid)
        {
            int key = (x * grid + y) * grid + z;
            int slot = (int)(((uint)key * 2654435761u) >> shift);
            while (true)
            {
                int k = keys[slot];
                if (k == key) { found = vals[slot]; break; }
                if (k == -1) break;
                slot = (slot + 1) & mask;
            }
        }
        nbr[idx] = found;
    }

    private Action<Index1D, IV>? _fillMinus1;
    private static void FillMinus1Impl(Index1D i, IV v) => v[i] = -1;

    /// <summary>An open-addressing voxel hash (key = (x*grid + y)*grid + z, value = voxel index) living on the device.</summary>
    public sealed class VoxelHash : IDisposable
    {
        internal MemoryBuffer1D<int, Stride1D.Dense> Keys = null!, Vals = null!;
        internal int Grid, Mask, Shift;
        public void Dispose() { Keys?.Dispose(); Vals?.Dispose(); }
    }

    /// <summary>Build the hash of <paramref name="n"/> voxels on a <paramref name="grid"/>^3 lattice (grid^3 must fit a 32-bit key).</summary>
    public VoxelHash BuildHash(IV coords, int n, int grid)
    {
        _hashInsert ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, int, int, int>(HashInsertImpl);
        _fillMinus1 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV>(FillMinus1Impl);
        // the key (x*grid + y)*grid + z is computed with wrapping 32-bit arithmetic and compared as a bit pattern, so it is exact while grid^3 < 2^32 - 1 (grid <= 1625: 1536^3 is 3.6e9); -1 (all bits set) never occurs and marks an empty slot
        if ((long)grid * grid * grid >= uint.MaxValue) throw new ArgumentOutOfRangeException(nameof(grid), "grid^3 must fit a 32-bit key (grid <= 1625)");
        int cap = 64, bits = 6;
        while (cap < 2L * n) { cap <<= 1; bits++; }
        var h = new VoxelHash { Keys = _acc.Allocate1D<int>(cap), Vals = _acc.Allocate1D<int>(cap), Grid = grid, Mask = cap - 1, Shift = 32 - bits };
        _fillMinus1(cap, h.Keys.View);
        _hashInsert(n, coords, h.Keys.View, h.Vals.View, grid, h.Mask, h.Shift);
        return h;
    }

    /// <summary>
    /// Build the 27-neighbour table for <paramref name="n"/> voxels on a <paramref name="grid"/>^3 lattice
    /// (grid^3 must fit a 32-bit key: grid &lt;= 1625). <paramref name="nbr"/> holds n*27 ints.
    /// </summary>
    public void BuildNeighbors(IV coords, int n, int grid, IV nbr)
    {
        _hashNeighbors ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV, int, int, int>(HashNeighborsImpl);
        using var h = BuildHash(coords, n, grid);
        _hashNeighbors(n * 27, coords, h.Keys.View, h.Vals.View, nbr, grid, h.Mask, h.Shift);
        _acc.Fence();   // the hash is freed on return
    }

    /// <summary>WebGPU: build the tiled GEMM's pipeline in the background (one kernel serves every variant; see <see cref="DiTKernels.PrecompileAsync"/>).</summary>
    public Task PrecompileAsync()
    {
        if (_acc is not global::SpawnDev.ILGPU.WebGPU.WebGPUAccelerator web || !UseTiled(TiledGemm.BK)) return Task.CompletedTask;
        return web.PrecompileAsync(TiledGemm.For(_acc), new Index3D(TiledGemm.Threads, 1, 1));   // already started by DiTKernels: deduplicated
    }

    // single lookup, for kernels that probe the hash themselves
    internal static int Lookup(IV keys, IV vals, int x, int y, int z, int grid, int mask, int shift)
    {
        if (x < 0 || x >= grid || y < 0 || y >= grid || z < 0 || z >= grid) return -1;
        int key = (x * grid + y) * grid + z;
        int slot = (int)(((uint)key * 2654435761u) >> shift);
        while (true)
        {
            int k = keys[slot];
            if (k == key) return vals[slot];
            if (k == -1) return -1;
            slot = (slot + 1) & mask;
        }
    }

    // =====================================================================================================
    // Submanifold convolution (3x3x3): out[m,n] = bias[n] + sum_kk sum_c x[nbr[m,kk], c] * W[n,kk,c]
    // =====================================================================================================
    private Action<KernelConfig, ArrayView1D<F4, Stride1D.Dense>, ArrayView1D<U4, Stride1D.Dense>, HV, FV, FV, IV, int, int, int, int, int, int, int>? _mmaConv;
    private Action<Index1D, FV, HV, HV, FV, IV, int, int, int, int, int>? _naiveConv;
    private static void NaiveConvImpl(Index1D idx, FV x, HV w, HV bias, FV o, IV nbr, int M, int cin, int cout, int hasBias, int taps)
    {
        int m = idx / cout, n = idx % cout;
        float acc = hasBias != 0 ? PrecisionConvert.ConvertToSingle(bias[n]) : 0f;
        for (int kk = 0; kk < taps; kk++)
        {
            int src = nbr[m * 27 + kk];
            if (src < 0) continue;
            long xb = (long)src * cin, wb = ((long)n * taps + kk) * cin;
            for (int c = 0; c < cin; c++) acc += x[xb + c] * PrecisionConvert.ConvertToSingle(w[wb + c]);
        }
        o[idx] = acc;
    }

    /// <summary>x [M, cin] fp32 -> o [M, cout] fp32; weight [cout][27][cin] fp16, bias [cout] fp16.</summary>
    public void Conv3(FV x, IV nbr, HV weight, HV bias, FV o, int M, int cin, int cout) => ConvTaps(x, nbr, weight, bias, o, M, cin, cout, 27);

    /// <summary>
    /// <see cref="Conv3"/> with <paramref name="taps"/> taps instead of 27 (a 3x3 convolution is 9, a causal temporal one 3, a pointwise one 1): the neighbour table keeps its row
    /// stride of 27 and fills the first <paramref name="taps"/> entries of every row, the weight is [cout][taps][cin]. The Wan 2.2 VAE's dense causal convolutions use it.
    /// </summary>
    public void ConvTaps(FV x, IV nbr, HV weight, HV bias, FV o, int M, int cin, int cout, int taps)
    {
        if (Cuda && cin % MmaGemm.BK == 0 && cout % 2 == 0)
        {
            _mmaConv ??= _acc.LoadStreamKernel<ArrayView1D<F4, Stride1D.Dense>, ArrayView1D<U4, Stride1D.Dense>, HV, FV, FV, IV, int, int, int, int, int, int, int>(MmaGemm.KernelF16Gather);
            _noRes ??= _acc.Allocate1D<float>(1);
            int tiles = ((M + MmaGemm.BM - 1) / MmaGemm.BM) * ((cout + MmaGemm.BN - 1) / MmaGemm.BN);
            _mmaConv(new KernelConfig(tiles, MmaGemm.Threads), x.Cast<float, F4>(), weight.Cast<global::ILGPU.Half, U4>(), bias, o, _noRes.View, nbr, M, taps * cin, cout, cin, 1, 0, 0);
            return;
        }
        if (UseTiled(taps * cin) && TiledGemm.Fits(M, taps * cin, cout, x.Length / cin, cin))
        {
            _tiledGemm ??= _acc.LoadStreamKernel<FV, UV, UV, FV, FV, IV, int, int, int, int, int, int, int, int>(TiledGemm.For(_acc));
            EnsureTiledDummies();
            int tiles = ((M + TiledGemm.BM - 1) / TiledGemm.BM) * ((cout + TiledGemm.BN - 1) / TiledGemm.BN);
            _tiledGemm(new KernelConfig(tiles, TiledGemm.Threads), x, weight.Cast<global::ILGPU.Half, uint>(), BiasWords(bias), o, _tiledDummyF!.View, nbr,
                M, taps * cin, cout, cin, 1, 0, 0, TiledGemm.ModeF16 | TiledGemm.ModeGather);
            return;
        }
        _naiveConv ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, HV, HV, FV, IV, int, int, int, int, int>(NaiveConvImpl);
        _naiveConv(M * cout, x, weight, bias, o, nbr, M, cin, cout, 1, taps);
    }

    private Action<KernelConfig, ArrayView1D<F4, Stride1D.Dense>, ArrayView1D<U4, Stride1D.Dense>, HV, FV, FV, IV, int, int, int, int, int, int, int>? _mmaLinearF16;
    private Action<Index1D, FV, HV, HV, FV, int, int, int>? _naiveLinearF16;
    private static void NaiveLinearF16Impl(Index1D idx, FV x, HV w, HV bias, FV o, int M, int K, int N)
    {
        int m = idx / N, n = idx % N;
        float acc = PrecisionConvert.ConvertToSingle(bias[n]);
        for (int k = 0; k < K; k++) acc += x[(long)m * K + k] * PrecisionConvert.ConvertToSingle(w[(long)n * K + k]);
        o[idx] = acc;
    }

    /// <summary>nn.Linear with fp16 weight [N, K] and fp16 bias [N]; x [M, K] fp32 -> o [M, N] fp32 (the pointwise convolutions of the Wan 2.2 VAE).</summary>
    public void LinearF16(FV x, HV weight, HV bias, FV o, int M, int K, int N)
    {
        if (Cuda && K % MmaGemm.BK == 0 && N % 2 == 0 && M >= 16)
        {
            _mmaLinearF16 ??= _acc.LoadStreamKernel<ArrayView1D<F4, Stride1D.Dense>, ArrayView1D<U4, Stride1D.Dense>, HV, FV, FV, IV, int, int, int, int, int, int, int>(MmaGemm.KernelF16Dense);
            _noRes ??= _acc.Allocate1D<float>(1); _noNbr ??= _acc.Allocate1D<int>(1);
            int tiles = ((M + MmaGemm.BM - 1) / MmaGemm.BM) * ((N + MmaGemm.BN - 1) / MmaGemm.BN);
            _mmaLinearF16(new KernelConfig(tiles, MmaGemm.Threads), x.Cast<float, F4>(), weight.Cast<global::ILGPU.Half, U4>(), bias, o, _noRes.View, _noNbr.View, M, K, N, 0, 1, 0, 0);
            return;
        }
        if (UseTiled(K) && TiledGemm.Fits(M, K, N))
        {
            _tiledGemm ??= _acc.LoadStreamKernel<FV, UV, UV, FV, FV, IV, int, int, int, int, int, int, int, int>(TiledGemm.For(_acc));
            EnsureTiledDummies();
            _noNbr ??= _acc.Allocate1D<int>(1);
            int tiles = ((M + TiledGemm.BM - 1) / TiledGemm.BM) * ((N + TiledGemm.BN - 1) / TiledGemm.BN);
            _tiledGemm(new KernelConfig(tiles, TiledGemm.Threads), x, weight.Cast<global::ILGPU.Half, uint>(), BiasWords(bias), o, _tiledDummyF!.View, _noNbr.View,
                M, K, N, 0, 1, 0, 0, TiledGemm.ModeF16);
            return;
        }
        _naiveLinearF16 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, HV, HV, FV, int, int, int>(NaiveLinearF16Impl);
        _naiveLinearF16(M * N, x, weight, bias, o, M, K, N);
    }

    // fp32-weight convolution (naive): the few convolutions whose checkpoint tensors are fp32 and tiny (SS decoder input / output)
    private Action<Index1D, FV, FV, FV, FV, IV, int, int, int>? _naiveConvF32;
    private static void NaiveConvF32Impl(Index1D idx, FV x, FV w, FV bias, FV o, IV nbr, int M, int cin, int cout)
    {
        int m = idx / cout, n = idx % cout;
        float acc = bias[n];
        for (int kk = 0; kk < 27; kk++)
        {
            int src = nbr[m * 27 + kk];
            if (src < 0) continue;
            long xb = (long)src * cin, wb = ((long)n * 27 + kk) * cin;
            for (int c = 0; c < cin; c++) acc += x[xb + c] * w[wb + c];
        }
        o[idx] = acc;
    }
    /// <summary>x [M, cin] -> o [M, cout] with fp32 weight [cout][27][cin] and fp32 bias.</summary>
    public void Conv3F32(FV x, IV nbr, FV weight, FV bias, FV o, int M, int cin, int cout)
    {
        _naiveConvF32 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, IV, int, int, int>(NaiveConvF32Impl);
        _naiveConvF32(M * cout, x, weight, bias, o, nbr, M, cin, cout);
    }

    // torch Conv3d weight [Co][Ci][27] (kernel index a0*9 + a1*3 + a2) -> the gather layout [Co][27][Ci]
    private Action<Index1D, HV, HV, int, int>? _permuteF16;
    private static void PermuteF16Impl(Index1D idx, HV src, HV dst, int cin, int cout)
    {
        int ci = idx % cin, kk = idx / cin % 27, co = idx / (cin * 27);
        dst[idx] = src[((long)co * cin + ci) * 27 + kk];
    }
    private Action<Index1D, HV, HV, int, int, int>? _permuteTapsF16;
    private static void PermuteTapsF16Impl(Index1D idx, HV src, HV dst, int cin, int taps, int dummy)
    {
        int ci = idx % cin, t = idx / cin % taps, co = idx / (cin * taps);
        dst[idx] = src[((long)co * cin + ci) * taps + t];
    }
    /// <summary>Torch / ONNX Conv weight [Co][Ci][taps] (taps = kH * kW, row-major) -> the gather layout [Co][taps][Ci] of <see cref="ConvTaps"/>, for any tap count.</summary>
    public void PermuteConvWeight(HV src, HV dst, int cin, int cout, int taps)
    {
        _permuteTapsF16 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, HV, HV, int, int, int>(PermuteTapsF16Impl);
        _permuteTapsF16(cout * taps * cin, src, dst, cin, taps, 0);
    }

    /// <summary>True when <see cref="ConvTapsExact"/> can run (the portable tiled GEMM, never the tensor-core path, which rounds the activations to fp16).</summary>
    public bool ConvTapsExactOk(int M, int taps, int cin, int cout, long xRows) => UseTiled(taps * cin) && TiledGemm.Fits(M, taps * cin, cout, xRows, cin);

    /// <summary>
    /// <see cref="ConvTaps"/> on the portable tiled GEMM with fp32 activations all the way (no tensor cores). <paramref name="addToOutput"/> adds the product to what <paramref name="o"/> already
    /// holds; no bias unless <paramref name="bias"/> is given (an fp16 one, added once: pass it to one call only): two calls with the high and the low fp16 halves of an fp32 weight
    /// give a convolution with the weight's full 22-bit mantissa; the Stable Diffusion VAE passes its fp16 bias (fp32 activations: SDXL's VAE overflows fp16).
    /// </summary>
    public void ConvTapsExact(FV x, IV nbr, HV weight, FV o, int M, int cin, int cout, int taps, bool addToOutput, HV? bias = null)
    {
        _tiledGemm ??= _acc.LoadStreamKernel<FV, UV, UV, FV, FV, IV, int, int, int, int, int, int, int, int>(TiledGemm.For(_acc));
        EnsureTiledDummies();
        int tiles = ((M + TiledGemm.BM - 1) / TiledGemm.BM) * ((cout + TiledGemm.BN - 1) / TiledGemm.BN);
        _tiledGemm(new KernelConfig(tiles, TiledGemm.Threads), x, weight.Cast<global::ILGPU.Half, uint>(), bias.HasValue ? BiasWords(bias.Value) : _tiledDummyU!.View, o, _tiledDummyF!.View, nbr,
            M, taps * cin, cout, cin, bias.HasValue ? 1 : 0, 0, addToOutput ? 2 : 0, TiledGemm.ModeF16 | TiledGemm.ModeGather);
    }

    /// <summary>True when <see cref="ConvTaps"/> would run a GEMM (tensor cores or the tiled kernel) for these sizes instead of its naive loop.</summary>
    public bool ConvTapsIsGemm(int M, int taps, int cin, int cout, long xRows)
        => (Cuda && cin % MmaGemm.BK == 0 && cout % 2 == 0) || (UseTiled(taps * cin) && TiledGemm.Fits(M, taps * cin, cout, xRows, cin));

    private Action<Index1D, FV, FV, int, int>? _permuteF32;
    private static void PermuteF32Impl(Index1D idx, FV src, FV dst, int cin, int cout)
    {
        int ci = idx % cin, kk = idx / cin % 27, co = idx / (cin * 27);
        dst[idx] = src[((long)co * cin + ci) * 27 + kk];
    }
    public void PermuteConvWeight(HV src, HV dst, int cin, int cout)
    {
        _permuteF16 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, HV, HV, int, int>(PermuteF16Impl);
        _permuteF16(cout * 27 * cin, src, dst, cin, cout);
    }
    public void PermuteConvWeight(FV src, FV dst, int cin, int cout)
    {
        _permuteF32 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int>(PermuteF32Impl);
        _permuteF32(cout * 27 * cin, src, dst, cin, cout);
    }

    // LayerNorm with fp32 affine parameters
    private Action<Index1D, FV, FV, FV, FV, int, float>? _layerNormF32;
    private static void LayerNormF32Impl(Index1D m, FV x, FV o, FV gamma, FV beta, int C, float eps)
    {
        long b = (long)(int)m * C;
        float mean = 0f;
        for (int c = 0; c < C; c++) mean += x[b + c];
        mean /= C;
        float var = 0f;
        for (int c = 0; c < C; c++) { float d = x[b + c] - mean; var += d * d; }
        float rstd = 1f / MathF.Sqrt(var / C + eps);
        for (int c = 0; c < C; c++) o[b + c] = (x[b + c] - mean) * rstd * gamma[c] + beta[c];
    }
    public void LayerNormF32(FV x, FV o, FV gamma, FV beta, int rows, int C, float eps)
    {
        _layerNormF32 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, int, float>(LayerNormF32Impl);
        _layerNormF32(rows, x, o, gamma, beta, C, eps);
    }

    // pixel_shuffle_3d on tokens: src [G^3, 8*cc] (channel c*8 + i*4 + j*2 + k) -> dst [(2G)^3, cc], cell (2x+i, 2y+j, 2z+k)
    private Action<Index1D, FV, FV, int, int>? _pixelShuffle;
    private static void PixelShuffleImpl(Index1D idx, FV src, FV dst, int g, int cc)
    {
        int c = idx % cc, t = idx / cc;
        int g2 = g * 2;
        int z2 = t % g2, y2 = t / g2 % g2, x2 = t / (g2 * g2);
        int n = ((x2 >> 1) * g + (y2 >> 1)) * g + (z2 >> 1);
        int sub = (x2 & 1) * 4 + (y2 & 1) * 2 + (z2 & 1);
        dst[idx] = src[(long)n * 8 * cc + c * 8 + sub];
    }
    public void PixelShuffle3d(FV src, FV dst, int g, int cc)
    {
        _pixelShuffle ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int>(PixelShuffleImpl);
        _pixelShuffle(8 * g * g * g * cc, src, dst, g, cc);
    }

    // occupancy: logits > 0, optional 2x max-pool, argwhere (x slowest)
    private Action<Index1D, FV, IV>? _flag;
    private static void FlagImpl(Index1D i, FV logits, IV flag) => flag[i] = logits[i] > 0f ? 1 : 0;
    private Action<Index1D, IV, IV, int>? _maxPool2;
    private static void MaxPool2Impl(Index1D i, IV src, IV dst, int g)
    {
        int h = g / 2;
        int z = i % h, y = i / h % h, x = i / (h * h);
        int any = 0;
        for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int c = 0; c < 2; c++)
            any |= src[((2 * x + a) * g + 2 * y + b) * g + 2 * z + c];
        dst[i] = any;
    }
    private Action<Index1D, IV, IV, IV, int>? _argwhere;
    private static void ArgwhereImpl(Index1D i, IV flag, IV offsets, IV coords, int g)
    {
        if (flag[i] == 0) return;
        int o = offsets[i];
        coords[o * 3] = i / (g * g); coords[o * 3 + 1] = i / g % g; coords[o * 3 + 2] = i % g;
    }

    /// <summary>
    /// Occupied cells of <paramref name="logits"/> (a [grid^3] volume, 1 channel): logit &gt; 0; when <paramref name="outGrid"/> is half of
    /// <paramref name="grid"/> the volume is first max-pooled 2x (any child set). Returns the coordinates [count, 3] in x-slowest order
    /// (torch.argwhere); the caller disposes the buffer.
    /// </summary>
    public MemoryBuffer1D<int, Stride1D.Dense> Occupancy(FV logits, int grid, int outGrid, out int count)
    {
        var (coords, c) = Task.Run(() => OccupancyAsync(logits, grid, outGrid)).GetAwaiter().GetResult();
        count = c;
        return coords;
    }

    public async Task<(MemoryBuffer1D<int, Stride1D.Dense> Coords, int Count)> OccupancyAsync(FV logits, int grid, int outGrid)
    {
        _flag ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV>(FlagImpl);
        _maxPool2 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, int>(MaxPool2Impl);
        _argwhere ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, int>(ArgwhereImpl);
        if (outGrid != grid && outGrid * 2 != grid) throw new ArgumentException("outGrid must equal grid or grid / 2");
        using var flag = _acc.Allocate1D<int>((long)grid * grid * grid);
        _flag(grid * grid * grid, logits, flag.View);
        MemoryBuffer1D<int, Stride1D.Dense>? pooled = null;
        var cells = flag.View;
        if (outGrid != grid)
        {
            pooled = _acc.Allocate1D<int>((long)outGrid * outGrid * outGrid);
            _maxPool2(outGrid * outGrid * outGrid, flag.View, pooled.View, grid);
            cells = pooled.View;
        }
        int n = outGrid * outGrid * outGrid;
        using var offsets = _acc.Allocate1D<int>(n + 1);
        int count = await ExclusiveScanAsync(cells, n, offsets.View);
        var coords = _acc.Allocate1D<int>(Math.Max(1L, (long)count * 3));
        _argwhere(n, cells, offsets.View, coords.View, outGrid);
        _acc.Fence();
        pooled?.Dispose();
        return (coords, count);
    }

    // =====================================================================================================
    // Linear with fp16 weight [N,K] and bias, optional activation and residual:  o = act(x W^T + b) (+ res)
    // =====================================================================================================
    private Action<KernelConfig, ArrayView1D<F4, Stride1D.Dense>, ArrayView1D<U4, Stride1D.Dense>, HV, FV, FV, IV, int, int, int, int, int, int, int>? _mmaLinear;
    private Action<Index1D, FV, HV, HV, FV, FV, int, int, int, int, int, int>? _naiveLinear;
    private static void NaiveLinearImpl(Index1D idx, FV x, HV w, HV bias, FV o, FV res, int M, int K, int N, int hasBias, int act, int hasRes)
    {
        int m = idx / N, n = idx % N;
        float acc = hasBias != 0 ? PrecisionConvert.ConvertToSingle(bias[n]) : 0f;
        long xb = (long)m * K, wb = (long)n * K;
        for (int k = 0; k < K; k++) acc += x[xb + k] * PrecisionConvert.ConvertToSingle(w[wb + k]);
        if (act == 1) acc = acc / (1f + MathF.Exp(-acc));
        if (act == 3)                                                              // exact GELU (erf), Abramowitz-Stegun 7.1.26
        {
            float z = acc * 0.7071067811865476f, az = MathF.Abs(z);
            float t = 1f / (1f + 0.3275911f * az);
            float poly = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
            float erfAbs = 1f - poly * MathF.Exp(-az * az);
            acc = 0.5f * acc * (1f + (z < 0f ? -erfAbs : erfAbs));
        }
        if (hasRes != 0) acc += res[idx];
        o[idx] = acc;
    }

    /// <summary>
    /// x [M,K] fp32; weight [N,K] fp16; act: 0 none, 1 SiLU; res (optional) [M,N] is added after the activation. <paramref name="exact"/> keeps the activations fp32 (the portable
    /// tiled GEMM instead of the tensor cores, which round them to fp16): the Stable Diffusion 2.x attention projections need it, as diffusers' upcast_attention.
    /// </summary>
    public void Linear(FV x, HV weight, HV? bias, FV o, int M, int K, int N, int act = 0, FV? res = null, bool exact = false)
    {
        _normDummyG ??= _acc.Allocate1D<global::ILGPU.Half>(2); _normDummyB ??= _acc.Allocate1D<global::ILGPU.Half>(2); EnsureTiledDummies();
        var b = bias ?? _normDummyG.View;   // separate placeholder: WebGPU forbids binding the weight twice
        if (!exact && Cuda && K % MmaGemm.BK == 0 && N >= 64 && N % 2 == 0 && M >= 16)
        {
            _mmaLinear ??= _acc.LoadStreamKernel<ArrayView1D<F4, Stride1D.Dense>, ArrayView1D<U4, Stride1D.Dense>, HV, FV, FV, IV, int, int, int, int, int, int, int>(MmaGemm.KernelF16Dense);
            _noNbr ??= _acc.Allocate1D<int>(1);
            int tiles = ((M + MmaGemm.BM - 1) / MmaGemm.BM) * ((N + MmaGemm.BN - 1) / MmaGemm.BN);
            _mmaLinear(new KernelConfig(tiles, MmaGemm.Threads), x.Cast<float, F4>(), weight.Cast<global::ILGPU.Half, U4>(), b, o, res ?? o, _noNbr.View, M, K, N, 0, bias.HasValue ? 1 : 0, act, res.HasValue ? 1 : 0);
            return;
        }
        if (UseTiled(K) && TiledGemm.Fits(M, K, N))
        {
            _tiledGemm ??= _acc.LoadStreamKernel<FV, UV, UV, FV, FV, IV, int, int, int, int, int, int, int, int>(TiledGemm.For(_acc));
            EnsureTiledDummies();
            int tiles = ((M + TiledGemm.BM - 1) / TiledGemm.BM) * ((N + TiledGemm.BN - 1) / TiledGemm.BN);
            // a residual that IS the output is added in place (hasRes 2): WebGPU forbids binding one buffer to two slots
            bool inPlace = res.HasValue && SameView(res.Value, o);
            _tiledGemm(new KernelConfig(tiles, TiledGemm.Threads), x, weight.Cast<global::ILGPU.Half, uint>(), bias.HasValue ? BiasWords(bias.Value) : _tiledDummyU!.View,
                o, res.HasValue && !inPlace ? res.Value : _tiledDummyF!.View, _tiledDummyI!.View, M, K, N, 0, bias.HasValue ? 1 : 0, act, !res.HasValue ? 0 : inPlace ? 2 : 1, TiledGemm.ModeF16);
            return;
        }
        _naiveLinear ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, HV, HV, FV, FV, int, int, int, int, int, int>(NaiveLinearImpl);
        _naiveLinear(M * N, x, weight, b, o, res ?? _tiledDummyF!.View, M, K, N, bias.HasValue ? 1 : 0, act, res.HasValue ? 1 : 0);
    }

    // The tiled GEMM reads an fp16 bias as 32-bit words (two halves each). Cast<Half, uint> of an ODD-length view drops the last half: with cout = 3 (the SD VAE's
    // conv_out) the third bias was read past its binding - bounds-checked on WebGPU (a wrong blue channel, image relRMS 3.4e-2), silently past the view on CUDA /
    // OpenCL (where it happened to be right). An odd length or odd start gets a zero-padded copy, made once per bias and kept (biases are never patched).
    private readonly Dictionary<(object, long, long), MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>> _paddedBias = new();
    private UV BiasWords(HV bias)
    {
        IContiguousArrayView cv = bias.BaseView;
        if ((bias.Length & 1) == 0 && (cv.Index & 1) == 0) return bias.Cast<global::ILGPU.Half, uint>();
        var key = ((object)cv.Buffer, cv.Index, bias.Length);
        if (!_paddedBias.TryGetValue(key, out var padded))
        {
            padded = _acc.Allocate1D<global::ILGPU.Half>(bias.Length + 1);
            padded.MemSetToZero();
            padded.View.SubView(0, bias.Length).CopyFrom(bias);
            _paddedBias[key] = padded;
        }
        return padded.View.Cast<global::ILGPU.Half, uint>();
    }

    private static bool SameView(FV a, FV b)
    {
        IContiguousArrayView va = a.BaseView, vb = b.BaseView;
        return va.Buffer == vb.Buffer && va.Index == vb.Index;
    }

    // Portable register-tiled path for every backend without the tensor-core kernels (see DiTKernels.UseTiled).
    private MemoryBuffer1D<global::ILGPU.Half, Stride1D.Dense>? _normDummyG, _normDummyB;
    private bool UseTiled(int K) => DiTKernels.EnableTiled && K % TiledGemm.BK == 0 && _acc.MaxNumThreadsPerGroup >= TiledGemm.Threads && _acc.AcceleratorType != AcceleratorType.WebGL;
    private Action<KernelConfig, FV, UV, UV, FV, FV, IV, int, int, int, int, int, int, int, int>? _tiledGemm;
    private MemoryBuffer1D<uint, Stride1D.Dense>? _tiledDummyU;
    private MemoryBuffer1D<float, Stride1D.Dense>? _tiledDummyF;
    private MemoryBuffer1D<int, Stride1D.Dense>? _tiledDummyI;
    private void EnsureTiledDummies() { _tiledDummyU ??= _acc.Allocate1D<uint>(1); _tiledDummyF ??= _acc.Allocate1D<float>(1); _tiledDummyI ??= _acc.Allocate1D<int>(1); }

    // =====================================================================================================
    // LayerNorm over channels. gamma/beta fp16 (affine) or none; eps per call (1e-6 in the blocks, 1e-5 for the final).
    // =====================================================================================================
    private Action<Index1D, FV, FV, HV, HV, int, int, float>? _layerNorm;
    private static void LayerNormImpl(Index1D m, FV x, FV o, HV gamma, HV beta, int C, int affine, float eps)
    {
        long b = (long)(int)m * C;
        float mean = 0f;
        for (int c = 0; c < C; c++) mean += x[b + c];
        mean /= C;
        float var = 0f;
        for (int c = 0; c < C; c++) { float d = x[b + c] - mean; var += d * d; }
        float rstd = 1f / MathF.Sqrt(var / C + eps);
        for (int c = 0; c < C; c++)
        {
            float v = (x[b + c] - mean) * rstd;
            if (affine != 0) v = v * PrecisionConvert.ConvertToSingle(gamma[c]) + PrecisionConvert.ConvertToSingle(beta[c]);
            o[b + c] = v;
        }
    }
    // group-per-row form (256 threads, two block reductions); the thread-per-row kernel above remains for the CPU accelerator
    private const int NormGroup = 256;
    private Action<KernelConfig, FV, FV, HV, HV, int, int, float>? _layerNormGroup;
    private static void LayerNormGroupImpl(FV x, FV o, HV gamma, HV beta, int C, int affine, float eps)
    {
        int row = Grid.IdxX, t = Group.IdxX;
        long b = (long)row * C;
        var sh = SharedMemory.Allocate<float>(NormGroup);
        float s = 0f;
        for (int c = t; c < C; c += NormGroup) s += x[b + c];
        sh[t] = s;
        Group.Barrier();
        for (int st = NormGroup / 2; st > 0; st >>= 1) { if (t < st) sh[t] += sh[t + st]; Group.Barrier(); }
        float mean = sh[0] / C;
        Group.Barrier();
        float q = 0f;
        for (int c = t; c < C; c += NormGroup) { float d = x[b + c] - mean; q += d * d; }
        sh[t] = q;
        Group.Barrier();
        for (int st = NormGroup / 2; st > 0; st >>= 1) { if (t < st) sh[t] += sh[t + st]; Group.Barrier(); }
        float rstd = 1f / MathF.Sqrt(sh[0] / C + eps);
        for (int c = t; c < C; c += NormGroup)
        {
            float v = (x[b + c] - mean) * rstd;
            if (affine != 0) v = v * PrecisionConvert.ConvertToSingle(gamma[c]) + PrecisionConvert.ConvertToSingle(beta[c]);
            o[b + c] = v;
        }
    }
    public void LayerNorm(FV x, FV o, HV? gamma, HV? beta, int rows, int C, float eps)
    {
        // absent parameters are bound to 1-element placeholders: WebGPU cannot bind an empty (default) view
        _normDummyG ??= _acc.Allocate1D<global::ILGPU.Half>(2); _normDummyB ??= _acc.Allocate1D<global::ILGPU.Half>(2);
        if (_acc.MaxGroupSize.X >= NormGroup)
        {
            _layerNormGroup ??= _acc.LoadStreamKernel<FV, FV, HV, HV, int, int, float>(LayerNormGroupImpl);
            _layerNormGroup(new KernelConfig(rows, NormGroup), x, o, gamma ?? _normDummyG.View, beta ?? _normDummyB.View, C, gamma.HasValue ? 1 : 0, eps);
            return;
        }
        _layerNorm ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, HV, HV, int, int, float>(LayerNormImpl);
        _layerNorm(rows, x, o, gamma ?? _normDummyG.View, beta ?? _normDummyB.View, C, gamma.HasValue ? 1 : 0, eps);
    }

    // =====================================================================================================
    // SiLU (in place) and residual add
    // =====================================================================================================
    private Action<Index1D, FV>? _siluInPlace;
    private static void SiluInPlaceImpl(Index1D i, FV x) => x[i] = x[i] / (1f + MathF.Exp(-x[i]));
    public void SiluInPlace(FV x, int count)
    {
        _siluInPlace ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV>(SiluInPlaceImpl);
        _siluInPlace(count, x);
    }

    // =====================================================================================================
    // Subdivision: logits [N,8] -> children. Child slot s of voxel i exists when logits[i*8+s] > 0.
    // New voxel order = parent-major, slot ascending (sub.nonzero() of the reference). Slot bit0 -> x, bit1 -> y, bit2 -> z.
    // =====================================================================================================
    private Action<Index1D, FV, IV>? _subCount;
    private static void SubCountImpl(Index1D i, FV logits, IV counts)
    {
        int c = 0;
        for (int s = 0; s < 8; s++) if (logits[i * 8 + s] > 0f) c++;
        counts[i] = c;
    }
    private Action<Index1D, IV, IV, int>? _serialScan;
    private static void SerialScanImpl(Index1D _, IV counts, IV offsets, int n)
    {
        int run = 0;
        for (int i = 0; i < n; i++) { offsets[i] = run; run += counts[i]; }
        offsets[n] = run;
    }
    private Action<Index1D, FV, IV, IV, IV, IV, IV>? _subWrite;
    private static void SubWriteImpl(Index1D i, FV logits, IV offsets, IV coords, IV newCoords, IV parent, IV slot)
    {
        int o = offsets[i];
        for (int s = 0; s < 8; s++)
        {
            if (logits[i * 8 + s] > 0f)
            {
                newCoords[o * 3] = coords[i * 3] * 2 + (s & 1);
                newCoords[o * 3 + 1] = coords[i * 3 + 1] * 2 + ((s >> 1) & 1);
                newCoords[o * 3 + 2] = coords[i * 3 + 2] * 2 + ((s >> 2) & 1);
                parent[o] = i;
                slot[o] = s;
                o++;
            }
        }
    }

    /// <summary>Exclusive prefix sum of <paramref name="counts"/> (n ints) into <paramref name="offsets"/> (n+1 ints); returns the total.</summary>
    public int ExclusiveScan(IV counts, int n, IV offsets) => Task.Run(() => ExclusiveScanAsync(counts, n, offsets)).GetAwaiter().GetResult();

    /// <summary><see cref="ExclusiveScan"/> with an awaited readback of the total (browser-safe).</summary>
    public async Task<int> ExclusiveScanAsync(IV counts, int n, IV offsets)
    {
        using var temp = ScanDevice(counts, n, offsets);
        var total = await offsets.SubView(n, 1).CopyToHostAsync();   // one scalar: sizes the output buffers (waits for the scan)
        return total[0];
    }

    // Blocked scan: one thread sums each block of ScanBlock, one thread scans the block sums, one thread per block writes its offsets.
    // (A single-thread scan of the dual-grid edge flags, ~4M ints, is a latency-bound serial loop on the device.)
    private const int ScanBlock = 1024;
    private Action<Index1D, IV, IV, int>? _blockSum;
    private static void BlockSumImpl(Index1D b, IV counts, IV sums, int n)
    {
        int start = b * ScanBlock, end = Math.Min(start + ScanBlock, n), s = 0;
        for (int i = start; i < end; i++) s += counts[i];
        sums[b] = s;
    }
    private Action<Index1D, IV, IV, IV, int>? _blockWrite;
    private static void BlockWriteImpl(Index1D b, IV counts, IV blockOffsets, IV offsets, int n)
    {
        int start = b * ScanBlock, end = Math.Min(start + ScanBlock, n), run = blockOffsets[b];
        for (int i = start; i < end; i++) { offsets[i] = run; run += counts[i]; }
        if (end == n) offsets[n] = run;
    }

    /// <summary>Queue the scan; the returned temporaries must outlive the dispatches (dispose after the next awaited readback).</summary>
    private IDisposable? ScanDevice(IV counts, int n, IV offsets)
    {
        _serialScan ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, int>(SerialScanImpl);
        if (n <= ScanBlock * 4) { _serialScan(1, counts, offsets, n); return null; }
        _blockSum ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, int>(BlockSumImpl);
        _blockWrite ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, int>(BlockWriteImpl);
        int nb = (n + ScanBlock - 1) / ScanBlock;
        var sums = _acc.Allocate1D<int>(nb);                   // two buffers: WebGPU forbids binding one buffer twice in a dispatch
        var blockOffsets = _acc.Allocate1D<int>(nb + 1L);
        _blockSum(nb, counts, sums.View, n);
        _serialScan(1, sums.View, blockOffsets.View, nb);
        _blockWrite(nb, counts, blockOffsets.View, offsets, n);
        return new ScanTemp(sums, blockOffsets);
    }
    private sealed class ScanTemp(IDisposable a, IDisposable b) : IDisposable { public void Dispose() { a.Dispose(); b.Dispose(); } }

    /// <summary>Per-voxel child offsets (n+1 entries; the last is the total) from the subdivision logits.</summary>
    public int CountChildren(FV logits, int n, IV offsets) => Task.Run(() => CountChildrenAsync(logits, n, offsets)).GetAwaiter().GetResult();

    /// <summary><see cref="CountChildren"/> with an awaited readback of the total (browser-safe).</summary>
    public async Task<int> CountChildrenAsync(FV logits, int n, IV offsets)
    {
        _subCount ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV>(SubCountImpl);
        _serialScan ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, int>(SerialScanImpl);
        using var counts = _acc.Allocate1D<int>(n);
        _subCount(n, logits, counts.View);
        using var temp = ScanDevice(counts.View, n, offsets);
        var total = await offsets.SubView(n, 1).CopyToHostAsync();   // one scalar: sizes the output buffers (waits for both kernels)
        return total[0];
    }

    /// <summary>Write the children: <paramref name="newCoords"/> [M,3], <paramref name="parent"/> [M], <paramref name="slot"/> [M].</summary>
    public void WriteChildren(FV logits, IV offsets, IV coords, int n, IV newCoords, IV parent, IV slot)
    {
        _subWrite ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, IV, IV, IV>(SubWriteImpl);
        _subWrite(n, logits, offsets, coords, newCoords, parent, slot);
    }

    // =====================================================================================================
    // Channel2Spatial feature gather: out[m, c] = in[parent[m], slot[m] * cc + c], in is [N, 8*cc]
    // =====================================================================================================
    private Action<Index1D, FV, IV, IV, FV, int>? _c2s;
    private static void C2sImpl(Index1D idx, FV x, IV parent, IV slot, FV o, int cc)
    {
        int m = idx / cc, c = idx % cc;
        o[idx] = x[(long)parent[m] * (8 * cc) + slot[m] * cc + c];
    }
    public void Channel2Spatial(FV x, IV parent, IV slot, FV o, int m, int cc)
    {
        _c2s ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, FV, int>(C2sImpl);
        _c2s(m * cc, x, parent, slot, o, cc);
    }

    // The same gather for a CHUNK of parents: `x` holds only the rows [parentBase, parentBase + rows) of the parents (their conv output), `parent` and `slot` are the sub-ranges of the children of those parents.
    private Action<Index1D, FV, IV, IV, FV, int, int>? _c2sChunk;
    private static void C2sChunkImpl(Index1D idx, FV x, IV parent, IV slot, FV o, int cc, int parentBase)
    {
        int m = idx / cc, c = idx % cc;
        o[idx] = x[(long)(parent[m] - parentBase) * (8 * cc) + slot[m] * cc + c];
    }
    public void Channel2SpatialChunk(FV x, IV parent, IV slot, FV o, int m, int cc, int parentBase)
    {
        _c2sChunk ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, FV, int, int>(C2sChunkImpl);
        _c2sChunk(m * cc, x, parent, slot, o, cc, parentBase);
    }

    // h[m, c] += skip[m, c / r]      (skip_connection = repeat_interleave(out / (C / 8)) followed by the residual add)
    private Action<Index1D, FV, FV, int, int>? _addRepeat;
    private static void AddRepeatImpl(Index1D idx, FV h, FV skip, int cout, int r)
    {
        int m = idx / cout, c = idx % cout;
        h[idx] = h[idx] + skip[(long)m * (cout / r) + c / r];
    }
    public void AddRepeat(FV h, FV skip, int m, int cout, int r)
    {
        _addRepeat ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int>(AddRepeatImpl);
        _addRepeat(m * cout, h, skip, cout, r);
    }
}
