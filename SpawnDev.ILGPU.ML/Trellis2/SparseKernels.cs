using ILGPU;
using ILGPU.Runtime;

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

    /// <summary>
    /// Build the 27-neighbour table for <paramref name="n"/> voxels on a <paramref name="grid"/>^3 lattice
    /// (grid^3 must fit an int: grid &lt;= 1290). <paramref name="nbr"/> holds n*27 ints.
    /// </summary>
    public void BuildNeighbors(IV coords, int n, int grid, IV nbr)
    {
        _hashInsert ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, int, int, int>(HashInsertImpl);
        _hashNeighbors ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV, int, int, int>(HashNeighborsImpl);
        _fillMinus1 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV>(FillMinus1Impl);
        if ((long)grid * grid * grid >= int.MaxValue) throw new ArgumentOutOfRangeException(nameof(grid), "grid^3 must fit an int key");
        int cap = 64, bits = 6;
        while (cap < 2L * n) { cap <<= 1; bits++; }
        using var keys = _acc.Allocate1D<int>(cap);
        using var vals = _acc.Allocate1D<int>(cap);
        _fillMinus1(cap, keys.View);
        _hashInsert(n, coords, keys.View, vals.View, grid, cap - 1, 32 - bits);
        _hashNeighbors(n * 27, coords, keys.View, vals.View, nbr, grid, cap - 1, 32 - bits);
        _acc.Synchronize();   // keys / vals are freed on return
    }

    // =====================================================================================================
    // Submanifold convolution (3x3x3): out[m,n] = bias[n] + sum_kk sum_c x[nbr[m,kk], c] * W[n,kk,c]
    // =====================================================================================================
    private Action<KernelConfig, FV, UV, HV, FV, FV, IV, int, int, int, int, int, int, int>? _mmaConv;
    private Action<Index1D, FV, HV, HV, FV, IV, int, int, int, int>? _naiveConv;
    private static void NaiveConvImpl(Index1D idx, FV x, HV w, HV bias, FV o, IV nbr, int M, int cin, int cout, int hasBias)
    {
        int m = idx / cout, n = idx % cout;
        float acc = hasBias != 0 ? PrecisionConvert.ConvertToSingle(bias[n]) : 0f;
        for (int kk = 0; kk < 27; kk++)
        {
            int src = nbr[m * 27 + kk];
            if (src < 0) continue;
            long xb = (long)src * cin, wb = ((long)n * 27 + kk) * cin;
            for (int c = 0; c < cin; c++) acc += x[xb + c] * PrecisionConvert.ConvertToSingle(w[wb + c]);
        }
        o[idx] = acc;
    }

    /// <summary>x [M, cin] fp32 -> o [M, cout] fp32; weight [cout][27][cin] fp16, bias [cout] fp16.</summary>
    public void Conv3(FV x, IV nbr, HV weight, HV bias, FV o, int M, int cin, int cout)
    {
        if (Cuda && cin % MmaGemm.BK == 0 && cout % 2 == 0)
        {
            _mmaConv ??= _acc.LoadStreamKernel<FV, UV, HV, FV, FV, IV, int, int, int, int, int, int, int>(MmaGemm.KernelF16Gather);
            _noRes ??= _acc.Allocate1D<float>(1);
            int tiles = ((M + MmaGemm.BM - 1) / MmaGemm.BM) * ((cout + MmaGemm.BN - 1) / MmaGemm.BN);
            _mmaConv(new KernelConfig(tiles, MmaGemm.Threads), x, weight.Cast<global::ILGPU.Half, uint>(), bias, o, _noRes.View, nbr, M, 27 * cin, cout, cin, 1, 0, 0);
            return;
        }
        _naiveConv ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, HV, HV, FV, IV, int, int, int, int>(NaiveConvImpl);
        _naiveConv(M * cout, x, weight, bias, o, nbr, M, cin, cout, 1);
    }

    // =====================================================================================================
    // Linear with fp16 weight [N,K] and bias, optional activation and residual:  o = act(x W^T + b) (+ res)
    // =====================================================================================================
    private Action<KernelConfig, FV, UV, HV, FV, FV, IV, int, int, int, int, int, int, int>? _mmaLinear;
    private Action<Index1D, FV, HV, HV, FV, FV, int, int, int, int, int, int>? _naiveLinear;
    private static void NaiveLinearImpl(Index1D idx, FV x, HV w, HV bias, FV o, FV res, int M, int K, int N, int hasBias, int act, int hasRes)
    {
        int m = idx / N, n = idx % N;
        float acc = hasBias != 0 ? PrecisionConvert.ConvertToSingle(bias[n]) : 0f;
        long xb = (long)m * K, wb = (long)n * K;
        for (int k = 0; k < K; k++) acc += x[xb + k] * PrecisionConvert.ConvertToSingle(w[wb + k]);
        if (act == 1) acc = acc / (1f + MathF.Exp(-acc));
        if (hasRes != 0) acc += res[idx];
        o[idx] = acc;
    }

    /// <summary>x [M,K] fp32; weight [N,K] fp16; act: 0 none, 1 SiLU; res (optional) [M,N] is added after the activation.</summary>
    public void Linear(FV x, HV weight, HV? bias, FV o, int M, int K, int N, int act = 0, FV? res = null)
    {
        var b = bias ?? weight.SubView(0, 1);
        if (Cuda && K % MmaGemm.BK == 0 && N >= 64 && N % 2 == 0 && M >= 16)
        {
            _mmaLinear ??= _acc.LoadStreamKernel<FV, UV, HV, FV, FV, IV, int, int, int, int, int, int, int>(MmaGemm.KernelF16Dense);
            _noNbr ??= _acc.Allocate1D<int>(1);
            int tiles = ((M + MmaGemm.BM - 1) / MmaGemm.BM) * ((N + MmaGemm.BN - 1) / MmaGemm.BN);
            _mmaLinear(new KernelConfig(tiles, MmaGemm.Threads), x, weight.Cast<global::ILGPU.Half, uint>(), b, o, res ?? o, _noNbr.View, M, K, N, 0, bias.HasValue ? 1 : 0, act, res.HasValue ? 1 : 0);
            return;
        }
        _naiveLinear ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, HV, HV, FV, FV, int, int, int, int, int, int>(NaiveLinearImpl);
        _naiveLinear(M * N, x, weight, b, o, res ?? o, M, K, N, bias.HasValue ? 1 : 0, act, res.HasValue ? 1 : 0);
    }

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
    public void LayerNorm(FV x, FV o, HV? gamma, HV? beta, int rows, int C, float eps)
    {
        _layerNorm ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, HV, HV, int, int, float>(LayerNormImpl);
        _layerNorm(rows, x, o, gamma ?? default, beta ?? default, C, gamma.HasValue ? 1 : 0, eps);
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

    /// <summary>Per-voxel child offsets (n+1 entries; the last is the total) from the subdivision logits.</summary>
    public int CountChildren(FV logits, int n, IV offsets)
    {
        _subCount ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV>(SubCountImpl);
        _serialScan ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, int>(SerialScanImpl);
        using var counts = _acc.Allocate1D<int>(n);
        _subCount(n, logits, counts.View);
        _serialScan(1, counts.View, offsets, n);
        _acc.Synchronize();
        var total = new int[1];
        offsets.SubView(n, 1).CopyToCPU(total);        // one scalar: sizes the output buffers
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
