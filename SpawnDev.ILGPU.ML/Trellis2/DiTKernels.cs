using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using BV = ArrayView1D<BFloat16, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;
using UV = ArrayView1D<uint, Stride1D.Dense>;

/// <summary>
/// The kernels of the TRELLIS.2 flow transformers (sparse-structure DiT and the two SLat DiTs): bf16 weights,
/// fp32 activations, fp32 accumulate. Every kernel here is the straightforward one-thread-per-output form: it
/// is the CORRECTNESS lane, checked element-for-element against tensors dumped from the real reference
/// (tools/trellis2-harness golden). The fast kernels (tiled / tensor-core GEMM, flash attention) are added
/// beside them and held to THESE as their oracle, never the other way round.
/// All entry points take the reference's own tensor layout, so a block reads like trellis2/modules/transformer.
/// </summary>
public sealed class DiTKernels
{
    private readonly Accelerator _acc;
    public DiTKernels(Accelerator accelerator) => _acc = accelerator;

    /// <summary>Epilogue of <see cref="Linear"/>.</summary>
    public enum Act { None = 0, Silu = 1, GeluTanh = 2, GeluErf = 3 }

    // ---------------------------------------------------------------------------------------------------
    // Linear: out[m,n] = act( bias[n] + sum_k x[m,k] * w[n,k] )        (nn.Linear, w stored [N,K])
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, BV, BV, FV, int, int, int, int, int>? _linear;
    private static void LinearImpl(Index1D i, FV x, BV w, BV bias, FV o, int M, int K, int N, int hasBias, int act)
    {
        int m = i / N, n = i % N;
        long xo = (long)m * K, wo = (long)n * K;
        float acc = 0f;
        for (int k = 0; k < K; k++)
            acc += x[xo + k] * PrecisionConvert.ConvertToSingle(w[wo + k]);
        if (hasBias != 0) acc += PrecisionConvert.ConvertToSingle(bias[n]);
        o[i] = Activate(acc, act);
    }

    /// <summary>
    /// Tensor-core path for <see cref="Linear"/> (CUDA, bf16 mma.sync, activations rounded to bf16 as the
    /// reference does). TRELLIS2_MMA=0 forces the fp32 reference kernel: the A/B for any numerical question.
    /// </summary>
    public static bool EnableTensorCores = Environment.GetEnvironmentVariable("TRELLIS2_MMA") != "0";
    private Action<KernelConfig, ArrayView1D<F4, Stride1D.Dense>, ArrayView1D<U4, Stride1D.Dense>, BV, FV, FV, IV, int, int, int, int, int, int, int>? _mmaLinear;
    private MemoryBuffer1D<int, Stride1D.Dense>? _noNbr;   // placeholder for the gather operand of the dense kernels (never read)

    private bool UseMma(int M, int K, int N)
        => EnableTensorCores && _acc.AcceleratorType == AcceleratorType.Cuda && K % MmaGemm.BK == 0 && N >= 64 && N % 2 == 0 && M >= 16;

    /// <summary>nn.Linear with a bf16 weight [N,K] and optional bf16 bias [N]; x is [M,K] fp32.</summary>
    public void Linear(FV x, BV w, BV? bias, FV o, int M, int K, int N, Act act = Act.None)
    {
        EnsureDummies();
        var b = bias ?? _dummyB!.View;   // never read when hasBias == 0 (a separate buffer: WebGPU forbids binding w twice)
        if (UseMma(M, K, N))
        {
            _mmaLinear ??= _acc.LoadStreamKernel<ArrayView1D<F4, Stride1D.Dense>, ArrayView1D<U4, Stride1D.Dense>, BV, FV, FV, IV, int, int, int, int, int, int, int>(MmaGemm.KernelBf16Dense);
            _noNbr ??= _acc.Allocate1D<int>(1);
            int tiles = ((M + MmaGemm.BM - 1) / MmaGemm.BM) * ((N + MmaGemm.BN - 1) / MmaGemm.BN);
            _mmaLinear(new KernelConfig(tiles, MmaGemm.Threads), x.Cast<float, F4>(), w.Cast<BFloat16, U4>(), b, o, o, _noNbr.View, M, K, N, 0, bias.HasValue ? 1 : 0, (int)act, 0);
            return;
        }
        if (UseTiled(K) && TiledGemm.Fits(M, K, N))
        {
            _tiledBf16 ??= _acc.LoadStreamKernel<FV, UV, UV, FV, FV, IV, int, int, int, int, int, int, int>(TiledGemm.Bf16Dense);
            EnsureDummies();
            int tiles = ((M + TiledGemm.BM - 1) / TiledGemm.BM) * ((N + TiledGemm.BN - 1) / TiledGemm.BN);
            _tiledBf16(new KernelConfig(tiles, TiledGemm.Threads), x, w.Cast<BFloat16, uint>(), bias.HasValue ? bias.Value.Cast<BFloat16, uint>() : _dummyU!.View,
                o, _dummyF!.View, _noNbr!.View, M, K, N, 0, bias.HasValue ? 1 : 0, (int)act, 0);
            return;
        }
        _linear ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, BV, BV, FV, int, int, int, int, int>(LinearImpl);
        _linear(M * N, x, w, b, o, M, K, N, bias.HasValue ? 1 : 0, (int)act);
    }

    /// <summary>
    /// Portable register-tiled path (<see cref="TiledGemm"/>) for every backend without the tensor-core kernels: WebGPU, OpenCL, CPU, and CUDA with
    /// TRELLIS2_MMA=0. TRELLIS2_TILED=0 forces the one-thread-per-output reference kernel.
    /// </summary>
    public static bool EnableTiled = Environment.GetEnvironmentVariable("TRELLIS2_TILED") != "0";
    internal bool UseTiled(int K) => EnableTiled && K % TiledGemm.BK == 0 && _acc.MaxNumThreadsPerGroup >= TiledGemm.Threads && _acc.AcceleratorType != AcceleratorType.WebGL;   // WebGL has no shared memory
    private Action<KernelConfig, FV, UV, UV, FV, FV, IV, int, int, int, int, int, int, int>? _tiledBf16;
    private MemoryBuffer1D<uint, Stride1D.Dense>? _dummyU;
    private MemoryBuffer1D<float, Stride1D.Dense>? _dummyF;
    private MemoryBuffer1D<BFloat16, Stride1D.Dense>? _dummyB, _dummyB2;
    private void EnsureDummies()
    {   // bound in place of absent operands: WebGPU forbids binding one buffer to two slots, so no operand may stand in for another
        _dummyU ??= _acc.Allocate1D<uint>(1); _dummyF ??= _acc.Allocate1D<float>(1); _noNbr ??= _acc.Allocate1D<int>(1);
        _dummyB ??= _acc.Allocate1D<BFloat16>(2); _dummyB2 ??= _acc.Allocate1D<BFloat16>(2);
    }

    // NOT inlined: the tiled GEMM epilogue calls this 64 times per thread; inlined, every call site carried all three activation bodies
    // (the WGSL of one GEMM ran to ~600 KB, and the browser compiles shaders on first use).
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    internal static float Activate(float v, int act)
    {
        if (act == 1) return v / (1f + MathF.Exp(-v));                              // SiLU
        if (act == 2)                                                              // GELU, tanh approximation
        {
            float z = 0.7978845608028654f * (v + 0.044715f * v * v * v);
            z = MathF.Max(-20f, MathF.Min(20f, z));
            float e = MathF.Exp(2f * z);
            return 0.5f * v * (1f + (e - 1f) / (e + 1f));
        }
        if (act == 3)                                                              // exact GELU (erf), Abramowitz-Stegun 7.1.26
        {
            float z = v * 0.7071067811865476f, az = MathF.Abs(z);
            float t = 1f / (1f + 0.3275911f * az);
            float poly = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
            float erfAbs = 1f - poly * MathF.Exp(-az * az);
            return 0.5f * v * (1f + (z < 0f ? -erfAbs : erfAbs));
        }
        return v;
    }

    // ---------------------------------------------------------------------------------------------------
    // SiLU on a vector (adaLN_modulation = SiLU -> Linear)
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, FV>? _silu;
    private static void SiluImpl(Index1D i, FV x, FV o) => o[i] = Activate(x[i], 1);
    public void Silu(FV x, FV o, int count)
    {
        _silu ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV>(SiluImpl);
        _silu(count, x, o);
    }

    // ---------------------------------------------------------------------------------------------------
    // modv = bf16 per-block modulation parameter + fp32 shared modulation   ((self.modulation + mod))
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, BV, FV, FV>? _addBf16;
    private static void AddBf16Impl(Index1D i, BV a, FV b, FV o) => o[i] = PrecisionConvert.ConvertToSingle(a[i]) + b[i];
    public void AddBf16(BV a, FV b, FV o, int count)
    {
        _addBf16 ??= _acc.LoadAutoGroupedStreamKernel<Index1D, BV, FV, FV>(AddBf16Impl);
        _addBf16(count, a, b, o);
    }

    // ---------------------------------------------------------------------------------------------------
    // LayerNorm over the last dim. modulate: out = LN(x) * (1 + mod[scaleOff+c]) + mod[shiftOff+c]
    //                              affine:   out = LN(x) * gamma[c] + beta[c]     (bf16)
    //                              plain:    out = LN(x)
    // eps is the reference's: 1e-6 inside the blocks, torch's default 1e-5 for the final F.layer_norm.
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, FV, FV, BV, BV, int, int, int, int, float>? _layerNorm;
    private static void LayerNormImpl(Index1D m, FV x, FV o, FV mod, BV gamma, BV beta, int C, int mode, int shiftOff, int scaleOff, float eps)
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
            if (mode == 1) v = v * (1f + mod[scaleOff + c]) + mod[shiftOff + c];
            else if (mode == 2) v = v * PrecisionConvert.ConvertToSingle(gamma[c]) + PrecisionConvert.ConvertToSingle(beta[c]);
            o[b + c] = v;
        }
    }

    // One 256-thread group per row (rows are 1536 wide): two block reductions (mean, then variance of the centred values), then the
    // elementwise part. The one-thread-per-row form above is kept for backends whose groups cannot be 256 wide (the CPU accelerator).
    private const int NormGroup = 256;
    private Action<KernelConfig, FV, FV, FV, BV, BV, int, int, int, int, float>? _layerNormGroup;
    private static void LayerNormGroupImpl(FV x, FV o, FV mod, BV gamma, BV beta, int C, int mode, int shiftOff, int scaleOff, float eps)
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
            if (mode == 1) v = v * (1f + mod[scaleOff + c]) + mod[shiftOff + c];
            else if (mode == 2) v = v * PrecisionConvert.ConvertToSingle(gamma[c]) + PrecisionConvert.ConvertToSingle(beta[c]);
            o[b + c] = v;
        }
    }

    private void LayerNormCore(FV x, FV o, FV mod, BV gamma, BV beta, int rows, int C, int mode, int shiftOff, int scaleOff, float eps)
    {
        // operands the mode does not read are bound to 1-element placeholders: WebGPU cannot bind an empty (default) view
        EnsureDummies();
        if (mode != 1) mod = _dummyF!.View;
        if (mode != 2) { gamma = _dummyB!.View; beta = _dummyB2!.View; }
        if (_acc.MaxGroupSize.X >= NormGroup)
        {
            _layerNormGroup ??= _acc.LoadStreamKernel<FV, FV, FV, BV, BV, int, int, int, int, float>(LayerNormGroupImpl);
            _layerNormGroup(new KernelConfig(rows, NormGroup), x, o, mod, gamma, beta, C, mode, shiftOff, scaleOff, eps);
            return;
        }
        _layerNorm ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, BV, BV, int, int, int, int, float>(LayerNormImpl);
        _layerNorm(rows, x, o, mod, gamma, beta, C, mode, shiftOff, scaleOff, eps);
    }

    /// <summary>norm(x) * (1 + scale) + shift with shift/scale read from <paramref name="mod"/> (adaLN, share_mod).</summary>
    public void LayerNormModulate(FV x, FV o, FV mod, int shiftOff, int scaleOff, int rows, int C, float eps)
        => LayerNormCore(x, o, mod, default, default, rows, C, 1, shiftOff, scaleOff, eps);

    /// <summary>norm(x) * gamma + beta (elementwise_affine=True), bf16 parameters.</summary>
    public void LayerNormAffine(FV x, FV o, BV gamma, BV beta, int rows, int C, float eps)
        => LayerNormCore(x, o, default, gamma, beta, rows, C, 2, 0, 0, eps);

    /// <summary>norm(x) with no parameters (F.layer_norm(h, h.shape[-1:])).</summary>
    public void LayerNormPlain(FV x, FV o, int rows, int C, float eps)
        => LayerNormCore(x, o, default, default, default, rows, C, 0, 0, 0, eps);

    // ---------------------------------------------------------------------------------------------------
    // x[m,c] += gate[c] * h[m,c]     (gateOff < 0: plain residual add)
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, FV, FV, int, int>? _gateResidual;
    private static void GateResidualImpl(Index1D i, FV x, FV h, FV mod, int gateOff, int C)
    {
        float g = gateOff < 0 ? 1f : mod[gateOff + i % C];
        x[i] = x[i] + g * h[i];
    }
    public void GateResidual(FV x, FV h, FV mod, int gateOff, int count, int C)
    {
        _gateResidual ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int>(GateResidualImpl);
        if (gateOff < 0) { EnsureDummies(); mod = _dummyF!.View; }   // plain residual: the unread gate binds a placeholder (WebGPU cannot bind an empty view)
        _gateResidual(count, x, h, mod, gateOff, C);
    }

    // ---------------------------------------------------------------------------------------------------
    // Per-head RMS norm (F.normalize(x, dim=-1) * gamma[h,:] * sqrt(D)), optional 3-D RoPE, IN PLACE.
    // Vector (n, h) lives at buf[n*stride + off + h*D .. + D). phases = [N, D/2, 2] (cos, sin).
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, BV, FV, int, int, int, int, int>? _headNorm;
    private static void HeadNormImpl(Index1D i, FV buf, BV gamma, FV phases, int stride, int off, int H, int D, int useRope)
    {
        int h = i % H, n = i / H;
        long b = (long)n * stride + off + h * D;
        float ss = 0f;
        for (int d = 0; d < D; d++) { float v = buf[b + d]; ss += v * v; }
        // torch F.normalize: x / max(||x||, 1e-12)
        float inv = 1f / MathF.Max(MathF.Sqrt(ss), 1e-12f);
        float scale = MathF.Sqrt((float)D);
        int pairs = D / 2;
        for (int p = 0; p < pairs; p++)
        {
            float x0 = buf[b + 2 * p] * inv * PrecisionConvert.ConvertToSingle(gamma[h * D + 2 * p]) * scale;
            float x1 = buf[b + 2 * p + 1] * inv * PrecisionConvert.ConvertToSingle(gamma[h * D + 2 * p + 1]) * scale;
            if (useRope != 0)
            {
                long pi = ((long)n * pairs + p) * 2;
                float c = phases[pi], s = phases[pi + 1];
                float r0 = x0 * c - x1 * s;
                float r1 = x0 * s + x1 * c;
                x0 = r0; x1 = r1;
            }
            buf[b + 2 * p] = x0;
            buf[b + 2 * p + 1] = x1;
        }
    }
    // One 32-lane warp per (token, head) vector: the lanes split the D/2 pairs, the sum of squares is a shuffle reduction.
    private Action<KernelConfig, FV, BV, FV, int, int, int, int, int>? _headNormWarp;
    private static void HeadNormWarpImpl(FV buf, BV gamma, FV phases, int stride, int off, int H, int D, int useRope)
    {
        int vec = Grid.IdxX, lane = Group.IdxX;
        int h = vec % H, n = vec / H;
        long b = (long)n * stride + off + h * D;
        int pairs = D / 2;
        float ss = 0f;
        for (int p = lane; p < pairs; p += 32)
        {
            float x0 = buf[b + 2 * p], x1 = buf[b + 2 * p + 1];
            ss += x0 * x0 + x1 * x1;
        }
        for (int m = 16; m > 0; m >>= 1) ss += Warp.ShuffleXor(ss, m);
        float inv = 1f / MathF.Max(MathF.Sqrt(ss), 1e-12f);
        float scale = MathF.Sqrt((float)D);
        for (int p = lane; p < pairs; p += 32)
        {
            float x0 = buf[b + 2 * p] * inv * PrecisionConvert.ConvertToSingle(gamma[h * D + 2 * p]) * scale;
            float x1 = buf[b + 2 * p + 1] * inv * PrecisionConvert.ConvertToSingle(gamma[h * D + 2 * p + 1]) * scale;
            if (useRope != 0)
            {
                long pi = ((long)n * pairs + p) * 2;
                float c = phases[pi], sn = phases[pi + 1];
                float r0 = x0 * c - x1 * sn;
                float r1 = x0 * sn + x1 * c;
                x0 = r0; x1 = r1;
            }
            buf[b + 2 * p] = x0;
            buf[b + 2 * p + 1] = x1;
        }
    }
    public void HeadNorm(FV buf, BV gamma, FV? phases, int N, int stride, int off, int H, int D)
    {
        if (_acc.AcceleratorType == AcceleratorType.Cuda)
        {
            _headNormWarp ??= _acc.LoadStreamKernel<FV, BV, FV, int, int, int, int, int>(HeadNormWarpImpl);
            EnsureDummies();
            _headNormWarp(new KernelConfig(N * H, 32), buf, gamma, phases ?? _dummyF!.View, stride, off, H, D, phases.HasValue ? 1 : 0);
            return;
        }
        _headNorm ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, BV, FV, int, int, int, int, int>(HeadNormImpl);
        EnsureDummies();   // an absent RoPE table binds a placeholder: WebGPU forbids binding buf twice
        _headNorm(N * H, buf, gamma, phases ?? _dummyF!.View, stride, off, H, D, phases.HasValue ? 1 : 0);
    }

    // ---------------------------------------------------------------------------------------------------
    // 3-D RoPE phase table from integer token coordinates. coords = [N,3]. Layout of the pair index p:
    // axis-major (axis = p / freqDim, j = p % freqDim), padded with angle 0 up to D/2 (RotaryPositionEmbedder).
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, IV, FV, int, int, float, float>? _ropePhases;
    private static void RopePhasesImpl(Index1D i, IV coords, FV phases, int pairs, int freqDim, float freqLo, float freqHi)
    {
        int n = i / pairs, p = i % pairs;
        float angle = 0f;
        if (p < 3 * freqDim)
        {
            int axis = p / freqDim, j = p % freqDim;
            float freq = freqLo / MathF.Pow(freqHi, (float)j / freqDim);
            angle = coords[n * 3 + axis] * freq;
        }
        phases[2 * i] = MathF.Cos(angle);
        phases[2 * i + 1] = MathF.Sin(angle);
    }
    public void RopePhases(IV coords, FV phases, int N, int headDim, float freqLo = 1f, float freqHi = 10000f)
    {
        _ropePhases ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, FV, int, int, float, float>(RopePhasesImpl);
        int pairs = headDim / 2;
        _ropePhases(N * pairs, coords, phases, pairs, pairs / 3, freqLo, freqHi);
    }

    // dense res^3 grid, 'ij' order (x slowest)
    private Action<Index1D, IV, int>? _gridCoords;
    private static void GridCoordsImpl(Index1D n, IV coords, int res)
    {
        coords[n * 3] = n / (res * res);
        coords[n * 3 + 1] = n / res % res;
        coords[n * 3 + 2] = n % res;
    }
    public void GridCoords(IV coords, int res)
    {
        _gridCoords ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, int>(GridCoordsImpl);
        _gridCoords(res * res * res, coords, res);
    }

    // ---------------------------------------------------------------------------------------------------
    // Layout moves for the dense latent: [C, N] <-> [N, C]
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, FV, int, int>? _transpose;
    private static void TransposeImpl(Index1D i, FV src, FV dst, int rows, int cols)
    {
        // dst[c,r] = src[r,c]   (src is rows x cols)
        int r = i / cols, c = i % cols;
        dst[c * rows + r] = src[i];
    }
    public void Transpose(FV src, FV dst, int rows, int cols)
    {
        _transpose ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int>(TransposeImpl);
        _transpose(rows * cols, src, dst, rows, cols);
    }

    // ---------------------------------------------------------------------------------------------------
    // REFERENCE attention (materialises the scores): the oracle the flash kernel is held to. Fine up to the
    // dense 4096-token sparse-structure stage; NOT the production path for the SLat stages.
    // q at q[i*qStride + qOff + h*D], k at k[j*kStride + kOff + h*D], v at v[j*kStride + vOff + h*D].
    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, FV, FV, int, int, int, int, int, int, int, int, float>? _scores;
    private static void ScoresImpl(Index1D idx, FV q, FV k, FV s, int Nq, int Nk, int H, int D, int qStride, int qOff, int kStride, int kOff, float scale)
    {
        long jj = idx % Nk; long rest = idx / Nk; int i = (int)(rest % Nq); int h = (int)(rest / Nq);
        long qb = (long)i * qStride + qOff + h * D, kb = jj * kStride + kOff + h * D;
        float acc = 0f;
        for (int d = 0; d < D; d++) acc += q[qb + d] * k[kb + d];
        s[idx] = acc * scale;
    }
    private Action<Index1D, FV, int>? _softmaxRows;
    private static void SoftmaxRowsImpl(Index1D row, FV s, int Nk)
    {
        long b = (long)(int)row * Nk;
        float mx = float.NegativeInfinity;
        for (int j = 0; j < Nk; j++) mx = MathF.Max(mx, s[b + j]);
        float sum = 0f;
        for (int j = 0; j < Nk; j++) { float e = MathF.Exp(s[b + j] - mx); s[b + j] = e; sum += e; }
        float inv = 1f / sum;
        for (int j = 0; j < Nk; j++) s[b + j] *= inv;
    }
    private Action<Index1D, FV, FV, FV, int, int, int, int, int, int>? _pv;
    private static void PvImpl(Index1D idx, FV p, FV v, FV o, int Nq, int Nk, int H, int D, int vStride, int vOff)
    {
        int d = idx % D; int rest = idx / D; int h = rest % H; int i = rest / H;
        long pb = ((long)h * Nq + i) * Nk;
        float acc = 0f;
        for (int j = 0; j < Nk; j++) acc += p[pb + j] * v[(long)j * vStride + vOff + h * D + d];
        o[idx] = acc;
    }

    private Action<KernelConfig, FV, UV, FV, int, int, int, int, int, int, int, int>? _flash;
    private Action<Index1D, FV, UV>? _toBf16Pairs;
    private MemoryBuffer1D<uint, Stride1D.Dense>? _kv16;

    private static uint Bf16Bits(float f)
    {
        uint bits = (uint)Interop.FloatAsInt(f);
        return (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
    }
    private static void ToBf16PairsImpl(Index1D i, FV src, UV dst) => dst[i] = Bf16Bits(src[2 * i]) | (Bf16Bits(src[2 * i + 1]) << 16);

    /// <summary>True when <see cref="Attention"/> runs a flash kernel (no score matrix, <c>scores</c> may be default): the tensor-core one or the portable one.</summary>
    public bool UsesFlashAttention(int headDim) => UsesMmaAttention(headDim) || UsesTiledAttention(headDim);
    private bool UsesMmaAttention(int headDim)
        => EnableTensorCores && _acc.AcceleratorType == AcceleratorType.Cuda && headDim == MmaAttention.HeadDim;
    /// <summary>Portable flash attention (<see cref="TiledAttention"/>): needs 256-thread groups and warp shuffles over groups of 4 lanes.</summary>
    private bool UsesTiledAttention(int headDim)
        => EnableTiled && (headDim == 128 || headDim == 64) && _acc.MaxNumThreadsPerGroup >= TiledAttention.Threads && _acc.WarpSize >= 4 && _acc.AcceleratorType != AcceleratorType.WebGL;
    private readonly Action<KernelConfig, FV, FV, FV, int, int, int, int, int, int, int, int>?[] _tiledCross = new Action<KernelConfig, FV, FV, FV, int, int, int, int, int, int, int, int>?[4];
    private readonly Action<KernelConfig, FV, FV, int, int, int, int, int, int, int, int>?[] _tiledSelf = new Action<KernelConfig, FV, FV, int, int, int, int, int, int, int, int>?[4];
    /// <summary>Null until the first attention: whether this backend compiles warp shuffles (NVIDIA's OpenCL does not; browsers need the subgroups feature).</summary>
    private bool? _shuffleOk;

    /// <summary>Which portable attention kernel this backend runs (after the first attention): for benchmarks and logs.</summary>
    public string TiledAttentionKind => _shuffleOk switch
    {
        true => $"shuffle {TiledAttention.LanesPerQuery} lanes x {TiledAttention.QueriesPerGroup * TiledAttention.LanesPerQuery / TiledAttention.Threads} queries",
        false => "shared-memory reduction",
        null => "not resolved",
    };

    private KernelConfig Cfg(int Nq, int H)
    {
        int qpg = _shuffleOk == true ? TiledAttention.QueriesPerGroup : TiledAttention.QueriesPerGroupSmem;
        return new KernelConfig(((Nq + qpg - 1) / qpg) * H, TiledAttention.Threads);
    }

    // The shuffle kernel is tried first; a backend that cannot compile Warp.ShuffleXor (or whose warps are narrower than the lanes a query
    // spans) gets the shared-memory reduction variant.
    private Action<KernelConfig, FV, FV, FV, int, int, int, int, int, int, int, int> CrossKernel(int key)
    {
        if (_tiledCross[key] != null) return _tiledCross[key]!;
        if (_acc.WarpSize < TiledAttention.LanesPerQuery) _shuffleOk = false;
        if (_shuffleOk != false)
        {
            try { _tiledCross[key] = _acc.LoadStreamKernel<FV, FV, FV, int, int, int, int, int, int, int, int>(key < 2 ? TiledAttention.CrossD128 : TiledAttention.CrossD64); _shuffleOk = true; return _tiledCross[key]!; }
            catch (Exception ex) when (ex is global::ILGPU.InternalCompilerException || ex.InnerException is global::ILGPU.Backends.InvalidCodeGenerationException || ex is NotSupportedException) { _shuffleOk = false; }
        }
        return _tiledCross[key] = _acc.LoadStreamKernel<FV, FV, FV, int, int, int, int, int, int, int, int>(key < 2 ? TiledAttention.CrossD128Smem : TiledAttention.CrossD64Smem);
    }
    private Action<KernelConfig, FV, FV, int, int, int, int, int, int, int, int> SelfKernel(int key)
    {
        if (_tiledSelf[key] != null) return _tiledSelf[key]!;
        if (_acc.WarpSize < TiledAttention.LanesPerQuery) _shuffleOk = false;
        if (_shuffleOk != false)
        {
            try { _tiledSelf[key] = _acc.LoadStreamKernel<FV, FV, int, int, int, int, int, int, int, int>(key < 2 ? TiledAttention.SelfD128 : TiledAttention.SelfD64); _shuffleOk = true; return _tiledSelf[key]!; }
            catch (Exception ex) when (ex is global::ILGPU.InternalCompilerException || ex.InnerException is global::ILGPU.Backends.InvalidCodeGenerationException || ex is NotSupportedException) { _shuffleOk = false; }
        }
        return _tiledSelf[key] = _acc.LoadStreamKernel<FV, FV, int, int, int, int, int, int, int, int>(key < 2 ? TiledAttention.SelfD128Smem : TiledAttention.SelfD64Smem);
    }

    /// <summary>
    /// Multi-head attention, no mask. q at q[i*qStride + qOff + h*D], k at kv[j*kvStride + kOff + h*D], v at
    /// kv[j*kvStride + vOff + h*D]; o is [Nq, H*D]. Flash kernel on CUDA (bf16 tensor cores, O(1) extra memory);
    /// otherwise the reference kernels, which need <paramref name="scores"/> to hold H*Nq*Nk floats.
    /// </summary>
    public void Attention(FV q, int qStride, int qOff, FV kv, int kvStride, int kOff, int vOff, FV scores, FV o, int Nq, int Nk, int H, int D)
    {
        if (!UsesMmaAttention(D) && UsesTiledAttention(D))
        {
            IContiguousArrayView qv = q.BaseView, kvv = kv.BaseView;
            bool self = qv.Buffer == kvv.Buffer;
            if (self && qv.Index != kvv.Index) throw new ArgumentException("q and kv share a buffer but start at different offsets; pass the same view with offsets");
            int key = (D == 128 ? 0 : 2) + (self ? 1 : 0);
            if (self)
            {
                var k = SelfKernel(key);                                    // resolves _shuffleOk, which sets the queries per group
                k(Cfg(Nq, H), q, o, Nq, Nk, H, qStride, qOff, kvStride, kOff, vOff);   // q and k/v are slices of one buffer: one binding
            }
            else
            {
                var k = CrossKernel(key);
                k(Cfg(Nq, H), q, kv, o, Nq, Nk, H, qStride, qOff, kvStride, kOff, vOff);
            }
            return;
        }
        if (UsesMmaAttention(D))
        {
            // K and V go to the kernel as bf16 pairs (one rounding pass here instead of one conversion per query tile inside the kernel)
            _flash ??= _acc.LoadStreamKernel<FV, UV, FV, int, int, int, int, int, int, int, int>(MmaAttention.Kernel);
            _toBf16Pairs ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, UV>(ToBf16PairsImpl);
            long pairs = ((long)(Nk - 1) * kvStride + Math.Max(kOff, vOff) + H * D + 1) / 2;
            if (_kv16 == null || _kv16.Length < pairs) { _kv16?.Dispose(); _kv16 = _acc.Allocate1D<uint>(pairs); }
            _toBf16Pairs((int)pairs, kv, _kv16.View);
            _flash(new KernelConfig(((Nq + MmaAttention.QTile - 1) / MmaAttention.QTile) * H, MmaAttention.Threads),
                q, _kv16.View, o, Nq, Nk, H, qStride, qOff, kvStride, kOff, vOff);
            return;
        }
        ReferenceAttention(q, qStride, qOff, kv, kvStride, kOff, vOff, scores, o, Nq, Nk, H, D);
    }

    /// <summary>scores must hold H*Nq*Nk floats. o is [Nq, H*D].</summary>
    public void ReferenceAttention(FV q, int qStride, int qOff, FV kv, int kvStride, int kOff, int vOff, FV scores, FV o, int Nq, int Nk, int H, int D)
    {
        _scores ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, int, int, int, int, int, float>(ScoresImpl);
        _softmaxRows ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, int>(SoftmaxRowsImpl);
        _pv ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, int, int, int>(PvImpl);
        float scale = 1f / MathF.Sqrt(D);
        _scores((int)((long)H * Nq * Nk), q, kv, scores, Nq, Nk, H, D, qStride, qOff, kvStride, kOff, scale);
        _softmaxRows(H * Nq, scores, Nk);
        _pv(Nq * H * D, scores, kv, o, Nq, Nk, H, D, kvStride, vOff);
    }
}
