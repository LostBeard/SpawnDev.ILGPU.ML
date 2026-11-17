using ILGPU;
using ILGPU.Runtime;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;
using BV = ILGPU.Runtime.ArrayView1D<ILGPU.BFloat16, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.AceStep;

/// <summary>
/// The few kernels ACE-Step 1.5's transformers need beyond the shared DiT ones: bidirectional grouped-query attention with an optional sliding window
/// (|i - j| &lt;= window, ACE-Step's create_4d_mask), RMS-norm modulation, SwiGLU on separate gate / up buffers, and the k2 / s2 Conv1d / ConvTranspose1d of the
/// DiT's input and output written as patchify / unpatchify around ordinary GEMMs.
/// </summary>
public sealed class AceStepKernels
{
    private readonly Accelerator _acc;
    public AceStepKernels(Accelerator accelerator) => _acc = accelerator;

    // ───────────────────────── attention ─────────────────────────
    private Action<Index1D, FV, FV, FV, int, int, int, int, int, int, int, int, float>? _scores;
    private Action<Index1D, FV, int>? _softmax;
    private Action<Index1D, FV, FV, FV, int, int, int, int, int, int>? _pv;

    // scores[h, t, j] = q[t, h] . k[j, h / rep] * scale, or -inf outside the window
    private static void ScoresImpl(Index1D i, FV q, FV k, FV s, int Nq, int Nk, int H, int D, int rep, int qStride, int kStride, int window, float scale)
    {
        long ii = (int)i;
        int j = (int)(ii % Nk), t = (int)(ii / Nk % Nq), h = (int)(ii / ((long)Nk * Nq));
        if (window >= 0 && (t - j > window || j - t > window)) { s[i] = float.NegativeInfinity; return; }
        long qo = (long)t * qStride + (long)h * D, ko = (long)j * kStride + (long)(h / rep) * D;
        float acc = 0f;
        for (int d = 0; d < D; d++) acc = float.MultiplyAddEstimate(q[qo + d], k[ko + d], acc);
        s[i] = acc * scale;
    }

    // one thread per row: max, exp, sum, divide (a fully masked row cannot occur: the diagonal is always in the window)
    private static void SoftmaxImpl(Index1D r, FV s, int n)
    {
        long o = (long)(int)r * n;
        float m = float.NegativeInfinity;
        for (int j = 0; j < n; j++) m = MathF.Max(m, s[o + j]);
        float sum = 0f;
        for (int j = 0; j < n; j++) { float e = MathF.Exp(s[o + j] - m); s[o + j] = e; sum += e; }
        float inv = 1f / sum;
        for (int j = 0; j < n; j++) s[o + j] *= inv;
    }

    private static void PvImpl(Index1D i, FV s, FV v, FV o, int Nq, int Nk, int H, int D, int rep, int vStride)
    {
        int C = H * D, t = i / C, c = i % C, h = c / D, d = c % D;
        long so = ((long)h * Nq + t) * Nk, vo = (long)(h / rep) * D + d;
        float acc = 0f;
        for (int j = 0; j < Nk; j++) acc = float.MultiplyAddEstimate(s[so + j], v[(long)j * vStride + vo], acc);
        o[i] = acc;
    }

    /// <summary>
    /// Bidirectional grouped-query attention: q [Nq, H D], k / v [Nk, (H / rep) D] (row strides given), o [Nq, H D]; <paramref name="window"/> &gt;= 0 keeps keys with
    /// |t - j| &lt;= window, &lt; 0 keeps all. <paramref name="scores"/> holds H Nq Nk floats. Scale 1 / sqrt(D).
    /// </summary>
    public void Gqa(FV q, int qStride, FV k, int kStride, FV v, int vStride, FV scores, FV o, int Nq, int Nk, int H, int D, int rep, int window)
    {
        _scores ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, int, int, int, int, int, float>(ScoresImpl);
        _softmax ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, int>(SoftmaxImpl);
        _pv ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, int, int, int>(PvImpl);
        _scores((int)((long)H * Nq * Nk), q, k, scores, Nq, Nk, H, D, rep, qStride, kStride, window, 1f / MathF.Sqrt(D));
        _softmax(H * Nq, scores, Nk);
        _pv(Nq * H * D, scores, v, o, Nq, Nk, H, D, rep, vStride);
    }

    // ───────────────────────── modulation, SwiGLU, adds ─────────────────────────
    private Action<Index1D, FV, FV, FV, int, int, int>? _modulate;
    private static void ModulateImpl(Index1D i, FV n, FV o, FV mod, int shiftOff, int scaleOff, int C)
    {
        int c = i % C;
        o[i] = float.MultiplyAddEstimate(n[i], 1f + mod[scaleOff + c], mod[shiftOff + c]);
    }
    /// <summary>o[r, c] = n[r, c] (1 + mod[scaleOff + c]) + mod[shiftOff + c] (n: an RMS-normed input).</summary>
    public void Modulate(FV n, FV o, FV mod, int shiftOff, int scaleOff, int rows, int C)
    {
        _modulate ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int>(ModulateImpl);
        _modulate(rows * C, n, o, mod, shiftOff, scaleOff, C);
    }

    private Action<Index1D, FV, FV, FV>? _siluMul;
    private static void SiluMulImpl(Index1D i, FV g, FV u, FV o) { float x = g[i]; o[i] = x / (1f + MathF.Exp(-x)) * u[i]; }
    /// <summary>o = silu(g) * u over n values (Qwen3MLP's gate / up).</summary>
    public void SiluMul(FV g, FV u, FV o, int n)
    {
        _siluMul ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV>(SiluMulImpl);
        _siluMul(n, g, u, o);
    }

    private Action<Index1D, FV, FV, FV>? _add;
    private static void AddImpl(Index1D i, FV a, FV b, FV o) => o[i] = a[i] + b[i];
    /// <summary>o = a + b over n values.</summary>
    public void Add(FV a, FV b, FV o, int n)
    {
        _add ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV>(AddImpl);
        _add(n, a, b, o);
    }

    private Action<Index1D, FV, FV, FV, float>? _axpy;
    private static void AxpyImpl(Index1D i, FV x, FV v, FV o, float a) => o[i] = float.MultiplyAddEstimate(a, v[i], x[i]);
    /// <summary>o = x + a v over n values (the denoised latent x - sigma v).</summary>
    public void Axpy(FV x, FV v, FV o, float a, int n)
    {
        _axpy ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, float>(AxpyImpl);
        _axpy(n, x, v, o, a);
    }

    private Action<Index1D, FV, FV, float>? _axpyInPlace;
    private static void AxpyInPlaceImpl(Index1D i, FV x, FV v, float a) => x[i] = float.MultiplyAddEstimate(a, v[i], x[i]);
    /// <summary>x += a v over n values (an Euler step), one binding per buffer (WebGPU).</summary>
    public void AxpyInPlace(FV x, FV v, float a, int n)
    {
        _axpyInPlace ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, float>(AxpyInPlaceImpl);
        _axpyInPlace(n, x, v, a);
    }

    private Action<Index1D, FV, FV, FV, int, int>? _concat;
    private static void ConcatImpl(Index1D i, FV a, FV b, FV o, int ca, int cb)
    {
        int w = ca + cb, r = i / w, c = i % w;
        o[i] = c < ca ? a[(long)r * ca + c] : b[(long)r * cb + c - ca];
    }
    /// <summary>o [rows, ca + cb] = [a [rows, ca] | b [rows, cb]] (the DiT's context: source latents | chunk mask).</summary>
    public void Concat(FV a, FV b, FV o, int rows, int ca, int cb)
    {
        _concat ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int>(ConcatImpl);
        _concat(rows * (ca + cb), a, b, o, ca, cb);
    }

    // ───────────────────────── the DiT's k2 / s2 input and output ─────────────────────────
    private Action<Index1D, FV, FV, FV, int, int, int>? _patchify;
    // P[p, c 2 + k] = X[2 p + k, c], X = [context (cc) | hidden (hc)] per frame, zero past T (ACE-Step pads the time to even)
    private static void PatchifyImpl(Index1D i, FV ctx, FV hid, FV p, int T, int cc, int hc)
    {
        int width = 2 * (cc + hc), row = i / width, r = i % width, c = r / 2, t = 2 * row + r % 2;
        p[i] = t >= T ? 0f : c < cc ? ctx[(long)t * cc + c] : hid[(long)t * hc + c - cc];
    }
    /// <summary>The Conv1d(k2, s2) input as rows: [ceil(T / 2), 2 (cc + hc)] from the time-major context [T, cc] and latent [T, hc].</summary>
    public void Patchify(FV ctx, FV hid, FV p, int T, int cc, int hc)
    {
        _patchify ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int>(PatchifyImpl);
        _patchify((T + 1) / 2 * 2 * (cc + hc), ctx, hid, p, T, cc, hc);
    }

    private Action<Index1D, FV, BV, FV, int>? _unpatchify;
    // out[t, c] = y[t / 2, c 2 + t % 2] + bias[c]
    private static void UnpatchifyImpl(Index1D i, FV y, BV bias, FV o, int C)
    {
        int t = i / C, c = i % C;
        o[i] = y[(long)(t / 2) * 2 * C + c * 2 + t % 2] + (float)bias[c];
    }
    /// <summary>The ConvTranspose1d(k2, s2) output, time-major [T, C], from the GEMM rows y [ceil(T / 2), 2 C] (column c 2 + k), plus the bias, cropped to T.</summary>
    public void Unpatchify(FV y, BV bias, FV o, int T, int C)
    {
        _unpatchify ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, BV, FV, int>(UnpatchifyImpl);
        _unpatchify(T * C, y, bias, o, C);
    }
}
