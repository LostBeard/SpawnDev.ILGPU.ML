using ILGPU;
using ILGPU.Runtime;
using FV = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;
using BV = ILGPU.Runtime.ArrayView1D<ILGPU.BFloat16, ILGPU.Stride1D.Dense>;

namespace SpawnDev.ILGPU.ML.Ltx2;

/// <summary>
/// The kernels of Gemma 3 (Hugging Face transformers' Gemma3, Apache-2.0) that the other text encoders do not have: the RMS norm that scales by <c>1 + w</c> (formed in bf16, as ComfyUI does)
/// (Gemma3RMSNorm), the GELU-tanh gated MLP, and LTX-2's per-layer normalisation of the stacked hidden states (diffusers' LTX-2 text connectors, Apache-2.0).
/// </summary>
public sealed class Gemma3Kernels
{
    private readonly Accelerator _acc;
    public Gemma3Kernels(Accelerator accelerator) => _acc = accelerator;

    // 1 + w formed in the weight's dtype (bf16, round to nearest even): MEASURED 2026-11-17 against ComfyUI's Gemma 3 layer 0 (float64 reference): the fp32 1 + w is
    // 2.6e-3 off, bf16(1 + w) 1.3e-7. (Hugging Face's Gemma3RMSNorm adds in fp32; ComfyUI, the reference SpawnGen matches, adds in the weight's dtype.)
    private static float W1(BV w, int c)
    {
        uint u = (uint)Interop.FloatAsInt(1f + PrecisionConvert.ConvertToSingle(w[c]));
        u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
        return Interop.IntAsFloat(u);
    }

    private Action<Index1D, FV, FV, BV, int, float>? _rms;
    private static void RmsImpl(Index1D row, FV x, FV o, BV w, int C, float eps)
    {
        long b = (long)(int)row * C;
        float ss = 0f;
        for (int c = 0; c < C; c++) { float v = x[b + c]; ss += v * v; }
        float inv = 1f / MathF.Sqrt(ss / C + eps);
        for (int c = 0; c < C; c++) o[b + c] = x[b + c] * inv * W1(w, c);
    }
    /// <summary>o[r, :] = x[r, :] / sqrt(mean(x[r, :]^2) + eps) * (1 + w)  (Gemma3RMSNorm, float32).</summary>
    public void RmsNorm(FV x, FV o, BV w, int rows, int C, float eps)
    {
        _rms ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, BV, int, float>(RmsImpl);
        _rms(rows, x, o, w, C, eps);
    }

    private Action<Index1D, FV, FV, BV, int, float>? _rmsAdd;
    private static void RmsAddImpl(Index1D row, FV h, FV x, BV w, int C, float eps)
    {
        long b = (long)(int)row * C;
        float ss = 0f;
        for (int c = 0; c < C; c++) { float v = x[b + c]; ss += v * v; }
        float inv = 1f / MathF.Sqrt(ss / C + eps);
        for (int c = 0; c < C; c++) h[b + c] += x[b + c] * inv * W1(w, c);
    }
    /// <summary>h[r, :] += RmsNorm(x[r, :]) (Gemma 3's post-attention / post-feedforward norm and the residual add, fused).</summary>
    public void RmsNormAddInto(FV h, FV x, BV w, int rows, int C, float eps)
    {
        _rmsAdd ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, BV, int, float>(RmsAddImpl);
        _rmsAdd(rows, h, x, w, C, eps);
    }

    private Action<Index1D, FV, BV, int, int, int, int, float>? _rmsHeads;
    private static void RmsHeadsImpl(Index1D i, FV buf, BV w, int stride, int off, int H, int D, float eps)
    {
        int h = i % H, n = i / H;
        long b = (long)n * stride + off + h * D;
        float ss = 0f;
        for (int d = 0; d < D; d++) { float v = buf[b + d]; ss += v * v; }
        float inv = 1f / MathF.Sqrt(ss / D + eps);
        for (int d = 0; d < D; d++) buf[b + d] = buf[b + d] * inv * W1(w, d);
    }
    /// <summary>In place (1 + w) RMS norm of every D-wide head vector of rows [n, stride] from column off (Gemma 3's q_norm / k_norm).</summary>
    public void RmsNormHeads(FV buf, BV w, int rows, int stride, int off, int H, int D, float eps)
    {
        _rmsHeads ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, BV, int, int, int, int, float>(RmsHeadsImpl);
        _rmsHeads(rows * H, buf, w, stride, off, H, D, eps);
    }

    private Action<Index1D, FV, FV, FV>? _geglu;
    private static void GegluImpl(Index1D i, FV gate, FV up, FV o)
    {
        float g = gate[i];
        // gelu(approximate = "tanh"); |g| > 10 saturates (the polynomial overflows the tanh argument's float range long after that)
        float a = g > 10f ? g : g < -10f ? 0f : 0.5f * g * (1f + MathF.Tanh(0.7978845608028654f * (g + 0.044715f * g * g * g)));
        o[i] = a * up[i];
    }
    /// <summary>o = gelu_tanh(gate) * up over count values (Gemma 3's MLP).</summary>
    public void GegluTanh(FV gate, FV up, FV o, int count)
    {
        _geglu ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV>(GegluImpl);
        _geglu(count, gate, up, o);
    }

    // ───────────────────────── LTX-2: per-slot normalisation of the stacked hidden states ─────────────────────────

    private const int StatThreads = 256;
    private Action<KernelConfig, FV, FV, long>? _stats;
    private static void StatsImpl(FV slots, FV stats, long count)
    {
        int s = Grid.IdxX, t = Group.IdxX;
        var sum = SharedMemory.Allocate<float>(StatThreads);
        var mn = SharedMemory.Allocate<float>(StatThreads);
        var mx = SharedMemory.Allocate<float>(StatThreads);
        long b = (long)s * count;
        float a = 0f, lo = float.MaxValue, hi = -float.MaxValue;
        for (long i = t; i < count; i += StatThreads) { float v = slots[b + i]; a += v; lo = MathF.Min(lo, v); hi = MathF.Max(hi, v); }
        sum[t] = a; mn[t] = lo; mx[t] = hi;
        Group.Barrier();
        for (int k = StatThreads / 2; k > 0; k >>= 1)
        {
            if (t < k) { sum[t] += sum[t + k]; mn[t] = MathF.Min(mn[t], mn[t + k]); mx[t] = MathF.Max(mx[t], mx[t + k]); }
            Group.Barrier();
        }
        if (t == 0) { stats[3 * s] = sum[0]; stats[3 * s + 1] = mn[0]; stats[3 * s + 2] = mx[0]; }
    }

    private Action<Index1D, FV, FV, FV, int, int, int, float>? _pack;
    private static void PackImpl(Index1D i, FV slots, FV stats, FV o, int n, int C, int S, float scale)
    {
        int s = i % S, rest = i / S, c = rest % C, t = rest / C;                 // o[t, c * S + s]: channel-major
        float mean = stats[3 * s] / ((float)n * C + 1e-6f);
        float range = stats[3 * s + 2] - stats[3 * s + 1];
        o[i] = scale * (slots[(long)s * n * C + (long)t * C + c] - mean) / (range + 1e-6f);
    }

    /// <summary>
    /// LTX-2's text features: <paramref name="slots"/> [S, n, C] (the embedding and every layer's hidden state of the n real tokens) normalised per slot over all n x C values,
    /// <c>scale (x - mean) / (max - min + 1e-6)</c> with mean = sum / (n C + 1e-6), written channel-major: o[t, c S + s]. <paramref name="stats"/> holds 3 S floats.
    /// </summary>
    public void NormalisedStack(FV slots, FV stats, FV o, int S, int n, int C, float scale)
    {
        _stats ??= _acc.LoadStreamKernel<FV, FV, long>(StatsImpl);
        _pack ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, int, int, int, float>(PackImpl);
        _stats(new KernelConfig(S, StatThreads), slots, stats, (long)n * C);
        _pack((int)((long)n * C * S), slots, stats, o, n, C, S, scale);
    }
}
