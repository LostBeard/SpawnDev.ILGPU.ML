using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using DV = ArrayView1D<double, Stride1D.Dense>;

/// <summary>
/// TRELLIS.2's FlowEulerGuidanceIntervalSampler (trellis2/pipelines/samplers/flow_euler.py + the classifier-free-guidance and
/// guidance-interval mixins), on the device. The latent never leaves the GPU: the only scalars computed on the host are the
/// timestep schedule; the CFG-rescale standard deviations are reduced on the device.
///   t_seq = rescale_t * s / (1 + (rescale_t - 1) * s), s = linspace(1, 0, steps + 1)
///   per step: v = model(x, 1000 t)          when guidance is off or t is outside [lo, hi]
///             v = s * v_cond + (1 - s) * v_neg, then optional guidance rescale (x0 std matching)   inside the interval
///   x = x - (t - t_prev) * v
/// </summary>
public sealed class FlowSampler
{
    /// <summary>The sampler configuration of one stage (pipeline.json "*_sampler.params").</summary>
    /// <remarks>The interval and RescaleT are float64 on purpose: the reference compares python floats, and 0.6f (float32) is larger
    /// than 0.6 (float64), which silently turned guidance off at the step where t is exactly 0.6 (rescale_t 3, step 8 of 12).</remarks>
    public readonly record struct Options(int Steps, float GuidanceStrength, float GuidanceRescale, double IntervalLo, double IntervalHi, double RescaleT, float SigmaMin = 1e-5f);

    /// <summary>Evaluate the velocity: <paramref name="useCond"/> false means the negative (zero) condition. t1000 = 1000 * t.</summary>
    public delegate void Model(FV x, float t1000, bool useCond, FV velocity);

    private readonly Accelerator _acc;
    public FlowSampler(Accelerator accelerator) => _acc = accelerator;

    /// <summary>The (t, t_prev) pairs of the schedule, in the reference's float64.</summary>
    public static (double T, double TPrev)[] Schedule(int steps, double rescaleT)
    {
        var pairs = new (double, double)[steps];
        double Rescale(double s) => rescaleT * s / (1 + (rescaleT - 1) * s);
        for (int i = 0; i < steps; i++)
        {
            pairs[i] = (Rescale(Linspace(i, steps)), Rescale(Linspace(i + 1, steps)));
        }
        return pairs;
    }

    /// <summary>np.linspace(1, 0, steps + 1)[i], bit for bit: start + i * step, with the endpoint forced to the stop value.</summary>
    private static double Linspace(int i, int steps) => i == steps ? 0.0 : i * (-1.0 / steps) + 1.0;

    /// <summary>Number of model evaluations <see cref="Sample"/> will make for these options.</summary>
    public static int ModelCalls(Options o)
    {
        int calls = 0;
        foreach (var (t, _) in Schedule(o.Steps, o.RescaleT))
            calls += UsesGuidance(o, t) ? 2 : 1;
        return calls;
    }

    private static bool UsesGuidance(Options o, double t)
        => o.GuidanceStrength != 1f && o.IntervalLo <= t && t <= o.IntervalHi;   // strength 0 (negative only) is never configured

    // ---------------------------------------------------------------------------------------------------
    private Action<Index1D, FV, FV, float>? _axpy;
    private static void AxpyImpl(Index1D i, FV x, FV v, float a) => x[i] = x[i] - a * v[i];

    private Action<Index1D, FV, FV, FV, float>? _cfg;
    private static void CfgImpl(Index1D i, FV pos, FV neg, FV pred, float s) => pred[i] = s * pos[i] + (1f - s) * neg[i];

    // partial double sums of x0(pos) and x0(cfg): sums[k*4 + {0..3}] = sumA, sumA2, sumB, sumB2 for strided slice k
    private Action<Index1D, FV, FV, FV, DV, int, float, float>? _partial;
    private const int Slices = 1024;
    private static void PartialImpl(Index1D k, FV x, FV pos, FV cfg, DV sums, int n, float c1, float c2)
    {
        double sa = 0, sa2 = 0, sb = 0, sb2 = 0;
        for (int i = k; i < n; i += Slices)
        {
            double a = c1 * x[i] - c2 * pos[i];     // x0 from pred: (1 - sigma) x - (sigma + (1 - sigma) t) pred
            double b = c1 * x[i] - c2 * cfg[i];
            sa += a; sa2 += a * a; sb += b; sb2 += b * b;
        }
        sums[k * 4] = sa; sums[k * 4 + 1] = sa2; sums[k * 4 + 2] = sb; sums[k * 4 + 3] = sb2;
    }

    // stds[0] = std(x0 pos), stds[1] = std(x0 cfg)  (unbiased, like torch.std)
    private Action<Index1D, DV, FV, int>? _finish;
    private static void FinishImpl(Index1D _, DV sums, FV stds, int n)
    {
        double sa = 0, sa2 = 0, sb = 0, sb2 = 0;
        for (int k = 0; k < Slices; k++) { sa += sums[k * 4]; sa2 += sums[k * 4 + 1]; sb += sums[k * 4 + 2]; sb2 += sums[k * 4 + 3]; }
        double va = (sa2 - sa * sa / n) / (n - 1), vb = (sb2 - sb * sb / n) / (n - 1);
        stds[0] = (float)Math.Sqrt(Math.Max(va, 0.0));
        stds[1] = (float)Math.Sqrt(Math.Max(vb, 0.0));
    }

    private Action<Index1D, FV, FV, FV, float, float, float>? _rescale;
    private static void RescaleImpl(Index1D i, FV x, FV pred, FV stds, float sigma, float t, float r)
    {
        float c2 = sigma + (1f - sigma) * t;
        float x0cfg = (1f - sigma) * x[i] - c2 * pred[i];
        float x0r = x0cfg * (stds[0] / stds[1]);
        float x0 = r * x0r + (1f - r) * x0cfg;
        pred[i] = ((1f - sigma) * x[i] - x0) / c2;
    }

    // counter-based Gaussian: splitmix64 -> two uniforms -> Box-Muller (one normal per element)
    private Action<Index1D, FV, ulong>? _randn;
    private static void RandnImpl(Index1D i, FV o, ulong seed)
    {
        ulong z = seed + (ulong)(int)i * 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; z ^= z >> 31;
        ulong w = z + 0x9E3779B97F4A7C15UL;
        w = (w ^ (w >> 30)) * 0xBF58476D1CE4E5B9UL; w = (w ^ (w >> 27)) * 0x94D049BB133111EBUL; w ^= w >> 31;
        float u1 = ((z >> 40) + 0.5f) * (1f / 16777216f);      // (0, 1)
        float u2 = (w >> 40) * (1f / 16777216f);
        o[i] = MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(6.2831853071795864f * u2);
    }

    /// <summary>Fill <paramref name="o"/> with standard normal noise from a counter-based generator (deterministic per seed).</summary>
    public void Randn(FV o, int count, ulong seed)
    {
        _randn ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, ulong>(RandnImpl);
        _randn(count, o, seed);
    }

    /// <summary>
    /// Integrate from noise to sample. <paramref name="x"/> holds the initial noise on entry and the sample on exit
    /// (<paramref name="count"/> elements). Model evaluations are issued synchronously by <paramref name="model"/>. <paramref name="afterStep"/> gets (step, the time x has reached) after every Euler step.
    /// </summary>
    public void Sample(Model model, FV x, int count, Options o, Action<int, int>? progress = null, Action<int, double>? afterStep = null)
    {
        _axpy ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, float>(AxpyImpl);
        _cfg ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, float>(CfgImpl);
        _partial ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, DV, int, float, float>(PartialImpl);
        _finish ??= _acc.LoadAutoGroupedStreamKernel<Index1D, DV, FV, int>(FinishImpl);
        _rescale ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, float, float, float>(RescaleImpl);

        using var vPos = _acc.Allocate1D<float>(count);
        using var vNeg = _acc.Allocate1D<float>(count);
        using var pred = _acc.Allocate1D<float>(count);
        using var sums = _acc.Allocate1D<double>(Slices * 4);
        using var stds = _acc.Allocate1D<float>(2);

        var steps = Schedule(o.Steps, o.RescaleT);
        float sigma = o.SigmaMin;
        for (int step = 0; step < steps.Length; step++)
        {
            var (t, tPrev) = steps[step];
            float t1000 = (float)(1000 * t);
            if (!UsesGuidance(o, t))
            {
                model(x, t1000, true, pred.View);
            }
            else
            {
                model(x, t1000, true, vPos.View);
                model(x, t1000, false, vNeg.View);
                _cfg(count, vPos.View, vNeg.View, pred.View, o.GuidanceStrength);
                if (o.GuidanceRescale > 0f)
                {
                    float tf = (float)t;
                    _partial(Slices, x, vPos.View, pred.View, sums.View, count, 1f - sigma, sigma + (1f - sigma) * tf);
                    _finish(1, sums.View, stds.View, count);
                    _rescale(count, x, pred.View, stds.View, sigma, tf, o.GuidanceRescale);
                }
            }
            _axpy(count, x, pred.View, (float)(t - tPrev));
            afterStep?.Invoke(step, tPrev);                     // x is now at time tPrev: RePaint-style editing puts the known part back here
            _acc.Fence();
            progress?.Invoke(step + 1, steps.Length);
        }
    }
}
