using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    // probe kernels for fractional bilinear sampling (the DeformConv WebGL failure): each isolates one step
    private static void ProbeFracImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> t, ArrayView1D<float, Stride1D.Dense> o) => o[i] = t[i] - MathF.Floor(t[i]);
    private static void ProbeIntFracImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> t, ArrayView1D<float, Stride1D.Dense> o) { int k = (int)MathF.Floor(t[i]); o[i] = t[i] - k; }
    private static void ProbeLerpImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> t, ArrayView1D<float, Stride1D.Dense> o) { float f = t[i] - MathF.Floor(t[i]); o[i] = 3f * (1f - f) + 7f * f; }
    private static void ProbeGatherLerpImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> t, ArrayView1D<float, Stride1D.Dense> src, ArrayView1D<float, Stride1D.Dense> o)
    {
        float x = t[i] + 4f;                       // 4.x: inside src
        float fl = MathF.Floor(x);
        int k = (int)fl;
        float f = x - fl;
        o[i] = src[k] * (1f - f) + src[k + 1] * f;
    }
    private static void ProbeMulChainImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> t, ArrayView1D<float, Stride1D.Dense> o)
    {
        float ty = t[i] - MathF.Floor(t[i]), tx = 1f - ty;
        o[i] = 2f * (1f - ty) * (1f - tx) + 3f * (1f - ty) * tx + 5f * ty * (1f - tx) + 7f * ty * tx;
    }

    /// <summary>Fractional bilinear steps on every backend against the host: which one a backend gets wrong (DeformConv was wrong on WebGL for fractional offsets only).</summary>
    [TestMethod(Timeout = 60000)]
    public async Task Bilinear_FractionalSteps_MatchHost() => await RunTest(async accelerator =>
    {
        var t = new float[] { 0.25f, 1.75f, -0.3f, -1.6f, 2.5f, 0.0f, 3.9f, -2.1f };
        var src = new float[16]; for (int i = 0; i < 16; i++) src[i] = i * i * 0.5f;
        using var tb = accelerator.Allocate1D(t); using var sb = accelerator.Allocate1D(src); using var ob = accelerator.Allocate1D<float>(t.Length);
        var report = new List<string>(); string? bad = null;
        async Task Check(string name, Action run, Func<float, float> host)
        {
            run();
            var got = await ob.CopyToHostAsync<float>(0, t.Length);
            double worst = 0; int at = 0;
            for (int i = 0; i < t.Length; i++) { double d = Math.Abs(got[i] - host(t[i])); if (!(d <= worst)) { worst = d; at = i; } }
            report.Add($"{name}: {worst:E1} (t {t[at]}: ours {got[at]:G5}, host {host(t[at]):G5})");
            if (worst > 1e-4 && bad == null) bad = name;
        }
        await Check("t - floor(t)", () => accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ProbeFracImpl)(t.Length, tb.View, ob.View), x => x - MathF.Floor(x));
        await Check("t - (int)floor(t)", () => accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ProbeIntFracImpl)(t.Length, tb.View, ob.View), x => x - (int)MathF.Floor(x));
        await Check("lerp", () => accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ProbeLerpImpl)(t.Length, tb.View, ob.View), x => { float f = x - MathF.Floor(x); return 3f * (1f - f) + 7f * f; });
        await Check("gather + lerp", () => accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ProbeGatherLerpImpl)(t.Length, tb.View, sb.View, ob.View),
            x => { float y = x + 4f; float fl = MathF.Floor(y); int k = (int)fl; float f = y - fl; return src[k] * (1f - f) + src[k + 1] * f; });
        await Check("4-corner weights", () => accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ProbeMulChainImpl)(t.Length, tb.View, ob.View),
            x => { float ty = x - MathF.Floor(x), tx = 1f - ty; return 2f * (1f - ty) * (1f - tx) + 3f * (1f - ty) * tx + 5f * ty * (1f - tx) + 7f * ty * tx; });
        Console.WriteLine($"[DeformDiag] probes {BackendName}: " + string.Join(" | ", report));
        if (bad != null) throw new Exception($"first wrong step: {bad}. " + string.Join(" | ", report));
    });

    /// <summary>
    /// The DeformConv kernel alone against a host reference of the same formula, in growing configurations (1x1 / zero offsets, 3x3 + pad, integer offsets,
    /// fractional offsets, + bias, + mask, groups + dilation + stride): the first configuration that breaks names the part a backend gets wrong.
    /// Every configuration must match on every backend.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task DeformConv_Kernel_MatchesHostReference_Configs() => await RunTest(async accelerator =>
    {
        var reg = new Operators.OperatorRegistry(accelerator);
        var rng = new Random(1108);
        var report = new List<string>();
        string? firstBad = null;
        foreach (var (name, kH, pad, offScale, offInt, bias, mask, groups, offGroups, stride, dil) in new[]
        {
            ("1x1 zero-offset", 1, 0, 0f, false, false, false, 1, 1, 1, 1),
            ("3x3 pad1 zero-offset", 3, 1, 0f, false, false, false, 1, 1, 1, 1),
            ("3x3 integer offsets", 3, 1, 1f, true, false, false, 1, 1, 1, 1),
            ("3x3 dy=0.5 const", 3, 1, -1f, false, false, false, 1, 1, 1, 1),
            ("3x3 dx=0.5 const", 3, 1, -2f, false, false, false, 1, 1, 1, 1),
            ("3x3 random dy only", 3, 1, -3f, false, false, false, 1, 1, 1, 1),
            ("3x3 random dx only", 3, 1, -4f, false, false, false, 1, 1, 1, 1),
            ("3x3 fractional offsets", 3, 1, 0.7f, false, false, false, 1, 1, 1, 1),
            ("+ bias", 3, 1, 0.7f, false, true, false, 1, 1, 1, 1),
            ("+ mask", 3, 1, 0.7f, false, true, true, 1, 1, 1, 1),
            ("groups 2, offset groups 2, stride 2, dilation 2", 3, 2, 0.7f, false, true, true, 2, 2, 2, 2),
        })
        {
            int N = 1, inC = 4, H = 6, W = 7, outC = 4, kW = kH;
            int outH = (H + 2 * pad - dil * (kH - 1) - 1) / stride + 1, outW = (W + 2 * pad - dil * (kW - 1) - 1) / stride + 1;
            int cPerG = inC / groups;
            float[] R(int n, float s) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * s; return a; }
            var x = R(N * inC * H * W, 1f);
            var wt = R(outC * cPerG * kH * kW, 1f);
            var off = R(N * offGroups * kH * kW * 2 * outH * outW, offScale);
            if (offInt) for (int i = 0; i < off.Length; i++) off[i] = MathF.Round(off[i] * 2f);
            if (offScale < 0)
            {
                // patterns: -1 dy 0.5 everywhere, -2 dx 0.5, -3 random dy only, -4 random dx only (layout [.., k*2 + {0 dy, 1 dx}, outH, outW])
                var rnd = R(off.Length, 0.7f);
                for (int i = 0; i < off.Length; i++)
                {
                    bool isDx = (i / (outH * outW)) % 2 == 1;
                    off[i] = offScale switch { -1f => isDx ? 0f : 0.5f, -2f => isDx ? 0.5f : 0f, -3f => isDx ? 0f : rnd[i], _ => isDx ? rnd[i] : 0f };
                }
            }
            var b = R(outC, 1f);
            var m = R(N * offGroups * kH * kW * outH * outW, 1f);
            // host reference: the kernel's own formula (bilinear, zero outside)
            var want = new float[N * outC * outH * outW];
            int hw = outH * outW;
            for (int n = 0; n < N; n++) for (int oc = 0; oc < outC; oc++) for (int oh = 0; oh < outH; oh++) for (int ow = 0; ow < outW; ow++)
            {
                int g = oc / (outC / groups);
                double sum = 0;
                for (int ic = g * cPerG; ic < (g + 1) * cPerG; ic++)
                {
                    int og = ic / (inC / offGroups);
                    for (int kh = 0; kh < kH; kh++) for (int kw = 0; kw < kW; kw++)
                    {
                        int k = kh * kW + kw;
                        int ob = ((n * offGroups + og) * kH * kW + k) * 2 * hw + oh * outW + ow;
                        float fy = oh * stride - pad + kh * dil + off[ob], fx = ow * stride - pad + kw * dil + off[ob + hw];
                        int y0 = (int)MathF.Floor(fy), x0 = (int)MathF.Floor(fx);
                        float ty = fy - y0, tx = fx - x0;
                        float V(int yy, int xx) => yy >= 0 && yy < H && xx >= 0 && xx < W ? x[((n * inC + ic) * H + yy) * W + xx] : 0f;
                        float v = V(y0, x0) * (1 - ty) * (1 - tx) + V(y0, x0 + 1) * (1 - ty) * tx + V(y0 + 1, x0) * ty * (1 - tx) + V(y0 + 1, x0 + 1) * ty * tx;
                        if (mask) v *= m[((n * offGroups + og) * kH * kW + k) * hw + oh * outW + ow];
                        sum += v * wt[((oc * cPerG + (ic - g * cPerG)) * kH + kh) * kW + kw];
                    }
                }
                want[((n * outC + oc) * outH + oh) * outW + ow] = (float)sum + (bias ? b[oc] : 0f);
            }
            using var xb = accelerator.Allocate1D(x); using var wb = accelerator.Allocate1D(wt); using var obuf = accelerator.Allocate1D(off);
            using var bb = accelerator.Allocate1D(bias ? b : new float[] { 0f, 0f }); using var mb = accelerator.Allocate1D(mask ? m : new float[] { 0f });
            using var yb = accelerator.Allocate1D<float>(want.Length);
            var prm = reg.ParamsView(new float[] { inC, H, W, outC, outH, outW, kH, kW, stride, stride, pad, pad, groups, offGroups, dil, dil, mask ? 1 : 0, bias ? 1 : 0 });
            reg.ElementWise.DeformConv(xb.View, wb.View, obuf.View, mb.View, bb.View, yb.View, prm, want.Length);
            var got = await yb.CopyToHostAsync<float>(0, want.Length);
            double worst = 0; int at = 0;
            for (int i = 0; i < want.Length; i++) { double d = Math.Abs(got[i] - want[i]); if (!(d <= worst)) { worst = d; at = i; } }
            report.Add($"{name}: max|diff| {worst:E1} at {at} (ours {got[at]:G5}, host {want[at]:G5})");
            if (!(worst < 1e-3) && firstBad == null) firstBad = name;
        }
        Console.WriteLine($"[DeformDiag] {BackendName}: " + string.Join(" | ", report));
        if (firstBad != null) throw new Exception($"first wrong configuration: {firstBad}. " + string.Join(" | ", report));
    });
}
