using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Training;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Training M1.3: every gradient kernel in <see cref="BackwardKernels"/> against central finite differences of a float64 CPU
/// forward (an independent check of the derivation, not a restatement of the formula), on every backend. The scalar being
/// differentiated is L = sum(dy * f(x)) with a random dy, so the expected gradient is the vector-Jacobian product.
/// </summary>
public abstract partial class MLTestBase
{
    private static double[] Fd(double[] x, Func<double[], double> f, double h = 1e-5)
    {
        var g = new double[x.Length];
        var xp = (double[])x.Clone();
        for (int i = 0; i < x.Length; i++)
        {
            double o = xp[i];
            xp[i] = o + h; double a = f(xp);
            xp[i] = o - h; double b = f(xp);
            xp[i] = o;
            g[i] = (a - b) / (2 * h);
        }
        return g;
    }

    private static double[] D(float[] v) => v.Select(x => (double)x).ToArray();
    private static float[] F(double[] v) => v.Select(x => (float)x).ToArray();

    private static void AssertGrad(string what, float[] got, double[] want, double tol = 2e-4)
    {
        double scale = Math.Max(1e-3, want.Max(Math.Abs));
        double worst = 0; int at = 0;
        for (int i = 0; i < want.Length; i++) { double d = Math.Abs(got[i] - want[i]); if (d > worst) { worst = d; at = i; } }
        if (worst > tol * Math.Max(1, scale))
            throw new Exception($"{what}: max |diff| {worst:E2} at {at} (got {got[at]:G6}, finite difference {want[at]:G6}, max |grad| {scale:G4})");
    }

    [TestMethod]
    public async Task Backward_RMSNorm_MatchesFiniteDifference() => await RunTest(async accelerator =>
    {
        int rows = 3, C = 40; float eps = 1e-5f;
        var x = RandomFloats(rows * C, seed: 501); var w = RandomFloats(C, seed: 502, scale: 0.5f); var dy = RandomFloats(rows * C, seed: 503);
        for (int i = 0; i < C; i++) w[i] += 1f;
        double L(double[] xv)
        {
            double s = 0;
            for (int r = 0; r < rows; r++)
            {
                double ss = 0; for (int c = 0; c < C; c++) ss += xv[r * C + c] * xv[r * C + c];
                double rstd = 1 / Math.Sqrt(ss / C + eps);
                for (int c = 0; c < C; c++) s += dy[r * C + c] * xv[r * C + c] * rstd * w[c];
            }
            return s;
        }
        using var reg = new OperatorRegistry(accelerator);
        using var xb = accelerator.Allocate1D(x); using var wb = accelerator.Allocate1D(w); using var db = accelerator.Allocate1D(dy);
        using var dx = accelerator.Allocate1D<float>(rows * C); using var scratch = accelerator.Allocate1D<float>(2 * rows);
        reg.Backward.RMSNorm(xb.View, wb.View, db.View, dx.View, scratch.View, rows, C, eps);
        AssertGrad("RMSNorm dx", await dx.View.CopyToHostAsync(), Fd(D(x), L));
        return "RMSNorm dx matches finite differences";
    });

    [TestMethod]
    public async Task Backward_SwiGLU_MatchesFiniteDifference() => await RunTest(async accelerator =>
    {
        int n = 64;
        var g = RandomFloats(n, seed: 511, scale: 6f); var u = RandomFloats(n, seed: 512, scale: 2f); var dy = RandomFloats(n, seed: 513);
        double Silu(double v) => v / (1 + Math.Exp(-v));
        double Lg(double[] gv) { double s = 0; for (int i = 0; i < n; i++) s += dy[i] * Silu(gv[i]) * u[i]; return s; }
        double Lu(double[] uv) { double s = 0; for (int i = 0; i < n; i++) s += dy[i] * Silu(g[i]) * uv[i]; return s; }
        using var reg = new OperatorRegistry(accelerator);
        using var gb = accelerator.Allocate1D(g); using var ub = accelerator.Allocate1D(u); using var db = accelerator.Allocate1D(dy);
        using var dg = accelerator.Allocate1D<float>(n); using var du = accelerator.Allocate1D<float>(n);
        reg.Backward.SwiGLU(gb.View, ub.View, db.View, dg.View, du.View, n);
        AssertGrad("SwiGLU dg", await dg.View.CopyToHostAsync(), Fd(D(g), Lg));
        AssertGrad("SwiGLU du", await du.View.CopyToHostAsync(), Fd(D(u), Lu));
        return "SwiGLU dg, du match finite differences";
    });

    /// <summary>RoPE is a rotation, so its backward is its inverse: RoPEGrad(RoPE(x)) must return x, through the engine's own forward kernel,
    /// for both pairings, a partial rotary dimension, rows_per_position (seq-major heads) and a position offset.</summary>
    [TestMethod]
    public async Task Backward_RoPE_InvertsForward() => await RunTest(async accelerator =>
    {
        int seq = 6, heads = 3, hd = 16;
        int rows = seq * heads;
        var x = RandomFloats(rows * hd, seed: 521);
        using var reg = new OperatorRegistry(accelerator);
        using var xb = accelerator.Allocate1D(x); using var yb = accelerator.Allocate1D<float>(x.Length); using var back = accelerator.Allocate1D<float>(x.Length);
        var notes = new List<string>();
        foreach (var (interleaved, rot, offset) in new[] { (true, 16, 0), (false, 16, 5), (true, 8, 3) })
        {
            reg.RoPE.Apply(xb.View, yb.View, rows, hd, offset, 10000f, rot, interleaved, heads);
            reg.Backward.RoPE(yb.View, back.View, null, rows, hd, offset, 10000f, rot, interleaved, heads);
            var y = await yb.View.CopyToHostAsync();
            var r = await back.View.CopyToHostAsync();
            double moved = x.Zip(y, (a, b) => Math.Abs(a - b)).Max();
            if (moved < 1e-2) throw new Exception($"RoPE forward did not rotate (interleaved={interleaved}): the round trip proves nothing");
            AssertGrad($"RoPE round trip interleaved={interleaved} rotary={rot} offset={offset}", r, D(x), 1e-5);
            notes.Add($"interleaved={interleaved} rotary={rot} offset={offset}");
        }
        return "RoPEGrad(RoPE(x)) == x: " + string.Join("; ", notes);
    });

    [TestMethod]
    public async Task Backward_SoftmaxCrossEntropy_MatchesFiniteDifference() => await RunTest(async accelerator =>
    {
        int rows = 5, C = 11, ignore = -100;
        var logits = RandomFloats(rows * C, seed: 531, scale: 3f);
        var labels = new float[] { 3, ignore, 0, 10, ignore };
        double L(double[] lv)
        {
            double s = 0; int n = 0;
            for (int r = 0; r < rows; r++)
            {
                if ((int)labels[r] == ignore) continue;
                double m = double.NegativeInfinity; for (int c = 0; c < C; c++) m = Math.Max(m, lv[r * C + c]);
                double z = 0; for (int c = 0; c < C; c++) z += Math.Exp(lv[r * C + c] - m);
                s += -(lv[r * C + (int)labels[r]] - m - Math.Log(z)); n++;
            }
            return s / n;
        }
        using var reg = new OperatorRegistry(accelerator);
        using var lb = accelerator.Allocate1D(logits); using var yb = accelerator.Allocate1D(labels);
        using var dl = accelerator.Allocate1D<float>(rows * C); using var cnt = accelerator.Allocate1D<float>(1 + rows);
        reg.Backward.SoftmaxCrossEntropy(lb.View, yb.View, dl.View, cnt.View, rows, C, ignore);
        AssertGrad("SoftmaxCE dlogits", await dl.View.CopyToHostAsync(), Fd(D(logits), L));
        return "SoftmaxCE dlogits (mean, ignore_index) matches finite differences";
    });

    /// <summary>Causal GQA attention in the seq-major layout GGUFGraphBuilder emits, with and without a kv offset and a window.</summary>
    [TestMethod]
    public async Task Backward_Attention_MatchesFiniteDifference() => await RunTest(async accelerator =>
    {
        int heads = 4, kvHeads = 2, seq = 5, hd = 8;
        var notes = new List<string>();
        foreach (var (kvOffset, window, seqKV) in new[] { (0, 0, 5), (2, 0, 7), (0, 3, 5) })
        {
            var q = RandomFloats(seq * heads * hd, seed: 541 + kvOffset); var k = RandomFloats(seqKV * kvHeads * hd, seed: 542 + window);
            var v = RandomFloats(seqKV * kvHeads * hd, seed: 543); var dout = RandomFloats(seq * heads * hd, seed: 544);
            double scale = 1 / Math.Sqrt(hd);
            double[] Fwd(double[] qv, double[] kv, double[] vv)
            {
                var o = new double[seq * heads * hd];
                for (int h = 0; h < heads; h++)
                {
                    int g = h / (heads / kvHeads);
                    for (int i = 0; i < seq; i++)
                    {
                        int qp = i + kvOffset;
                        var sc = new double[seqKV]; double m = double.NegativeInfinity;
                        for (int j = 0; j < seqKV; j++)
                        {
                            bool ok = j <= qp && (window == 0 || qp - j < window);
                            if (!ok) { sc[j] = double.NegativeInfinity; continue; }
                            double s = 0; for (int e = 0; e < hd; e++) s += qv[(i * heads + h) * hd + e] * kv[(j * kvHeads + g) * hd + e];
                            sc[j] = s * scale; m = Math.Max(m, sc[j]);
                        }
                        double z = 0; for (int j = 0; j < seqKV; j++) z += double.IsNegativeInfinity(sc[j]) ? 0 : Math.Exp(sc[j] - m);
                        for (int j = 0; j < seqKV; j++)
                        {
                            if (double.IsNegativeInfinity(sc[j])) continue;
                            double p = Math.Exp(sc[j] - m) / z;
                            for (int e = 0; e < hd; e++) o[(i * heads + h) * hd + e] += p * vv[(j * kvHeads + g) * hd + e];
                        }
                    }
                }
                return o;
            }
            double Dot(double[] o) { double s = 0; for (int i = 0; i < o.Length; i++) s += dout[i] * o[i]; return s; }
            var outp = F(Fwd(D(q), D(k), D(v)));
            using var reg = new OperatorRegistry(accelerator);
            using var qb = accelerator.Allocate1D(q); using var kb = accelerator.Allocate1D(k); using var vb = accelerator.Allocate1D(v);
            using var ob = accelerator.Allocate1D(outp); using var dob = accelerator.Allocate1D(dout);
            using var dq = accelerator.Allocate1D<float>(q.Length); using var dk = accelerator.Allocate1D<float>(k.Length); using var dv = accelerator.Allocate1D<float>(v.Length);
            using var stats = accelerator.Allocate1D<float>(2 * heads * seq);
            reg.Backward.Attention(qb.View, kb.View, vb.View, ob.View, dob.View, dq.View, dk.View, dv.View, stats.View,
                new BackwardKernels.AttnDims { Heads = heads, KvHeads = kvHeads, SeqQ = seq, SeqKV = seqKV, HeadDim = hd, Causal = 1, Window = window, KvOffset = kvOffset, Scale = (float)scale });
            string tag = $"kvOffset={kvOffset} window={window} seqKV={seqKV}";
            AssertGrad($"attention dq ({tag})", await dq.View.CopyToHostAsync(), Fd(D(q), qv => Dot(Fwd(qv, D(k), D(v)))));
            AssertGrad($"attention dk ({tag})", await dk.View.CopyToHostAsync(), Fd(D(k), kv => Dot(Fwd(D(q), kv, D(v)))));
            AssertGrad($"attention dv ({tag})", await dv.View.CopyToHostAsync(), Fd(D(v), vv => Dot(Fwd(D(q), D(k), vv))));
            notes.Add(tag);
        }
        return "attention dq, dk, dv match finite differences: " + string.Join("; ", notes);
    });

    /// <summary>
    /// The ONNX SoftmaxCrossEntropyLoss op (rewritten onto the device 2026-11-04): mean over the rows NOT ignored, sum, none, and the
    /// log_prob output, against a float64 CPU reference. The old body divided by every row even when rows were masked.
    /// </summary>
    [TestMethod]
    public async Task SoftmaxCrossEntropyLoss_Op_MatchesReference() => await RunTest(async accelerator =>
    {
        int rows = 6, C = 9, ignore = -100;
        var logits = RandomFloats(rows * C, seed: 551, scale: 3f);
        var labels = new float[] { 4, ignore, 0, 8, ignore, 2 };
        double Lse(int r) { double m = double.NegativeInfinity; for (int c = 0; c < C; c++) m = Math.Max(m, logits[r * C + c]); double z = 0; for (int c = 0; c < C; c++) z += Math.Exp(logits[r * C + c] - m); return m + Math.Log(z); }
        var perRow = Enumerable.Range(0, rows).Select(r => (int)labels[r] == ignore ? 0.0 : Lse(r) - logits[r * C + (int)labels[r]]).ToArray();
        int kept = labels.Count(l => (int)l != ignore);
        using var reg = new OperatorRegistry(accelerator);
        using var lb = accelerator.Allocate1D(logits); using var yb = accelerator.Allocate1D(labels);
        using var loss = accelerator.Allocate1D<float>(rows); using var lp = accelerator.Allocate1D<float>(rows * C); using var scratch = accelerator.Allocate1D<float>(1 + rows);
        var notes = new List<string>();
        foreach (var (red, name, want) in new[] { (1, "mean", new[] { perRow.Sum() / kept }), (2, "sum", new[] { perRow.Sum() }), (0, "none", perRow) })
        {
            reg.Backward.SoftmaxCrossEntropyLoss(lb.View, yb.View, loss.View, lp.View, scratch.View, rows, C, ignore, red);
            var got = await loss.View.SubView(0, want.Length).CopyToHostAsync();
            AssertGrad($"CE loss reduction={name}", got, want, 1e-5);
            notes.Add($"{name} {got[0]:F5}");
        }
        var gotLp = await lp.View.CopyToHostAsync();
        AssertGrad("CE log_prob", gotLp, Enumerable.Range(0, rows * C).Select(i => logits[i] - Lse(i / C)).ToArray(), 1e-5);
        return "SoftmaxCrossEntropyLoss: " + string.Join(", ", notes) + ", log_prob OK";
    });
}
