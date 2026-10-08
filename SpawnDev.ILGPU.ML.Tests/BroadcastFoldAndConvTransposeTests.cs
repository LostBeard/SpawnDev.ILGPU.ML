using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Tests;

/// <summary>
/// Two bugs big-LaMa (Carve/LaMa-ONNX lama_fp32.onnx) found, 2026-10-08, both silent until a crash 9000 nodes later:
/// <list type="number">
/// <item>Compile-time folding of a binary op on two constants used a flat modulo (<c>a[j % lenA]</c>) and kept the
/// LARGER input's shape - right for a scalar or equal shapes, wrong for a real N-D broadcast. LaMa builds its DFT
/// matrices as <c>[64,1] * [33]</c>: the fold returned [64,1] and every Fourier unit collapsed to one frequency.
/// GraphCompiler.BroadcastFold maps it per axis.</item>
/// <item>2-D ConvTranspose ignored <c>output_padding</c> in shape inference and in the kernel: LaMa's three stride-2
/// decoders went 64 -> 127 -> 253 -> 505 instead of 512.</item>
/// </list>
/// With both fixed, LaMa on OpenCL matches onnxruntime to 0.003 on a 0..255 output.
/// </summary>
public class BroadcastFoldAndConvTransposeTests : KernelTestBase
{
    private readonly OperatorRegistry _reg;
    private readonly BufferPool _pool;

    public BroadcastFoldAndConvTransposeTests(AcceleratorFixture fixture) : base(fixture)
    {
        _reg = new OperatorRegistry(Accelerator);
        _pool = new BufferPool(Accelerator);
    }

    [Fact]
    public void BroadcastFold_ColumnTimesRow_IsTheOuterProductShape()
    {
        var f = GraphCompiler.BroadcastFold(new[] { new[] { 4, 1 }, new[] { 3 } }, 4, 3);
        Assert.NotNull(f);
        Assert.Equal(new[] { 4, 3 }, f!.Value.Shape);
        // out[i, j] reads a[i] and b[j]
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
            {
                Assert.Equal(i, f.Value.IdxA[i * 3 + j]);
                Assert.Equal(j, f.Value.IdxB[i * 3 + j]);
            }
    }

    [Fact]
    public void BroadcastFold_LeavesScalarsAndUnknownShapesToTheFlatFold()
    {
        // A scalar against a vector broadcasts to the vector (and the flat fold agrees).
        var s = GraphCompiler.BroadcastFold(new[] { Array.Empty<int>(), new[] { 5 } }, 1, 5);
        Assert.NotNull(s);
        Assert.Equal(new[] { 5 }, s!.Value.Shape);
        Assert.All(s.Value.IdxA, i => Assert.Equal(0, i));
        // Shape and value count disagree: no claim, the caller keeps its old behaviour.
        Assert.Null(GraphCompiler.BroadcastFold(new[] { new[] { 2, 2 }, new[] { 3 } }, 5, 3));
        // Not broadcastable.
        Assert.Null(GraphCompiler.BroadcastFold(new[] { new[] { 2, 3 }, new[] { 4 } }, 6, 4));
    }

    static float[] ConvTransposeReference(float[] x, int inC, int inH, int inW, float[] w, int outC, int k, int stride, int pad, int outPad, float[] bias)
    {
        int outH = (inH - 1) * stride - 2 * pad + k + outPad, outW = (inW - 1) * stride - 2 * pad + k + outPad;
        var y = new float[outC * outH * outW];
        for (int oc = 0; oc < outC; oc++)
            for (int i = 0; i < outH * outW; i++) y[oc * outH * outW + i] = bias[oc];
        // Scatter form (the definition): every input pixel adds the kernel at stride * position - pad.
        for (int ic = 0; ic < inC; ic++)
            for (int iy = 0; iy < inH; iy++)
                for (int ix = 0; ix < inW; ix++)
                    for (int oc = 0; oc < outC; oc++)
                        for (int ky = 0; ky < k; ky++)
                            for (int kx = 0; kx < k; kx++)
                            {
                                int oy = iy * stride - pad + ky, ox = ix * stride - pad + kx;
                                if (oy < 0 || ox < 0 || oy >= outH || ox >= outW) continue;
                                y[(oc * outH + oy) * outW + ox] += x[(ic * inH + iy) * inW + ix] * w[((ic * outC + oc) * k + ky) * k + kx];
                            }
        return y;
    }

    [Fact]
    public void ConvTranspose2D_OutputPadding_LandsOnAnExactDoubling()
    {
        const int inC = 2, outC = 3, H = 4, W = 5, k = 3, stride = 2, pad = 1, outPad = 1;
        var rng = new Random(9);
        float[] R(int n) => Enumerable.Range(0, n).Select(_ => (float)rng.NextDouble() - 0.5f).ToArray();
        var xv = R(inC * H * W); var wv = R(inC * outC * k * k); var bv = R(outC);
        var attrs = new Dictionary<string, object>
        {
            ["strides"] = new long[] { stride, stride }, ["pads"] = new long[] { pad, pad, pad, pad },
            ["output_padding"] = new long[] { outPad, outPad }, ["kernel_shape"] = new long[] { k, k },
        };
        var op = _reg.Resolve("ConvTranspose");
        var shape = op.InferOutputShapes(new[] { new[] { 1, inC, H, W }, new[] { inC, outC, k, k }, new[] { outC } }, attrs)[0];
        Assert.Equal(new[] { 1, outC, 2 * H, 2 * W }, shape);

        var x = _pool.AllocatePermanent(xv, new[] { 1, inC, H, W });
        var wt = _pool.AllocatePermanent(wv, new[] { inC, outC, k, k });
        var b = _pool.AllocatePermanent(bv, new[] { outC });
        var y = _pool.AllocatePermanent(new float[outC * 2 * H * 2 * W], shape);
        op.Execute(new OnnxOpContext
        {
            Inputs = new[] { x, wt, b }, Outputs = new[] { y }, Attributes = attrs, Pool = _pool, Registry = _reg,
            InputNames = new[] { "x", "w", "b" },
        });
        Accelerator.Synchronize();
        var got = new float[y.ElementCount];
        y.Data.SubView(0, y.ElementCount).CopyToCPU(got);
        var want = ConvTransposeReference(xv, inC, H, W, wv, outC, k, stride, pad, outPad, bv);
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++) Assert.True(MathF.Abs(want[i] - got[i]) < 1e-5f, $"element {i}: {got[i]} vs {want[i]}");
    }
}
