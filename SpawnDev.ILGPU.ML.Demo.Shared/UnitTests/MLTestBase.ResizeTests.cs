using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Resize OPERATOR tests — the production node path (Resolve -> Execute) vs a CPU oracle. These exist
/// because <see cref="ResizeOperator"/> used to IGNORE the ONNX <c>mode</c> attribute and ALWAYS
/// bilinear-resize. SD-Turbo's VAE decoder (and depth/style/super-res models) upsample with
/// <c>mode="nearest"</c>; bilinear-smoothing a nearest Resize low-passes the whole image, so EVERY
/// generated image came out blurry. The discriminator test below is the regression guard: it uses a 2x2
/// input whose nearest and bilinear upsamples DIFFER at every interior pixel, so a regression to "always
/// bilinear" fails loudly. (Found by diffing our VAE decode against the ONNX Runtime oracle on an identical
/// latent — ORT sharp, ours soft — then reading the op.)
/// </summary>
public abstract partial class MLTestBase
{
    // CPU oracle: ONNX Resize mode="nearest", asymmetric/floor (the SD/diffusers upsample). Each output
    // pixel copies the input pixel at floor(outIdx * inDim/outDim) per axis. NCHW.
    private static float[] ResizeNearestCpu(float[] x, int C, int inH, int inW, int outH, int outW)
    {
        var o = new float[C * outH * outW];
        for (int c = 0; c < C; c++)
            for (int oy = 0; oy < outH; oy++)
            {
                int iy = (int)((long)oy * inH / outH); if (iy >= inH) iy = inH - 1;
                for (int ox = 0; ox < outW; ox++)
                {
                    int ix = (int)((long)ox * inW / outW); if (ix >= inW) ix = inW - 1;
                    o[(c * outH + oy) * outW + ox] = x[(c * inH + iy) * inW + ix];
                }
            }
        return o;
    }

    [TestMethod]
    public async Task Resize_NearestMode_MatchesCpu_NotBilinear() => await RunTest(async accelerator =>
    {
        // 1 channel, 2x2 -> 4x4 nearest. Distinct corner values so nearest != bilinear at interior pixels.
        const int C = 1, inH = 2, inW = 2, outH = 4, outW = 4;
        var x = new float[] { 0f, 10f, 20f, 30f }; // [[0,10],[20,30]]
        var expected = ResizeNearestCpu(x, C, inH, inW, outH, outW);

        using var inBuf = accelerator.Allocate1D(x);
        using var outBuf = accelerator.Allocate1D<float>(C * outH * outW);
        using var registry = new OperatorRegistry(accelerator);
        var op = registry.Resolve("Resize");
        var pool = new BufferPool(accelerator);
        op.Execute(new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 1, C, inH, inW }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 1, C, outH, outW }) },
            Attributes = new Dictionary<string, object> { ["mode"] = "nearest" },
            Pool = pool,
            InputNames = new[] { "x" },
        });
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, C * outH * outW);

        AssertClose(expected, got, 1e-4f, "Resize nearest ");

        // Discriminator: a "0.5" interior pixel (bilinear avg of 0 and 10 = 5) would mean a regression to
        // always-bilinear. Nearest gives exactly 0 or 10 there — never 5.
        // got[row0] must be [0,0,10,10] for nearest (NOT [0, ~3.3, ~6.6, 10] bilinear).
        if (MathF.Abs(got[1] - 0f) > 1e-4f)
            throw new Exception($"Resize nearest regressed to bilinear: out[0,1]={got[1]} (nearest=0, bilinear≈3.3). " +
                "ResizeOperator must honor mode=\"nearest\".");
        Console.WriteLine($"[Resize] nearest mode matches CPU oracle + is NOT bilinear ({BackendName})");
    });

    /// <summary>
    /// Resize mode="cubic" against onnxruntime for every coordinate_transformation_mode the operator implements, with exclude_outside and a
    /// non-default cubic_coeff_a: a 5x7 input to 8x4 (H up, W down). The operator used to fall back to BILINEAR for cubic align_corners, which
    /// CLAP's HTSAT uses (torch bicubic, align_corners=True) to stretch its log-mel 1001 -> 1024 frames: 2.6% off at that Resize and the audio
    /// tower at cosine 0.9996 to onnxruntime (MEASURED 2026-11-03). Reference: tools/gen_resize_cubic_reference.py (onnxruntime 1.24.3).
    /// </summary>
    [TestMethod]
    public async Task Resize_CubicModes_MatchOnnxRuntime() => await RunTest(async accelerator =>
    {
        const int inH = 5, inW = 7, outH = 8, outW = 4;
        var x = Enumerable.Range(0, inH * inW).Select(i => (float)(Math.Sin(i * 0.7) * 3 + i % 3)).ToArray();
        var cases = new (string Mode, float A, int Exclude, float[] Want)[]
        {
            ("half_pixel", -0.75f, 0, new float[] { 1.12085f, 5.28076f, 1.85396f, -1.71805f, -0.0540311f, 3.07515f, 3.74567f, -0.128395f, -1.26013f, 0.0395078f, 4.83087f, 2.29099f, -1.06745f, -1.54187f, 1.7685f, 4.37539f, 0.407842f, -1.37914f, -1.42577f, 3.64925f, 2.18071f, 0.217433f, -2.23861f, 0.628197f, 3.47599f, 1.63819f, -0.449975f, -1.58687f, 4.04646f, 2.44966f, 1.47278f, -2.67624f }),
            ("align_corners", -0.75f, 0, new float[] { 0f, 4.95635f, 2.00496f, -2.61473f, -1.42872f, 2.31871f, 4.23248f, -0.278547f, -1.77755f, -0.612886f, 4.63562f, 2.74517f, -0.114956f, -1.93272f, 1.73525f, 4.90341f, 1.62152f, -1.49082f, -1.08546f, 4.11988f, 2.33654f, 0.359609f, -1.96058f, 0.936745f, 2.92256f, 1.84206f, -0.991812f, -1.22232f, 3.04589f, 2.50997f, 0.807333f, -1.9154f }),
            ("asymmetric", -0.5f, 0, new float[] { 0f, 4.78043f, 2.21546f, 0.1497f, -1.48274f, 1.75232f, 3.88928f, 1.44003f, -1.54416f, -0.972622f, 3.65167f, 2.77021f, 0.569752f, -2.32961f, 0.571148f, 3.67107f, 1.86511f, -0.755337f, -1.81166f, 2.24712f, 2.65828f, 1.62479f, -2.42741f, -0.0906126f, 2.98093f, 2.82094f, 0.603577f, -2.22165f, 3.08315f, 3.2342f, 1.81659f, -3.0019f }),
            ("pytorch_half_pixel", -0.75f, 1, new float[] { 1.06613f, 5.36305f, 1.80234f, -1.91301f, -0.0615335f, 3.24305f, 3.59255f, -0.32794f, -1.32254f, 0.0395078f, 4.83087f, 2.29274f, -0.98271f, -1.54187f, 1.7685f, 4.42284f, 0.532675f, -1.37914f, -1.42577f, 3.68402f, 2.20665f, 0.217433f, -2.23861f, 0.613942f, 3.46201f, 1.69816f, -0.298406f, -1.64552f, 3.97796f, 2.4837f, 1.5337f, -2.66826f }),
            ("half_pixel", -0.5f, 1, new float[] { 1.01615f, 5.1879f, 1.88388f, -1.76705f, -0.0363229f, 3.15176f, 3.46039f, -0.224186f, -1.30994f, 0.0755415f, 4.83129f, 2.24861f, -0.812171f, -1.50058f, 1.57786f, 4.23466f, 0.480388f, -1.33135f, -1.0624f, 3.57592f, 2.2443f, 0.366386f, -2.13282f, 0.478569f, 3.35342f, 1.72985f, -0.299281f, -1.50964f, 3.86951f, 2.47708f, 1.28681f, -2.49036f }),
        };
        using var inBuf = accelerator.Allocate1D(x);
        using var outBuf = accelerator.Allocate1D<float>(outH * outW);
        using var registry = new OperatorRegistry(accelerator);
        var op = registry.Resolve("Resize");
        var pool = new BufferPool(accelerator);
        var notes = new List<string>();
        foreach (var (mode, a, exclude, want) in cases)
        {
            op.Execute(new OnnxOpContext
            {
                Inputs = new[] { new Tensor(inBuf.View, new[] { 1, 1, inH, inW }) },
                Outputs = new[] { new Tensor(outBuf.View, new[] { 1, 1, outH, outW }) },
                Attributes = new Dictionary<string, object> { ["mode"] = "cubic", ["coordinate_transformation_mode"] = mode, ["cubic_coeff_a"] = a, ["exclude_outside"] = (long)exclude },
                Pool = pool,
                InputNames = new[] { "x" },
            });
            await accelerator.SynchronizeAsync();
            var got = await outBuf.CopyToHostAsync<float>(0, outH * outW);
            double worst = want.Zip(got).Max(z => Math.Abs(z.First - z.Second));
            if (!(worst <= 1e-4)) throw new Exception($"Resize cubic {mode} a={a} exclude_outside={exclude}: worst |d| {worst:G4} to onnxruntime; got [{string.Join(", ", got.Take(6).Select(v => v.ToString("G5")))}] want [{string.Join(", ", want.Take(6).Select(v => v.ToString("G5")))}]");
            notes.Add($"{mode}/{a}/{exclude}: {worst:G2}");
        }
        return string.Join("; ", notes);
    });

    [TestMethod]
    public async Task Resize_LinearMode_IsBilinear() => await RunTest(async accelerator =>
    {
        // mode="linear" must still bilinear-interpolate (the non-default path stays correct).
        const int C = 1, inH = 2, inW = 2, outH = 4, outW = 4;
        var x = new float[] { 0f, 10f, 20f, 30f };

        using var inBuf = accelerator.Allocate1D(x);
        using var outBuf = accelerator.Allocate1D<float>(C * outH * outW);
        using var registry = new OperatorRegistry(accelerator);
        var op = registry.Resolve("Resize");
        var pool = new BufferPool(accelerator);
        op.Execute(new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 1, C, inH, inW }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 1, C, outH, outW }) },
            Attributes = new Dictionary<string, object> { ["mode"] = "linear" },
            Pool = pool,
            InputNames = new[] { "x" },
        });
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, C * outH * outW);

        // Linear must produce at least one interior value strictly between the input corners (smoothing) —
        // i.e. NOT the pure nearest set {0,10,20,30}. Proves the linear path is still wired.
        bool anyInterpolated = got.Any(v => v > 0.01f && v < 9.99f) || got.Any(v => v > 10.01f && v < 19.99f);
        if (!anyInterpolated)
            throw new Exception("Resize mode=\"linear\" produced no interpolated values — bilinear path broken.");
        Console.WriteLine($"[Resize] linear mode still bilinear-interpolates ({BackendName})");
    });
}
