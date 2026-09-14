using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// 1-D transposed convolution against a CPU oracle - the upsampler every neural vocoder is built from.
/// </summary>
/// <remarks>
/// <para>
/// 🔴 THE KERNEL IS NEW, SO "IT RAN" PROVES NOTHING. A transposed convolution that indexes its weight
/// wrongly still produces a full, plausible, correctly-shaped buffer - and in a vocoder that is audio
/// which sounds like a bad recording rather than like a bug. The ONLY thing that separates the two is a
/// CPU reference, so every case here compares against one.
/// </para>
/// <para>
/// ⚠️ THE WEIGHT LAYOUT IS THE TRAP. ONNX ConvTranspose weight is <c>[inC, outC/groups, kL]</c> - the
/// INPUT channel leads - which is the opposite of Conv's <c>[outC, inC/groups, kL]</c>. With square
/// channel counts the two index the same number of elements and differ only in WHICH, so a transposition
/// is invisible to any shape check and to any test that uses inC == outC. The asymmetric case below is
/// there to make it visible.
/// </para>
/// </remarks>
public abstract partial class MLTestBase
{
    /// <summary>Reference 1-D transposed convolution. Deliberately the naive scatter form.</summary>
    /// <remarks>
    /// ⚠️ Written as a SCATTER while the kernel is a GATHER, on purpose. An oracle that shares the
    /// kernel's structure shares its mistakes - transcribing the same index arithmetic into both is how a
    /// test agrees with a bug. Accumulating from the input side is the textbook definition and derives the
    /// mapping independently.
    /// </remarks>
    private static float[] CpuConvTranspose1D(float[] input, float[] weight, float[] bias,
        int batch, int inC, int inL, int outC, int kL,
        int stride, int padBegin, int padEnd, int dilation, int groups)
    {
        var outL = ConvTranspose1DKernel.OutputLength(inL, kL, stride, padBegin, padEnd, dilation);
        var outCPerGroup = outC / groups;
        var inCPerGroup = inC / groups;
        var output = new float[batch * outC * outL];

        for (var n = 0; n < batch; n++)
            for (var ic = 0; ic < inC; ic++)
            {
                var group = ic / inCPerGroup;
                for (var ix = 0; ix < inL; ix++)
                {
                    var v = input[n * inC * inL + ic * inL + ix];
                    for (var ocLocal = 0; ocLocal < outCPerGroup; ocLocal++)
                        for (var kx = 0; kx < kL; kx++)
                        {
                            var ox = ix * stride + kx * dilation - padBegin;
                            if (ox < 0 || ox >= outL) continue;
                            var oc = group * outCPerGroup + ocLocal;
                            output[n * outC * outL + oc * outL + ox] +=
                                v * weight[ic * outCPerGroup * kL + ocLocal * kL + kx];
                        }
                }
            }

        if (bias.Length > 0)
            for (var n = 0; n < batch; n++)
                for (var oc = 0; oc < outC; oc++)
                    for (var ox = 0; ox < outL; ox++)
                        output[n * outC * outL + oc * outL + ox] += bias[oc];
        return output;
    }

    /// <summary>Stride 1, the simplest case - proves the taps and the weight layout.</summary>
    [TestMethod]
    public async Task ConvTranspose1D_Stride1MatchesCpu() => await RunTest(async accelerator =>
    {
        // ⚠️ inC != outC deliberately. Equal channel counts hide a transposed weight index.
        int batch = 1, inC = 6, inL = 9, outC = 4, kL = 3;
        var input = RandomFloats(batch * inC * inL, seed: 4101, scale: 0.7f);
        var weight = RandomFloats(inC * outC * kL, seed: 4102, scale: 0.3f);
        var bias = RandomFloats(outC, seed: 4103, scale: 0.05f);
        var expected = CpuConvTranspose1D(input, weight, bias, batch, inC, inL, outC, kL, 1, 0, 0, 1, 1);

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(expected.Length);

        using var k = new ConvTranspose1DKernel(accelerator);
        k.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, batch, inC, inL, outC, kL, 1, 0, 0, 1, 1);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View, expected, inC * 2e-5f, "ConvTranspose1D stride1: ");
    });

    /// <summary>Stride 2 with padding - the shape a vocoder actually upsamples with.</summary>
    /// <remarks>
    /// 🔴 STRIDE IS WHERE A TRANSPOSED CONVOLUTION IS DIFFERENT, and where a gather implementation can be
    /// subtly wrong: only the taps whose offset divides evenly by the stride contribute to a given output,
    /// and getting that test wrong produces output that is smooth and plausible instead of correct. A
    /// stride-1 test cannot catch it, because at stride 1 every tap qualifies.
    /// </remarks>
    [TestMethod]
    public async Task ConvTranspose1D_Stride2WithPaddingMatchesCpu() => await RunTest(async accelerator =>
    {
        int batch = 1, inC = 5, inL = 11, outC = 3, kL = 4, stride = 2, padBegin = 1, padEnd = 1;
        var input = RandomFloats(batch * inC * inL, seed: 4111, scale: 0.6f);
        var weight = RandomFloats(inC * outC * kL, seed: 4112, scale: 0.25f);
        var bias = RandomFloats(outC, seed: 4113, scale: 0.05f);
        var expected = CpuConvTranspose1D(input, weight, bias, batch, inC, inL, outC, kL,
            stride, padBegin, padEnd, 1, 1);

        // The length formula is part of what is under test - a wrong one silently truncates the tail.
        var outL = ConvTranspose1DKernel.OutputLength(inL, kL, stride, padBegin, padEnd);
        if (expected.Length != batch * outC * outL)
            throw new Exception($"the oracle and the length formula disagree: {expected.Length} vs "
                + $"{batch * outC * outL}");

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(expected.Length);

        using var k = new ConvTranspose1DKernel(accelerator);
        k.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, batch, inC, inL, outC, kL,
            stride, padBegin, padEnd, 1, 1);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View, expected, inC * 2e-5f, "ConvTranspose1D stride2: ");
    });

    /// <summary>Grouped, batched, and dilated at once - the combination nothing else covers.</summary>
    /// <remarks>
    /// ⚠️ Grouping is where the weight layout bites hardest: outC is <c>w[1] * groups</c>, not
    /// <c>w[1]</c>, and the two agree exactly when groups == 1 - so a grouped case is the only thing that
    /// can catch that. Batch is here for the same reason: N == 1 makes the per-sample offset vanish.
    /// </remarks>
    [TestMethod]
    public async Task ConvTranspose1D_GroupedBatchedDilatedMatchesCpu() => await RunTest(async accelerator =>
    {
        int batch = 2, groups = 2, inC = 6, inL = 7, outC = 4, kL = 3, stride = 2, dilation = 2;
        var input = RandomFloats(batch * inC * inL, seed: 4121, scale: 0.5f);
        var weight = RandomFloats(inC * (outC / groups) * kL, seed: 4122, scale: 0.3f);
        var bias = RandomFloats(outC, seed: 4123, scale: 0.05f);
        var expected = CpuConvTranspose1D(input, weight, bias, batch, inC, inL, outC, kL,
            stride, 0, 0, dilation, groups);

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(expected.Length);

        using var k = new ConvTranspose1DKernel(accelerator);
        k.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, batch, inC, inL, outC, kL,
            stride, 0, 0, dilation, groups);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View, expected, inC * 2e-5f, "ConvTranspose1D grouped: ");
    });

    /// <summary>Without a bias the result is the convolution alone, not the convolution plus garbage.</summary>
    /// <remarks>
    /// The operator rents a zeroed buffer when the ONNX node has no bias input. An unzeroed rental would
    /// add whatever the pool last held - a per-run offset that looks like noise in the audio and would
    /// change between runs, which is the hardest kind of defect to pin.
    /// </remarks>
    [TestMethod]
    public async Task ConvTranspose1D_NoBiasAddsNothing() => await RunTest(async accelerator =>
    {
        int batch = 1, inC = 3, inL = 5, outC = 2, kL = 3;
        var input = RandomFloats(batch * inC * inL, seed: 4131, scale: 0.8f);
        var weight = RandomFloats(inC * outC * kL, seed: 4132, scale: 0.4f);
        var expected = CpuConvTranspose1D(input, weight, Array.Empty<float>(),
            batch, inC, inL, outC, kL, 1, 0, 0, 1, 1);

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D<float>(1);   // an EMPTY-length view is what the kernel tests
        using var outBuf = accelerator.Allocate1D<float>(expected.Length);

        using var k = new ConvTranspose1DKernel(accelerator);
        k.Forward(inBuf.View, wBuf.View, bBuf.View.SubView(0, 0), outBuf.View,
            batch, inC, inL, outC, kL, 1, 0, 0, 1, 1);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View, expected, inC * 2e-5f, "ConvTranspose1D no bias: ");
    });
}
