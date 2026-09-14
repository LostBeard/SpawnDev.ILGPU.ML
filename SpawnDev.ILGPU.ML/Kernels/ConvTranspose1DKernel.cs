using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// 1D transposed convolution ("deconvolution") for audio models.
/// Layout: [N, C, L]. Weight: [inC, outC/groups, kL] - ONNX's ConvTranspose layout.
/// </summary>
/// <remarks>
/// <para>
/// 🔴 WHY THIS EXISTS. <c>ConvTransposeOperator</c> refused anything that was not rank 4 -
/// "ConvTranspose expects 4D input [N,C,H,W]" - while <c>Conv</c> had had a 1-D path for ages. That gap
/// stops every neural vocoder in existence: HiFiGAN, and everything built on it, upsamples with stacked
/// ConvTranspose1d. Kokoro dies at node 1331 of 2323 without this.
/// </para>
/// <para>
/// ⭐ GATHER FORM, NOT SCATTER. The textbook transposed convolution scatters each input sample into
/// several outputs, which needs atomics to accumulate and makes the result order-dependent - on a GPU
/// that means non-deterministic float summation, which is exactly the class of defect that reads as "the
/// model is slightly wrong sometimes". One thread per OUTPUT element, gathering the inputs that reach it,
/// needs no atomics and is bit-reproducible. It is also the form the 2-D kernel here already uses.
/// </para>
/// <para>
/// The mapping is the inverse of a forward convolution: output position <c>ox</c> is written by input
/// position <c>ix</c> through tap <c>kx</c> exactly when <c>ox + padBegin - kx*dilation == ix*stride</c>.
/// So a thread walks the taps, keeps the ones where that difference divides evenly by the stride, and
/// ignores the rest - which is where a transposed convolution's characteristic zero-stuffing comes from.
/// </para>
/// <para>
/// ⚠️ The inner loop is FLATTENED over (ic, kx) for the same reason <c>Conv1DKernel</c> flattens its own:
/// the WGSL/GLSL backends miscompile triple-nested loops. See the WORKAROUND note there.
/// </para>
/// </remarks>
public class ConvTranspose1DKernel : IDisposable
{
    private readonly Accelerator _accelerator;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,   // input  [N * inC * inL]
        ArrayView1D<float, Stride1D.Dense>,   // weight [inC * outC/groups * kL]
        ArrayView1D<float, Stride1D.Dense>,   // bias   [outC] (or empty)
        ArrayView1D<float, Stride1D.Dense>,   // output [N * outC * outL]
        ArrayView1D<int, Stride1D.Dense>>?    // packed params
        _kernel;

    /// <summary>One buffer per distinct param set - every vocoder layer shares this kernel instance.</summary>
    private readonly ParamBufferCache<int> _paramsCache = new();

    public ConvTranspose1DKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Output length for a 1-D transposed convolution, per the ONNX formula.
    /// </summary>
    public static int OutputLength(int inL, int kL, int stride, int padBegin, int padEnd,
        int dilation = 1, int outputPadding = 0)
        => (inL - 1) * stride - padBegin - padEnd + dilation * (kL - 1) + 1 + outputPadding;

    /// <summary>
    /// Forward pass.
    /// </summary>
    /// <param name="input">[N, inC, inL].</param>
    /// <param name="weight">[inC, outC/groups, kL] - note the INPUT channel leads, unlike Conv.</param>
    /// <param name="bias">[outC], or an empty view.</param>
    /// <param name="output">[N, outC, outL].</param>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int batch, int inC, int inL, int outC, int kL,
        int stride = 1, int padBegin = 0, int padEnd = 0, int dilation = 1, int groups = 1,
        int outputPadding = 0)
    {
        var outL = OutputLength(inL, kL, stride, padBegin, padEnd, dilation, outputPadding);
        if (outL <= 0)
            throw new InvalidOperationException(
                $"ConvTranspose1D would produce a length of {outL} (inL={inL}, kL={kL}, stride={stride}, "
                + $"pads={padBegin}/{padEnd}, dilation={dilation}) - check the pads, which are the usual cause");

        var inCPerGroup = inC / groups;
        var outCPerGroup = outC / groups;
        var kernelLoopSize = inCPerGroup * kL;
        var totalOutput = batch * outC * outL;

        EnsureLoaded();

        var paramsData = new[]
        {
            inC, inL, outC, outL, kL, stride, padBegin, dilation,
            inCPerGroup, outCPerGroup, kernelLoopSize,
        };
        // Reuse when unchanged - see Conv1DKernel for why a fresh allocation per call makes CUDA graph
        // capture impossible (cuMemAlloc inside a capture window faults uncatchably).
        var paramsView = _paramsCache.Get(_accelerator, paramsData);

        _kernel!(totalOutput, input, weight, bias, output, paramsView);
    }

    /// <summary>One thread per output element, gathering every input that reaches it.</summary>
    private static void ConvTranspose1DFlatImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int inC = p[0]; int inL = p[1]; int outC = p[2]; int outL = p[3];
        int kL = p[4]; int stride = p[5]; int padBegin = p[6]; int dilation = p[7];
        int inCPerGroup = p[8]; int outCPerGroup = p[9]; int kernelLoopSize = p[10];

        int perSample = outC * outL;
        int n = idx / perSample;
        int rem = idx % perSample;
        int oc = rem / outL;
        int ox = rem % outL;

        int group = oc / outCPerGroup;
        int ocLocal = oc % outCPerGroup;
        int icStart = group * inCPerGroup;

        int inputBase = n * inC * inL;
        float sum = 0f;

        for (int i = 0; i < kernelLoopSize; i++)
        {
            int icLocal = i / kL;
            int kx = i % kL;

            // The inverse mapping: this tap contributes only when the stride divides evenly.
            int num = ox + padBegin - kx * dilation;
            if (num < 0) continue;
            if (num % stride != 0) continue;
            int ix = num / stride;
            if (ix >= inL) continue;

            int ic = icStart + icLocal;
            float inputVal = input[inputBase + ic * inL + ix];
            // ⚠️ WEIGHT IS [inC, outC/groups, kL] - the input channel leads. Conv's is [outC, inC/g, kL].
            // Transposing those two in the index is a silent, plausible-sounding wrongness.
            float weightVal = weight[ic * outCPerGroup * kL + ocLocal * kL + kx];
            sum += inputVal * weightVal;
        }

        if (bias.Length > 0) sum += bias[oc];
        output[idx] = sum;
    }

    private void EnsureLoaded()
    {
        _kernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(ConvTranspose1DFlatImpl);
    }

    public void Dispose() => _paramsCache.Dispose();
}
