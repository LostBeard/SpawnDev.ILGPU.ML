using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    // These two kernels were one-thread-per-row writing every element of the row in a loop =
    // multi-store-per-thread, which silently corrupts on the WebGL Transform-Feedback path (only one
    // store per thread lands). The tests use rows with DISTINCT values so corruption can't pass by
    // coincidence, and assert the per-row invariant (unit norm / sums-to-1) on every backend.

    [TestMethod(Timeout = 30000)]
    public async Task PostProcess_L2NormalizeRows_MatchesCPU() => await RunTest(async accelerator =>
    {
        int rows = 3, dim = 4;
        // Row norms: 5, 3, 4 (distinct, so a per-row scale error shows).
        var data = new float[] { 3f, 4f, 0f, 0f, 1f, 2f, 2f, 0f, 2f, 2f, 2f, 2f };
        using var buf = accelerator.Allocate1D(data);

        var pp = new PostProcessingKernels(accelerator);
        pp.L2NormalizeRows(buf.View, rows, dim);
        await accelerator.SynchronizeAsync();
        var result = await buf.CopyToHostAsync<float>(0, rows * dim);

        for (int r = 0; r < rows; r++)
        {
            // CPU reference: original_value / original_norm.
            float origSumSq = 0f;
            for (int c = 0; c < dim; c++) { float v = data[r * dim + c]; origSumSq += v * v; }
            float origNorm = MathF.Sqrt(origSumSq);

            float gpuSumSq = 0f;
            for (int c = 0; c < dim; c++)
            {
                float got = result[r * dim + c];
                float expected = data[r * dim + c] / origNorm;
                gpuSumSq += got * got;
                if (MathF.Abs(got - expected) > 1e-3f)
                    throw new Exception($"L2NormalizeRows[{r},{c}]={got:F4}, expected {expected:F4}");
            }
            float gpuNorm = MathF.Sqrt(gpuSumSq);
            if (MathF.Abs(gpuNorm - 1f) > 1e-3f)
                throw new Exception($"L2NormalizeRows row {r} norm={gpuNorm:F4}, expected 1.0");
        }
    });

    [TestMethod(Timeout = 30000)]
    public async Task PostProcess_SoftmaxRows_MatchesCPU() => await RunTest(async accelerator =>
    {
        int rows = 2, rowSize = 3;
        var data = new float[] { 2f, 1f, 0.1f, 0.1f, 0.5f, 3f };
        using var buf = accelerator.Allocate1D(data);

        var pp = new PostProcessingKernels(accelerator);
        pp.SoftmaxRows(buf.View, rows, rowSize);
        await accelerator.SynchronizeAsync();
        var result = await buf.CopyToHostAsync<float>(0, rows * rowSize);

        for (int r = 0; r < rows; r++)
        {
            // CPU reference softmax.
            float max = float.NegativeInfinity;
            for (int i = 0; i < rowSize; i++)
            {
                float v = data[r * rowSize + i];
                if (v > max) max = v;
            }
            float sum = 0f;
            var exp = new float[rowSize];
            for (int i = 0; i < rowSize; i++) { exp[i] = MathF.Exp(data[r * rowSize + i] - max); sum += exp[i]; }

            float gpuSum = 0f;
            for (int i = 0; i < rowSize; i++)
            {
                float expected = exp[i] / sum;
                float got = result[r * rowSize + i];
                gpuSum += got;
                if (MathF.Abs(got - expected) > 1e-3f)
                    throw new Exception($"SoftmaxRows[{r},{i}]={got:F4}, expected {expected:F4}");
            }
            if (MathF.Abs(gpuSum - 1f) > 1e-3f)
                throw new Exception($"SoftmaxRows row {r} sum={gpuSum:F4}, expected 1.0");
        }
    });

    // NCHW<->NHWC were SCATTER kernels (wrote output[reindexed] = input[thread]); WebGL TF can only
    // write at the thread's own index, so they produced garbage there. Distinct values + a roundtrip
    // make any reindex error visible.
    [TestMethod(Timeout = 30000)]
    public async Task TensorLayout_NCHW_NHWC_Roundtrip_MatchesCPU() => await RunTest(async accelerator =>
    {
        int C = 3, H = 2, W = 2;
        int n = C * H * W;
        var nchw = new float[n];
        for (int i = 0; i < n; i++) nchw[i] = i + 1; // 1..12 distinct

        using var nchwBuf = accelerator.Allocate1D(nchw);
        using var nhwcBuf = accelerator.Allocate1D<float>(n);
        using var backBuf = accelerator.Allocate1D<float>(n);

        var tl = new TensorLayoutKernel(accelerator);
        tl.NCHWToNHWC(nchwBuf.View, nhwcBuf.View, C, H, W);
        tl.NHWCToNCHW(nhwcBuf.View, backBuf.View, C, H, W);
        await accelerator.SynchronizeAsync();
        var nhwc = await nhwcBuf.CopyToHostAsync<float>(0, n);
        var back = await backBuf.CopyToHostAsync<float>(0, n);

        for (int c = 0; c < C; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                {
                    float expected = nchw[c * H * W + y * W + x];
                    float got = nhwc[y * W * C + x * C + c];
                    if (got != expected)
                        throw new Exception($"NCHWToNHWC[c{c},y{y},x{x}]={got}, expected {expected}");
                }
        for (int i = 0; i < n; i++)
            if (back[i] != nchw[i])
                throw new Exception($"NCHW->NHWC->NCHW roundtrip[{i}]={back[i]}, expected {nchw[i]}");
    });
}
