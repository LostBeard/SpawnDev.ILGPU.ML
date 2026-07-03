using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Verify fused linear (MatMul + Bias + ReLU) matches separate operations.
    /// </summary>
    [TestMethod]
    public async Task FusedLinear_Relu_MatchesSeparate() => await RunTest(async accelerator =>
    {
        int M = 32, K = 64, N = 48;
        var input = RandomFloats(M * K, seed: 300);
        var weights = RandomFloats(K * N, seed: 301, scale: 0.1f);
        var bias = RandomFloats(N, seed: 302, scale: 0.5f);

        // CPU reference: MatMul + Bias + ReLU
        var expected = new float[M * N];
        for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
        {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += input[r * K + k] * weights[k * N + c];
            float val = sum + bias[c];
            expected[r * N + c] = val > 0 ? val : 0; // ReLU
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weights);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(M * N);

        var fused = new FusedLinearKernel(accelerator);
        fused.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, M, K, N, FusedActivation.ReLU);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, M * N), expected, 1e-3f, "FusedLinear+ReLU: ");
    });

    /// <summary>
    /// Verify fused linear with GELU matches separate operations.
    /// </summary>
    [TestMethod]
    public async Task FusedLinear_Gelu_MatchesSeparate() => await RunTest(async accelerator =>
    {
        int M = 16, K = 32, N = 24;
        var input = RandomFloats(M * K, seed: 310);
        var weights = RandomFloats(K * N, seed: 311, scale: 0.1f);
        var bias = RandomFloats(N, seed: 312, scale: 0.3f);

        // CPU reference: MatMul + Bias + GELU
        var expected = new float[M * N];
        for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
        {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += input[r * K + k] * weights[k * N + c];
            float x = sum + bias[c];
            // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            if (x > 10f) expected[r * N + c] = x;
            else if (x < -10f) expected[r * N + c] = 0f;
            else expected[r * N + c] = 0.5f * x * (1f + ErfApprox(x * 0.7071067811865475f));
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weights);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(M * N);

        var fused = new FusedLinearKernel(accelerator);
        fused.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, M, K, N, FusedActivation.GELU);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, M * N), expected, 1e-2f, "FusedLinear+GELU: ");
    });

    /// <summary>
    /// Verify fused linear with no activation matches plain MatMul + Bias.
    /// </summary>
    [TestMethod]
    public async Task FusedLinear_None_MatchesMatMulPlusBias() => await RunTest(async accelerator =>
    {
        int M = 8, K = 16, N = 12;
        var input = RandomFloats(M * K, seed: 320);
        var weights = RandomFloats(K * N, seed: 321, scale: 0.2f);
        var bias = RandomFloats(N, seed: 322);

        // CPU reference: MatMul + Bias
        var expected = CpuMatMul(input, weights, M, K, N);
        for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
            expected[r * N + c] += bias[c];

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weights);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(M * N);

        var fused = new FusedLinearKernel(accelerator);
        fused.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, M, K, N, FusedActivation.None);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, M * N), expected, 1e-3f, "FusedLinear+None: ");
    });

    // ── Register-blocked path (M,N >= 64) — the GPT-2 FFN regime (768→3072 / 3072→768) ──
    // The tests above use sub-64 dims and hit the per-element kernels; these force the
    // register-blocked fused GEMM on capable backends (CPU/WebGL fall back to per-element,
    // both compared to the same CPU reference). Dims are non-multiples of 64/16 to exercise
    // the partial-tile bounds guards.

    /// <summary>
    /// Fused linear, no activation, at register-blocked sizes (M,N >= 64) — matches MatMul + Bias.
    /// </summary>
    [TestMethod]
    public async Task FusedLinear_None_RegBlocked_LargeMatchesReference() => await RunTest(async accelerator =>
    {
        int M = 70, K = 130, N = 100; // >= 64, non-tile-multiple → register-blocked path + partial tiles
        var input = RandomFloats(M * K, seed: 330);
        var weights = RandomFloats(K * N, seed: 331, scale: 0.1f);
        var bias = RandomFloats(N, seed: 332, scale: 0.5f);

        var expected = CpuMatMul(input, weights, M, K, N);
        for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
            expected[r * N + c] += bias[c];

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weights);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(M * N);

        var fused = new FusedLinearKernel(accelerator);
        fused.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, M, K, N, FusedActivation.None);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, M * N), expected, 1e-3f, "FusedLinear+None (reg-blocked): ");
    });

    /// <summary>
    /// ALIGNED-SHAPE + ragged-M coverage of the register-blocked path: K%16==0 and N%64==0 (exact K-tiles,
    /// full N-tiles) with M=70 ragged rows - the DAv3 linear regime (K/N are 16/64-multiples, M = token
    /// count). The sibling tests above use K=130/N=100 on purpose (partial-tile coverage); this is the
    /// aligned twin, covering None + GELU write-backs vs the CPU reference. HISTORY (2026-07-03): a vec4
    /// F4/AsAligned16 tile-load variant of the reg-blocked kernel was built, gated on exactly these shapes,
    /// passed this test 62/62 - and measured NEUTRAL-TO-NEGATIVE in the DAv3 WebGPU frame (11.0 -> 11.9ms;
    /// the shared-tile staging is already fully coalesced, so load width is not this kernel's bottleneck,
    /// unlike the per-thread-streaming attention loop where vec4+hoist gave 3.7x). Reverted per measurement;
    /// don't re-add without a new attribution case.
    /// </summary>
    [TestMethod]
    public async Task FusedLinear_RegBlocked_AlignedShape_RaggedM_MatchesReference() => await RunTest(async accelerator =>
    {
        int M = 70, K = 128, N = 128; // K%16==0, N%64==0 (aligned tiles); M ragged
        var input = RandomFloats(M * K, seed: 430);
        var weights = RandomFloats(K * N, seed: 431, scale: 0.1f);
        var bias = RandomFloats(N, seed: 432, scale: 0.5f);

        var expectedNone = CpuMatMul(input, weights, M, K, N);
        for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
            expectedNone[r * N + c] += bias[c];

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weights);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(M * N);

        var fused = new FusedLinearKernel(accelerator);
        fused.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, M, K, N, FusedActivation.None);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, M * N), expectedNone, 1e-3f, "FusedLinear+None (reg-blocked, aligned tiles, ragged M): ");

        var expectedGelu = new float[M * N];
        for (int i = 0; i < M * N; i++)
        {
            float x = expectedNone[i];
            if (x > 10f) expectedGelu[i] = x;
            else if (x < -10f) expectedGelu[i] = 0f;
            else expectedGelu[i] = 0.5f * x * (1f + ErfApprox(x * 0.7071067811865475f));
        }
        fused.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, M, K, N, FusedActivation.GELU);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, M * N), expectedGelu, 1e-2f, "FusedLinear+GELU (reg-blocked, aligned tiles, ragged M): ");
    });

    /// <summary>
    /// REGISTER-BLOCKED FusedLinear with a NATIVE low-precision weight (fp16/bf16) at M,N >= 64 — the SD
    /// ResNet/FFN regime. ForwardLowP routes the large low-p linear to the 64×64-tile / 4×4-register kernel
    /// (weight decoded once on the shared-mem load, bias+activation fused in the write-back) instead of the
    /// per-element kernel. Covers None / ReLU / GELU / SiLU vs a fp32 reference using the same low-p-rounded
    /// weights, on all 6 backends (CPU/WebGL fall back to per-element via the same gate as the fp32 path).
    /// </summary>
    [TestMethod]
    public async Task FusedLinear_LowP_RegBlocked_LargeMatchesReference() => await RunTest(async accelerator =>
    {
        int M = 70, K = 130, N = 100; // >= 64, non-tile-multiple → register-blocked path + partial tiles
        var input = RandomFloats(M * K, seed: 350, scale: 1.5f);
        var w = RandomFloats(K * N, seed: 351, scale: 0.2f);
        var bias = RandomFloats(N, seed: 352, scale: 0.5f);
        var fused = new FusedLinearKernel(accelerator);

        async Task Check<T>(Func<float, T> toLowP, Func<T, float> toF32, FusedActivation act, string name)
            where T : unmanaged, System.Numerics.INumber<T>
        {
            var wLowP = new T[w.Length];
            for (int i = 0; i < w.Length; i++) wLowP[i] = toLowP(w[i]);
            var expected = new float[M * N];
            for (int r = 0; r < M; r++)
                for (int c = 0; c < N; c++)
                {
                    float s = 0f;
                    for (int k = 0; k < K; k++) s += input[r * K + k] * toF32(wLowP[k * N + c]);
                    float x = s + bias[c];
                    expected[r * N + c] = act switch
                    {
                        FusedActivation.ReLU => x > 0f ? x : 0f,
                        FusedActivation.GELU => x > 10f ? x : x < -10f ? 0f : 0.5f * x * (1f + ErfApprox(x * 0.7071067811865475f)),
                        FusedActivation.SiLU => x / (1f + MathF.Exp(-x)),
                        _ => x,
                    };
                }
            using var inBuf = accelerator.Allocate1D(input);
            using var wBuf = accelerator.Allocate1D(wLowP);
            using var bBuf = accelerator.Allocate1D(bias);
            using var outBuf = accelerator.Allocate1D<float>(M * N);
            fused.ForwardLowP(inBuf.View, wBuf.View, bBuf.View, outBuf.View, M, K, N, act); // M,N>=64 → reg-blocked low-p
            await accelerator.SynchronizeAsync();
            await AssertCloseGpu(accelerator, outBuf.View.SubView(0, M * N), expected, 2e-2f,
                $"FusedLinear+{act} low-p<{name}> (reg-blocked): ");
        }

        foreach (var act in new[] { FusedActivation.None, FusedActivation.ReLU, FusedActivation.GELU, FusedActivation.SiLU })
        {
            await Check<global::ILGPU.BFloat16>(f => (global::ILGPU.BFloat16)f, b => (float)b, act, "BFloat16");
            await Check<global::ILGPU.Half>(f => (global::ILGPU.Half)f, h => (float)h, act, "Half");
        }
    });

    /// <summary>
    /// Fused linear with SiLU at register-blocked sizes (M,N >= 64) — the SD ResNet/FFN path now has a
    /// register-blocked f32 variant (SiLU added to the RB gate + FusedActivate). Matches the per-element /
    /// x·sigmoid(x) reference.
    /// </summary>
    [TestMethod]
    public async Task FusedLinear_Silu_RegBlocked_LargeMatchesReference() => await RunTest(async accelerator =>
    {
        int M = 70, K = 130, N = 100; // >= 64, non-tile-multiple → register-blocked SiLU path
        var input = RandomFloats(M * K, seed: 360, scale: 1.5f);
        var weights = RandomFloats(K * N, seed: 361, scale: 0.2f);
        var bias = RandomFloats(N, seed: 362, scale: 0.5f);

        var expected = new float[M * N];
        for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
        {
            float sum = 0;
            for (int k = 0; k < K; k++) sum += input[r * K + k] * weights[k * N + c];
            float x = sum + bias[c];
            expected[r * N + c] = x / (1f + MathF.Exp(-x)); // SiLU
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weights);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(M * N);
        var fused = new FusedLinearKernel(accelerator);
        fused.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, M, K, N, FusedActivation.SiLU);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, M * N), expected, 1e-2f, "FusedLinear+SiLU (reg-blocked): ");
    });

    /// <summary>
    /// Fused linear with erf-GELU at register-blocked sizes (M,N >= 64) — the GPT-2 decoder FFN path.
    /// Guards that the register-blocked erf-GELU is bit-faithful to the per-element / ORT-matched form.
    /// </summary>
    [TestMethod]
    public async Task FusedLinear_Gelu_RegBlocked_LargeMatchesReference() => await RunTest(async accelerator =>
    {
        int M = 70, K = 130, N = 100; // >= 64, non-tile-multiple → register-blocked erf-GELU path
        var input = RandomFloats(M * K, seed: 340, scale: 2f);   // wider spread → some pre-activations in the GELU tails
        var weights = RandomFloats(K * N, seed: 341, scale: 0.2f);
        var bias = RandomFloats(N, seed: 342, scale: 0.5f);

        var expected = new float[M * N];
        for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
        {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += input[r * K + k] * weights[k * N + c];
            float x = sum + bias[c];
            if (x > 10f) expected[r * N + c] = x;
            else if (x < -10f) expected[r * N + c] = 0f;
            else expected[r * N + c] = 0.5f * x * (1f + ErfApprox(x * 0.7071067811865475f));
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weights);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(M * N);

        var fused = new FusedLinearKernel(accelerator);
        fused.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, M, K, N, FusedActivation.GELU);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, M * N), expected, 1e-2f, "FusedLinear+GELU (reg-blocked): ");
    });
}
