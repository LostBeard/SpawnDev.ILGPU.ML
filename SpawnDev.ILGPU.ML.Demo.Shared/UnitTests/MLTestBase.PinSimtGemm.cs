using SpawnDev.ILGPU.ML.Kernels;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Turns <see cref="FusedDequantMatMul.EnableMmaGemm"/> off until disposed: for tests of OTHER kernels or of path equivalence that the tensor-core GEMM
    /// (CUDA sm_75+, local.13+) would otherwise take over.
    /// </summary>
    /// <remarks>
    /// The tensor-core GEMM rounds activations and weights to f16 (about 5e-4 relative; llama.cpp's MMQ rounds activations to 8 bits): within the llama.cpp
    /// log-probability gate on real models, outside the tolerances of tests built for fp32 SIMT kernels. FOUND 2026-11-03 by the first full sweep after it
    /// (local.13 / local.14 shipped on focused gates): it took the M &gt; 1 rows of the RegBlocked / MultiRow / tiled-GEMM oracle tests (their kernels never
    /// ran), and the decode-vs-recompute tests compared an fp32 M = 1 GEMV with an f16 M &gt; 1 GEMM on synthetic weights with logits near 1e3 (an LFM2 argmax
    /// even flipped). Each of those tests is about something else (the named kernel, the KV / conv-state cache), so they pin the SIMT route; the tensor-core
    /// GEMM has its own oracle (Gemm_MmaKQuant_MatchesOracle_AndIsTaken) and the real-model llama.cpp gate.
    /// </remarks>
    protected static IDisposable PinSimtGemm() => new SimtGemmScope();

    private sealed class SimtGemmScope : IDisposable
    {
        private readonly bool _saved = FusedDequantMatMul.EnableMmaGemm;
        public SimtGemmScope() => FusedDequantMatMul.EnableMmaGemm = false;
        public void Dispose() => FusedDequantMatMul.EnableMmaGemm = _saved;
    }
}
