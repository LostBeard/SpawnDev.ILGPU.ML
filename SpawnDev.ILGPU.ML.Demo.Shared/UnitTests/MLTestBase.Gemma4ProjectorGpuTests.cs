using SpawnDev.ILGPU.ML.Multimodal;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Equivalence tests for the gemma4 multimodal projector GPU port (<see cref="Gemma4MultimodalProjectorGpu"/>)
/// against the verified CPU reference (<see cref="Gemma4MultimodalProjector"/>). Both are built from the SAME
/// synthetic <see cref="Gemma4ProjectorWeights"/> (no GGUF file needed — backend-portable, runs in the browser
/// lanes too), then fed identical inputs; the device-side forward must match the host forward within float
/// tolerance. This is the Rule-1 minimum bar (CPU reference comparison) for the Rule-4 zero-copy port: it
/// locks the LayerNorm / ggml-layout linear / AddBias / factorized-pos-embd / weightless-RMSNorm composition.
///
/// Dims are small (D=128, PatchLen=147, PosTable=8) so every backend (incl. the slow Wasm/WebGL lanes) runs
/// quickly; the math paths are identical to the real 3840/6912 model — only the sizes differ.
/// </summary>
public abstract partial class MLTestBase
{
    private const int GpuProjD = 128;        // EmbedDim
    private const int GpuProjPatchLen = 147; // 7*7*3
    private const int GpuProjPosTable = 8;
    private const int GpuProjAudioLen = 64;

    /// <summary>Random projector weights at the small test dims (vision + audio both present).</summary>
    private static Gemma4ProjectorWeights MakeSyntheticWeights(int seed)
    {
        var rng = new Random(seed);
        float[] Rand(int n, double scale) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * scale); return a; }
        // Norm gains near 1, biases near 0 (realistic LayerNorm params); projection weights small.
        float[] Gain(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 0.6 + 0.7); return a; }

        return new Gemma4ProjectorWeights
        {
            EmbedDim = GpuProjD,
            PatchLen = GpuProjPatchLen,
            AudioFrameLen = GpuProjAudioLen,
            PosTableLen = GpuProjPosTable,

            PatchEmbdW = Rand(GpuProjD * GpuProjPatchLen, 0.08),
            PatchEmbdB = Rand(GpuProjD, 0.05),
            PatchNorm1W = Gain(GpuProjPatchLen), PatchNorm1B = Rand(GpuProjPatchLen, 0.05),
            PatchNorm2W = Gain(GpuProjD), PatchNorm2B = Rand(GpuProjD, 0.05),
            PatchNorm3W = Gain(GpuProjD), PatchNorm3B = Rand(GpuProjD, 0.05),
            PosEmbd = Rand(GpuProjD * GpuProjPosTable * 2, 0.1),
            MmInputProjW = Rand(GpuProjD * GpuProjD, 0.06),

            MmAInputProjW = Rand(GpuProjD * GpuProjAudioLen, 0.06),
        };
    }

    [TestMethod]
    public async Task Gemma4ProjectorGpu_Image_MatchesCpu() => await RunTest(async accelerator =>
    {
        var weights = MakeSyntheticWeights(seed: 1337);
        const int nCols = 3, nRows = 2, nPatches = nCols * nRows;

        var rng = new Random(24);
        var patches = new float[nPatches * GpuProjPatchLen];
        for (int i = 0; i < patches.Length; i++) patches[i] = (float)rng.NextDouble(); // /255-style [0,1]

        var cpu = new Gemma4MultimodalProjector(weights);
        using var gpu = new Gemma4MultimodalProjectorGpu(accelerator, weights);

        if (!gpu.SupportsVision) throw new Exception("GPU projector reports no vision encoder for vision weights.");
        if (gpu.EmbedDim != GpuProjD || gpu.PatchLen != GpuProjPatchLen || gpu.PosTableLen != GpuProjPosTable)
            throw new Exception($"GPU projector dims wrong: D={gpu.EmbedDim} PatchLen={gpu.PatchLen} PosTable={gpu.PosTableLen}.");

        var expected = cpu.EncodeImage(patches, nPatches, nCols, nRows);
        var got = await gpu.EncodeImageAsync(patches, nPatches, nCols, nRows);

        if (got.Length != expected.Length) throw new Exception($"length {got.Length} != {expected.Length}");
        AssertCloseQuant(got, expected, 1e-3f, "Gemma4 GPU projector image");
        Console.WriteLine($"[Gemma4ProjGpu] image {nPatches}x{GpuProjPatchLen}->{GpuProjD} matches CPU reference ({BackendName})");
    });

    [TestMethod]
    public async Task Gemma4ProjectorGpu_Audio_MatchesCpu() => await RunTest(async accelerator =>
    {
        var weights = MakeSyntheticWeights(seed: 2024);
        const int nFrames = 5;

        var rng = new Random(99);
        var frames = new float[nFrames * GpuProjAudioLen];
        for (int i = 0; i < frames.Length; i++) frames[i] = (float)(rng.NextDouble() * 2 - 1); // waveform [-1,1]

        var cpu = new Gemma4MultimodalProjector(weights);
        using var gpu = new Gemma4MultimodalProjectorGpu(accelerator, weights);

        if (!gpu.SupportsAudio) throw new Exception("GPU projector reports no audio encoder for audio weights.");

        var expected = cpu.EncodeAudio(frames, nFrames);
        var got = await gpu.EncodeAudioAsync(frames, nFrames);

        if (got.Length != expected.Length) throw new Exception($"length {got.Length} != {expected.Length}");
        AssertCloseQuant(got, expected, 1e-3f, "Gemma4 GPU projector audio");
        Console.WriteLine($"[Gemma4ProjGpu] audio {nFrames}x{GpuProjAudioLen}->{GpuProjD} matches CPU reference ({BackendName})");
    });
}
