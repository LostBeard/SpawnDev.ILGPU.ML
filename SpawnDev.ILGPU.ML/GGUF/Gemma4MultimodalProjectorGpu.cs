using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Kernels;

namespace SpawnDev.ILGPU.ML.Multimodal;

/// <summary>
/// GPU (Rule 4 zero-copy) port of the gemma4 "Unified" multimodal projector — the device-side twin of the
/// CPU reference <see cref="Gemma4MultimodalProjector"/>. Turns preprocessed image patches and raw audio
/// frames into LLM-embedding-space vectors [N, EmbedDim] entirely on the accelerator, composed from the
/// existing, individually-verified ML kernels (<see cref="LayerNormKernel"/> double-Welford,
/// <see cref="FusedScaledMatMulKernel"/> as the ggml-layout linear, <see cref="NormalizationKernels"/>
/// weightless RMSNorm, <see cref="ElementWiseKernels.AddBias"/>) plus one small factorized 2D
/// position-embedding add kernel. Bit-for-bit close to the CPU reference (verified by
/// <c>Gemma4ProjectorGpu_*_MatchesCpu</c> PMT tests); the few-hundred media rows are small next to the 7 GB
/// decoder, but keeping them on the GPU removes the per-projection CPU matmul over the 6912→3840 and
/// 3840→3840 weights and is the foundation for a fully GPU-resident media splice.
///
/// Weights are uploaded once at construction and reused for every projection. The output of
/// <see cref="EncodeImageToBufferAsync"/> / <see cref="EncodeAudioToBufferAsync"/> is a GPU buffer the caller
/// owns and disposes (the zero-copy primitive); the <c>*Async</c> host-returning helpers exist for the
/// current host-side splice path and the equivalence tests.
///
/// Forward spec + verbatim llama.cpp citations: Plans/gemma4-multimodal-bringup.md and
/// Plans/gemma4-llamacpp-reference-snippets.md. Math (and the gemma4 quirks it encodes) is identical to the
/// CPU reference — see that class for the per-step commentary.
/// </summary>
public sealed class Gemma4MultimodalProjectorGpu : IDisposable
{
    // gemma4 graph epsilons (verbatim): patch LayerNorms use PyTorch default 1e-5; the weightless
    // pre-projection RMSNorm uses hparams.eps = 1e-6. Identical to the CPU reference.
    private const float LayerNormEps = 1e-5f;
    private const float RmsNormEps = 1e-6f;

    private readonly Accelerator _accel;
    private readonly LayerNormKernel _layerNorm;
    private readonly FusedScaledMatMulKernel _linear;   // ggml-layout linear: out[r,o] = Σ_i x[r,i]·W[o,i] (scale=1)
    private readonly ElementWiseKernels _elem;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int>? _addPosKernel;

    // Weightless RMSNorm (two-pass, double-precision sum-of-squares — identical math to the CPU reference and
    // NormalizationKernels). Own the per-row invRms scratch and REUSE it (resize on growth) instead of the
    // append-forever temp-buffer list NormalizationKernels keeps — that would leak one small buffer per image.
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, float>? _rmsStatsKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>? _rmsApplyKernel;
    private MemoryBuffer1D<float, Stride1D.Dense>? _invRms;

    // Uploaded weights (owned, disposed in Dispose).
    private readonly MemoryBuffer1D<float, Stride1D.Dense>? _patchEmbdW, _patchEmbdB;
    private readonly MemoryBuffer1D<float, Stride1D.Dense>? _n1w, _n1b, _n2w, _n2b, _n3w, _n3b;
    private readonly MemoryBuffer1D<float, Stride1D.Dense>? _posEmbd, _mmW;
    private readonly MemoryBuffer1D<float, Stride1D.Dense>? _mmAW;

    /// <summary>Embedding dim of the LLM (projection output), 3840 for gemma4 12B.</summary>
    public int EmbedDim { get; }
    /// <summary>Flattened patch length the vision path consumes (6912 = 48*48*3).</summary>
    public int PatchLen { get; }
    /// <summary>Raw audio frame length (640 samples).</summary>
    public int AudioFrameLen { get; }
    /// <summary>Position-embedding table length (1120) — caps the resized grid to this many patches per axis.</summary>
    public int PosTableLen { get; }

    /// <summary>True if the vision projector (gemma4uv) weights are loaded.</summary>
    public bool SupportsVision => _patchEmbdW != null;
    /// <summary>True if the audio projector (gemma4ua) weights are loaded.</summary>
    public bool SupportsAudio => _mmAW != null;

    public Gemma4MultimodalProjectorGpu(Accelerator accel, MmprojModel mm)
        : this(accel, Gemma4ProjectorWeights.FromMmproj(mm)) { }

    public Gemma4MultimodalProjectorGpu(Accelerator accel, Gemma4ProjectorWeights w)
    {
        _accel = accel;
        EmbedDim = w.EmbedDim;
        PatchLen = w.PatchLen;
        AudioFrameLen = w.AudioFrameLen;
        PosTableLen = w.PosTableLen;

        _layerNorm = new LayerNormKernel(accel);
        _linear = new FusedScaledMatMulKernel(accel);
        _elem = new ElementWiseKernels(accel);

        MemoryBuffer1D<float, Stride1D.Dense>? Up(float[]? a) => a == null ? null : accel.Allocate1D(a);
        if (w.HasVision)
        {
            _patchEmbdW = Up(w.PatchEmbdW);
            _patchEmbdB = Up(w.PatchEmbdB);
            _n1w = Up(w.PatchNorm1W); _n1b = Up(w.PatchNorm1B);
            _n2w = Up(w.PatchNorm2W); _n2b = Up(w.PatchNorm2B);
            _n3w = Up(w.PatchNorm3W); _n3b = Up(w.PatchNorm3B);
            _posEmbd = Up(w.PosEmbd);
            _mmW = Up(w.MmInputProjW);
        }
        if (w.HasAudio)
            _mmAW = Up(w.MmAInputProjW);
    }

    /// <summary>
    /// Vision forward on the GPU. Input <paramref name="patchesGpu"/> = [nPatches, PatchLen] preprocessed
    /// patch vectors (channel-outermost row-major, /255). Returns a [nPatches, EmbedDim] GPU buffer (caller
    /// owns + disposes) of LLM-space embeddings, to be spliced RAW (no sqrt(n_embd) scale). All intermediate
    /// buffers are disposed after a single drain so this is safe on the command-batched browser backends.
    /// </summary>
    public async Task<MemoryBuffer1D<float, Stride1D.Dense>> EncodeImageToBufferAsync(
        ArrayView1D<float, Stride1D.Dense> patchesGpu, int nPatches, int nCols, int nRows)
    {
        if (_patchEmbdW == null) throw new InvalidOperationException("mmproj has no vision encoder.");
        if (patchesGpu.Length != (long)nPatches * PatchLen)
            throw new ArgumentException($"patches length {patchesGpu.Length} != nPatches({nPatches})*PatchLen({PatchLen}).");
        if (nCols > PosTableLen || nRows > PosTableLen)
            throw new ArgumentException($"grid {nCols}x{nRows} exceeds position table length {PosTableLen}.");

        int D = EmbedDim;
        EnsurePosKernel();
        EnsureRmsKernels();

        // Intermediates: distinct buffers everywhere (no input==output aliasing — WebGPU forbids binding the
        // same buffer to two slots). All kept alive until the post-drain disposal below.
        var x1 = _accel.Allocate1D<float>((long)nPatches * PatchLen); // patch_norm.1 output
        var x2 = _accel.Allocate1D<float>((long)nPatches * D);        // patch_embd + bias
        var x3 = _accel.Allocate1D<float>((long)nPatches * D);        // patch_norm.2 (+ pos in place)
        var x5 = _accel.Allocate1D<float>((long)nPatches * D);        // patch_norm.3
        var x6 = _accel.Allocate1D<float>((long)nPatches * D);        // weightless RMSNorm
        var outp = _accel.Allocate1D<float>((long)nPatches * D);      // mm.input_projection

        // 1. patch_norm.1 — LayerNorm over the PatchLen patch dim.
        _layerNorm.Forward(patchesGpu, x1.View, _n1w!.View, _n1b!.View, nPatches, PatchLen, LayerNormEps);
        // 2. patch_embd matmul (PatchLen -> D) + bias.
        _linear.Forward(x1.View, _patchEmbdW.View, x2.View, nPatches, PatchLen, D, 1f);
        _elem.AddBias(x2.View, _patchEmbdB!.View, nPatches * D, D);
        // 3. patch_norm.2 — LayerNorm over D.
        _layerNorm.Forward(x2.View, x3.View, _n2w!.View, _n2b!.View, nPatches, D, LayerNormEps);
        // 4. + factorized 2D position embedding (in place on x3, one store per thread at its own index).
        _addPosKernel!(nPatches * D, x3.View, _posEmbd!.View, D, nCols, PosTableLen);
        // 5. patch_norm.3 — LayerNorm over D (post-position).
        _layerNorm.Forward(x3.View, x5.View, _n3w!.View, _n3b!.View, nPatches, D, LayerNormEps);
        // 6. weightless RMSNorm (embedding_pre_projection_norm).
        RmsNormWeightless(x5.View, x6.View, nPatches, D, RmsNormEps);
        // 7. mm.input_projection (D -> D), no bias/activation.
        _linear.Forward(x6.View, _mmW!.View, outp.View, nPatches, D, D, 1f);

        await _accel.SynchronizeAsync(); // drain before disposing intermediates (browser-safe)
        x1.Dispose(); x2.Dispose(); x3.Dispose(); x5.Dispose(); x6.Dispose();
        return outp;
    }

    /// <summary>
    /// Audio forward on the GPU. Input <paramref name="framesGpu"/> = [nFrames, AudioFrameLen] raw-waveform
    /// frames. Returns a [nFrames, EmbedDim] GPU buffer (caller owns + disposes). Graph:
    /// weightless RMSNorm(AudioFrameLen) -> mm.a.input_projection(AudioFrameLen -> EmbedDim).
    /// </summary>
    public async Task<MemoryBuffer1D<float, Stride1D.Dense>> EncodeAudioToBufferAsync(
        ArrayView1D<float, Stride1D.Dense> framesGpu, int nFrames)
    {
        if (_mmAW == null) throw new InvalidOperationException("mmproj has no audio encoder.");
        if (framesGpu.Length != (long)nFrames * AudioFrameLen)
            throw new ArgumentException($"frames length {framesGpu.Length} != nFrames({nFrames})*AudioFrameLen({AudioFrameLen}).");

        EnsureRmsKernels();
        var aRms = _accel.Allocate1D<float>((long)nFrames * AudioFrameLen);
        var outp = _accel.Allocate1D<float>((long)nFrames * EmbedDim);

        RmsNormWeightless(framesGpu, aRms.View, nFrames, AudioFrameLen, RmsNormEps);
        _linear.Forward(aRms.View, _mmAW.View, outp.View, nFrames, AudioFrameLen, EmbedDim, 1f);

        await _accel.SynchronizeAsync();
        aRms.Dispose();
        return outp;
    }

    /// <summary>Host-returning vision forward (uploads patches, runs on GPU, reads back). Matches the CPU
    /// reference's <see cref="Gemma4MultimodalProjector.EncodeImage"/> signature for the current host splice.</summary>
    public async Task<float[]> EncodeImageAsync(float[] patches, int nPatches, int nCols, int nRows)
    {
        using var pin = _accel.Allocate1D(patches);
        using var buf = await EncodeImageToBufferAsync(pin.View, nPatches, nCols, nRows);
        return await buf.CopyToHostAsync<float>(0, (long)nPatches * EmbedDim);
    }

    /// <summary>Host-returning audio forward. Matches <see cref="Gemma4MultimodalProjector.EncodeAudio"/>.</summary>
    public async Task<float[]> EncodeAudioAsync(float[] frames, int nFrames)
    {
        using var pin = _accel.Allocate1D(frames);
        using var buf = await EncodeAudioToBufferAsync(pin.View, nFrames);
        return await buf.CopyToHostAsync<float>(0, (long)nFrames * EmbedDim);
    }

    private void EnsurePosKernel()
    {
        _addPosKernel ??= _accel.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int>(AddPositionEmbeddingImpl);
    }

    private void EnsureRmsKernels()
    {
        _rmsStatsKernel ??= _accel.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, float>(RmsNormStatsImpl);
        _rmsApplyKernel ??= _accel.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int>(RmsNormApplyImpl);
    }

    /// <summary>Weightless RMSNorm: out[r,c] = in[r,c] / sqrt(mean(in[r,:]^2) + eps). Two-pass (WebGL-TF safe:
    /// each thread writes one output at its own index), double-precision sum-of-squares. Reuses the per-row
    /// invRms scratch (resize on growth). Caller must keep <paramref name="input"/> != <paramref name="output"/>.</summary>
    private void RmsNormWeightless(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output, int rows, int C, float eps)
    {
        if (_invRms == null || _invRms.Length < rows)
        {
            _invRms?.Dispose();
            _invRms = _accel.Allocate1D<float>(rows);
        }
        _rmsStatsKernel!(rows, input, _invRms.View, C, eps);
        _rmsApplyKernel!(rows * C, input, output, _invRms.View, C);
    }

    private static void RmsNormStatsImpl(Index1D row,
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> invRms, int C, float eps)
    {
        int offset = row * C;
        double sumSq = 0.0;
        for (int i = 0; i < C; i++) { double v = input[offset + i]; sumSq += v * v; }
        invRms[row] = 1f / MathF.Sqrt((float)(sumSq / C) + eps);
    }

    private static void RmsNormApplyImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> invRms, int C)
    {
        int row = idx / C;
        output[idx] = input[idx] * invRms[row];
    }

    /// <summary>Add the factorized 2D position embedding in place: row p (grid (p%nCols, p/nCols)) gets
    /// tbl_x[:, p%nCols] + tbl_y[:, p/nCols]. posEmbd flat = ne0=D (contiguous), ne1=PosTableLen positions,
    /// ne2=2 axis (0=x at base 0, 1=y at base D*PosTableLen). One thread per element → write index == thread
    /// index (WebGL Transform-Feedback safe). Mirrors CPU AddPositionEmbedding exactly.</summary>
    private static void AddPositionEmbeddingImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> x,
        ArrayView1D<float, Stride1D.Dense> posEmbd,
        int D, int nCols, int posTableLen)
    {
        int p = idx / D;
        int e = idx % D;
        int px = p % nCols;
        int py = p / nCols;
        int yAxisBase = D * posTableLen;
        x[idx] += posEmbd[px * D + e] + posEmbd[yAxisBase + py * D + e];
    }

    public void Dispose()
    {
        _patchEmbdW?.Dispose(); _patchEmbdB?.Dispose();
        _n1w?.Dispose(); _n1b?.Dispose(); _n2w?.Dispose(); _n2b?.Dispose(); _n3w?.Dispose(); _n3b?.Dispose();
        _posEmbd?.Dispose(); _mmW?.Dispose(); _mmAW?.Dispose();
        _invRms?.Dispose();
        _layerNorm.Dispose();
        (_elem as IDisposable)?.Dispose();
    }
}
