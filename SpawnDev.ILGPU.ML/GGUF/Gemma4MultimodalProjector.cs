using SpawnDev.ILGPU.ML.GGUF;

namespace SpawnDev.ILGPU.ML.Multimodal;

/// <summary>
/// Gemma 4 12B "Unified" (encoder-free) multimodal projector. Turns preprocessed image patches and raw
/// audio frames into LLM-embedding-space vectors [N, 3840] that get spliced into the decoder's token
/// embedding stream. There is NO vision tower / audio encoder — just the lightweight linear layers held
/// in the mmproj GGUF (see <see cref="MmprojModel"/>).
///
/// This is the CORRECTNESS-FIRST reference forward (plain C#, runs on the few-hundred-row media tensors
/// where it is not a bottleneck — the 7 GB decoder dominates). A GPU/zero-copy port (Rule 4) follows once
/// the embeddings are verified bit-for-bit against the llama.cpp mtmd oracle.
///
/// Forward spec + verbatim llama.cpp citations: Plans/gemma4-multimodal-bringup.md and
/// Plans/gemma4-llamacpp-reference-snippets.md.
/// </summary>
public sealed class Gemma4MultimodalProjector
{
    // gemma4 graph epsilons (verbatim from llama.cpp): patch LayerNorms use PyTorch default 1e-5;
    // the weightless pre-projection RMSNorms use hparams.eps = 1e-6.
    private const float LayerNormEps = 1e-5f;
    private const float RmsNormEps = 1e-6f;

    private readonly MmprojModel _mm;

    // Vision weights (raw ggml layout; the matmul indexes W[o,i] = data[o*in + i] directly, no transpose).
    private readonly float[]? _patchEmbdW;   // [out=3840, in=6912]
    private readonly float[]? _patchEmbdB;   // [3840]
    private readonly float[]? _patchNorm1W, _patchNorm1B; // [6912]
    private readonly float[]? _patchNorm2W, _patchNorm2B; // [3840]
    private readonly float[]? _patchNorm3W, _patchNorm3B; // [3840]
    private readonly float[]? _posEmbd;      // [3840*1120*2] flat (ne0=3840, ne1=1120 pos, ne2=2 axis)
    private readonly float[]? _mmInputProjW; // [out=3840, in=3840]

    // Audio weights.
    private readonly float[]? _mmAInputProjW; // [out=3840, in=640]

    /// <summary>True if the mmproj carries the vision projector (gemma4uv).</summary>
    public bool SupportsVision => _patchEmbdW != null;
    /// <summary>True if the mmproj carries the audio projector (gemma4ua).</summary>
    public bool SupportsAudio => _mmAInputProjW != null;

    /// <summary>Embedding dim of the LLM (projection output), 3840 for gemma4 12B.</summary>
    public int EmbedDim { get; }
    /// <summary>Flattened patch length the vision path consumes (6912 = 48*48*3).</summary>
    public int PatchLen { get; }
    /// <summary>Raw audio frame length (640 samples).</summary>
    public int AudioFrameLen { get; }
    /// <summary>Position-embedding table length (1120) — caps the resized grid to 1120 patches per axis.</summary>
    public int PosTableLen { get; }

    public Gemma4MultimodalProjector(MmprojModel mm)
    {
        _mm = mm;
        EmbedDim = mm.VisionProjectionDim;        // 3840
        PatchLen = mm.GetTensorShape("v.patch_embd.weight")?[0] ?? 6912; // ne0 = in = 6912
        AudioFrameLen = mm.AudioFrameLength;       // 640
        PosTableLen = mm.GetTensorShape("v.position_embd.weight")?[1] ?? 1120; // ne1 = positions

        if (mm.HasVisionEncoder)
        {
            _patchEmbdW = mm.GetTensorF32("v.patch_embd.weight");
            _patchEmbdB = mm.GetTensorF32("v.patch_embd.bias");
            _patchNorm1W = mm.GetTensorF32("v.patch_norm.1.weight");
            _patchNorm1B = mm.GetTensorF32("v.patch_norm.1.bias");
            _patchNorm2W = mm.GetTensorF32("v.patch_norm.2.weight");
            _patchNorm2B = mm.GetTensorF32("v.patch_norm.2.bias");
            _patchNorm3W = mm.GetTensorF32("v.patch_norm.3.weight");
            _patchNorm3B = mm.GetTensorF32("v.patch_norm.3.bias");
            _posEmbd = mm.GetTensorF32("v.position_embd.weight");
            _mmInputProjW = mm.GetTensorF32("mm.input_projection.weight");
        }
        if (mm.HasAudioEncoder)
            _mmAInputProjW = mm.GetTensorF32("mm.a.input_projection.weight");
    }

    /// <summary>
    /// Vision forward. Input: <paramref name="patches"/> = [nPatches, 6912] preprocessed patch vectors
    /// (channel-outermost row-major, /255 normalized; produced by the image preprocessor). nCols/nRows are
    /// the 48px-grid dimensions (nCols = resizedWidth/48), so patch p sits at grid (p%nCols, p/nCols).
    /// Output: [nPatches, 3840] LLM-space embeddings, to be spliced RAW (no sqrt(n_embd) scale).
    /// </summary>
    public float[] EncodeImage(float[] patches, int nPatches, int nCols, int nRows)
    {
        if (_patchEmbdW == null) throw new InvalidOperationException("mmproj has no vision encoder.");
        if (patches.Length != (long)nPatches * PatchLen)
            throw new ArgumentException($"patches length {patches.Length} != nPatches({nPatches})*PatchLen({PatchLen}).");
        if (nCols > PosTableLen || nRows > PosTableLen)
            throw new ArgumentException($"grid {nCols}x{nRows} exceeds position table length {PosTableLen}.");

        int D = EmbedDim;
        // 1. patch_norm.1 — LayerNorm over the 6912 patch dim.
        var x1 = LayerNorm(patches, nPatches, PatchLen, _patchNorm1W!, _patchNorm1B!, LayerNormEps);
        // 2. patch_embd matmul (6912 -> 3840) + bias.
        var x2 = MatMulW(x1, nPatches, PatchLen, D, _patchEmbdW!);
        AddBias(x2, nPatches, D, _patchEmbdB!);
        // 3. patch_norm.2 — LayerNorm over 3840.
        var x3 = LayerNorm(x2, nPatches, D, _patchNorm2W!, _patchNorm2B!, LayerNormEps);
        // 4. + factorized 2D position embedding (tbl_x[p%nCols] + tbl_y[p/nCols]).
        AddPositionEmbedding(x3, nPatches, D, nCols);
        // 5. patch_norm.3 — LayerNorm over 3840 (post-position).
        var x5 = LayerNorm(x3, nPatches, D, _patchNorm3W!, _patchNorm3B!, LayerNormEps);
        // 6. weightless RMSNorm (embedding_pre_projection_norm).
        RmsNormInPlace(x5, nPatches, D, RmsNormEps);
        // 7. mm.input_projection (3840 -> 3840), no bias/activation.
        return MatMulW(x5, nPatches, D, D, _mmInputProjW!);
    }

    /// <summary>
    /// Audio forward. Input: <paramref name="frames"/> = [nFrames, 640] raw-waveform frames (16 kHz, 640
    /// samples/frame = 40 ms, non-overlapping, last zero-padded). Output: [nFrames, 3840] LLM-space
    /// embeddings, spliced RAW. Graph: RMSNorm(640) -> mm.a.input_projection(640 -> 3840). No conv/attention.
    /// </summary>
    public float[] EncodeAudio(float[] frames, int nFrames)
    {
        if (_mmAInputProjW == null) throw new InvalidOperationException("mmproj has no audio encoder.");
        if (frames.Length != (long)nFrames * AudioFrameLen)
            throw new ArgumentException($"frames length {frames.Length} != nFrames({nFrames})*AudioFrameLen({AudioFrameLen}).");

        var a = (float[])frames.Clone();
        RmsNormInPlace(a, nFrames, AudioFrameLen, RmsNormEps);
        return MatMulW(a, nFrames, AudioFrameLen, EmbedDim, _mmAInputProjW!);
    }

    // ── primitive ops (CPU reference) ───────────────────────────────────────────────────────────────

    /// <summary>y[r,o] = sum_i x[r,i] * W[o,i], with W in ggml layout (W[o,i] = w[o*in + i]). out = [rows, outDim].</summary>
    private static float[] MatMulW(float[] x, int rows, int inDim, int outDim, float[] w)
    {
        var y = new float[(long)rows * outDim];
        for (int r = 0; r < rows; r++)
        {
            int xBase = r * inDim;
            int yBase = r * outDim;
            for (int o = 0; o < outDim; o++)
            {
                int wBase = o * inDim;
                float s = 0f;
                for (int i = 0; i < inDim; i++) s += x[xBase + i] * w[wBase + i];
                y[yBase + o] = s;
            }
        }
        return y;
    }

    /// <summary>PyTorch LayerNorm per row over C: (x-mean)/sqrt(var+eps)*gamma + beta. var is the biased (population) variance.</summary>
    private static float[] LayerNorm(float[] x, int rows, int C, float[] gamma, float[] beta, float eps)
    {
        var y = new float[(long)rows * C];
        for (int r = 0; r < rows; r++)
        {
            int b = r * C;
            double mean = 0;
            for (int c = 0; c < C; c++) mean += x[b + c];
            mean /= C;
            double var = 0;
            for (int c = 0; c < C; c++) { double d = x[b + c] - mean; var += d * d; }
            var /= C;
            float invStd = (float)(1.0 / Math.Sqrt(var + eps));
            for (int c = 0; c < C; c++)
                y[b + c] = (float)((x[b + c] - mean) * invStd) * gamma[c] + beta[c];
        }
        return y;
    }

    /// <summary>Weightless RMSNorm per row over C: x / sqrt(mean(x^2)+eps). In place.</summary>
    private static void RmsNormInPlace(float[] x, int rows, int C, float eps)
    {
        for (int r = 0; r < rows; r++)
        {
            int b = r * C;
            double ms = 0;
            for (int c = 0; c < C; c++) { double v = x[b + c]; ms += v * v; }
            ms /= C;
            float inv = (float)(1.0 / Math.Sqrt(ms + eps));
            for (int c = 0; c < C; c++) x[b + c] *= inv;
        }
    }

    private static void AddBias(float[] x, int rows, int C, float[] bias)
    {
        for (int r = 0; r < rows; r++)
        {
            int b = r * C;
            for (int c = 0; c < C; c++) x[b + c] += bias[c];
        }
    }

    /// <summary>Add the factorized 2D position embedding: row p gets tbl_x[:, p%nCols] + tbl_y[:, p/nCols].
    /// _posEmbd flat = ne0=D (contiguous), ne1=PosTableLen positions, ne2=2 axis (0=x, 1=y).</summary>
    private void AddPositionEmbedding(float[] x, int rows, int D, int nCols)
    {
        int yAxisBase = D * PosTableLen; // start of axis=1 (tbl_y) block
        for (int p = 0; p < rows; p++)
        {
            int px = p % nCols;
            int py = p / nCols;
            int xb = p * D;
            int tx = px * D;             // tbl_x[:, px]
            int ty = yAxisBase + py * D; // tbl_y[:, py]
            for (int e = 0; e < D; e++)
                x[xb + e] += _posEmbd![tx + e] + _posEmbd[ty + e];
        }
    }
}
