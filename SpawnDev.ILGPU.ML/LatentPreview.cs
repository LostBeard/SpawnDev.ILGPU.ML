namespace SpawnDev.ILGPU.ML;

/// <summary>
/// A live preview of a video sampler: a model-space latent (the sampler's current x0 estimate) to colours without the VAE, one pixel per latent voxel, by a linear map fitted
/// by least squares over clips this library decoded (<c>tools/ltx-harness/fit_latent_preview.py</c>; the tables are in LatentPreview.Factors.g.cs). Cheap enough to show at
/// every denoising step: an LTX latent of a 704x480 x 97 clip is 13 frames of 22 x 15 voxels.
/// </summary>
public sealed partial class LatentPreview
{
    private readonly float[] _w, _b;
    private static LatentPreview? _ltx, _wan;

    /// <summary>LTX-Video 0.9.8: 128 channels, one latent frame per 8 video frames (the first alone), 32 x 32 pixels per voxel.</summary>
    public static LatentPreview Ltx => _ltx ??= new LatentPreview(128, 8, 32, LtxFactors, LtxBias);
    /// <summary>Wan 2.2 TI2V-5B: 48 channels, one latent frame per 4 video frames (the first alone), 16 x 16 pixels per voxel.</summary>
    public static LatentPreview Wan => _wan ??= new LatentPreview(48, 4, 16, WanFactors, WanBias);

    public int Channels { get; }
    /// <summary>Video frames per latent frame (after the first).</summary>
    public int TemporalRatio { get; }
    /// <summary>Pixels per latent voxel side.</summary>
    public int SpatialRatio { get; }

    private LatentPreview(int channels, int temporal, int spatial, float[] factors, float[] bias)
    {
        Channels = channels; TemporalRatio = temporal; SpatialRatio = spatial; _w = factors; _b = bias;
        if (factors.Length != channels * 3 || bias.Length != 3) throw new ArgumentException("the preview table does not match the channel count");
    }

    /// <summary>
    /// Latent frame <paramref name="f"/> of <paramref name="latent"/> ([Channels, <paramref name="frames"/>, <paramref name="h"/>, <paramref name="w"/>] channel-first, model space) as RGBA
    /// bytes, <paramref name="w"/> x <paramref name="h"/> pixels.
    /// </summary>
    public byte[] FrameRgba(float[] latent, int frames, int h, int w, int f)
    {
        int hw = h * w;
        if (latent.Length != (long)Channels * frames * hw) throw new ArgumentException("the latent must be [Channels, frames, h, w]");
        var o = new byte[hw * 4];
        for (int p = 0; p < hw; p++)
        {
            float r = _b[0], g = _b[1], b = _b[2];
            for (int c = 0; c < Channels; c++)
            {
                float v = latent[((long)c * frames + f) * hw + p];
                r += v * _w[c * 3]; g += v * _w[c * 3 + 1]; b += v * _w[c * 3 + 2];
            }
            o[p * 4] = ToByte(r); o[p * 4 + 1] = ToByte(g); o[p * 4 + 2] = ToByte(b); o[p * 4 + 3] = 255;
        }
        return o;
    }

    private static byte ToByte(float v) => (byte)Math.Clamp((int)MathF.Round((v + 1f) * 127.5f), 0, 255);
}
