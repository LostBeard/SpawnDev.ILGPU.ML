namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Tensor layout conversion utilities.
/// Convert between NCHW (channel-first) and NHWC (channel-last) formats,
/// interleaved and planar representations, and batch dimensions.
/// </summary>
public static class TensorLayout
{
    /// <summary>
    /// Convert NCHW float tensor to NHWC float tensor.
    /// Input shape: [C, H, W]. Output shape: [H, W, C].
    /// </summary>
    public static float[] NCHWToNHWC(float[] nchw, int channels, int height, int width)
    {
        int hw = height * width;
        var nhwc = new float[channels * hw];
        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    nhwc[(y * width + x) * channels + c] = nchw[c * hw + y * width + x];
                }
            }
        }
        return nhwc;
    }

    /// <summary>
    /// Convert NHWC float tensor to NCHW float tensor.
    /// Input shape: [H, W, C]. Output shape: [C, H, W].
    /// </summary>
    public static float[] NHWCToNCHW(float[] nhwc, int channels, int height, int width)
    {
        int hw = height * width;
        var nchw = new float[channels * hw];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    nchw[c * hw + y * width + x] = nhwc[(y * width + x) * channels + c];
                }
            }
        }
        return nchw;
    }

    /// <summary>
    /// Convert interleaved RGB bytes [R,G,B,R,G,B,...] to planar float [RRR..., GGG..., BBB...] in [0,1].
    /// Output is NCHW layout [3, H, W].
    /// </summary>
    public static float[] InterleavedRGBToPlanarFloat(byte[] rgb, int width, int height)
    {
        int hw = width * height;
        var planar = new float[3 * hw];
        for (int i = 0; i < hw; i++)
        {
            planar[0 * hw + i] = rgb[i * 3 + 0] / 255f;
            planar[1 * hw + i] = rgb[i * 3 + 1] / 255f;
            planar[2 * hw + i] = rgb[i * 3 + 2] / 255f;
        }
        return planar;
    }

    /// <summary>
    /// Convert interleaved RGBA bytes to planar float [3, H, W] in [0,1], dropping alpha.
    /// </summary>
    public static float[] InterleavedRGBAToPlanarFloat(byte[] rgba, int width, int height)
    {
        int hw = width * height;
        var planar = new float[3 * hw];
        for (int i = 0; i < hw; i++)
        {
            planar[0 * hw + i] = rgba[i * 4 + 0] / 255f;
            planar[1 * hw + i] = rgba[i * 4 + 1] / 255f;
            planar[2 * hw + i] = rgba[i * 4 + 2] / 255f;
        }
        return planar;
    }

    /// <summary>
    /// Convert planar float [3, H, W] in [0,1] to interleaved RGBA bytes.
    /// </summary>
    public static byte[] PlanarFloatToInterleavedRGBA(float[] planar, int width, int height)
    {
        int hw = width * height;
        var rgba = new byte[hw * 4];
        for (int i = 0; i < hw; i++)
        {
            rgba[i * 4 + 0] = ClampByte(planar[0 * hw + i] * 255f);
            rgba[i * 4 + 1] = ClampByte(planar[1 * hw + i] * 255f);
            rgba[i * 4 + 2] = ClampByte(planar[2 * hw + i] * 255f);
            rgba[i * 4 + 3] = 255;
        }
        return rgba;
    }

    /// <summary>
    /// Add batch dimension to a tensor. Prepends a dimension of size 1.
    /// E.g., [3, 224, 224] -> conceptually [1, 3, 224, 224] (data unchanged, shape metadata only).
    /// </summary>
    public static float[] AddBatchDimension(float[] tensor) => tensor; // Data is identical, shape changes

    /// <summary>
    /// Remove batch dimension from a tensor. Strips leading dimension of size 1.
    /// Data is identical.
    /// </summary>
    public static float[] RemoveBatchDimension(float[] tensor) => tensor; // Data is identical, shape changes

    /// <summary>
    /// Transpose a 2D matrix stored as flat array. [rows, cols] -> [cols, rows].
    /// </summary>
    public static float[] Transpose2D(float[] data, int rows, int cols)
    {
        var result = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[c * rows + r] = data[r * cols + c];
        return result;
    }

    private static byte ClampByte(float v) => (byte)Math.Clamp((int)(v + 0.5f), 0, 255);
}
