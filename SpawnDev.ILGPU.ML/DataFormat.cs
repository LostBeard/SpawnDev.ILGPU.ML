namespace SpawnDev.ILGPU.ML;

/// <summary>
/// Data layout format for spatial tensors.
/// ONNX models use NCHW, TFLite models use NHWC.
/// Operators use this to index channels and spatial dimensions correctly.
/// </summary>
public enum DataFormat
{
    /// <summary>[Batch, Channels, Height, Width] — ONNX standard</summary>
    NCHW,
    /// <summary>[Batch, Height, Width, Channels] — TFLite standard</summary>
    NHWC
}

/// <summary>
/// Helper for layout-aware dimension indexing.
/// </summary>
public static class LayoutHelper
{
    /// <summary>Extract N, C, H, W from a 4D shape regardless of layout.</summary>
    public static (int N, int C, int H, int W) GetDims(int[] shape, DataFormat fmt)
    {
        if (shape.Length < 4) return (shape.Length > 0 ? shape[0] : 1, 1, 1, shape.Length > 1 ? shape[^1] : 1);
        return fmt == DataFormat.NCHW
            ? (shape[0], shape[1], shape[2], shape[3])
            : (shape[0], shape[3], shape[1], shape[2]);
    }

    public static int ChannelAxis(DataFormat fmt) => fmt == DataFormat.NCHW ? 1 : 3;
    public static int HeightAxis(DataFormat fmt) => fmt == DataFormat.NCHW ? 2 : 1;
    public static int WidthAxis(DataFormat fmt) => fmt == DataFormat.NCHW ? 3 : 2;

    /// <summary>Get channel dim from weight shape. NCHW weights: [outC,inC,kH,kW]. NHWC weights: [outC,kH,kW,inC].</summary>
    public static (int outC, int inC, int kH, int kW) GetWeightDims(int[] wShape, DataFormat fmt)
    {
        return fmt == DataFormat.NCHW
            ? (wShape[0], wShape[1], wShape[2], wShape[3])
            : (wShape[0], wShape[3], wShape[1], wShape[2]);
    }
}
