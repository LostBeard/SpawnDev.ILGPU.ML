namespace SpawnDev.ILGPU.ML;

/// <summary>How a session stores its weights on the device.</summary>
public enum WeightStorage
{
    /// <summary>In the model file's own precision (FP16 sources stay FP16 where a native low-precision op reads them).</summary>
    Source,
    /// <summary>
    /// Also store FP32 weights as FP16 when they are read only as the weight operand of a low-precision-capable op
    /// (FusedLinear, rank-2 MatMul, Gemm, Conv group=1). Half the device memory and weight bandwidth; compute and
    /// activations stay FP32. Weights round to FP16's 11-bit significand, so outputs move slightly - opt-in.
    /// </summary>
    Half,
}
