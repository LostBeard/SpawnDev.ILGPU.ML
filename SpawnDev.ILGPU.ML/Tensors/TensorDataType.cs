namespace SpawnDev.ILGPU.ML.Tensors;

/// <summary>
/// Storage element type of a <see cref="Tensor"/>. Replaces the old <c>bool IsHalf</c> — a bool can only say
/// fp16-or-fp32, but weights now stay native in any low-precision type (Geordi's <c>ILGPU.Half</c>/
/// <c>BFloat16</c>/<c>Float8E4M3</c>/<c>Float8E5M2</c> at 100% parity on all 6 backends, FP4/INT4 next). Op
/// kernels read the native type directly and convert to f32 ONLY at the arithmetic, in registers, via
/// <c>ILGPU.PrecisionConvert</c> — never a managed f32 temp buffer (the no-needless-low-p→f32-conversion rule).
///
/// Values are the ONNX <c>TensorProto.DataType</c> codes so the loader maps its parsed dtype straight to this
/// enum (a cast) with no lookup table.
/// </summary>
public enum TensorDataType
{
    /// <summary>32-bit IEEE float — activations and fp32 weights (the default; data lives in <c>Tensor.Data</c>).</summary>
    Float32 = 1,
    /// <summary>16-bit IEEE half (<c>ILGPU.Half</c>).</summary>
    Float16 = 10,
    /// <summary>bfloat16 (<c>ILGPU.BFloat16</c>) — 8-bit exponent, 7-bit mantissa.</summary>
    BFloat16 = 16,
    /// <summary>OCP FP8 E4M3 (<c>ILGPU.Float8E4M3</c>).</summary>
    Float8E4M3 = 17,
    /// <summary>OCP FP8 E5M2 (<c>ILGPU.Float8E5M2</c>).</summary>
    Float8E5M2 = 19,
}
