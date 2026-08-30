namespace SpawnDev.ILGPU.ML.Onnx;

/// <summary>
/// Decodes the two FNUZ FP8 formats - ONNX <c>FLOAT8E4M3FNUZ</c> (18) and <c>FLOAT8E5M2FNUZ</c> (20).
/// </summary>
/// <remarks>
/// These are NOT the OCP formats <c>ILGPU.Float8E4M3</c> / <c>ILGPU.Float8E5M2</c> carry, and mapping them
/// onto those types would silently mis-decode every weight. They differ in three ways:
/// <list type="bullet">
/// <item>exponent bias is ONE HIGHER - 8 for E4M3FNUZ (vs 7), 16 for E5M2FNUZ (vs 15)</item>
/// <item>no infinities in either (the "FN" = finite)</item>
/// <item>"UZ" = unsigned zero: there is no negative zero, and the bit pattern 0x80 that would encode it is
///       the ONLY NaN</item>
/// </list>
/// So the same byte means different numbers in the two families - e.g. 0x38 is 1.0 in OCP E4M3 and 0.5 in
/// E4M3FNUZ. That is exactly why these dtypes were left unsupported rather than aliased.
/// <para>
/// ⚠️ Decoding is to fp32, and that IS an expansion - unavoidable here, because ILGPU has no FNUZ type to
/// store natively. Do not read this as license to upcast formats that DO have one (Half / BFloat16 /
/// Float8E4M3 / Float8E5M2 all stay native; see <c>BufferPool.AllocateLowPWeightFromStreamAsync</c>).
/// </para>
/// </remarks>
public static class Fp8Fnuz
{
    /// <summary>Decode one <c>FLOAT8E4M3FNUZ</c> byte (1 sign, 4 exponent, 3 mantissa; bias 8).</summary>
    /// <param name="raw">The encoded byte.</param>
    /// <returns>The value, or NaN for 0x80.</returns>
    public static float E4M3FnuzToFloat(byte raw) => FnuzToFloat(raw, expBits: 4, mantBits: 3, bias: 8);

    /// <summary>Decode one <c>FLOAT8E5M2FNUZ</c> byte (1 sign, 5 exponent, 2 mantissa; bias 16).</summary>
    /// <param name="raw">The encoded byte.</param>
    /// <returns>The value, or NaN for 0x80.</returns>
    public static float E5M2FnuzToFloat(byte raw) => FnuzToFloat(raw, expBits: 5, mantBits: 2, bias: 16);

    private static float FnuzToFloat(byte raw, int expBits, int mantBits, int bias)
    {
        // 0x80 is the sole NaN in both FNUZ formats (it is the slot that would otherwise be negative zero).
        if (raw == 0x80) return float.NaN;
        if (raw == 0x00) return 0f;

        int sign = raw >> 7;
        int mantMask = (1 << mantBits) - 1;
        int exp = (raw >> mantBits) & ((1 << expBits) - 1);
        int mant = raw & mantMask;

        float mag;
        if (exp == 0)
        {
            // Subnormal: no implicit leading 1, and the exponent is fixed at 1 - bias.
            mag = mant / (float)(1 << mantBits) * MathF.Pow(2f, 1 - bias);
        }
        else
        {
            // Normal: implicit leading 1. No inf/NaN exponent to exclude - FNUZ has neither.
            mag = (1f + mant / (float)(1 << mantBits)) * MathF.Pow(2f, exp - bias);
        }
        return sign != 0 ? -mag : mag;
    }
}
