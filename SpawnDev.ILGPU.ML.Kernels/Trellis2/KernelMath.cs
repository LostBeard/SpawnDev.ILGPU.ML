namespace SpawnDev.ILGPU.ML.Trellis2;

/// <summary>
/// Device functions shared by the generated kernels in this assembly and the hand-written kernels in SpawnDev.ILGPU.ML (which sits above this assembly, so
/// a function both use lives here).
/// </summary>
internal static class KernelMath
{
    /// <summary>The activation <paramref name="act"/> (DiTKernels.Act's values): 0 none, 1 SiLU, 2 GELU (tanh), 3 GELU (erf).</summary>
    // NOT inlined: the tiled GEMM epilogue calls this 64 times per thread; inlined, every call site carried all three activation bodies
    // (the WGSL of one GEMM ran to ~600 KB, and the browser compiles shaders on first use).
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    internal static float Activate(float v, int act)
    {
        if (act == 1) return v / (1f + MathF.Exp(-v));                              // SiLU
        if (act == 2)                                                              // GELU, tanh approximation
        {
            float z = 0.7978845608028654f * (v + 0.044715f * v * v * v);
            z = MathF.Max(-20f, MathF.Min(20f, z));
            float e = MathF.Exp(2f * z);
            return 0.5f * v * (1f + (e - 1f) / (e + 1f));
        }
        if (act == 3)                                                              // exact GELU (erf), Abramowitz-Stegun 7.1.26
        {
            float z = v * 0.7071067811865476f, az = MathF.Abs(z);
            float t = 1f / (1f + 0.3275911f * az);
            float poly = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
            float erfAbs = 1f - poly * MathF.Exp(-az * az);
            return 0.5f * v * (1f + (z < 0f ? -erfAbs : erfAbs));
        }
        return v;
    }
}
