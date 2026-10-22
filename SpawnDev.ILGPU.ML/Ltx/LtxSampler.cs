namespace SpawnDev.ILGPU.ML.Ltx;

/// <summary>
/// The sampler of the LTX-Video 2B 0.9.8 distilled model: flow-matching Euler on the fixed sigma list of the official first pass (<c>configs/ltxv-2b-0.9.8-distilled.yaml</c>), one DiT
/// forward per step (guidance 1: no negative prompt). The model timestep is <c>sigma * 1000</c> (FlowMatchEulerDiscreteScheduler's <c>timesteps</c>), the update is
/// <c>x += (sigma_next - sigma) * v</c> and the last sigma steps to 0.
/// </summary>
public static class LtxSampler
{
    /// <summary>The first pass of the distilled recipe, at 2/3 of the final resolution: 7 steps.</summary>
    public static readonly float[] DistilledFirstPass = { 1.0000f, 0.9937f, 0.9875f, 0.9812f, 0.9750f, 0.9094f, 0.7250f };

    /// <summary>The second pass (after the x2 latent upscaler, re-noised to the first sigma): 3 steps.</summary>
    public static readonly float[] DistilledSecondPass = { 0.9094f, 0.7250f, 0.4219f };

    /// <summary>The sigmas of a pass with the terminal 0 appended: step i goes from <c>[i]</c> to <c>[i + 1]</c>.</summary>
    public static float[] WithTerminal(float[] sigmas)
    {
        var s = new float[sigmas.Length + 1];
        Array.Copy(sigmas, s, sigmas.Length);
        return s;
    }

    /// <summary>The model timestep of a sigma, in float32 like the reference scheduler.</summary>
    public static float Timestep(float sigma) => sigma * 1000f;

    /// <summary>One Euler step, in place on the host latent: <c>x += (sigmaNext - sigma) * v</c>.</summary>
    public static void Step(float[] x, float[] v, float sigma, float sigmaNext)
    {
        float dt = sigmaNext - sigma;
        for (int i = 0; i < x.Length; i++) x[i] += dt * v[i];
    }
}
