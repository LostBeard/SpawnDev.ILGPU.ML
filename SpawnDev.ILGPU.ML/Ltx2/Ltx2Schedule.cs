namespace SpawnDev.ILGPU.ML.Ltx2;

/// <summary>
/// LTX-2's sigma schedules: ComfyUI's LTXVScheduler (MEASURED black box by SpawnGen against 22 ComfyUI schedules: steps 1 to 30, shifts, stretch on / off, terminals,
/// five latent sizes; this is SpawnGen's function, MIT) and the distilled recipe's fixed lists (diffusers' pipelines/ltx2/utils.py, Apache-2.0).
/// </summary>
public static class Ltx2Schedule
{
    /// <summary>The distilled LoRA's 8-step schedule (guidance 1), terminal 0 included.</summary>
    public static readonly float[] Distilled = { 1f, 0.99375f, 0.9875f, 0.98125f, 0.975f, 0.909375f, 0.725f, 0.421875f, 0f };

    /// <summary>The second pass's schedule after the x2 latent upscale (the t2v template's ManualSigmas), terminal 0 included.</summary>
    public static readonly float[] SecondPass = { 0.909375f, 0.725f, 0.421875f, 0f };

    /// <summary>
    /// ComfyUI's LTXVScheduler: <paramref name="steps"/> + 1 sigmas from 1 to 0. <c>t_i = 1 - i / steps</c>, <c>sigma = e^mu / (e^mu + (1 / t - 1))</c> with
    /// <c>mu = baseShift + (maxShift - baseShift) (tokens - 1024) / (4096 - 1024)</c> (not clamped) and <paramref name="videoTokens"/> = latent frames x height x width
    /// of the video latent; with <paramref name="stretch"/> the non-zero sigmas are mapped so that the last one is <paramref name="terminal"/>:
    /// <c>1 - (1 - sigma) (1 - terminal) / (1 - last)</c>.
    /// <para>Computed in float32 the way the measurements say ComfyUI does (37 of its schedules, _DevComms 2026-11-18-0800 / -1300): the stretch is reproduced bit for
    /// bit from ComfyUI's own pre-stretch values (192 of 192); the pre-stretch sigmas (t from a float32 linspace built from both ends, e^mu as a float32 scalar) match
    /// exactly on 80% of the entries and are one float32 step off on the rest; whole schedules: 292 of 365 values identical, the worst 8.9e-7 (in double: 145, 8.3e-7).</para>
    /// One step with stretch: ComfyUI divides 0 by 0 (NaN, except at mu 2.05 by rounding); this returns the limit [terminal, 0].
    /// </summary>
    public static float[] Sigmas(int steps, int videoTokens, double maxShift = 2.05, double baseShift = 0.95, bool stretch = true, double terminal = 0.1)
    {
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps), "at least one step");
        double mu = baseShift + (maxShift - baseShift) * (videoTokens - 1024) / (4096.0 - 1024.0);
        float e = (float)Math.Exp(mu);
        int n = steps + 1, half = n / 2;
        float step = (0f - 1f) / (n - 1);                               // linspace(1, 0, n) in float32: the first half from the start, the second from the end
        var s = new float[n];
        for (int i = 0; i < n; i++)
        {
            float t = i < half ? 1f + step * i : 0f - step * (n - 1 - i);
            s[i] = t != 0f ? e / (e + (1f / t - 1f)) : 0f;
        }
        if (stretch)
        {
            int nonZero = 0;
            for (int i = 0; i < n; i++) if (s[i] != 0f) nonZero++;
            float last = 1f - s[nonZero - 1];
            if (last == 0f) { for (int i = 0; i < nonZero; i++) s[i] = (float)terminal; }
            else
            {
                float scale = last / (float)(1.0 - terminal);
                for (int i = 0; i < nonZero; i++) s[i] = 1f - (1f - s[i]) / scale;
            }
        }
        return s;
    }
}
