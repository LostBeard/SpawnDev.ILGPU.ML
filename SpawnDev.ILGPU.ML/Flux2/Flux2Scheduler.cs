namespace SpawnDev.ILGPU.ML.Flux2;

/// <summary>
/// FlowMatchEulerDiscreteScheduler as FLUX.2 klein configures it (dynamic exponential shift, no stochastic sampling): sigmas run linearly from 1 down to 1/steps, are shifted by
/// mu (which depends on the image token count and the step count, see <see cref="EmpiricalMu"/>) as sigma' = e^mu / (e^mu + 1/sigma - 1), and end at 0. The model sees sigma' * 1000; a step is
/// the Euler update x += (sigma_next - sigma) * prediction. Checked against the reference's sigmas and timesteps (golden_klein.py).
/// </summary>
public static class Flux2Scheduler
{
    /// <summary>diffusers' compute_empirical_mu: a straight-line fit in the token count, interpolated between the 10-step and 200-step lines.</summary>
    public static double EmpiricalMu(int imageSeqLen, int steps)
    {
        const double a1 = 8.73809524e-05, b1 = 1.89833333, a2 = 0.00016927, b2 = 0.45666666;
        if (imageSeqLen > 4300) return a2 * imageSeqLen + b2;
        double m200 = a2 * imageSeqLen + b2, m10 = a1 * imageSeqLen + b1;
        double a = (m200 - m10) / 190.0, b = m200 - 200.0 * a;
        return a * steps + b;
    }

    /// <summary>The shifted sigmas, <paramref name="steps"/> + 1 of them (the last is 0).</summary>
    public static double[] Sigmas(int imageSeqLen, int steps)
    {
        double mu = EmpiricalMu(imageSeqLen, steps);
        var s = new double[steps + 1];
        for (int i = 0; i < steps; i++)
        {
            double sigma = steps == 1 ? 1.0 : 1.0 + (1.0 / steps - 1.0) * i / (steps - 1);     // np.linspace(1, 1/steps, steps)
            s[i] = Math.Exp(mu) / (Math.Exp(mu) + (1.0 / sigma - 1.0));
        }
        s[steps] = 0;
        return s;
    }

    /// <summary>The timestep the model is called with for step i: sigma_i * 1000.</summary>
    public static float Timestep(double[] sigmas, int i) => (float)(sigmas[i] * 1000.0);
}
