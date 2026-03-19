namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Noise schedulers for diffusion models (Stable Diffusion, SDXL, etc.).
/// Implements DDIM and Euler schedulers for iterative denoising.
/// Pure math — no GPU dependency. The UNet inference runs on GPU,
/// but the scheduling (timestep selection, noise mixing) is CPU-side.
/// </summary>
public static class DiffusionScheduler
{
    /// <summary>
    /// Generate the beta schedule used by most diffusion models.
    /// Returns alphas_cumprod[numTrainTimesteps] — the cumulative product of (1 - beta).
    /// </summary>
    public static float[] ComputeAlphasCumprod(int numTrainTimesteps = 1000, float betaStart = 0.00085f, float betaEnd = 0.012f, string schedule = "scaled_linear")
    {
        float[] betas;

        if (schedule == "scaled_linear")
        {
            // Stable Diffusion default: linear in sqrt space
            float sqrtStart = MathF.Sqrt(betaStart);
            float sqrtEnd = MathF.Sqrt(betaEnd);
            betas = new float[numTrainTimesteps];
            for (int i = 0; i < numTrainTimesteps; i++)
            {
                float t = (float)i / (numTrainTimesteps - 1);
                float sqrtBeta = sqrtStart + t * (sqrtEnd - sqrtStart);
                betas[i] = sqrtBeta * sqrtBeta;
            }
        }
        else // "linear"
        {
            betas = new float[numTrainTimesteps];
            for (int i = 0; i < numTrainTimesteps; i++)
            {
                betas[i] = betaStart + (float)i / (numTrainTimesteps - 1) * (betaEnd - betaStart);
            }
        }

        // Compute alphas_cumprod = cumprod(1 - betas)
        var alphasCumprod = new float[numTrainTimesteps];
        alphasCumprod[0] = 1f - betas[0];
        for (int i = 1; i < numTrainTimesteps; i++)
        {
            alphasCumprod[i] = alphasCumprod[i - 1] * (1f - betas[i]);
        }

        return alphasCumprod;
    }

    /// <summary>
    /// Select timesteps for inference. Evenly spaced from the training schedule.
    /// </summary>
    /// <param name="numInferenceSteps">Number of denoising steps (e.g., 20, 50)</param>
    /// <param name="numTrainTimesteps">Total training timesteps (usually 1000)</param>
    public static int[] GetTimesteps(int numInferenceSteps, int numTrainTimesteps = 1000)
    {
        float stepRatio = (float)numTrainTimesteps / numInferenceSteps;
        var timesteps = new int[numInferenceSteps];
        for (int i = 0; i < numInferenceSteps; i++)
        {
            timesteps[i] = (int)Math.Round((numInferenceSteps - 1 - i) * stepRatio);
        }
        return timesteps;
    }

    // ──────────────────────────────────────────────
    //  DDIM Scheduler
    // ──────────────────────────────────────────────

    /// <summary>
    /// DDIM scheduler step: given the model's noise prediction, compute the previous latent.
    /// This is the core denoising step called once per inference timestep.
    /// </summary>
    /// <param name="modelOutput">UNet predicted noise [B, C, H, W] as flat array</param>
    /// <param name="sample">Current noisy latent [B, C, H, W]</param>
    /// <param name="timestep">Current timestep index</param>
    /// <param name="prevTimestep">Previous timestep index (-1 for final step)</param>
    /// <param name="alphasCumprod">Pre-computed alphas_cumprod schedule</param>
    /// <param name="eta">DDIM eta parameter (0 = deterministic, 1 = DDPM-like)</param>
    /// <returns>Previous (less noisy) latent</returns>
    public static float[] DDIMStep(float[] modelOutput, float[] sample, int timestep, int prevTimestep,
        float[] alphasCumprod, float eta = 0f)
    {
        int length = sample.Length;
        float alphaProdT = alphasCumprod[timestep];
        float alphaProdTPrev = prevTimestep >= 0 ? alphasCumprod[prevTimestep] : 1f;

        float betaProdT = 1f - alphaProdT;
        float betaProdTPrev = 1f - alphaProdTPrev;

        float sqrtAlphaProdT = MathF.Sqrt(alphaProdT);
        float sqrtBetaProdT = MathF.Sqrt(betaProdT);
        float sqrtAlphaProdTPrev = MathF.Sqrt(alphaProdTPrev);

        // Predicted x0 from noise prediction
        // x0_pred = (sample - sqrt(beta_prod_t) * model_output) / sqrt(alpha_prod_t)
        var result = new float[length];

        // Compute variance for stochastic DDIM (eta > 0)
        float variance = 0;
        if (eta > 0 && prevTimestep >= 0)
        {
            variance = betaProdTPrev / betaProdT * (1f - alphaProdT / alphaProdTPrev);
            variance = eta * eta * variance;
        }

        float sqrtVariance = MathF.Sqrt(variance);
        float dirCoeff = MathF.Sqrt(betaProdTPrev - variance);

        var rng = eta > 0 ? new Random() : null;

        for (int i = 0; i < length; i++)
        {
            // Predicted original sample
            float predX0 = (sample[i] - sqrtBetaProdT * modelOutput[i]) / sqrtAlphaProdT;

            // Clip predicted x0 to [-1, 1] for stability
            predX0 = Math.Clamp(predX0, -1f, 1f);

            // Direction pointing to x_t
            float dirXt = dirCoeff * modelOutput[i];

            // Previous sample
            result[i] = sqrtAlphaProdTPrev * predX0 + dirXt;

            // Add noise for stochastic DDIM
            if (eta > 0 && rng != null)
            {
                result[i] += sqrtVariance * (float)NormalRandom(rng);
            }
        }

        return result;
    }

    // ──────────────────────────────────────────────
    //  Euler Scheduler (used by SD Turbo for single-step)
    // ──────────────────────────────────────────────

    /// <summary>
    /// Euler scheduler step. Simpler than DDIM, used by fast models like SD Turbo.
    /// </summary>
    public static float[] EulerStep(float[] modelOutput, float[] sample, float sigma, float sigmaNext)
    {
        int length = sample.Length;
        var result = new float[length];

        float dt = sigmaNext - sigma;

        for (int i = 0; i < length; i++)
        {
            // Euler method: x_{t-1} = x_t + (x_t - denoised) / sigma * dt
            // With the noise prediction parameterization:
            result[i] = sample[i] + modelOutput[i] * dt;
        }

        return result;
    }

    /// <summary>
    /// Convert timesteps to sigmas for Euler scheduler.
    /// </summary>
    public static float[] TimestepsToSigmas(int[] timesteps, float[] alphasCumprod)
    {
        var sigmas = new float[timesteps.Length + 1];
        for (int i = 0; i < timesteps.Length; i++)
        {
            float alphaCumprod = alphasCumprod[timesteps[i]];
            sigmas[i] = MathF.Sqrt((1f - alphaCumprod) / alphaCumprod);
        }
        sigmas[^1] = 0; // Final sigma is 0
        return sigmas;
    }

    // ──────────────────────────────────────────────
    //  Latent utilities
    // ──────────────────────────────────────────────

    /// <summary>
    /// Generate random Gaussian noise for initial latent.
    /// Shape: [1, channels, height, width] (SD: [1, 4, 64, 64] for 512x512 output)
    /// </summary>
    public static float[] GenerateNoise(int channels, int height, int width, int? seed = null)
    {
        var rng = seed.HasValue ? new Random(seed.Value) : new Random();
        var noise = new float[channels * height * width];
        for (int i = 0; i < noise.Length; i++)
        {
            noise[i] = (float)NormalRandom(rng);
        }
        return noise;
    }

    /// <summary>
    /// Scale the initial noise by the first sigma (for Euler scheduler).
    /// </summary>
    public static float[] ScaleNoise(float[] noise, float sigma)
    {
        var scaled = new float[noise.Length];
        for (int i = 0; i < noise.Length; i++)
            scaled[i] = noise[i] * sigma;
        return scaled;
    }

    /// <summary>Box-Muller transform for generating standard normal random values.</summary>
    private static double NormalRandom(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
}
