namespace SpawnDev.ILGPU.ML.Wan;

/// <summary>
/// UniPC multistep sampler for flow matching, as Wan 2.2 uses it (<c>FlowUniPCMultistepScheduler</c> of the reference, solver order 2, B(h) variant bh2,
/// <c>predict_x0</c>, <c>flow_prediction</c>, lower-order final step, timestep shift): the model output is converted to an x0 prediction, a UniC corrector refines
/// the sample the previous step produced, and a UniP predictor takes the step. The math runs in double on the host over flat <c>float[]</c> latents (a few
/// vector operations per step); the reference does the same operations on float32 scalars, so results agree to float rounding.
/// </summary>
public sealed class FlowUniPCScheduler
{
    private readonly int _numTrain, _solverOrder;
    private readonly bool _lowerOrderFinal;
    private float[] _sigmas = Array.Empty<float>();
    private long[] _timesteps = Array.Empty<long>();
    private readonly float[]?[] _modelOutputs;
    private float[]? _lastSample;
    private int _stepIndex, _lowerOrderNums, _thisOrder;

    /// <summary>
    /// Reproduce the reference's arithmetic instead of the exact one: every scalar operation rounded to float32 (it computes <c>1 - sigma</c>, the lambdas and the solved
    /// weights on float32 scalars), the latent operations in float32, and the <c>einsum</c> that applies the UniP / UniC weights to the model-output differences in bf16:
    /// the reference calls the scheduler inside <c>torch.autocast(bfloat16)</c>, which runs <c>einsum</c> in bf16 and rounds that correction term (and the product of
    /// the 0-dim scale with it) to bf16. Off by default: the double-precision path is the more accurate arithmetic and differs from the reference's by that bf16 rounding
    /// (about 1e-4 relative on a latent); on, the result follows the reference to float rounding.
    /// </summary>
    public bool EmulateReference { get; init; }
    private double F(double v) => EmulateReference ? (double)(float)v : v;

    /// <summary>float32 to bfloat16 and back, round to nearest even.</summary>
    private static float Bf16(float v)
    {
        int bits = BitConverter.SingleToInt32Bits(v);
        if ((bits & 0x7F800000) == 0x7F800000) return v;
        bits = (int)(((uint)bits + 0x7FFFu + (((uint)bits >> 16) & 1u)) & 0xFFFF0000u);
        return BitConverter.Int32BitsToSingle(bits);
    }

    /// <summary>Sigmas, one per step plus the final 0 (float32 like the reference).</summary>
    public IReadOnlyList<float> Sigmas => _sigmas;
    /// <summary>Integer timesteps (sigma * 1000, truncated), one per step: what the transformer is called with.</summary>
    public IReadOnlyList<long> Timesteps => _timesteps;
    public int Steps => _timesteps.Length;

    public FlowUniPCScheduler(int numTrainTimesteps = 1000, int solverOrder = 2, bool lowerOrderFinal = true)
    {
        if (solverOrder is < 1 or > 2) throw new NotSupportedException("solver orders 1 and 2 (the order Wan uses is 2)");
        _numTrain = numTrainTimesteps; _solverOrder = solverOrder; _lowerOrderFinal = lowerOrderFinal;
        _modelOutputs = new float[]?[solverOrder];
    }

    /// <summary>set_timesteps(steps, shift): sigmas linspace(sigma_max, sigma_min, steps + 1)[:-1], shifted by <paramref name="shift"/>, then the final sigma 0.</summary>
    public void SetTimesteps(int steps, double shift)
    {
        // the training sigmas: 1 - linspace(1, 1/N, N)[::-1] as float32; sigma_max = its first value (0.999 as float32), sigma_min = its last (0)
        double sigmaMax = (double)(float)(1.0 - 1.0 / _numTrain), sigmaMin = 0.0;
        var sig = new double[steps];
        for (int i = 0; i < steps; i++)
        {
            double s = sigmaMax + (sigmaMin - sigmaMax) * i / steps;               // np.linspace(max, min, steps + 1)[:-1]
            sig[i] = shift * s / (1 + (shift - 1) * s);
        }
        _timesteps = new long[steps];
        _sigmas = new float[steps + 1];
        for (int i = 0; i < steps; i++) { _timesteps[i] = (long)(sig[i] * _numTrain); _sigmas[i] = (float)sig[i]; }
        _sigmas[steps] = 0f;
        Array.Clear(_modelOutputs);
        _lastSample = null; _stepIndex = 0; _lowerOrderNums = 0; _thisOrder = 0;
    }

    /// <summary>One sampler step: <paramref name="modelOutput"/> is the (guidance-combined) flow prediction at <paramref name="sample"/>; returns the sample of the next sigma.</summary>
    public float[] Step(float[] modelOutput, float[] sample)
    {
        if (_timesteps.Length == 0) throw new InvalidOperationException("SetTimesteps first");
        int n = sample.Length;
        double sigmaCur = _sigmas[_stepIndex];
        // convert_model_output (predict_x0, flow_prediction): x0 = sample - sigma * model_output
        var x0 = new float[n];
        for (int i = 0; i < n; i++) x0[i] = (float)(sample[i] - sigmaCur * modelOutput[i]);

        if (_stepIndex > 0 && _lastSample != null)
            sample = Corrector(x0, _lastSample, sample, _thisOrder);

        for (int i = 0; i < _solverOrder - 1; i++) _modelOutputs[i] = _modelOutputs[i + 1];
        _modelOutputs[_solverOrder - 1] = x0;

        int order = _lowerOrderFinal ? Math.Min(_solverOrder, _timesteps.Length - _stepIndex) : _solverOrder;
        _thisOrder = Math.Min(order, _lowerOrderNums + 1);
        _lastSample = sample;
        var prev = Predictor(sample, _thisOrder);
        if (_lowerOrderNums < _solverOrder) _lowerOrderNums++;
        _stepIndex++;
        return prev;
    }

    private double Lambda(double sigma) => F(F(Math.Log(F(1 - sigma))) - F(Math.Log(sigma)));

    /// <summary>The coefficients shared by the predictor and the corrector: h, rk (order 2), h_phi_1, B_h and the solved weights.</summary>
    private void Coefficients(double lambdaT, double lambdaS0, double lambdaSi, int order, out double hPhi1, out double bH, out double[] rho, bool corrector)
    {
        double h = F(lambdaT - lambdaS0), hh = -h;                           // predict_x0
        hPhi1 = F(double.ExpM1(hh));
        bH = F(double.ExpM1(hh));                                            // bh2
        double hPhiK = F(F(hPhi1 / hh) - 1), factorial = 1;
        double rk = order >= 2 ? F(F(lambdaSi - lambdaS0) / h) : double.NaN;
        // R rows: rks^(i-1), rks = [rk (order 2), 1]; b_i = h_phi_k * i! / B_h
        var b = new double[order];
        for (int i = 1; i <= order; i++)
        {
            b[i - 1] = F(F(hPhiK * factorial) / bH);
            factorial *= i + 1;
            hPhiK = F(F(hPhiK / hh) - F(1 / factorial));
        }
        if (!corrector)
            rho = order == 2 ? new[] { 0.5 } : Array.Empty<double>();        // UniP: 0.5 for order 2 (no D1 for order 1)
        else if (order == 1)
            rho = new[] { 0.5 };
        else
        {
            // solve [[1, 1], [rk, 1]] * rho = b
            double r0 = F(F(b[0] - b[1]) / F(1 - rk));
            rho = new[] { r0, F(b[0] - r0) };
        }
    }

    private float[] Predictor(float[] x, int order)
    {
        double sigmaT = _sigmas[_stepIndex + 1], sigmaS0 = _sigmas[_stepIndex];
        double alphaT = F(1 - sigmaT);
        double lambdaT = Lambda(sigmaT), lambdaS0 = Lambda(sigmaS0);
        var m0 = _modelOutputs[_solverOrder - 1]!;
        double lambdaSi = order >= 2 ? Lambda(_sigmas[_stepIndex - 1]) : 0;
        Coefficients(lambdaT, lambdaS0, lambdaSi, order, out double hPhi1, out double bH, out double[] rho, corrector: false);
        double rk = order >= 2 ? F(F(lambdaSi - lambdaS0) / F(lambdaT - lambdaS0)) : 1;
        var prev = order >= 2 ? _modelOutputs[_solverOrder - 2] : null;
        var o = new float[x.Length];
        double a = F(sigmaT / sigmaS0), b = F(alphaT * hPhi1), c = F(alphaT * bH);
        if (EmulateReference)
        {
            // the reference's float32 tensor arithmetic; pred_res = einsum(rho, D1) is computed in bf16 and the 0-dim scale times it stays bf16
            float af = (float)a, bf = (float)b, cf = (float)c, rkf = (float)rk, rhoBf = Bf16((float)(order >= 2 ? rho[0] : 0));
            for (int i = 0; i < x.Length; i++)
            {
                float xt = af * x[i] - bf * m0[i];
                if (order >= 2)
                {
                    float d1 = (prev![i] - m0[i]) / rkf;
                    float predRes = Bf16(rhoBf * Bf16(d1));
                    xt -= Bf16(cf * predRes);
                }
                o[i] = xt;
            }
            return o;
        }
        for (int i = 0; i < x.Length; i++)
        {
            double xt = a * x[i] - b * m0[i];
            if (order >= 2) xt -= c * (rho[0] * ((prev![i] - m0[i]) / rk));
            o[i] = (float)xt;
        }
        return o;
    }

    private float[] Corrector(float[] modelT, float[] lastSample, float[] thisSample, int order)
    {
        double sigmaT = _sigmas[_stepIndex], sigmaS0 = _sigmas[_stepIndex - 1];
        double alphaT = F(1 - sigmaT);
        double lambdaT = Lambda(sigmaT), lambdaS0 = Lambda(sigmaS0);
        var m0 = _modelOutputs[_solverOrder - 1]!;
        double lambdaSi = order >= 2 ? Lambda(_sigmas[_stepIndex - 2]) : 0;
        Coefficients(lambdaT, lambdaS0, lambdaSi, order, out double hPhi1, out double bH, out double[] rho, corrector: true);
        double rk = order >= 2 ? F(F(lambdaSi - lambdaS0) / F(lambdaT - lambdaS0)) : 1;
        var prev = order >= 2 ? _modelOutputs[_solverOrder - 2] : null;
        var o = new float[lastSample.Length];
        double a = F(sigmaT / sigmaS0), b = F(alphaT * hPhi1), c = F(alphaT * bH);
        double rhoLast = rho[^1];
        if (EmulateReference)
        {
            // float32 tensor arithmetic; corr_res = einsum(rho[:-1], D1s) is bf16, added to the float32 rho_last * D1_t, then scaled by the float32 0-dim alpha_t * B_h
            float af = (float)a, bf = (float)b, cf = (float)c, rkf = (float)rk, rhoLastF = (float)rhoLast, rhoBf = Bf16((float)(order >= 2 ? rho[0] : 0));
            for (int i = 0; i < o.Length; i++)
            {
                float xt = af * lastSample[i] - bf * m0[i];
                float corr = order >= 2 ? Bf16(rhoBf * Bf16((prev![i] - m0[i]) / rkf)) : 0f;
                float d1t = modelT[i] - m0[i];
                xt -= cf * (corr + rhoLastF * d1t);
                o[i] = xt;
            }
            return o;
        }
        for (int i = 0; i < o.Length; i++)
        {
            double xt = a * lastSample[i] - b * m0[i];
            double corr = order >= 2 ? rho[0] * ((prev![i] - m0[i]) / rk) : 0;
            double d1t = modelT[i] - m0[i];
            xt -= c * (corr + rhoLast * d1t);
            o[i] = (float)xt;
        }
        return o;
    }
}
