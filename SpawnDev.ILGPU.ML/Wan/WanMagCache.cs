namespace SpawnDev.ILGPU.ML.Wan;

/// <summary>
/// MagCache for Wan 2.2 (<see cref="WanRequest.MagCache"/>; Ma et al., "MagCache: Fast Video Generation with Magnitude-Aware Cache", NeurIPS 2025). The residual of the
/// transformer's blocks (output minus input) changes in magnitude by a nearly constant, prompt-independent ratio from step to step; while the accumulated error
/// sum |1 - prod ratio| of the steps since the last full call stays under <see cref="Threshold"/> and at most <see cref="MaxSkips"/> calls in a row are skipped, a call
/// reuses the cached residual instead of running the 30 blocks. The first <see cref="RetentionRatio"/> of the calls always run. Each guidance branch (conditional,
/// unconditional) keeps its own state. Defaults and ratios are MagCache's for TI2V-5B (github.com/Zehong-Ma/MagCache, MagCache4Wan2.2, Apache-2.0); <see cref="Ratios"/>
/// replaces the table (one value per call, conditional and unconditional interleaved; a different step count is nearest-interpolated, as MagCache does).
/// </summary>
public sealed record WanMagCache(double Threshold = 0.04, int MaxSkips = 2, double RetentionRatio = 0.2, double[]? Ratios = null)
{
    /// <summary>
    /// MagCache's calibrated ratios for Wan2.2 TI2V-5B text to video, 50 steps (from MagCache4Wan2.2/magcache_generate.py at df81cb18, Apache-2.0): the first two calls
    /// have no previous residual (1.0 is prepended, as MagCache's init_magcache does).
    /// </summary>
    public static readonly double[] Ti2v5bTextToVideo =
    {
        1.0, 1.0,
        0.99505, 0.99389, 0.99441, 0.9957, 0.99558, 0.99551, 0.99499, 0.9945, 0.99534, 0.99548, 0.99468, 0.9946, 0.99463, 0.99458,
        0.9946, 0.99453, 0.99408, 0.99404, 0.9945, 0.99441, 0.99409, 0.99398, 0.99403, 0.99397, 0.99382, 0.99377, 0.99349, 0.99343,
        0.99377, 0.99378, 0.9933, 0.99328, 0.99303, 0.99301, 0.99217, 0.99216, 0.992, 0.99201, 0.99201, 0.99202, 0.99133, 0.99132,
        0.99112, 0.9911, 0.99155, 0.99155, 0.98958, 0.98957, 0.98959, 0.98958, 0.98838, 0.98835, 0.98826, 0.98825, 0.9883, 0.98828,
        0.98711, 0.98709, 0.98562, 0.98561, 0.98511, 0.9851, 0.98414, 0.98412, 0.98284, 0.98282, 0.98104, 0.98101, 0.97981, 0.97979,
        0.97849, 0.97849, 0.97557, 0.97554, 0.97398, 0.97395, 0.97171, 0.97166, 0.96917, 0.96913, 0.96511, 0.96507, 0.96263, 0.96257,
        0.95839, 0.95835, 0.95483, 0.95475, 0.94942, 0.94936, 0.9468, 0.94678, 0.94583, 0.94594, 0.94843, 0.94872, 0.96949, 0.97015,
    };

    /// <summary>MagCache's calibrated ratios for Wan2.2 TI2V-5B image to video, 50 steps (the same source).</summary>
    public static readonly double[] Ti2v5bImageToVideo =
    {
        1.0, 1.0,
        0.99512, 0.99559, 0.99559, 0.99561, 0.99595, 0.99577, 0.99512, 0.99512, 0.99546, 0.99534, 0.99543, 0.99531, 0.99496, 0.99491,
        0.99504, 0.99499, 0.99444, 0.99449, 0.99481, 0.99481, 0.99435, 0.99435, 0.9943, 0.99431, 0.99411, 0.99406, 0.99373, 0.99376,
        0.99413, 0.99405, 0.99363, 0.99359, 0.99335, 0.99331, 0.99244, 0.99243, 0.99229, 0.99229, 0.99239, 0.99236, 0.99163, 0.9916,
        0.99149, 0.99151, 0.99191, 0.99192, 0.9898, 0.98981, 0.9899, 0.98987, 0.98849, 0.98849, 0.98846, 0.98846, 0.98861, 0.98861,
        0.9874, 0.98738, 0.98588, 0.98589, 0.98539, 0.98534, 0.98444, 0.98439, 0.9831, 0.98309, 0.98119, 0.98118, 0.98001, 0.98,
        0.97862, 0.97859, 0.97555, 0.97558, 0.97392, 0.97388, 0.97152, 0.97145, 0.96871, 0.9687, 0.96435, 0.96434, 0.96129, 0.96127,
        0.95639, 0.95638, 0.95176, 0.95175, 0.94446, 0.94452, 0.93972, 0.93974, 0.93575, 0.9359, 0.93537, 0.93552, 0.96655, 0.96616,
    };

    /// <summary>The per-call ratios for <paramref name="steps"/> steps: the table as is, or each branch nearest-interpolated (MagCache's nearest_interp).</summary>
    public static double[] ForSteps(double[] table, int steps)
    {
        if (table.Length == 2 * steps) return table;
        var o = new double[2 * steps];
        int n = table.Length / 2;
        for (int b = 0; b < 2; b++)
            for (int i = 0; i < steps; i++)
            {
                double scale = steps == 1 ? 0 : (n - 1) / (double)(steps - 1);                 // numpy: np.round(np.arange(steps) * scale), half to even
                int src = steps == 1 ? n - 1 : (int)Math.Round(i * scale, MidpointRounding.ToEven);
                o[2 * i + b] = table[2 * src + b];
            }
        return o;
    }

    /// <summary>
    /// The skip decisions of one video, MagCache's rule: call <c>cnt</c> = 2 * step + branch; calls before (int)(2 * steps * RetentionRatio) always run; otherwise the
    /// branch's ratio product, step count and error advance, and the call is skipped while the error is under the threshold and the run of skips is at most MaxSkips (a
    /// call that runs resets the branch's state).
    /// </summary>
    public sealed class State
    {
        private readonly WanMagCache _c;
        private readonly double[] _ratios;
        private readonly int _calls;
        private readonly double[] _accRatio = { 1, 1 }, _accErr = { 0, 0 };
        private readonly int[] _accSteps = { 0, 0 };

        public State(WanMagCache config, double[] table, int steps)
        {
            _c = config; _calls = 2 * steps; _ratios = ForSteps(config.Ratios ?? table, steps);
        }

        /// <summary>Calls skipped so far.</summary>
        public int Skipped { get; private set; }

        /// <summary>Whether call (<paramref name="step"/>, <paramref name="branch"/>) reuses the cached residual.</summary>
        public bool Skip(int step, int branch)
        {
            int cnt = 2 * step + branch;
            if (cnt < (int)(_calls * _c.RetentionRatio)) return false;
            _accRatio[branch] *= _ratios[cnt];
            _accSteps[branch]++;
            _accErr[branch] += Math.Abs(1 - _accRatio[branch]);
            if (_accErr[branch] < _c.Threshold && _accSteps[branch] <= _c.MaxSkips) { Skipped++; return true; }
            _accErr[branch] = 0; _accSteps[branch] = 0; _accRatio[branch] = 1;
            return false;
        }
    }
}
