using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    // tools/clap-reference/probe_reference.py: 24 unit rows x 8, labels, and AssistZ fit_logistic's weights on them (numpy, float64)
    private static readonly float[][] ProbeRefX =
    {
        new float[] { 0.453291893f, -0.183061093f, -0.153504834f, 0.175070778f, 0.449396491f, -0.362312257f, 0.43556723f, 0.428783625f },
        new float[] { -0.268874079f, -0.17514132f, -0.165534049f, 0.287447214f, 0.424070299f, 0.303815782f, -0.500503182f, -0.514161229f },
        new float[] { -0.0823044851f, 0.272187889f, 0.103568889f, -0.334070563f, -0.603900254f, 0.0779866725f, -0.423199505f, 0.496925354f },
        new float[] { 0.614269078f, -0.0731058866f, 0.633835316f, 0.0785124153f, 0.419232309f, -0.0528960489f, -0.100161597f, -0.144331321f },
        new float[] { 0.360063434f, -0.410400897f, 0.378356844f, 0.265139401f, 0.150860354f, -0.446930498f, -0.219445735f, 0.466701716f },
        new float[] { 0.144574925f, -0.109664932f, -0.612487793f, 0.48750791f, -0.454265296f, -0.0259074494f, -0.117221437f, -0.365372956f },
        new float[] { 0.0902817771f, -0.536244988f, -0.307627022f, 0.429299265f, -0.357653111f, 0.119424418f, 0.141604781f, 0.512960434f },
        new float[] { 0.191492617f, -0.221258149f, -0.636530042f, 0.0212969612f, -0.299305677f, 0.616883934f, -0.0112654371f, 0.196200907f },
        new float[] { 0.170894936f, -0.0534320362f, 0.0653856173f, -0.174479619f, 0.415776134f, -0.656990945f, -0.41707623f, 0.393398136f },
        new float[] { 0.635553062f, -0.471723735f, 0.0479491279f, 0.0173834153f, -0.223360851f, 0.236171335f, 0.39571622f, 0.329680741f },
        new float[] { -0.187114999f, -0.289144278f, -0.43748197f, 0.122119024f, 0.505846262f, 0.457899719f, 0.412655294f, -0.198098525f },
        new float[] { -0.13693884f, 0.359675109f, 0.292878956f, -0.473695308f, -0.18712613f, -0.346167952f, -0.511249781f, 0.354247779f },
        new float[] { 0.795259178f, -0.0998657793f, 0.0398430079f, 0.155818f, -0.32765606f, -0.200671539f, 0.377268642f, -0.204361841f },
        new float[] { -0.0118507389f, 0.270631313f, 0.262183905f, -0.209578812f, -0.521811366f, -0.63678509f, 0.241999075f, -0.278583258f },
        new float[] { -0.350225866f, 0.368783444f, -0.3262555f, 0.216013119f, 0.466295928f, -0.256126344f, 0.388272256f, 0.392999172f },
        new float[] { 0.205021411f, 0.664096713f, 0.0718860328f, -0.0253761522f, -0.486762762f, 0.177163228f, -0.106414363f, 0.481125206f },
        new float[] { -0.104125299f, -0.0583677404f, -0.5419631f, 0.327100933f, -0.0321883708f, 0.179584667f, 0.582651079f, -0.460720211f },
        new float[] { 0.519371152f, -0.0101463208f, -0.357181937f, -0.268761724f, 0.114017896f, 0.588091671f, -0.226760983f, 0.34650597f },
        new float[] { 0.0676947385f, -0.0958643407f, 0.281756669f, -0.0324022211f, 0.436445713f, -0.585461795f, -0.439555794f, -0.423474938f },
        new float[] { 0.580651164f, -0.164551929f, 0.317476034f, -0.373637825f, -0.182234049f, 0.106672637f, 0.447505265f, 0.387970805f },
        new float[] { 0.168331996f, 0.321132928f, -0.237496436f, -0.660551012f, -0.375480324f, -0.335234612f, 0.325122774f, 0.129356995f },
        new float[] { 0.330799937f, 0.489901155f, -0.337073445f, -0.0720020533f, -0.379506052f, 0.24874559f, -0.279791951f, -0.497576833f },
        new float[] { -0.205719754f, 0.075738959f, -0.409106135f, -0.150062099f, -0.608451068f, 0.427757949f, 0.15397726f, -0.430299878f },
        new float[] { 0.000773616834f, 0.0399630591f, 0.425656378f, -0.63688159f, 0.153506145f, 0.139159888f, 0.0824898854f, -0.601553559f },
    };
    private static readonly int[] ProbeRefY = { 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0 };
    private static readonly float[] ProbeRefA = { 1.26840766f, 0.0375801288f, 0.474286762f, 0.82301389f, -0.0675619602f, -0.151927804f, 0.0338043839f, 0.0251645096f };
    private const float ProbeRefB = -0.268255078f;

    /// <summary>
    /// <see cref="ClapProbe.FitLogistic"/> for two classes is AssistZ's Lab/probe_eval.py fit_logistic (class-balanced, full batch, L2 / N): on
    /// tools/clap-reference/probe_reference.py's dataset (24 x 8, unbalanced 8 / 16) its weights and bias equal numpy's within 1e-5; a three-class
    /// fit separates three clusters; a JSON round trip predicts identically. Host math: CPU lane only. Added 2026-11-03.
    /// </summary>
    [TestMethod]
    public async Task ClapProbe_FitMatchesNumpyReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.CPU) throw new UnsupportedTestException("host math; gated once, on the CPU lane");
        var labels = ProbeRefY.Select(v => v == 1 ? "gunfire" : "other").ToArray();
        var probe = ClapProbe.FitLogistic(ProbeRefX, labels, classes: new[] { "other", "gunfire" }, checkpoint: ClapAudioClassifier.DefaultRepo);
        double worst = Math.Abs(probe.Bias[0] - ProbeRefB);
        for (int k = 0; k < ProbeRefA.Length; k++) worst = Math.Max(worst, Math.Abs(probe.Weights[0][k] - ProbeRefA[k]));
        if (!(worst <= 1e-5)) throw new Exception($"two-class fit off numpy by {worst:G3} (w0 {probe.Weights[0][0]:G6} vs {ProbeRefA[0]:G6}, b {probe.Bias[0]:G6} vs {ProbeRefB:G6})");
        int correct = ProbeRefX.Select((x, i) => probe.PredictLabel(x) == labels[i]).Count(ok => ok);

        // three clusters around three axes of an 8-d space, unit rows
        var rng = new Random(9);
        var x3 = new List<float[]>(); var y3 = new List<string>();
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < 12; i++)
            {
                var v = Enumerable.Range(0, 8).Select(k => (float)((k == c * 2 ? 1.5 : 0) + 0.3 * (rng.NextDouble() * 2 - 1))).ToArray();
                double nrm = Math.Sqrt(v.Sum(t => (double)t * t));
                x3.Add(v.Select(t => (float)(t / nrm)).ToArray()); y3.Add("c" + c);
            }
        var probe3 = ClapProbe.FitLogistic(x3, y3);
        int correct3 = x3.Select((x, i) => probe3.PredictLabel(x) == y3[i]).Count(ok => ok);
        if (correct3 != x3.Count) throw new Exception($"three-class fit: {correct3}/{x3.Count} training examples right");
        foreach (var x in x3) { double s = probe3.Predict(x).Sum(); if (Math.Abs(s - 1) > 1e-5) throw new Exception($"three-class probabilities sum to {s}"); }

        foreach (var (p, xs) in new[] { (probe, (IReadOnlyList<float[]>)ProbeRefX), (probe3, x3) })
        {
            var back = ClapProbe.FromJson(p.ToJson());
            foreach (var x in xs)
                if (!back.Predict(x).SequenceEqual(p.Predict(x))) throw new Exception("JSON round trip predicts differently");
            if (back.Checkpoint != p.Checkpoint || !back.Classes.SequenceEqual(p.Classes)) throw new Exception("JSON round trip lost classes / checkpoint");
        }
        await Task.CompletedTask;
        return $"two-class fit = numpy within {worst:G2} ({correct}/{labels.Length} train right); three-class {correct3}/{x3.Count}; JSON round trips exact";
    });
}
