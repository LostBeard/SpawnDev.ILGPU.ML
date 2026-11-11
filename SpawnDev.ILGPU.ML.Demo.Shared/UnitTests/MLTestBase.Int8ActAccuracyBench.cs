using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    // original prose for teacher forcing (no third-party text)
    private const string Int8AccText =
        "The ferry left the harbor a little after six, while the gulls were still arguing over the fish market's scraps. Mara stood at the rail with her " +
        "coat buttoned to the chin and watched the town shrink into a line of wet roofs and chimney smoke. She had crossed this water a hundred times as a " +
        "child, always with her grandfather, who timed the trip by the lighthouse on the point and swore the boat was late whenever the beam swept past " +
        "before the engine changed its note. Now the lighthouse was automated, the keeper's cottage had become a cafe, and the engine was electric and " +
        "made no note at all. Halfway across, the wind turned and brought the smell of rain. A boy in a yellow jacket asked his father how deep the water " +
        "was, and the father, who clearly did not know, said it was deep enough to hide a whale. The boy considered this seriously and then asked whether " +
        "the whale could see the ferry from below, and if so, whether it minded. Mara smiled. Her grandfather would have had an answer for that, something " +
        "about whales preferring quiet boats and being glad the old diesel engines were gone. On the far shore the island's single road climbed past the " +
        "post office, the school with its three classrooms, and the field where the summer fair still set up its tents every August. She had come back to " +
        "sort through the house, to decide what to keep and what to give away, and she had told herself it would take a weekend. Looking at the hill now, " +
        "with the house just visible behind its hedge, she suspected it would take considerably longer, and that she would not mind if it did.";

    /// <summary>
    /// End-to-end accuracy of the OPT-IN int8 activations (<c>FusedDequantMatMul.Int8ActGemv</c>, Ollama's q8_1 MMVQ scheme) on a real model, in llama.cpp's own
    /// quantization metrics (llama-perplexity --kl-divergence): teacher-forced over a fixed ~300-token text, every step's distribution with int8 activations
    /// against the default fp32 activations - mean / p99 / max KL divergence, top-1 agreement, and both perplexities. A measurement (not gated): what a user who
    /// opts in gives up. Model: an Ollama-cache GGUF (env INT8ACC_MODEL, default qwen3:1.7b = Q4_K_M), CUDA (dp4a).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> Bench_Int8Act_TeacherForced_KLD() => await RunTest(async accelerator =>
    {
        if (OperatingSystem.IsBrowser() || accelerator.AcceleratorType != AcceleratorType.Cuda) throw new UnsupportedTestException("dp4a (CUDA), reads the local Ollama cache");
        string root = Environment.GetEnvironmentVariable("OLLAMA_MODELS") is { Length: > 0 } env ? env
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".ollama", "models");
        string modelTag = Environment.GetEnvironmentVariable("INT8ACC_MODEL") is { Length: > 0 } mt ? mt : "qwen3:1.7b";
        var mparts = modelTag.Split(':');
        var manifest = Path.Combine(root, "manifests", "registry.ollama.ai", "library", mparts[0], mparts.Length > 1 ? mparts[1] : "latest");
        if (!File.Exists(manifest)) throw new UnsupportedTestException($"{modelTag} not in the Ollama cache");
        string? path = null;
        using (var doc = JsonDocument.Parse(File.ReadAllText(manifest)))
            foreach (var layer in doc.RootElement.GetProperty("layers").EnumerateArray())
                if (layer.GetProperty("mediaType").GetString()!.EndsWith(".model")) path = Path.Combine(root, "blobs", layer.GetProperty("digest").GetString()!.Replace(':', '-'));

        using var fs = new FileStream(path!, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
        using var dec = await GgufBatchDecoder.CreateAsync(accelerator, fs, maxSeq: 1024, maxBatch: 1);
        var tok = SentencePieceTokenizer.FromGGUF(dec.Model)!;
        var ids = tok.Encode(Int8AccText);                                // the text as the tokenizer encodes it (both passes see the same ids)
        int steps = ids.Length - 1;
        if (steps < 64) throw new Exception($"text tokenized to only {ids.Length} ids");

        // one teacher-forced pass: the distribution after each prefix ids[0..t], t = 0 .. steps - 1 (log-softmax kept per step), and the forced token's NLL
        async Task<(float[][] LogP, double Nll)> Pass(bool int8)
        {
            SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv = int8;
            dec.InvalidateCapturedSteps();
            var rows = new float[steps][];
            double nll = 0;
            await dec.PrefillAsync(new[] { ids[0] }, 0f);
            for (int t = 0; t < steps; t++)
            {
                if (t > 0) await dec.StepAsync(new[] { ids[t] }, 0f);
                var l = await dec.ReadLogitsAsync();
                double mx = double.NegativeInfinity; for (int i = 0; i < l.Length; i++) if (l[i] > mx) mx = l[i];
                double z = 0; for (int i = 0; i < l.Length; i++) z += Math.Exp(l[i] - mx);
                double lz = mx + Math.Log(z);
                var lp = new float[l.Length];
                for (int i = 0; i < l.Length; i++) lp[i] = (float)(l[i] - lz);
                rows[t] = lp;
                nll -= lp[ids[t + 1]];
            }
            return (rows, nll);
        }
        bool saved = SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv;
        (float[][] LogP, double Nll) fp, i8;
        try
        {
            fp = await Pass(false);
            i8 = await Pass(true);
        }
        finally { SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv = saved; dec.InvalidateCapturedSteps(); }

        var kls = new double[steps];
        int top1Same = 0;
        for (int t = 0; t < steps; t++)
        {
            float[] p = fp.LogP[t], q = i8.LogP[t];
            double kl = 0; int ap = 0, aq = 0;
            for (int i = 0; i < p.Length; i++)
            {
                double pi = Math.Exp(p[i]);
                if (pi > 0) kl += pi * (p[i] - q[i]);
                if (p[i] > p[ap]) ap = i;
                if (q[i] > q[aq]) aq = i;
            }
            kls[t] = Math.Max(0, kl);
            if (ap == aq) top1Same++;
        }
        // the prefill path (M > 1 rows: the int8 tensor-core GEMM, Int8ActGemm): prefixes of 32, 64, ... tokens prefilled whole, the last row's distribution
        var pre = new List<double>();
        int preTop1 = 0;
        try
        {
            for (int len = 32; len <= ids.Length; len += 32)
            {
                var prefix = ids.Take(len).ToArray();
                float[]? lp0 = null;
                foreach (bool int8 in new[] { false, true })
                {
                    SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv = int8;
                    dec.InvalidateCapturedSteps();
                    await dec.PrefillAsync(prefix, 0f);
                    var l = await dec.ReadLogitsAsync();
                    double mx = double.NegativeInfinity; for (int i = 0; i < l.Length; i++) if (l[i] > mx) mx = l[i];
                    double z = 0; for (int i = 0; i < l.Length; i++) z += Math.Exp(l[i] - mx);
                    double lz = mx + Math.Log(z);
                    var lp = new float[l.Length]; for (int i = 0; i < l.Length; i++) lp[i] = (float)(l[i] - lz);
                    if (!int8) { lp0 = lp; continue; }
                    double kl = 0; int a0 = 0, a1 = 0;
                    for (int i = 0; i < lp.Length; i++) { double pi = Math.Exp(lp0![i]); if (pi > 0) kl += pi * (lp0[i] - lp[i]); if (lp0[i] > lp0[a0]) a0 = i; if (lp[i] > lp[a1]) a1 = i; }
                    pre.Add(Math.Max(0, kl)); if (a0 == a1) preTop1++;
                }
            }
        }
        finally { SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv = saved; dec.InvalidateCapturedSteps(); }
        string preNote = pre.Count == 0 ? "" : $"; PREFILL (int8 tensor-core GEMM, {pre.Count} prefixes of 32..{32 * pre.Count} tokens, last row): KL mean {pre.Average():E2}, max {pre.Max():E2}, top-1 same {preTop1} of {pre.Count}";
        var sorted = kls.OrderBy(v => v).ToArray();
        string report = $"{modelTag} ({Path.GetFileName(path)}), {steps} teacher-forced steps, int8 activations vs fp32: KL mean {kls.Average():E2}, " +
            $"median {sorted[steps / 2]:E2}, p99 {sorted[(int)(steps * 0.99)]:E2}, max {sorted[^1]:E2}; top-1 same {100.0 * top1Same / steps:F1}%; " +
            $"perplexity fp32 {Math.Exp(fp.Nll / steps):F4}, int8 {Math.Exp(i8.Nll / steps):F4} ({100.0 * (Math.Exp((i8.Nll - fp.Nll) / steps) - 1):+0.000;-0.000}%)" + preNote;
        Console.WriteLine("[Int8ActKLD] " + report);
        return report;
    });
}
