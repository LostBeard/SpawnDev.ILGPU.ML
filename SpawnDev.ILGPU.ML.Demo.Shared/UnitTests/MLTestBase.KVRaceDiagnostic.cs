using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// INVESTIGATION HARNESS (not a PMT test): tight-loop repro for the intermittent CPU-backend
/// non-determinism in <c>GGUFDecodeKVCache_IncrementalMatchesFullRecompute</c>.
///
/// CPU kernel launches are SYNCHRONOUS + serialized (CPUAccelerator.Launch blocks on
/// finishedEventPerMultiprocessor.SignalAndWait under a SemaphoreSlim(1)), so a CopyFrom can NOT
/// race a producing kernel on CPU — the flake must be an INTRA-dispatch multi-thread data race or
/// an uninitialized-memory read. This harness discriminates WHICH path is non-deterministic by
/// running, per iteration: full-recompute twice (refA vs refB), decode twice (decA vs decB), and
/// decode vs reference (the actual test invariant).
///
/// Session + KV cache are built ONCE and reused (the race is per-dispatch, not per-build), so the
/// loop runs forwards only — fast enough for thousands of iterations. Env knobs:
///   KVRACE_BURN = background CPU-burner thread count (0 = none; default = ProcessorCount) to
///     perturb the scheduler the way a loaded PMT run does.
/// </summary>
public partial class MLTestBase
{
    public async Task DiagnoseKVDecodeRace(int iters)
    {
        const int embd = 256, vocab = 32, ffn = 320, ctx = 64;
        var bytes = BuildTinyQuantizedLlamaGGUF(embd, vocab, ffn, ctx, new Random(7));
        var model = GGUFParser.Parse(bytes);

        int nLayers = (int)model.BlockCount, nHeads = (int)model.AttentionHeadCount;
        int defNKV = (int)model.AttentionHeadCountKV; if (defNKV == 0) defNKV = nHeads;
        int defHd = embd / nHeads;
        var kvHeadsArr = new int[nLayers]; var hdArr = new int[nLayers];
        for (int L = 0; L < nLayers; L++)
        { var cfg = GGUFGraphBuilder.GetLayerAttnConfig(model, L, nHeads, defNKV, defHd); kvHeadsArr[L] = cfg.NKVHeads; hdArr[L] = cfg.HeadDim; }

        var seq = new float[] { 3, 9, 21, 5 };

        int burn = int.TryParse(Environment.GetEnvironmentVariable("KVRACE_BURN"), out var bb) ? bb : Environment.ProcessorCount;
        using var burnCts = new System.Threading.CancellationTokenSource();
        var burners = new List<Task>();
        for (int b = 0; b < burn; b++)
            burners.Add(Task.Run(() => { double x = 1.0001; var ct = burnCts.Token; while (!ct.IsCancellationRequested) { x = Math.Sin(x) + 1.0001; } GC.KeepAlive(x); }));

        int refRefMismatch = 0, decDecMismatch = 0, decRefMismatch = 0, crossIterRefMismatch = 0;
        float[]? canonicalRef = null;

        Console.WriteLine($"[KVRaceDiag:{BackendName}] iters={iters}, layers={nLayers}, nHeads={nHeads}, kvHeads={kvHeadsArr[0]}, hd={hdArr[0]}, seqLen={seq.Length}, burnerThreads={burn}, procs={Environment.ProcessorCount}");

        var (context, accelerator) = await CreateAcceleratorAsync();
        // Build the session + cache ONCE; reuse across iterations (kernels precompiled, pools warm).
        using var session = InferenceSession.CreateFromGGUF(accelerator, bytes);
        using var kv = new GGUFDecodeKVCache(accelerator, kvHeadsArr, hdArr, maxSeqLen: ctx);
        try
        {
            for (int it = 0; it < iters; it++)
            {
                float[] refA = await RunFull(session, accelerator, seq, vocab);
                float[] refB = await RunFull(session, accelerator, seq, vocab);
                float[] decA = await RunDecode(session, accelerator, kv, seq, vocab);
                float[] decB = await RunDecode(session, accelerator, kv, seq, vocab);

                if (FirstDivergence(refA, refB, seq.Length, vocab, out var rr))
                { refRefMismatch++; Console.WriteLine($"  it{it}: REF-vs-REF mismatch {rr}"); }
                if (FirstDivergence(decA, decB, seq.Length, vocab, out var dd))
                { decDecMismatch++; Console.WriteLine($"  it{it}: DEC-vs-DEC mismatch {dd}"); }
                if (FirstDivergence(decA, refA, seq.Length, vocab, out var dr))
                { decRefMismatch++; Console.WriteLine($"  it{it}: DEC-vs-REF mismatch {dr}"); }

                if (canonicalRef == null) canonicalRef = refA;
                else if (FirstDivergence(refA, canonicalRef, seq.Length, vocab, out var ci))
                { crossIterRefMismatch++; Console.WriteLine($"  it{it}: CROSS-ITER REF mismatch {ci}"); }

                if (it > 0 && it % 100 == 0)
                    Console.WriteLine($"  ...{it}/{iters} (rr={refRefMismatch} dd={decDecMismatch} dr={decRefMismatch} ci={crossIterRefMismatch})");
            }
        }
        finally
        {
            try { await accelerator.SynchronizeAsync(); } catch { }
            burnCts.Cancel();
            try { await Task.WhenAll(burners); } catch { }
            try { accelerator.Dispose(); } catch { }
            try { context.Dispose(); } catch { }
        }

        Console.WriteLine($"[KVRaceDiag:{BackendName}] DONE. ref-vs-ref={refRefMismatch}/{iters}, dec-vs-dec={decDecMismatch}/{iters}, dec-vs-ref={decRefMismatch}/{iters}, cross-iter-ref={crossIterRefMismatch}/{Math.Max(0, iters - 1)}");
        Console.WriteLine(refRefMismatch == 0 && decDecMismatch == 0 && decRefMismatch == 0 && crossIterRefMismatch == 0
            ? "[KVRaceDiag] NO non-determinism observed this run."
            : "[KVRaceDiag] NON-DETERMINISM REPRODUCED — see per-path counts above to localize.");
    }

    private static async Task<float[]> RunFull(InferenceSession session, Accelerator accelerator, float[] seq, int vocab)
    {
        using var inFull = accelerator.Allocate1D(seq);
        var outFull = await session.RunAsync(new Dictionary<string, Tensor>
        { ["input_ids"] = new Tensor(inFull.View, new[] { 1, seq.Length }, "input_ids") });
        var logitsT = outFull.TryGetValue("logits", out var lf) ? lf : outFull.Values.First();
        using var read = accelerator.Allocate1D<float>(seq.Length * vocab);
        await read.View.CopyFromAsync(logitsT.Data.SubView(0, seq.Length * vocab));
        await accelerator.SynchronizeAsync();
        return await read.CopyToHostAsync<float>(0, seq.Length * vocab);
    }

    private static async Task<float[]> RunDecode(InferenceSession session, Accelerator accelerator, GGUFDecodeKVCache kv, float[] seq, int vocab)
    {
        session.EnableGGUFDecode(kv); // resets DecodePastLen to 0; decode overwrites cache tokens 0..N-1
        var all = new float[seq.Length * vocab];
        for (int pos = 0; pos < seq.Length; pos++)
        {
            using var inTok = accelerator.Allocate1D(new[] { seq[pos] });
            var outStep = await session.RunDecodeStepAsync(new Dictionary<string, Tensor>
            { ["input_ids"] = new Tensor(inTok.View, new[] { 1, 1 }, "input_ids") });
            var stepT = outStep.TryGetValue("logits", out var ls) ? ls : outStep.Values.First();
            using var read = accelerator.Allocate1D<float>(vocab);
            await read.View.CopyFromAsync(stepT.Data.SubView(0, vocab));
            await accelerator.SynchronizeAsync();
            var stepLogits = await read.CopyToHostAsync<float>(0, vocab);
            Array.Copy(stepLogits, 0, all, pos * vocab, vocab);
        }
        return all;
    }

    /// <summary>Returns true if the two logit blocks diverge by argmax or beyond the test tolerance,
    /// reporting the first diverging (position, vocab) and the values.</summary>
    private static bool FirstDivergence(float[] a, float[] b, int positions, int vocab, out string detail)
    {
        for (int pos = 0; pos < positions; pos++)
        {
            int argA = 0, argB = 0;
            for (int v = 1; v < vocab; v++)
            {
                if (a[pos * vocab + v] > a[pos * vocab + argA]) argA = v;
                if (b[pos * vocab + v] > b[pos * vocab + argB]) argB = v;
            }
            if (argA != argB)
            { detail = $"@pos{pos}: argmax {argA} vs {argB} (a={a[pos * vocab + argA]:F5}, b={b[pos * vocab + argB]:F5})"; return true; }
            for (int v = 0; v < vocab; v++)
            {
                float av = a[pos * vocab + v], bv = b[pos * vocab + v];
                float tol = MathF.Max(2e-3f, MathF.Abs(av) * 2e-3f);
                if (MathF.Abs(av - bv) > tol)
                { detail = $"@pos{pos} v{v}: {av:F5} vs {bv:F5} (tol {tol:E1})"; return true; }
            }
        }
        detail = "";
        return false;
    }
}
