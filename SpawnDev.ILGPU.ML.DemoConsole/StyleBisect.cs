using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.DemoConsole;

/// <summary>
/// CPU-vs-CUDA per-node bisection for the CPU-backend style-transfer correctness bug.
/// CUDA = ground truth (matches ORT 1.24.3). Runs the same style model on CUDA and the
/// ILGPU CPU backend with GraphExecutor.CapturedOutputs enabled, then diffs each node's
/// captured fingerprint to find the FIRST node where the CPU backend diverges.
///
/// Invoke: dotnet run --project SpawnDev.ILGPU.ML.DemoConsole -- STYLEBISECT [style-model-name]
/// (this is an investigation diagnostic, NOT a PMT-substitute test runner.)
/// </summary>
public static class StyleBisect
{
    // The Demo's wwwroot holds the models + ORT reference I/O used by the reference tests.
    const string WwwRoot = @"D:\users\tj\Projects\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML.Demo\wwwroot";

    static readonly string[] StyleModels =
        { "style-pointilism", "style-rain-princess", "style-udnie", "style-mosaic", "style-candy" };

    public static async Task Run(string[] args)
    {
        // STYLECPUCHECK mode: run the EXACT production reference-assertion path
        // (CreateFromFile + RunAsync + CompareOnGpuAsync vs ORT, tol 5.0) on the CPU
        // backend in isolation for every style model. Mirrors Reference_Style*_MatchesOnnxRuntime.
        if (args.Length > 1 && args[1] == "CPUCHECK")
        {
            await CpuCheckAll();
            return;
        }

        // INSPECTORCHECK mode: run the Model Inspector demo's exact path
        // (ModelInspectorHelper.Inspect + CheckCompatibility) against every model file on disk.
        // Fast dev-iteration bug finder before the formal [TestMethod] suite + PMT.
        if (args.Length > 1 && args[1] == "INSPECTORCHECK")
        {
            InspectorCheckAll();
            return;
        }

        // STREAMCHECK mode: exercise InspectAsync(Stream) — real SafeTensors vs byte[], the
        // header-only proof (data section throws), and a non-seekable ONNX fallback.
        if (args.Length > 1 && args[1] == "STREAMCHECK")
        {
            await StreamCheck();
            return;
        }

        string modelName = args.Length > 1 ? args[1] : "style-pointilism";
        string modelPath = Path.Combine(WwwRoot, "models", modelName, "model.onnx");
        string inputPath = Path.Combine(WwwRoot, "references", modelName, "cat_input_nchw.bin");
        string expectedPath = Path.Combine(WwwRoot, "references", modelName, "cat_output_nchw.bin");

        if (!File.Exists(modelPath)) { Console.WriteLine($"[StyleBisect] model not found: {modelPath}"); return; }

        var modelBytes = File.ReadAllBytes(modelPath);
        var inputFloats = ReadFloats(inputPath);
        var expected = File.Exists(expectedPath) ? ReadFloats(expectedPath) : Array.Empty<float>();

        Console.WriteLine($"[StyleBisect] model={modelName} bytes={modelBytes.Length} inputElems={inputFloats.Length} expectedElems={expected.Length}");

        // Run CUDA (ground truth) then CPU. Each returns ordered per-node captured fingerprints.
        var cuda = await RunBackend("CUDA", modelBytes, inputFloats);
        if (cuda == null) { Console.WriteLine("[StyleBisect] CUDA unavailable — cannot bisect."); return; }
        var cpu = await RunBackend("CPU", modelBytes, inputFloats);
        if (cpu == null) { Console.WriteLine("[StyleBisect] CPU run failed."); return; }

        // Final-output mean error vs ORT expected (captured covers first N elems of final node).
        if (expected.Length > 0)
        {
            Console.WriteLine();
            ReportFinalVsExpected("CUDA", cuda, expected);
            ReportFinalVsExpected("CPU ", cpu, expected);
        }

        // ── Per-node bisection table ──
        var keys = cuda.Keys.Intersect(cpu.Keys)
            .OrderBy(k => k, StringComparer.Ordinal).ToList();
        Console.WriteLine();
        Console.WriteLine($"[StyleBisect] {keys.Count} common captured nodes. Per-node CPU-vs-CUDA diff:");
        Console.WriteLine($"{"node",-52} {"cudaMean",12} {"cudaAbsMax",12} {"cpuMean",12} {"cpuAbsMax",12} {"maxAbsDiff",12} {"meanAbsDiff",12}");

        string? firstDivergent = null;
        foreach (var k in keys)
        {
            var a = cuda[k]; var b = cpu[k];
            int n = Math.Min(a.Length, b.Length);
            double cuMean = 0, cuAbsMax = 0, cpMean = 0, cpAbsMax = 0, maxDiff = 0, sumDiff = 0;
            bool anyNan = false;
            for (int i = 0; i < n; i++)
            {
                float av = a[i], bv = b[i];
                if (float.IsNaN(bv) || float.IsNaN(av) || float.IsInfinity(bv) || float.IsInfinity(av)) anyNan = true;
                cuMean += av; cpMean += bv;
                cuAbsMax = Math.Max(cuAbsMax, Math.Abs(av));
                cpAbsMax = Math.Max(cpAbsMax, Math.Abs(bv));
                double d = Math.Abs((double)av - bv);
                maxDiff = Math.Max(maxDiff, d);
                sumDiff += d;
            }
            cuMean /= n; cpMean /= n;
            double meanDiff = sumDiff / n;

            // Divergence threshold: a node has diverged when the per-element max abs diff is
            // large relative to the magnitude of the values at that node (avoids flagging
            // tiny float rounding). 1e-2 absolute OR >1% of cudaAbsMax.
            bool diverged = anyNan || (maxDiff > 1e-2 && maxDiff > 0.01 * Math.Max(1e-6, cuAbsMax));
            if (diverged && firstDivergent == null) firstDivergent = k;

            string flag = diverged ? (firstDivergent == k ? "  <== FIRST DIVERGENT" : "  <== diverged") : "";
            if (anyNan) flag += " [NaN/Inf]";
            Console.WriteLine($"{Trunc(k,52),-52} {cuMean,12:G5} {cuAbsMax,12:G5} {cpMean,12:G5} {cpAbsMax,12:G5} {maxDiff,12:G5} {meanDiff,12:G5}{flag}");
        }

        Console.WriteLine();
        if (firstDivergent != null)
        {
            Console.WriteLine($"[StyleBisect] FIRST DIVERGENT NODE: {firstDivergent}");
            DumpFirst8(cuda, cpu, firstDivergent);
        }
        else
        {
            Console.WriteLine("[StyleBisect] No node diverged beyond threshold — CPU matches CUDA per-node. " +
                "Divergence may be in elements beyond the capture window, or in the final readback/postprocess.");
        }
    }

    /// <summary>
    /// Run the production reference-assertion path on CPU for every style model, in isolation.
    /// Exactly mirrors RunReferenceComparisonGpu: CreateFromFile + RunAsync + CompareOnGpuAsync
    /// vs ORT expected at the test's tol=5.0. Reports PASS/FAIL per model.
    /// </summary>
    static async Task CpuCheckAll()
    {
        Console.WriteLine("[StyleCpuCheck] Running production CPU reference assertion (tol meanErr<=5.0) per style model:");
        foreach (var m in StyleModels)
        {
            string modelPath = Path.Combine(WwwRoot, "models", m, "model.onnx");
            string inputPath = Path.Combine(WwwRoot, "references", m, "cat_input_nchw.bin");
            string expectedPath = Path.Combine(WwwRoot, "references", m, "cat_output_nchw.bin");
            if (!File.Exists(modelPath) || !File.Exists(inputPath) || !File.Exists(expectedPath))
            {
                Console.WriteLine($"  {m,-22} SKIP (missing files)");
                continue;
            }
            Context? context = null; Accelerator? acc = null;
            try
            {
                context = MLContext.CreateContext();
                acc = context.CreateCPUAccelerator(0);
                var modelBytes = File.ReadAllBytes(modelPath);
                var inputFloats = ReadFloats(inputPath);
                var expected = ReadFloats(expectedPath);

                using var session = InferenceSession.CreateFromFile(acc, modelBytes);
                using var inBuf = acc.Allocate1D(inputFloats);
                var inputTensor = new Tensor(inBuf.View, new[] { 1, 3, 224, 224 });
                var outputs = await session.RunAsync(new Dictionary<string, Tensor>
                {
                    [session.InputNames[0]] = inputTensor
                });
                var output = outputs[session.OutputNames[0]];
                int elems = Math.Min(output.ElementCount, expected.Length);
                using var expectedBuf = acc.Allocate1D(expected);
                var ew = new ElementWiseKernels(acc);
                var (meanErr, maxErr) = await ew.CompareOnGpuAsync(
                    output.Data.SubView(0, elems), expectedBuf.View.SubView(0, elems), elems);
                bool pass = meanErr <= 5.0f;
                Console.WriteLine($"  {m,-22} {(pass ? "PASS" : "FAIL")}  meanErr={meanErr:F4} maxErr={maxErr:F4} (elems={elems})");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  {m,-22} ERROR {ex.GetType().Name}: {ex.Message}");
            }
            finally
            {
                try { acc?.Dispose(); } catch { }
                try { context?.Dispose(); } catch { }
            }
        }
    }

    /// <summary>
    /// Run the Model Inspector demo's exact path against every model file under wwwroot.
    /// Reports Inspect() + CheckCompatibility() success/failure per file. Pure parsing — no GPU.
    /// </summary>
    static void InspectorCheckAll()
    {
        var roots = new[]
        {
            Path.Combine(WwwRoot, "models"),
            Path.Combine(WwwRoot, "test-models"),
        };
        var exts = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            { ".onnx", ".tflite", ".gguf", ".safetensors" };

        var files = new List<string>();
        foreach (var root in roots)
        {
            if (!Directory.Exists(root)) continue;
            files.AddRange(Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories)
                .Where(f => exts.Contains(Path.GetExtension(f))));
        }
        files.Sort(StringComparer.OrdinalIgnoreCase);

        Console.WriteLine($"[InspectorCheck] {files.Count} model files. Inspect() + CheckCompatibility():");
        Console.WriteLine($"{"file",-46} {"fmt",-7} {"nodes",6} {"in/out",7} {"ops",4} {"params",10} {"inspect",8} {"compat",18}");

        foreach (var f in files)
        {
            var rel = f.Replace(WwwRoot, "").TrimStart('\\', '/');
            byte[] bytes;
            try { bytes = File.ReadAllBytes(f); }
            catch (Exception ex) { Console.WriteLine($"{Trunc(rel,46),-46} READ-ERROR {ex.Message}"); continue; }

            string fmt = "?";
            try { fmt = SpawnDev.ILGPU.ML.InferenceSession.DetectModelFormat(bytes).ToString(); } catch { }

            string inspectStatus, compatStatus;
            int nodes = -1, ops = -1; long prms = -1; int nin = -1, nout = -1;
            try
            {
                var r = SpawnDev.ILGPU.ML.Onnx.ModelInspectorHelper.Inspect(bytes);
                nodes = r.NodeCount; ops = r.Operators.Length; prms = r.TotalParameters;
                nin = r.Inputs.Length; nout = r.Outputs.Length;
                bool sane = !string.IsNullOrEmpty(r.GraphName) && (r.NodeCount > 0 || r.InitializerCount > 0);
                inspectStatus = sane ? "OK" : "EMPTY";
            }
            catch (Exception ex) { inspectStatus = "THROW"; compatStatus = $"(skipped: {ex.GetType().Name})"; PrintRow(rel, fmt, nodes, nin, nout, ops, prms, inspectStatus, compatStatus); continue; }

            try
            {
                var c = SpawnDev.ILGPU.ML.Onnx.ModelInspectorHelper.CheckCompatibility(bytes);
                compatStatus = $"{c.CompatibilityPercent:F0}% {c.SupportedOps.Length}/{c.TotalOpsUsed}";
            }
            catch (Exception ex) { compatStatus = $"THROW:{ex.GetType().Name}"; }

            PrintRow(rel, fmt, nodes, nin, nout, ops, prms, inspectStatus, compatStatus);
        }
    }

    static async Task StreamCheck()
    {
        // 1) Real SafeTensors: InspectAsync(stream) == Inspect(byte[]).
        var stPath = Path.Combine(WwwRoot, "test-models", "test.safetensors");
        if (File.Exists(stPath))
        {
            var bytes = File.ReadAllBytes(stPath);
            var fb = SpawnDev.ILGPU.ML.Onnx.ModelInspectorHelper.Inspect(bytes);
            using var ms = new MemoryStream(bytes);
            var fs = await SpawnDev.ILGPU.ML.Onnx.ModelInspectorHelper.InspectAsync(ms);
            bool ok = fs.InitializerCount == fb.InitializerCount && fs.TotalParameters == fb.TotalParameters && fs.NodeCount == fb.NodeCount;
            Console.WriteLine($"[StreamCheck] SafeTensors stream==bytes: {(ok ? "OK" : "MISMATCH")} (tensors {fs.InitializerCount}/{fb.InitializerCount}, params {fs.TotalParameters}/{fb.TotalParameters})");
        }

        // 2) Header-only proof: 4 tensors claiming ~1GB data, served by a stream that throws past header.
        var entries = new List<string>(); long off = 0; const long per = 256L * 1024 * 1024;
        for (int i = 0; i < 4; i++) { long end = off + per; entries.Add($"\"layer_{i}.weight\":{{\"dtype\":\"F32\",\"shape\":[8192,8192],\"data_offsets\":[{off},{end}]}}"); off = end; }
        var json = "{" + string.Join(",", entries) + "}";
        var jb = System.Text.Encoding.UTF8.GetBytes(json);
        var header = new byte[8 + jb.Length];
        BitConverter.GetBytes((long)jb.Length).CopyTo(header, 0); jb.CopyTo(header, 8);
        using (var ts = new HeaderThrowStream(header, 8 + off))
        {
            try
            {
                var r = await SpawnDev.ILGPU.ML.Onnx.ModelInspectorHelper.InspectAsync(ts);
                bool ok = r.InitializerCount == 4 && r.NodeCount == 0 && r.TotalParameters == 4L * 8192 * 8192 && ts.PastHeader == 0;
                Console.WriteLine($"[StreamCheck] Header-only proof: {(ok ? "OK" : "FAIL")} (header={header.Length}B, tensors={r.InitializerCount}, params={r.TotalParameters}, readPastHeader={ts.PastHeader})");
            }
            catch (Exception ex) { Console.WriteLine($"[StreamCheck] Header-only proof THREW: {ex.GetType().Name}: {ex.Message}"); }
        }

        // 2b) GGUF seekable header-only: InspectAsync(stream) == Inspect(byte[]).
        var ggufPath = Path.Combine(WwwRoot, "test-models", "test.gguf");
        if (File.Exists(ggufPath))
        {
            var bytes = File.ReadAllBytes(ggufPath);
            var fb = SpawnDev.ILGPU.ML.Onnx.ModelInspectorHelper.Inspect(bytes);
            using var ms = new MemoryStream(bytes); // seekable
            var fs = await SpawnDev.ILGPU.ML.Onnx.ModelInspectorHelper.InspectAsync(ms);
            bool ok = fs.InitializerCount == fb.InitializerCount && fs.NodeCount == fb.NodeCount && fs.TotalParameters == fb.TotalParameters;
            Console.WriteLine($"[StreamCheck] GGUF seekable stream==bytes: {(ok ? "OK" : "MISMATCH")} (tensors {fs.NodeCount}/{fb.NodeCount}, params {fs.TotalParameters}/{fb.TotalParameters})");
        }

        // 3) Non-seekable ONNX fallback == byte[].
        var onnx = Path.Combine(WwwRoot, "models", "squeezenet", "model.onnx");
        if (File.Exists(onnx))
        {
            var bytes = File.ReadAllBytes(onnx);
            var fb = SpawnDev.ILGPU.ML.Onnx.ModelInspectorHelper.Inspect(bytes);
            using var fwd = new FwdOnlyStream(bytes);
            var fs = await SpawnDev.ILGPU.ML.Onnx.ModelInspectorHelper.InspectAsync(fwd);
            bool ok = fs.NodeCount == fb.NodeCount && fs.Operators.Length == fb.Operators.Length && fs.TotalParameters == fb.TotalParameters;
            Console.WriteLine($"[StreamCheck] ONNX non-seekable fallback==bytes: {(ok ? "OK" : "MISMATCH")} (nodes {fs.NodeCount}/{fb.NodeCount}, ops {fs.Operators.Length}/{fb.Operators.Length})");
        }
    }

    sealed class HeaderThrowStream : Stream
    {
        readonly byte[] _h; readonly long _len; long _pos;
        public long PastHeader { get; private set; }
        public HeaderThrowStream(byte[] h, long len) { _h = h; _len = len; }
        public override int Read(byte[] b, int o, int c)
        {
            if (_pos >= _h.Length) { PastHeader += c; throw new IOException("read past header"); }
            int n = (int)Math.Min(c, _h.Length - _pos); Array.Copy(_h, _pos, b, o, n); _pos += n; return n;
        }
        public override bool CanRead => true; public override bool CanSeek => true; public override bool CanWrite => false;
        public override long Length => _len; public override long Position { get => _pos; set => _pos = value; }
        public override long Seek(long o, SeekOrigin or) => _pos; public override void Flush() { }
        public override void SetLength(long v) => throw new NotSupportedException();
        public override void Write(byte[] b, int o, int c) => throw new NotSupportedException();
    }

    sealed class FwdOnlyStream : Stream
    {
        readonly byte[] _d; int _pos; public FwdOnlyStream(byte[] d) { _d = d; }
        public override int Read(byte[] b, int o, int c) { int n = Math.Min(c, _d.Length - _pos); if (n <= 0) return 0; Array.Copy(_d, _pos, b, o, n); _pos += n; return n; }
        public override bool CanRead => true; public override bool CanSeek => false; public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException(); public override long Position { get => _pos; set => throw new NotSupportedException(); }
        public override long Seek(long o, SeekOrigin or) => throw new NotSupportedException(); public override void Flush() { }
        public override void SetLength(long v) => throw new NotSupportedException();
        public override void Write(byte[] b, int o, int c) => throw new NotSupportedException();
    }

    static void PrintRow(string rel, string fmt, int nodes, int nin, int nout, int ops, long prms, string inspect, string compat)
    {
        string io = (nin < 0 ? "-" : nin.ToString()) + "/" + (nout < 0 ? "-" : nout.ToString());
        string p = prms < 0 ? "-" : prms.ToString("N0");
        Console.WriteLine($"{Trunc(rel,46),-46} {Trunc(fmt,7),-7} {(nodes<0?"-":nodes.ToString()),6} {io,7} {(ops<0?"-":ops.ToString()),4} {p,10} {inspect,8} {compat,18}");
    }

    static void DumpFirst8(Dictionary<string, float[]> cuda, Dictionary<string, float[]> cpu, string key)
    {
        var a = cuda[key]; var b = cpu[key];
        int n = Math.Min(8, Math.Min(a.Length, b.Length));
        Console.WriteLine($"  first {n}: CUDA=[{string.Join(",", a.Take(n).Select(v => v.ToString("F5")))}]");
        Console.WriteLine($"           CPU =[{string.Join(",", b.Take(n).Select(v => v.ToString("F5")))}]");
    }

    static void ReportFinalVsExpected(string name, Dictionary<string, float[]> caps, float[] expected)
    {
        var lastKey = caps.Keys.OrderBy(k => k, StringComparer.Ordinal).LastOrDefault();
        if (lastKey == null) return;
        var vals = caps[lastKey];
        int n = Math.Min(vals.Length, expected.Length);
        double sum = 0, max = 0; bool nan = false;
        for (int i = 0; i < n; i++)
        {
            if (float.IsNaN(vals[i]) || float.IsInfinity(vals[i])) nan = true;
            double d = Math.Abs((double)vals[i] - expected[i]);
            sum += d; max = Math.Max(max, d);
        }
        Console.WriteLine($"[StyleBisect] {name} final node '{Trunc(lastKey,40)}' vs ORT: meanErr={sum / n:F4} maxErr={max:F4}{(nan ? " [NaN/Inf!]" : "")} (over {n} elems)");
    }

    static async Task<Dictionary<string, float[]>?> RunBackend(string name, byte[] modelBytes, float[] inputFloats)
    {
        Context? context = null;
        Accelerator? acc = null;
        try
        {
            if (name == "CUDA")
            {
                context = MLContext.Create().ToContext();
                var devs = context.GetCudaDevices();
                if (devs.Count == 0) { context.Dispose(); return null; }
                acc = devs[0].CreateCudaAccelerator(context);
            }
            else
            {
                context = MLContext.CreateContext();
                acc = context.CreateCPUAccelerator(0);
            }

            GraphExecutor.CapturedOutputs = new Dictionary<string, float[]>();
            GraphExecutor.CaptureMaxElements = 8192;

            using var session = InferenceSession.CreateFromFile(acc, modelBytes);
            using var inBuf = acc.Allocate1D(inputFloats);
            var inputTensor = new Tensor(inBuf.View, new[] { 1, 3, 224, 224 });
            var outputs = await session.RunAsync(new Dictionary<string, Tensor>
            {
                [session.InputNames[0]] = inputTensor
            });
            await acc.SynchronizeAsync();

            var snapshot = new Dictionary<string, float[]>(GraphExecutor.CapturedOutputs);
            Console.WriteLine($"[StyleBisect] {name}: captured {snapshot.Count} nodes, output='{session.OutputNames[0]}'");
            return snapshot;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[StyleBisect] {name} ERROR: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
        finally
        {
            GraphExecutor.CapturedOutputs = null;
            try { acc?.Dispose(); } catch { }
            try { context?.Dispose(); } catch { }
        }
    }

    static float[] ReadFloats(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var f = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, f, 0, bytes.Length);
        return f;
    }

    static string Trunc(string s, int len) => s.Length <= len ? s : s.Substring(0, len);
}
