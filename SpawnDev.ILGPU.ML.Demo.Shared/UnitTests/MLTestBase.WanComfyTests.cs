using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.Wan;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// ComfyUI's repackaged Wan 2.2 files (Comfy-Org/Wan_2.2_ComfyUI_Repackaged, Apache-2.0) loaded through <see cref="Wan22Pipeline.LoadWeightsAsync"/>, which rewrites
/// ComfyUI's layout into ours (<see cref="WanComfyFormat"/>), must equal our own export of the same weights. Desktop; the files in WAN_DIR\comfy.
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// The VAE: every fp16 tensor (the convolutions permuted from torch's [co, ci, k...] to [co, taps, ci], the biases) bit-identical to our export, the RMS-norm
    /// gammas (fp16 there, fp32 in ours) within fp16 rounding, the latent normalisation exactly the reference's constants, and the same set of names.
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Wan_ComfyVae_LoadsAsOurExport() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL))
            throw new UnsupportedTestException("reads the files from disk: a desktop GPU");
        string comfyDir = Path.Combine(WanDir, "comfy");
        if (!File.Exists(Path.Combine(comfyDir, "wan2.2_vae.safetensors")) || !File.Exists(Path.Combine(WanDir, "wan22_vae_f16.safetensors")))
            throw new UnsupportedTestException($"needs {comfyDir}\\wan2.2_vae.safetensors (ComfyUI's) and our wan22_vae_f16.safetensors in {WanDir}");
        using var comfy = await Wan22Pipeline.LoadWeightsAsync(accelerator, new LocalFolderModelSource(comfyDir), "", "wan2.2_vae.safetensors");
        using var ours = await Wan22Pipeline.LoadWeightsAsync(accelerator, new LocalFolderModelSource(WanDir), "", "wan22_vae_f16.safetensors");
        var a = ours.Names.OrderBy(n => n, StringComparer.Ordinal).ToArray();
        var b = comfy.Names.OrderBy(n => n, StringComparer.Ordinal).ToArray();
        if (!a.SequenceEqual(b)) throw new Exception($"names differ: only ours {string.Join(", ", a.Except(b).Take(5))}; only ComfyUI's {string.Join(", ", b.Except(a).Take(5))}");
        int f16Exact = 0, f32Close = 0;
        double worstGamma = 0;
        foreach (var name in a)
        {
            if (!ours.Shape(name).SequenceEqual(comfy.Shape(name))) throw new Exception($"{name}: shape [{string.Join(",", ours.Shape(name))}] vs [{string.Join(",", comfy.Shape(name))}]");
            if (ours.IsF16(name))
            {
                if (!comfy.IsF16(name)) throw new Exception($"{name}: fp16 in ours, not in the adapted ComfyUI file");
                var x = await ours.F16(name).CopyToHostAsync(); var y = await comfy.F16(name).CopyToHostAsync();
                for (int i = 0; i < x.Length; i++)
                    if (x[i].RawValue != y[i].RawValue) throw new Exception($"{name}[{i}]: {(float)x[i]} vs {(float)y[i]} (fp16 bits differ)");
                f16Exact++;
            }
            else if (ours.IsF32(name))
            {
                if (!comfy.IsF32(name)) throw new Exception($"{name}: fp32 in ours, not in the adapted ComfyUI file");
                var x = await ours.F32(name).CopyToHostAsync(); var y = await comfy.F32(name).CopyToHostAsync();
                bool latent = name.StartsWith("latents_", StringComparison.Ordinal);
                for (int i = 0; i < x.Length; i++)
                {
                    double rel = Math.Abs(x[i] - y[i]) / Math.Max(1e-30, Math.Abs(x[i]));
                    if (latent ? x[i] != y[i] : rel > 1.0 / 2048) throw new Exception($"{name}[{i}]: {x[i]} vs {y[i]}");
                    if (!latent) worstGamma = Math.Max(worstGamma, rel);
                }
                f32Close++;
            }
            else throw new Exception($"{name}: neither fp16 nor fp32 in our export");
        }
        return $"{a.Length} tensors: {f16Exact} fp16 bit-identical, {f32Close} fp32 (gammas within fp16 rounding, worst rel {worstGamma:E1}; latent mean / std exact)";
    });

    /// <summary>
    /// The DiT: ComfyUI's fp16 file adapted (q / k / v fused, fp16 -> bf16, the reshapes) against our bf16 export of the same reference weights, read from disk
    /// tensor by tensor (both are 10 GB: only the adapted one is on the device). Both are roundings of the reference's fp32 weights: ours fp32 -> bf16 directly,
    /// ComfyUI's fp32 -> fp16 (a finer grid) -> bf16 here, and that double rounding can land one bf16 step away (found 2026-11-12: e.g. bits BC73 vs BC74). So
    /// every value must be bit-identical or one bf16 step away with the same sign, or, below fp16's normal range (|x| &lt; 6.1e-5), within fp16's subnormal step.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Wan_ComfyDit_LoadsAsOurExport() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL))
            throw new UnsupportedTestException("reads the files from disk: a desktop GPU");
        string comfyDir = Path.Combine(WanDir, "comfy"), oursPath = Path.Combine(WanDir, "wan22_ti2v_5b_dit_bf16.safetensors");
        if (!File.Exists(Path.Combine(comfyDir, "wan2.2_ti2v_5B_fp16.safetensors")) || !File.Exists(oursPath))
            throw new UnsupportedTestException($"needs {comfyDir}\\wan2.2_ti2v_5B_fp16.safetensors (ComfyUI's) and our wan22_ti2v_5b_dit_bf16.safetensors in {WanDir}");
        using var comfy = await Wan22Pipeline.LoadWeightsAsync(accelerator, new LocalFolderModelSource(comfyDir), "", "wan2.2_ti2v_5B_fp16.safetensors");
        using var ours = SpawnDev.ILGPU.ML.SafeTensors.SafeTensorsFileReader.Open(oursPath);
        var oursByName = ours.Tensors.ToDictionary(t => t.Name);
        var a = oursByName.Keys.OrderBy(n => n, StringComparer.Ordinal).ToArray();
        var b = comfy.Names.OrderBy(n => n, StringComparer.Ordinal).ToArray();
        if (!a.SequenceEqual(b)) throw new Exception($"names differ: only ours {string.Join(", ", a.Except(b).Take(5))}; only ComfyUI's {string.Join(", ", b.Except(a).Take(5))}");
        long values = 0, subnormal = 0, oneStep = 0;
        foreach (var name in a)
        {
            var t = oursByName[name];
            if (!t.Shape.SequenceEqual(comfy.Shape(name))) throw new Exception($"{name}: shape [{string.Join(",", t.Shape)}] vs [{string.Join(",", comfy.Shape(name))}]");
            if (t.DType != "BF16") throw new Exception($"{name}: our export is {t.DType}, expected BF16");
            var raw = new byte[t.DataLength];
            ours.ReadBytes(t, 0, raw);
            var yArr = await comfy.BF16(name).CopyToHostAsync();
            var (st, sub) = CompareBf16ThroughFp16(name, raw, yArr);
            oneStep += st; subnormal += sub;
            values += raw.Length / 2;
        }
        return $"{a.Length} tensors, {values:N0} values: {values - oneStep - subnormal:N0} bit-identical, {oneStep:N0} one bf16 step away ({100.0 * oneStep / values:F3}%: the double rounding through fp16), {subnormal:N0} below fp16's normal range";
    });

    // ---- umT5 (ComfyUI's umt5_xxl_fp16 / umt5_xxl_fp8_e4m3fn_scaled: Hugging Face T5 names; UmT5Encoder rewrites each layer as it streams) ----

    // the F8_E4M3 value of a byte, written independently of the kernel (powers, not bit assembly); NaN for S.1111.111
    private static float WanF8Reference(byte b)
    {
        int e = (b >> 3) & 15, m = b & 7;
        if ((b & 0x7F) == 0x7F) return float.NaN;
        double v = e == 0 ? m / 8.0 * Math.Pow(2, -6) : (1 + m / 8.0) * Math.Pow(2, e - 7);
        return (float)((b & 0x80) != 0 ? -v : v);
    }
    // float -> bf16 bits, round to nearest even
    private static ushort WanBf16Rne(float f) { uint u = BitConverter.SingleToUInt32Bits(f); return (ushort)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16); }

    /// <summary>
    /// The F8_E4M3 decode kernel (<see cref="WanComfyFormat.DecodeF8E4M3"/>): every byte value at every position of a word, times several scales, must be the bf16
    /// rounding (to nearest even) of value * scale, bit for bit (the subnormals, both zeros, the largest value 448, NaN stays NaN). The count is not a multiple of 4.
    /// Red check: the exponent bias off by one fails every normal value.
    /// </summary>
    [TestMethod]
    public async Task Wan_F8E4M3_Decode_AllBytes_Exact() => await RunTest(async accelerator =>
    {
        const int N = 1027;
        var bytes = new byte[(N + 3) / 4 * 4];
        for (int i = 0; i < N; i++) bytes[i] = (byte)(i * 37 + 11 + (i >> 8));          // all 256 values in each run of 256, shifted by one per run: each value at each of the 4 word positions
        using var packed = accelerator.Allocate1D(System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(bytes).ToArray());
        using var dst = accelerator.Allocate1D<BFloat16>(N);
        int checkedValues = 0;
        foreach (float scale in new[] { 1f, 0.0123456f, -3.7182817f, 1.0e-3f })
        {
            WanComfyFormat.DecodeF8E4M3(accelerator, packed.View, dst.View, scale);
            await accelerator.SynchronizeAsync();
            var got = System.Runtime.InteropServices.MemoryMarshal.Cast<BFloat16, ushort>(await dst.View.CopyToHostAsync()).ToArray();
            for (int i = 0; i < N; i++)
            {
                float want = WanF8Reference(bytes[i]) * scale;
                if (float.IsNaN(want))
                {
                    if (!((got[i] & 0x7F80) == 0x7F80 && (got[i] & 0x7F) != 0)) throw new Exception($"byte {bytes[i]:X2} x {scale}: NaN expected, got bits {got[i]:X4}");
                }
                else if (got[i] != WanBf16Rne(want)) throw new Exception($"byte {bytes[i]:X2} at {i} x {scale}: bits {got[i]:X4}, expected {WanBf16Rne(want):X4} ({want:E6})");
                checkedValues++;
            }
        }
        return $"{checkedValues} values (256 bytes x 4 word positions x 4 scales) bit-exact";
    });

    // the tensors of one layer (our names) of a umT5 file as bf16 device weights: our export as stored, or a Hugging Face layout file through WanComfyFormat
    private static async Task<GpuWeights> WanT5LayerAsync(Accelerator accelerator, string path, Func<string, bool> oursFilter)
    {
        using var empty = System.Text.Json.JsonDocument.Parse("{}");
        bool hf;
        using (var r = SpawnDev.ILGPU.ML.SafeTensors.SafeTensorsFileReader.Open(path)) hf = WanComfyFormat.IsComfyT5(r.Tensors.Select(t => t.Name));
        await using var stream = File.OpenRead(path);
        if (!hf) return await GpuWeights.LoadAsync(accelerator, stream, empty.RootElement, path, oursFilter);
        var w = await GpuWeights.LoadAsync(accelerator, stream, empty.RootElement, path, n => WanComfyFormat.T5OurName(n) is { } o && o != "token_embedding.weight" && oursFilter(o));
        try { await WanComfyFormat.AdaptT5Async(accelerator, w); return w; }
        catch { w.Dispose(); throw; }
    }
    private static IEnumerable<(string Label, Func<string, bool> Filter)> WanT5Layers()
    {
        for (int b = 0; b < UmT5Encoder.DefaultLayers; b++) { string p = $"blocks.{b}."; yield return (p, n => n.StartsWith(p, StringComparison.Ordinal)); }
        yield return ("norm.weight", n => n == "norm.weight");
    }

    /// <summary>
    /// ComfyUI's fp16 umT5 against our bf16 export of the reference's (bf16) encoder, layer by layer (all 24 and the final norm) and the whole embedding table: the same
    /// names and shapes, q / k / v fused in that order, and every value bit-identical, one bf16 step away (when their fp16 was rounded from a wider master) or below
    /// fp16's normal range (see <see cref="CompareBf16ThroughFp16"/>). The counts say which.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Wan_ComfyUmT5_Fp16_LayersAsOurExport() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL))
            throw new UnsupportedTestException("reads the files from disk: a desktop GPU");
        string comfyPath = Path.Combine(WanDir, "comfy", "umt5_xxl_fp16.safetensors"), oursPath = Path.Combine(WanDir, "umt5_xxl_enc_bf16.safetensors");
        if (!File.Exists(comfyPath) || !File.Exists(oursPath)) throw new UnsupportedTestException($"needs {comfyPath} (ComfyUI's) and {oursPath} (ours)");
        long values = 0, oneStep = 0, subnormal = 0; int tensors = 0;
        foreach (var (label, filter) in WanT5Layers())
        {
            using var ours = await WanT5LayerAsync(accelerator, oursPath, filter);
            using var comfy = await WanT5LayerAsync(accelerator, comfyPath, filter);
            var a = ours.Names.OrderBy(n => n, StringComparer.Ordinal).ToArray();
            var b = comfy.Names.OrderBy(n => n, StringComparer.Ordinal).ToArray();
            if (a.Length == 0 || !a.SequenceEqual(b)) throw new Exception($"{label}: names differ: ours [{string.Join(", ", a)}]; ComfyUI's adapted [{string.Join(", ", b)}]");
            foreach (var name in a)
            {
                if (!ours.Shape(name).SequenceEqual(comfy.Shape(name))) throw new Exception($"{name}: shape [{string.Join(",", ours.Shape(name))}] vs [{string.Join(",", comfy.Shape(name))}]");
                var x = await ours.BF16(name).CopyToHostAsync(); var y = await comfy.BF16(name).CopyToHostAsync();
                var (st, sub) = CompareBf16ThroughFp16(name, System.Runtime.InteropServices.MemoryMarshal.AsBytes(x.AsSpan()).ToArray(), y);
                oneStep += st; subnormal += sub; values += x.Length; tensors++;
            }
        }
        // the embedding table, straight from the two files (the encoder reads its rows from the file): fp16 rows rounded to bf16 against our bf16 rows
        long rows = 0;
        using (var ro = SpawnDev.ILGPU.ML.SafeTensors.SafeTensorsFileReader.Open(oursPath))
        using (var rc = SpawnDev.ILGPU.ML.SafeTensors.SafeTensorsFileReader.Open(comfyPath))
        {
            var to = ro.Get("token_embedding.weight"); var tc = rc.Get("shared.weight");
            if (!to.Shape.SequenceEqual(tc.Shape) || to.DType != "BF16" || tc.DType != "F16") throw new Exception($"embedding: ours {to.DType} [{string.Join(",", to.Shape)}], ComfyUI's {tc.DType} [{string.Join(",", tc.Shape)}]");
            const int RowsPerChunk = 4096; int rowBytes = UmT5Encoder.Dim * 2;
            var bo = new byte[RowsPerChunk * rowBytes]; var bc = new byte[RowsPerChunk * rowBytes];
            for (long r = 0; r < to.Shape[0]; r += RowsPerChunk)
            {
                int n = (int)Math.Min(RowsPerChunk, to.Shape[0] - r);
                var so = bo.AsSpan(0, n * rowBytes); var sc = bc.AsSpan(0, n * rowBytes);
                ro.ReadBytes(to, r * rowBytes, so); rc.ReadBytes(tc, r * rowBytes, sc);
                var h = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, System.Half>(sc);
                var y = new BFloat16[h.Length];
                var yb = System.Runtime.InteropServices.MemoryMarshal.Cast<BFloat16, ushort>(y.AsSpan());
                for (int i = 0; i < h.Length; i++) yb[i] = WanBf16Rne((float)h[i]);
                var (st, sub) = CompareBf16ThroughFp16($"embedding rows {r}..", so.ToArray(), y);
                oneStep += st; subnormal += sub; values += h.Length; rows += n;
            }
        }
        string report = $"{tensors} layer tensors + {rows:N0} embedding rows, {values:N0} values: {values - oneStep - subnormal:N0} bit-identical, {oneStep:N0} one bf16 step away ({100.0 * oneStep / values:F3}%), {subnormal:N0} below fp16's normal range";
        Console.WriteLine("[Wan] ComfyUI fp16 umT5 vs our bf16 export: " + report);
        return report;
    });

    /// <summary>
    /// ComfyUI's scaled 8-bit float umT5 against its fp16 file, layer by layer: the norms and bias tables (fp16 in both) bit-identical, and every matrix
    /// (value * scale_weight, decoded on the device) within the 8-bit format's rounding of the fp16 one, q, k and v each on their own (a scale paired with the wrong
    /// matrix, or q / k / v fused out of order, is a relative error near 1, not a few percent).
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Wan_ComfyUmT5_Fp8Scaled_LayersNearFp16() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL))
            throw new UnsupportedTestException("reads the files from disk: a desktop GPU");
        string p8 = Path.Combine(WanDir, "comfy", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"), p16 = Path.Combine(WanDir, "comfy", "umt5_xxl_fp16.safetensors");
        if (!File.Exists(p8) || !File.Exists(p16)) throw new UnsupportedTestException($"needs {p8} and {p16} (ComfyUI's)");
        double worst = 0, best = double.MaxValue; string worstName = ""; int matrices = 0, exact = 0;
        foreach (var (label, filter) in WanT5Layers())
        {
            using var w16 = await WanT5LayerAsync(accelerator, p16, filter);
            using var w8 = await WanT5LayerAsync(accelerator, p8, filter);
            var a = w16.Names.OrderBy(n => n, StringComparer.Ordinal).ToArray();
            var b = w8.Names.OrderBy(n => n, StringComparer.Ordinal).ToArray();
            if (a.Length == 0 || !a.SequenceEqual(b)) throw new Exception($"{label}: names differ: fp16 [{string.Join(", ", a)}]; fp8 [{string.Join(", ", b)}]");
            foreach (var name in a)
            {
                if (!w16.Shape(name).SequenceEqual(w8.Shape(name))) throw new Exception($"{name}: shape [{string.Join(",", w16.Shape(name))}] vs [{string.Join(",", w8.Shape(name))}]");
                var x = System.Runtime.InteropServices.MemoryMarshal.Cast<BFloat16, ushort>(await w16.BF16(name).CopyToHostAsync()).ToArray();
                var y = System.Runtime.InteropServices.MemoryMarshal.Cast<BFloat16, ushort>(await w8.BF16(name).CopyToHostAsync()).ToArray();
                bool matrix = name.Contains(".attn.qkv.") || name.Contains(".attn.o.") || name.Contains(".ffn.");
                if (!matrix)
                {
                    for (int i = 0; i < x.Length; i++) if (x[i] != y[i]) throw new Exception($"{name}[{i}]: bits {x[i]:X4} (fp16 file) vs {y[i]:X4} (fp8 file); fp16 in both files");
                    exact++;
                    continue;
                }
                int parts = name.Contains(".attn.qkv.") ? 3 : 1, len = x.Length / parts;
                for (int part = 0; part < parts; part++)
                {
                    double se = 0, sr = 0;
                    for (int i = part * len; i < (part + 1) * len; i++)
                    {
                        double xv = BitConverter.Int32BitsToSingle(x[i] << 16), yv = BitConverter.Int32BitsToSingle(y[i] << 16);
                        se += (xv - yv) * (xv - yv); sr += xv * xv;
                    }
                    double rel = Math.Sqrt(se / sr);
                    string id = parts == 3 ? $"{name} part {"qkv"[part]}" : name;
                    if (!(rel < WanF8MatrixBound)) throw new Exception($"{id}: relRMS {rel:E2} against the fp16 file (bound {WanF8MatrixBound:E1})");
                    if (rel > worst) { worst = rel; worstName = id; }
                    best = Math.Min(best, rel); matrices++;
                }
            }
        }
        Console.WriteLine($"[Wan] ComfyUI fp8 scaled umT5 vs fp16: {matrices} matrices relRMS {best:E2} .. {worst:E2} (worst {worstName}); {exact} fp16 tensors bit-identical");
        return $"{matrices} matrices relRMS {best:E2} .. {worst:E2} (worst {worstName}); {exact} norm / bias tensors bit-identical";
    });
    // F8_E4M3 keeps 4 significant bits. MEASURED on the host (numpy, 2026-11-12, blocks 0 / 11 / 23, all 7 matrices): value * scale against the fp16 file is
    // 2.646e-2 .. 2.655e-2 relative RMS on every matrix; the bf16 rounding of both sides adds about 1.5e-3 in quadrature
    private const double WanF8MatrixBound = 0.03;

    /// <summary>
    /// ComfyUI's fp16 umT5 through the encoder (its layers rewritten as they stream) against the reference's encoder outputs. Against the reference as it runs: the
    /// bound of our own bf16 export (MEASURED 2026-11-12: 2.3e-2 to 3.2e-2, ours 2.1e-2 to 2.8e-2). Against exact math on the reference's bf16 weights the weights
    /// themselves differ: ComfyUI's fp16 was rounded from a wider master (only 1 value in 8 is a bf16 value), so the embedding rows (read at fp16 precision) are
    /// 1.7e-3 relative RMS from the reference's and 1 matrix weight in 16 lands one bf16 step away (the double rounding), which the 24 layers carry to
    /// 7.8e-3 .. 1.57e-2 in the context (MEASURED, CUDA and OpenCL, three prompts; the 16-token prompt on OpenCL is the largest). A wrong mapping is near 1.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Wan_ComfyUmT5_Fp16_MatchesReference() => await RunTest(accelerator => WanUmT5Check(accelerator, Path.Combine("comfy", "umt5_xxl_fp16.safetensors"), 0.05, 0.03));

    /// <summary>ComfyUI's scaled 8-bit float umT5 through the encoder against the reference's encoder outputs: the 8-bit format's error on top of the rounding.</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Wan_ComfyUmT5_Fp8Scaled_MatchesReference() => await RunTest(accelerator => WanUmT5Check(accelerator, Path.Combine("comfy", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"), WanF8ContextBoundBf16, WanF8ContextBoundExact));
    // MEASURED 2026-11-12 (CUDA and OpenCL, three prompts): 5.96e-2 .. 7.42e-2 against the reference as it runs, 5.50e-2 .. 6.87e-2 against exact math: the
    // 8-bit floats (2.65e-2 per matrix) through 24 layers. For scale: our 8-bit block export (q8) is 2.6e-2 .. 3.4e-2 against the reference.
    private const double WanF8ContextBoundBf16 = 0.09, WanF8ContextBoundExact = 0.085;

    // our bf16 bytes vs the adapted ComfyUI bf16 values: equal bits, one bf16 step apart with the same sign (the double rounding fp32 -> fp16 -> bf16), or below
    // fp16's normal range within its subnormal step; returns (one step, subnormal) counts
    private static (long OneStep, long Subnormal) CompareBf16ThroughFp16(string name, byte[] raw, BFloat16[] yArr)
    {
        var x = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, ushort>(raw);
        var y = System.Runtime.InteropServices.MemoryMarshal.Cast<BFloat16, ushort>(yArr);
        if (x.Length != y.Length) throw new Exception($"{name}: {x.Length} vs {y.Length} values");
        long step = 0, sub = 0;
        for (int i = 0; i < x.Length; i++)
        {
            if (x[i] == y[i]) continue;
            if ((x[i] & 0x8000) == (y[i] & 0x8000) && Math.Abs(x[i] - y[i]) == 1) { step++; continue; }
            float xv = BitConverter.Int32BitsToSingle(x[i] << 16), yv = BitConverter.Int32BitsToSingle(y[i] << 16);
            if (!(Math.Abs(xv) < 6.11e-5f && Math.Abs(xv - yv) <= 6.0e-8f))
                throw new Exception($"{name}[{i}]: ours {xv:E4} vs ComfyUI's {yv:E4} (bits {x[i]:X4} / {y[i]:X4}): neither one bf16 step nor an fp16-subnormal rounding");
            sub++;
        }
        return (step, sub);
    }

    /// <summary>
    /// The tokenizer read from ComfyUI's umT5 files (the SentencePiece model in their U8 tensor <c>spiece_model</c>, <see cref="Wan22Pipeline.LoadTokenizerAsync"/>) against the one
    /// from google/umt5-xxl's tokenizer.json: the same pieces with bit-identical scores (256,000 from the model plus the 300 sentinels), the same unknown and end-of-sequence ids, and
    /// the same ids for the reference's three prompts (equal to the reference's own ids) and for edge cases (space runs, a character no piece covers, a sentinel, Chinese).
    /// Host only. Red check: a sentinel count of 299 or a score read from the wrong field fails it.
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Wan_Tokenizer_ComfySpiece_MatchesTokenizerJson() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.WebGPU) throw new UnsupportedTestException("reads the files from disk");
        string comfyDir = Path.Combine(WanDir, "comfy"), tokPath = Path.Combine(WanDir, "google", "umt5-xxl", "tokenizer.json");
        var files = new[] { "umt5_xxl_fp16.safetensors", "umt5_xxl_fp8_e4m3fn_scaled.safetensors" }.Where(f => File.Exists(Path.Combine(comfyDir, f))).ToArray();
        if (files.Length == 0 || !File.Exists(tokPath)) throw new UnsupportedTestException($"needs ComfyUI's umT5 files in {comfyDir} and {tokPath}");
        var json = await Wan22Pipeline.LoadTokenizerAsync(new LocalFolderModelSource(WanDir), "", "google/umt5-xxl/tokenizer.json");
        var refTexts = new List<(string Text, int[] Ids)>();
        foreach (var name in new[] { "0", "1", "neg" })
        {
            using var fx = System.IO.Compression.ZipFile.OpenRead(Path.Combine(WanDir, "dump", $"t5_{name}.npz"));
            var (ids, _) = WanNpz(fx, "ids"); var (mask, _) = WanNpz(fx, "mask");
            var textEntry = fx.GetEntry("text.npy")!; using var ts = textEntry.Open(); using var tms = new MemoryStream(); ts.CopyTo(tms);
            var raw = tms.ToArray(); int hlen = BitConverter.ToUInt16(raw, 8); var bytes = raw.AsSpan(10 + hlen);
            var sb = new System.Text.StringBuilder(); for (int i = 0; i + 3 < bytes.Length; i += 4) { int cp = BitConverter.ToInt32(bytes.Slice(i, 4)); if (cp == 0) break; sb.Append(char.ConvertFromUtf32(cp)); }
            refTexts.Add((sb.ToString(), ids.Take((int)mask.Sum()).Select(v => (int)v).ToArray()));
        }
        var edge = new[] { "", "a  b   c", "  leading and trailing  ", "\U0001F98A a fox in the snow \U0001F98A", "<extra_id_0> sentinel <extra_id_299>", "一只狐狸在雪地里奔跑，镜头缓缓推进。", "Ünïcödé façade naïve café", "x" };
        var report = new List<string>();
        foreach (var file in files)
        {
            var spm = await Wan22Pipeline.LoadTokenizerAsync(new LocalFolderModelSource(comfyDir), "", file);
            if (spm.VocabSize != json.VocabSize || spm.EosId != json.EosId || spm.UnkId != json.UnkId)
                throw new Exception($"{file}: vocab {spm.VocabSize} eos {spm.EosId} unk {spm.UnkId} vs tokenizer.json vocab {json.VocabSize} eos {json.EosId} unk {json.UnkId}");
            for (int i = 0; i < json.VocabSize; i++)
                if (spm.Piece(i) != json.Piece(i) || BitConverter.SingleToInt32Bits(spm.Score(i)) != BitConverter.SingleToInt32Bits(json.Score(i)))
                    throw new Exception($"{file}: piece {i} '{spm.Piece(i)}' {spm.Score(i):R} vs tokenizer.json '{json.Piece(i)}' {json.Score(i):R}");
            int tokens = 0;
            foreach (var (text, want) in refTexts)
            {
                var got = Wan22Pipeline.TokenizePrompt(spm, text);
                if (!got.SequenceEqual(want)) throw new Exception($"{file}: reference prompt differs at {Enumerable.Range(0, Math.Min(got.Length, want.Length)).FirstOrDefault(i => got[i] != want[i])} ({got.Length} vs {want.Length} tokens)");
                tokens += got.Length;
            }
            foreach (var text in edge)
            {
                var a = spm.Encode(Preprocessing.UnigramTokenizer.CleanWhitespace(text)); var b = json.Encode(Preprocessing.UnigramTokenizer.CleanWhitespace(text));
                if (!a.SequenceEqual(b)) throw new Exception($"{file}: '{text}': [{string.Join(",", a)}] vs tokenizer.json [{string.Join(",", b)}]");
                tokens += a.Length;
            }
            report.Add($"{file}: {spm.VocabSize} pieces identical (scores bit-exact), eos {spm.EosId}, unk {spm.UnkId}; {refTexts.Count} reference prompts = the reference's ids, {edge.Length} edge cases = tokenizer.json's ({tokens} tokens)");
        }
        foreach (var line in report) Console.WriteLine("[Wan] ComfyUI spiece tokenizer: " + line);
        return string.Join("; ", report);
    });
}
