using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.UniRig;
using UniRigHarness;

// UniRig auto-rigging gate. UNIRIG_DIR (default E:/spawndev-unirig) holds weights/ (export_weights.py) and golden/ (golden_skeleton.py).
//   encoder     the perceiver encoder against the reference latents
//   UNIRIG_ACCELERATOR=cuda|opencl|cpu picks the device (default: preferred)
var dir = Environment.GetEnvironmentVariable("UNIRIG_DIR") ?? "E:/spawndev-unirig";
var command = args.Length > 0 ? args[0].ToLowerInvariant() : "";
return command switch
{
    "encoder" => Gates.Encoder(dir),
    "tokenizer" => Gates.Tokenizer(dir),
    "decoder" => Gates.Decoder(dir),
    "generate" => Gates.Generate(dir),
    "voxelskin" => Gates.VoxelSkin(dir),
    "riggl" => Gates.RigGlb(dir, args.Length > 2 ? args[1] : throw new ArgumentException("riggl <in.glb> <out.glb> [seed]"), args[2], args.Length > 3 ? int.Parse(args[3]) : 0),
    "rig" => Gates.Rig(dir, args.Length > 1 ? args[1] : throw new ArgumentException("rig <mesh.obj> [out.json] [seed]"), args.Length > 2 ? args[2] : null, args.Length > 3 ? int.Parse(args[3]) : 0),
    _ => Usage(command),
};

static int Usage(string bad)
{
    Console.WriteLine($"unknown command '{bad}'. commands: encoder | tokenizer | decoder | generate | rig <mesh.obj> [out.json] [seed]");
    return 2;
}

namespace UniRigHarness
{
    internal static class Gpu
    {
        public static (IDisposable Context, Accelerator Accelerator) Create()
        {
            var builder = MLContext.Create();
            builder.AllAcceleratorsAsync().GetAwaiter().GetResult();
            var ctx = builder.ToContext();
            var want = Environment.GetEnvironmentVariable("UNIRIG_ACCELERATOR");
            Accelerator? accelerator = null;
            if (!string.IsNullOrWhiteSpace(want))
            {
                var match = ctx.Devices.FirstOrDefault(d => string.Equals(d.AcceleratorType.ToString(), want, StringComparison.OrdinalIgnoreCase))
                    ?? throw new InvalidOperationException($"UNIRIG_ACCELERATOR='{want}' matches no device. Available: " + string.Join(", ", ctx.Devices.Select(d => d.AcceleratorType.ToString())));
                accelerator = match.CreateAccelerator(ctx);
            }
            accelerator ??= ctx.CreatePreferredAcceleratorAsync().GetAwaiter().GetResult() ?? throw new InvalidOperationException("no accelerator available");
            Console.WriteLine($"engine : {accelerator.AcceleratorType} / {accelerator.Name}");
            return (ctx, accelerator);
        }
    }

    internal static class Gates
    {
        /// <summary>The C# encoder on the golden's points, against the reference latents. bf16 weights alone move the result by relRMS ~2e-3 (check_export.py).</summary>
        public static int Encoder(string dir)
        {
            var g = Path.Combine(dir, "golden", "skeleton");
            var (pts, ps) = Npy.Load<float>(Path.Combine(g, "points.npy"));
            var (nrm, _) = Npy.Load<float>(Path.Combine(g, "normals.npy"));
            var (pre, _) = Npy.Load<long>(Path.Combine(g, "presample_idx.npy"));
            var (fps, _) = Npy.Load<long>(Path.Combine(g, "fps_idx.npy"));
            var (gold, _) = Npy.Load<float>(Path.Combine(g, "latents.npy"));
            int n = ps[0];
            var rows = fps.Select(i => (int)pre[i]).ToArray();           // queries = pre-sampled points at the FPS indices

            var (ctx, accel) = Gpu.Create();
            using var _c = ctx; using var _a = accel;
            var sw = Stopwatch.StartNew();
            using var weights = GpuWeights.LoadFile(accel, Path.Combine(dir, "weights", "skeleton.safetensors"),
                name => name.StartsWith("mesh_encoder.encoder.") || name.StartsWith("output_proj."));
            Console.WriteLine($"weights  : {weights.Bytes / 1e6:F0} MB on the device in {sw.Elapsed.TotalSeconds:F1}s");
            using var p = accel.Allocate1D(pts); using var q = accel.Allocate1D(nrm);
            using var enc = new UniRigEncoder(accel, weights);

            double[] times = new double[3]; float[] got = Array.Empty<float>();
            for (int run = 0; run < times.Length; run++)
            {
                sw.Restart();
                var o = enc.Encode(p.View, q.View, n, rows);
                accel.Synchronize();
                times[run] = sw.Elapsed.TotalSeconds;
                if (run == 0) got = o.GetAsArray1D();
            }
            double num = 0, den = 0, max = 0;
            for (int i = 0; i < gold.Length; i++) { double d = got[i] - gold[i]; num += d * d; den += (double)gold[i] * gold[i]; max = Math.Max(max, Math.Abs(d)); }
            double rel = Math.Sqrt(num / den);
            Console.WriteLine($"encoder  : {QueryText(got)}  relRMS vs reference {rel:E2}   max |diff| {max:E2}   time {times[0]:F2}s (cold), {times[1]:F2}s, {times[2]:F2}s");
            bool ok = rel < 0.02;
            Console.WriteLine(ok ? "PASS" : "FAIL (relRMS >= 2e-2)");
            return ok ? 0 : 1;
        }

        /// <summary>The grammar's allowed set at every step of the reference's real generation, and the detokenized skeleton, against the reference's.</summary>
        public static int Tokenizer(string dir)
        {
            var g = Path.Combine(dir, "golden", "skeleton");
            var (full64, _) = Npy.Load<long>(Path.Combine(g, "full_ids.npy"));
            var full = full64.Select(v => (int)v).ToArray();
            var allowed = System.Text.Json.JsonSerializer.Deserialize<List<List<int>>>(File.ReadAllText(Path.Combine(g, "allowed.json")))!;
            int bad = 0;
            for (int r = 1; r < full.Length; r++)
            {
                var mine = UniRigTokenizer.Allowed(UniRigTokenizer.StateAfter(full.AsSpan(0, r + 1)));
                if (!mine.SequenceEqual(allowed[r - 1])) { if (bad++ < 3) Console.WriteLine($"  step {r}: allowed set differs (mine {mine.Count} ids, reference {allowed[r - 1].Count})"); }
            }
            Console.WriteLine($"grammar  : {full.Length - 1} steps, {bad} differ from the reference's allowed sets");

            var (bones, bs) = Npy.Load<float>(Path.Combine(g, "bones.npy"));              // [J, 6]: parent position, joint position
            var (refParents, _) = Npy.Load<long>(Path.Combine(g, "parents.npy"));
            var sk = UniRigTokenizer.Detokenize(full.Append(UniRigTokenizer.Eos).ToArray());
            double maxDiff = 0; int parentDiff = 0;
            for (int j = 0; j < bs[0]; j++)
            {
                for (int c = 0; c < 3; c++) maxDiff = Math.Max(maxDiff, Math.Abs(sk.Joints[j * 3 + c] - bones[j * 6 + 3 + c]));
                if (sk.Parents[j] != (int)refParents[j]) parentDiff++;
            }
            Console.WriteLine($"detokenize: {sk.Count} joints (reference {bs[0]}), max joint position diff {maxDiff:E2}, parents differing {parentDiff}");
            bool ok = bad == 0 && sk.Count == bs[0] && parentDiff == 0 && maxDiff < 1e-6;
            Console.WriteLine(ok ? "PASS" : "FAIL");
            return ok ? 0 : 1;
        }

        /// <summary>
        /// The OPT decoder, teacher-forced: the REFERENCE's encoder latents and the golden [bos, cls, ...] sequence in one prefill; its logits for every step against the
        /// reference's. Judged on relRMS and on whether the grammar-constrained argmax at each step is the token the reference generated.
        /// </summary>
        public static int Decoder(string dir)
        {
            var g = Path.Combine(dir, "golden", "skeleton");
            var (lat, _) = Npy.Load<float>(Path.Combine(g, "latents.npy"));                 // [1, 1024, 1024] from the reference encoder
            var (full64, _) = Npy.Load<long>(Path.Combine(g, "full_ids.npy"));
            var (gold, gs) = Npy.Load<float>(Path.Combine(g, "logits_teacher.npy"));       // [93, 267]
            var allowed = System.Text.Json.JsonSerializer.Deserialize<List<List<int>>>(File.ReadAllText(Path.Combine(g, "allowed.json")))!;
            var full = full64.Select(v => (int)v).ToArray();
            int steps = gs[0], V = gs[1];

            var (ctx, accel) = Gpu.Create();
            using var _c = ctx; using var _a = accel;
            var sw = Stopwatch.StartNew();
            using var weights = GpuWeights.LoadFile(accel, Path.Combine(dir, "weights", "skeleton.safetensors"), name => name.StartsWith("transformer."));
            Console.WriteLine($"weights  : {weights.Bytes / 1e6:F0} MB on the device in {sw.Elapsed.TotalSeconds:F1}s");
            using var cond = accel.Allocate1D(lat);
            using var dec = new UniRigDecoder(accel, weights);
            float[] got = Array.Empty<float>();
            double[] times = new double[2];
            for (int run = 0; run < times.Length; run++)
            {
                sw.Restart();
                got = dec.PrefillAsync(cond.View, 1024, full, steps).GetAwaiter().GetResult();
                times[run] = sw.Elapsed.TotalSeconds;
            }
            double num = 0, den = 0; int argAgree = 0, maskedAgree = 0;
            for (int r = 0; r < steps; r++)
            {
                int bestU = 0, bestM = -1; float vu = float.NegativeInfinity, vm = float.NegativeInfinity, rv = float.NegativeInfinity; int bestR = 0;
                var ok = new HashSet<int>(allowed[r]);
                for (int v = 0; v < V; v++)
                {
                    float a = got[r * V + v], b = gold[r * V + v];
                    num += (double)(a - b) * (a - b); den += (double)b * b;
                    if (a > vu) { vu = a; bestU = v; }
                    if (ok.Contains(v) && a > vm) { vm = a; bestM = v; }
                    if (b > rv) { rv = b; bestR = v; }
                }
                int expect = r + 2 < full.Length ? full[r + 2] : UniRigTokenizer.Eos;
                if (bestU == bestR) argAgree++;
                if (bestM == expect) maskedAgree++;
            }
            double rel = Math.Sqrt(num / den);
            Console.WriteLine($"decoder  : logits [{steps},{V}] relRMS vs reference {rel:E2}; unmasked argmax agrees {argAgree}/{steps}; grammar-constrained argmax reproduces the generated token {maskedAgree}/{steps}; {times[0]:F2}s cold, {times[1]:F2}s");
            bool pass = rel < 0.05 && maskedAgree >= steps - 2;
            Console.WriteLine(pass ? "PASS" : "FAIL");
            return pass ? 0 : 1;
        }

        /// <summary>The whole skeleton stage on the golden's points (C# encoder + C# decoder + grammar, greedy) against the reference's generated skeleton.</summary>
        public static int Generate(string dir)
        {
            var g = Path.Combine(dir, "golden", "skeleton");
            var (pts, ps) = Npy.Load<float>(Path.Combine(g, "points.npy"));
            var (nrm, _) = Npy.Load<float>(Path.Combine(g, "normals.npy"));
            var (pre, _) = Npy.Load<long>(Path.Combine(g, "presample_idx.npy"));
            var (fps, _) = Npy.Load<long>(Path.Combine(g, "fps_idx.npy"));
            var (full64, _) = Npy.Load<long>(Path.Combine(g, "full_ids.npy"));
            var (bones, bs) = Npy.Load<float>(Path.Combine(g, "bones.npy"));
            var (refParents, _) = Npy.Load<long>(Path.Combine(g, "parents.npy"));
            var gold = full64.Select(v => (int)v).Append(UniRigTokenizer.Eos).ToArray();           // [bos, cls, ..., eos]
            int n = ps[0];
            var rows = fps.Select(i => (int)pre[i]).ToArray();

            var (ctx, accel) = Gpu.Create();
            using var _c = ctx; using var _a = accel;
            using var weights = GpuWeights.LoadFile(accel, Path.Combine(dir, "weights", "skeleton.safetensors"));
            using var p = accel.Allocate1D(pts); using var q = accel.Allocate1D(nrm);
            using var gen = new UniRigSkeletonGenerator(accel, weights);
            UniRigSkeletonGenerator.Result r = null!;
            var times = new double[2];
            for (int run = 0; run < times.Length; run++)
            {
                var sw = Stopwatch.StartNew();
                r = gen.GenerateAsync(p.View, q.View, n, rows).GetAwaiter().GetResult();
                times[run] = sw.Elapsed.TotalSeconds;
            }
            int common = 0; while (common < gold.Length && common < r.Tokens.Length && gold[common] == r.Tokens[common]) common++;
            Console.WriteLine($"tokens   : {r.Tokens.Length} generated (reference {gold.Length}); identical for the first {common}; {times[0]:F2}s cold, {times[1]:F2}s");
            if (common < Math.Min(gold.Length, r.Tokens.Length))
            {
                // is the first difference a near-tie in the REFERENCE's own logits? row (common - 2) of logits_teacher is the distribution that chose token `common`
                var (lg, ls) = Npy.Load<float>(Path.Combine(g, "logits_teacher.npy"));
                int row = common - 2;
                if (row >= 0 && row < ls[0])
                {
                    float a = lg[row * ls[1] + gold[common]], b = lg[row * ls[1] + r.Tokens[common]];
                    Console.WriteLine($"first difference at token {common}: reference chose {gold[common]}, C# chose {r.Tokens[common]}; the reference's own logits for them: {a:F3} vs {b:F3} (margin {a - b:F3}; logit std ~{StdOf(lg, row * ls[1], ls[1]):F2})");
                }
            }
            Console.WriteLine($"skeleton : {r.Skeleton.Count} joints (reference {bs[0]})");
            double worst = 0, sum = 0; int cmp = Math.Min(r.Skeleton.Count, bs[0]), badParents = 0;
            for (int j = 0; j < cmp; j++)
            {
                double dx = r.Skeleton.Joints[j * 3] - bones[j * 6 + 3], dy = r.Skeleton.Joints[j * 3 + 1] - bones[j * 6 + 4], dz = r.Skeleton.Joints[j * 3 + 2] - bones[j * 6 + 5];
                double d = Math.Sqrt(dx * dx + dy * dy + dz * dz); worst = Math.Max(worst, d); sum += d;
                if (r.Skeleton.Parents[j] != (int)refParents[j]) badParents++;
            }
            Console.WriteLine($"           joint position error vs reference over the first {cmp}: mean {sum / cmp:F4}, worst {worst:F4} (one coordinate bin = {2.0 / 256:F4}); parents differing {badParents}");
            // The decoder's logits agree with the reference to relRMS ~2e-3 (decoder gate), but greedy decoding flips on a near-tie: the first difference above was a 0.016 margin on a logit
            // std of 2.9, adjacent coordinate bins, and every later token follows from it. So the gate is the SKELETON, not the token stream: same joints, same tree, positions within a few bins.
            const double Bin = 2.0 / 256;
            bool pass = r.Skeleton.Count == bs[0] && badParents == 0 && sum / cmp <= Bin && worst <= 6 * Bin;
            Console.WriteLine(pass ? "PASS" : "FAIL");
            return pass ? 0 : 1;
        }

        /// <summary>A mesh (plain v / f OBJ) to a skeleton in the mesh's own space, all in C#: sampler, encoder, decoder, grammar. Writes joints + parents as JSON.</summary>
        public static int Rig(string dir, string objPath, string? outJson, int seed)
        {
            var pos = new List<float>(); var idx = new List<int>();
            foreach (var line in File.ReadLines(objPath))
            {
                var t = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (t.Length >= 4 && t[0] == "v") for (int k = 1; k <= 3; k++) pos.Add(float.Parse(t[k], System.Globalization.CultureInfo.InvariantCulture));
                else if (t.Length >= 4 && t[0] == "f") for (int k = 1; k <= 3; k++) idx.Add(int.Parse(t[k].Split('/')[0]) - 1);
            }
            var (ctx, accel) = Gpu.Create();
            using var _c = ctx; using var _a = accel;
            using var weights = GpuWeights.LoadFile(accel, Path.Combine(dir, "weights", "skeleton.safetensors"));
            using var gen = new UniRigSkeletonGenerator(accel, weights);
            var sw = Stopwatch.StartNew();
            var (sk, tokens) = gen.RigMeshAsync(accel, pos.ToArray(), idx.ToArray(), seed).GetAwaiter().GetResult();
            Console.WriteLine($"rig      : {pos.Count / 3} verts, {idx.Count / 3} faces -> {sk.Count} joints from {tokens.Length} tokens in {sw.Elapsed.TotalSeconds:F2}s (includes kernel compile)");
            for (int j = 0; j < sk.Count; j++)
                Console.WriteLine($"  {j,3} parent {sk.Parents[j],3}  ({sk.Joints[j * 3]:F3}, {sk.Joints[j * 3 + 1]:F3}, {sk.Joints[j * 3 + 2]:F3})");
            if (outJson != null)
            {
                File.WriteAllText(outJson, System.Text.Json.JsonSerializer.Serialize(new { joints = sk.Joints, parents = sk.Parents, tokens }));
                Console.WriteLine($"wrote {outJson}");
            }
            return 0;
        }

        /// <summary>The skinning prior: C# graph + Dijkstra + weights on the reference's own voxels against the reference's skin, then the C# voxelisation against the same voxel set.</summary>
        public static int VoxelSkin(string dir)
        {
            var g = Path.Combine(dir, "golden", "voxel_skin");
            var (v, vs) = Npy.Load<float>(Path.Combine(g, "vertices.npy"));
            var (fi, _) = Npy.Load<int>(Path.Combine(g, "faces.npy"));
            var (jn, js) = Npy.Load<float>(Path.Combine(g, "joints.npy"));
            var (gc, _) = Npy.Load<float>(Path.Combine(g, "grid_coords.npy"));
            var (gold, _) = Npy.Load<float>(Path.Combine(g, "skin.npy"));
            int grid = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(g, "manifest.json"))).RootElement.GetProperty("grid").GetInt32();
            int n = vs[0], J = js[0];

            var sw = Stopwatch.StartNew();
            var refVox = UniRigVoxelSkin.FromCoords(gc, grid);
            var skin = UniRigVoxelSkin.Compute(v, fi, jn, refVox);
            Console.WriteLine($"skin     : {n} vertices x {J} joints, {refVox.Count} voxels, graph + Dijkstra {sw.Elapsed.TotalSeconds:F2}s");
            double max = 0, sum = 0; int worstRow = -1;
            for (int i = 0; i < gold.Length; i++) { double d = Math.Abs(skin[i] - gold[i]); sum += d; if (d > max) { max = d; worstRow = i / J; } }
            Console.WriteLine($"           vs reference skin: max |diff| {max:E2} (vertex {worstRow}), mean {sum / gold.Length:E2}; dominant bone differs on {Enumerable.Range(0, n).Count(i => ArgMax(skin, i * J, J) != ArgMax(gold, i * J, J))} vertices");

            sw.Restart();
            var mine = UniRigVoxelSkin.Voxelize(v, fi, grid);
            var theirs = new HashSet<long>(); for (int q = 0; q < refVox.Count; q++) theirs.Add(((long)refVox.Cells[q * 3] * grid + refVox.Cells[q * 3 + 1]) * grid + refVox.Cells[q * 3 + 2]);
            int common = 0; for (int q = 0; q < mine.Count; q++) if (theirs.Contains(((long)mine.Cells[q * 3] * grid + mine.Cells[q * 3 + 1]) * grid + mine.Cells[q * 3 + 2])) common++;
            Console.WriteLine($"voxelize : {mine.Count} voxels (reference-side implementation {refVox.Count}), {common} in common, {sw.Elapsed.TotalSeconds:F2}s");
            bool pass = max < 1e-4 && common >= 0.999 * Math.Max(mine.Count, refVox.Count);
            Console.WriteLine(pass ? "PASS" : "FAIL");
            return pass ? 0 : 1;
        }

        /// <summary>A generated character GLB to a rigged GLB: skeleton + skin weights, textures untouched.</summary>
        public static int RigGlb(string dir, string inGlb, string outGlb, int seed)
        {
            var glb = File.ReadAllBytes(inGlb);
            var (ctx, accel) = Gpu.Create();
            using var _c = ctx; using var _a = accel;
            using var weights = GpuWeights.LoadFile(accel, Path.Combine(dir, "weights", "skeleton.safetensors"));
            using var rigger = new UniRigRigger(accel, weights);
            var sw = Stopwatch.StartNew();
            var r = rigger.RigGlbAsync(glb, seed).GetAwaiter().GetResult();
            Console.WriteLine($"riggl    : {r.Skeleton.Count} joints, {glb.Length / 1024} KiB -> {r.Glb.Length / 1024} KiB in {sw.Elapsed.TotalSeconds:F2}s (includes kernel compile)");
            File.WriteAllBytes(outGlb, r.Glb);
            Console.WriteLine($"wrote {outGlb}");
            return 0;
        }

        private static int ArgMax(float[] a, int start, int count) { int b = 0; for (int i = 1; i < count; i++) if (a[start + i] > a[start + b]) b = i; return b; }

        private static double StdOf(float[] a, int start, int count)
        {
            double s = 0, ss = 0;
            for (int i = 0; i < count; i++) { s += a[start + i]; ss += (double)a[start + i] * a[start + i]; }
            double m = s / count; return Math.Sqrt(ss / count - m * m);
        }

        private static string QueryText(float[] a) { double s = 0, ss = 0; foreach (var v in a) { s += v; ss += (double)v * v; } double m = s / a.Length; return $"[1024,1024] mean {m:F4} std {Math.Sqrt(ss / a.Length - m * m):F4}"; }
    }
}
