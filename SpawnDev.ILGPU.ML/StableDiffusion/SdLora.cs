using System.Text.RegularExpressions;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.SafeTensors;

namespace SpawnDev.ILGPU.ML.StableDiffusion;

using HV = ArrayView1D<global::ILGPU.Half, Stride1D.Dense>;

/// <summary>One LoRA to merge: where its .safetensors comes from, the UNet strength and the text encoder strength (ComfyUI's strength_model / strength_clip).</summary>
public sealed record SdLoraSpec(Func<CancellationToken, Task<Stream>> Open, string Name, float Strength = 1f, float ClipStrength = 1f)
{
    public static SdLoraSpec From(IModelSource source, string repo, string file, float strength = 1f, float clipStrength = 1f)
        => new(ct => source.OpenAsync(repo, file, ct), $"{repo}/{file}", strength, clipStrength);
}

/// <summary>What <see cref="SdCheckpoint.ApplyLorasAsync"/> did: the modules patched per LoRA and the LoRA keys that matched nothing in this checkpoint.</summary>
public sealed record SdLoraReport(IReadOnlyList<(string Lora, int Patched, IReadOnlyList<string> Unmatched)> Loras);

/// <summary>
/// LoRA merging into the resident fp16 weights: W += strength * alpha / rank * (up @ down), on the device, into the fused q|k|v / k|v rows where the projection was fused.
/// Key formats: kohya (<c>lora_unet_*</c>, <c>lora_te_*</c>, <c>lora_te1_*</c> / <c>lora_te2_*</c>, diffusers or LDM module paths with underscores; <c>.lora_down / .lora_up / .alpha</c>),
/// PEFT / diffusers (<c>unet.* / text_encoder.* / text_encoder_2.*</c> with <c>.lora_A / .lora_B</c> or <c>.lora.down / .lora.up</c>) and LDM dotted names (<c>diffusion_model.*</c>).
/// Linear and convolution (LoCon) modules. What a LoRA set is taken off to is kept by the store's <see cref="SdPatchState"/>, so a new set of LoRAs (or new strengths)
/// restores before merging.
/// </summary>
internal sealed class SdLoraMerger
{
    private readonly Accelerator _acc;
    private readonly SdKernels _k;
    private readonly SdWeights? _unet, _text;
    private readonly SdPatchState? _unetState, _textState;
    private readonly SdFamily _family;
    private Dictionary<string, (SdWeights Store, string Source, bool Clip)>? _index;

    public SdLoraMerger(Accelerator acc, SdPatchState? unet, SdPatchState? text, SdFamily family)
    {
        _acc = acc; _k = new SdKernels(acc); _unetState = unet; _textState = text; _unet = unet?.Weights; _text = text?.Weights; _family = family;
    }

    // ---------------------------------------------------------------------------------------------------
    // name index: every way a LoRA file may name a module -> the store and the file-level tensor name (".weight" stripped)
    // ---------------------------------------------------------------------------------------------------
    private Dictionary<string, (SdWeights, string, bool)> Index()
    {
        if (_index != null) return _index;
        var ix = new Dictionary<string, (SdWeights, string, bool)>(StringComparer.Ordinal);
        if (_unet != null)
            foreach (var n in _unet.SourceNames.Where(n => n.EndsWith(".weight", StringComparison.Ordinal)).ToList())
            {
                string ldm = n[..^".weight".Length];
                var e = (_unet, n, false);
                ix.TryAdd("lora_unet_" + ldm.Replace('.', '_'), e);
                ix.TryAdd("diffusion_model." + ldm, e);
                ix.TryAdd("model.diffusion_model." + ldm, e);
                var diff = SdNames.LdmToDiffusersUNet(ldm);
                if (diff != null)
                {
                    ix.TryAdd("lora_unet_" + diff.Replace('.', '_'), e);
                    ix.TryAdd("unet." + diff, e);
                    ix.TryAdd(diff, e);
                }
            }
        if (_text != null)
        {
            // HF CLIP module names per tower, mapped to the normalised names (OpenCLIP towers hold q|k|v as one in_proj: the parts are rows of "qkv")
            void Tower(string prefix, string[] kohya, string[] peft)
            {
                for (int i = 0; _text.Has($"{prefix}layers.{i}.out.weight"); i++)
                    foreach (var (hf, local) in new[] { ("self_attn.q_proj", "q"), ("self_attn.k_proj", "k"), ("self_attn.v_proj", "v"), ("self_attn.out_proj", "out"), ("mlp.fc1", "fc1"), ("mlp.fc2", "fc2") })
                    {
                        string full = $"text_model.encoder.layers.{i}.{hf}";
                        var e = (_text, $"{prefix}layers.{i}.{local}.weight", true);
                        foreach (var k in kohya) ix.TryAdd(k + full.Replace('.', '_'), e);
                        foreach (var p in peft) ix.TryAdd(p + full, e);
                    }
            }
            if (_family == SdFamily.SD1) Tower("l.", new[] { "lora_te_" }, new[] { "text_encoder.", "" });
            else if (_family == SdFamily.SD2) Tower("h.", new[] { "lora_te_" }, new[] { "text_encoder.", "" });
            else { Tower("l.", new[] { "lora_te1_" }, new[] { "text_encoder." }); Tower("g.", new[] { "lora_te2_" }, new[] { "text_encoder_2." }); }
        }
        return _index = ix;
    }

    /// <summary>Where a normalised text-encoder part (<c>g.layers.3.k.weight</c>) lives: the HF-format towers fused q|k|v at load (Locate knows), the OpenCLIP ones store in_proj rows.</summary>
    private (string Name, int RowOffset, int Rows)? LocateText(SdWeights s, string source)
    {
        var loc = s.Locate(source);
        if (loc != null) return loc;
        var m = Regex.Match(source, @"^(.+\.)(q|k|v)\.weight$");
        if (!m.Success) return null;
        string qkv = m.Groups[1].Value + "qkv.weight";
        if (!s.Has(qkv)) return null;
        int C = s.Shape(qkv)[0] / 3, part = "qkv".IndexOf(m.Groups[2].Value[0]);
        return (qkv, part * C, C);
    }

    // ---------------------------------------------------------------------------------------------------
    // apply
    // ---------------------------------------------------------------------------------------------------
    private static readonly string[] DownSuffix = { ".lora_down.weight", ".lora_A.weight", ".lora.down.weight", ".lora_A.default.weight", ".lora_linear_layer.down.weight", "_lora.down.weight" };
    private static readonly string[] UpSuffix = { ".lora_up.weight", ".lora_B.weight", ".lora.up.weight", ".lora_B.default.weight", ".lora_linear_layer.up.weight", "_lora.up.weight" };

    private static (string Module, int Kind)? Split(string key)
    {
        foreach (var s in DownSuffix) if (key.EndsWith(s, StringComparison.Ordinal)) return (Module(key[..^s.Length]), 0);
        foreach (var s in UpSuffix) if (key.EndsWith(s, StringComparison.Ordinal)) return (Module(key[..^s.Length]), 1);
        if (key.EndsWith(".alpha", StringComparison.Ordinal)) return (key[..^".alpha".Length], 2);
        return null;
    }

    // the old diffusers attention-processor format names a module "<attention>.processor.to_k" (key "...processor.to_k_lora.down.weight"):
    // the module itself is "<attention>.to_k", and its output projection "to_out" is "to_out.0"
    private static string Module(string m)
    {
        int p = m.LastIndexOf(".processor.", StringComparison.Ordinal);
        if (p < 0) return m;
        var proj = m[(p + ".processor.".Length)..];
        return m[..p] + "." + (proj == "to_out" ? "to_out.0" : proj);
    }

    public async Task RestoreAsync()
    {
        if (_unetState != null) await _unetState.RestoreLorasAsync();
        if (_textState != null) await _textState.RestoreLorasAsync();
    }

    public async Task<SdLoraReport> ApplyAsync(IReadOnlyList<SdLoraSpec> loras, CancellationToken ct)
    {
        await RestoreAsync();
        var ix = Index();
        var results = new List<(string, int, IReadOnlyList<string>)>();
        foreach (var spec in loras)
        {
            await using var stream = await spec.Open(ct);
            var lenBytes = new byte[8];
            await stream.ReadExactlyAsync(lenBytes, ct);
            long headerSize = BitConverter.ToInt64(lenBytes);
            if (headerSize <= 0 || headerSize > 100_000_000) throw new InvalidDataException($"'{spec.Name}': not a safetensors file");
            var header = new byte[headerSize];
            await stream.ReadExactlyAsync(header, ct);
            long pos = 8 + headerSize, length = -1;
            try { if (stream.CanSeek) length = stream.Length - pos; } catch (NotSupportedException) { }
            var (tensors, _) = SafeTensorsFileReader.ParseHeader(spec.Name, header, pos, length);
            // group the keys by module; keep only modules this checkpoint has (the rest are reported, never uploaded)
            var modules = new Dictionary<string, (string? Down, string? Up, string? Alpha)>(StringComparer.Ordinal);
            var unmatched = new List<string>();
            foreach (var t in tensors)
            {
                var sp = Split(t.Name);
                if (sp == null) { unmatched.Add(t.Name); continue; }
                modules.TryGetValue(sp.Value.Module, out var e);
                if (sp.Value.Kind == 0) e.Down = t.Name; else if (sp.Value.Kind == 1) e.Up = t.Name; else e.Alpha = t.Name;
                modules[sp.Value.Module] = e;
            }
            var keep = new HashSet<string>(StringComparer.Ordinal);
            foreach (var (mod, e) in modules)
            {
                if (!ix.ContainsKey(mod) || e.Down == null || e.Up == null) { unmatched.Add(mod); continue; }
                keep.Add(e.Down); keep.Add(e.Up); if (e.Alpha != null) keep.Add(e.Alpha);
            }
            using var lw = new SdWeights();
            var alphas = new Dictionary<string, float>(StringComparer.Ordinal);
            using (var st = new SdTensorStreamer(_acc))
                await st.StreamAsync(stream, tensors, pos, n => keep.Contains(n) ? new SdTensorStreamer.Target(n.EndsWith(".alpha", StringComparison.Ordinal) ? null : lw, n, SdLayout.AsIs) : null,
                    null, ct, (n, v) => alphas[n] = v);
            int patched = 0;
            foreach (var (mod, e) in modules)
            {
                if (!ix.TryGetValue(mod, out var target) || e.Down == null || e.Up == null) continue;
                var (store, source, clip) = target;
                var loc = store == _text ? LocateText(store, source) : store.Locate(source);
                if (loc == null) { unmatched.Add(mod); continue; }
                var (tname, rowOff, rows) = loc.Value;
                var ds = lw.Shape(e.Down); var us = lw.Shape(e.Up);
                int rank = ds[0], K = 1; for (int d = 1; d < ds.Length; d++) K *= ds[d];
                int N = us[0];
                var ts = store.Shape(tname);
                int cols = 1; for (int d = 1; d < ts.Length; d++) cols *= ts[d];
                if (N != rows || K != cols || us.Length < 2 || us[1] != rank)
                    throw new InvalidDataException($"'{spec.Name}': {mod} is up {string.Join("x", us)} @ down {string.Join("x", ds)}, the weight is {string.Join("x", ts)} (rows {rowOff}+{rows})");
                int taps = ts.Length == 4 ? ts[2] * ts[3] : 1, cin = ts.Length == 4 ? ts[1] : cols;
                float alpha = e.Alpha != null && alphas.TryGetValue(e.Alpha, out var a) ? a : rank;
                float scale = (clip ? spec.ClipStrength : spec.Strength) * alpha / rank;
                if (scale == 0f) continue;
                (store == _text ? _textState : _unetState)!.BeforeLora(tname);
                _k.MergeLora(store[tname], lw[e.Up], lw[e.Down], N, K, rank, rowOff, cols, taps, cin, scale);
                patched++;
            }
            await _acc.SynchronizeAsync();      // the LoRA tensors go with lw: the merges must have run
            results.Add((spec.Name, patched, unmatched));
        }
        return new SdLoraReport(results);
    }
}
