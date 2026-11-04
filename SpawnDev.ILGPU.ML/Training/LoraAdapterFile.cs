using System.Text;
using System.Text.Json;
using SpawnDev.ILGPU.ML.SafeTensors;

namespace SpawnDev.ILGPU.ML.Training;

/// <summary>
/// LoRA adapters in Hugging Face PEFT's on-disk format: <c>adapter_model.safetensors</c> (F32 tensors named
/// <c>base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight</c> [rank, in] and <c>...lora_B.weight</c> [out, rank]) plus
/// <c>adapter_config.json</c>. An adapter trained here loads in PEFT (and llama.cpp's convert_lora_to_gguf), and one trained by PEFT loads
/// here as <see cref="LoraOptions.Initial"/>.
/// </summary>
public static class LoraAdapterFile
{
    /// <summary>
    /// Serialize <paramref name="tensors"/> (PEFT names, row-major F32) with their shapes as a safetensors file: an 8-byte little-endian
    /// header length, the JSON header (padded with spaces to a multiple of 8, as the reference implementation writes it), then the data.
    /// Tensors are written in name order so the file is deterministic.
    /// </summary>
    public static byte[] WriteSafeTensors(IReadOnlyDictionary<string, (float[] Data, int[] Shape)> tensors, IReadOnlyDictionary<string, string>? metadata = null)
    {
        var names = tensors.Keys.OrderBy(n => n, StringComparer.Ordinal).ToList();
        // Utf8JsonWriter, not JsonSerializer over object graphs: trim-safe (no reflection) in a trimmed Blazor app.
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms))
        {
            w.WriteStartObject();
            if (metadata != null && metadata.Count > 0)
            {
                w.WriteStartObject("__metadata__");
                foreach (var (k, v) in metadata) w.WriteString(k, v);
                w.WriteEndObject();
            }
            long offset = 0;
            foreach (var n in names)
            {
                var (data, shape) = tensors[n];
                long count = shape.Aggregate(1L, (a, b) => a * b);
                if (count != data.Length) throw new ArgumentException($"{n}: shape [{string.Join(",", shape)}] holds {count} values, data has {data.Length}");
                w.WriteStartObject(n);
                w.WriteString("dtype", "F32");
                w.WriteStartArray("shape"); foreach (var d in shape) w.WriteNumberValue(d); w.WriteEndArray();
                w.WriteStartArray("data_offsets"); w.WriteNumberValue(offset); w.WriteNumberValue(offset + 4L * data.Length); w.WriteEndArray();
                w.WriteEndObject();
                offset += 4L * data.Length;
            }
            w.WriteEndObject();
        }
        var json = ms.ToArray();
        long total = tensors.Values.Sum(t => 4L * t.Data.Length);
        int padded = (json.Length + 7) & ~7;
        var result = new byte[8 + padded + total];
        BitConverter.GetBytes((ulong)padded).CopyTo(result, 0);
        json.CopyTo(result, 8);
        for (int i = json.Length; i < padded; i++) result[8 + i] = (byte)' ';
        long at = 8 + padded;
        foreach (var n in names)
        {
            var data = tensors[n].Data;
            Buffer.BlockCopy(data, 0, result, (int)at, data.Length * 4);
            at += data.Length * 4L;
        }
        return result;
    }

    /// <summary>The adapter tensors of <paramref name="safetensors"/> by name (any stored float dtype, read as F32).</summary>
    public static Dictionary<string, float[]> ReadSafeTensors(byte[] safetensors)
    {
        var file = SafeTensorsParser.Parse(safetensors);
        var r = new Dictionary<string, float[]>();
        foreach (var t in file.Tensors)
            r[t.Name] = file.GetTensorFloat32(t) ?? Array.Empty<float>();
        return r;
    }

    /// <summary>PEFT's <c>adapter_config.json</c> for adapters of <paramref name="rank"/> / <paramref name="alpha"/> on <paramref name="adapters"/>.</summary>
    public static string PeftConfigJson(IEnumerable<LoraAdapter> adapters, int rank, float alpha, string? baseModel = null)
    {
        var modules = adapters.Select(a => a.PeftName[(a.PeftName.LastIndexOf('.') + 1)..]).Distinct().OrderBy(m => m, StringComparer.Ordinal).ToArray();
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = true }))
        {
            w.WriteStartObject();
            w.WriteString("peft_type", "LORA");
            w.WriteString("task_type", "CAUSAL_LM");
            w.WriteNumber("r", rank);
            w.WriteNumber("lora_alpha", alpha);
            w.WriteNumber("lora_dropout", 0.0);
            w.WriteString("bias", "none");
            w.WriteBoolean("fan_in_fan_out", false);
            w.WriteStartArray("target_modules"); foreach (var m in modules) w.WriteStringValue(m); w.WriteEndArray();
            w.WriteString("base_model_name_or_path", baseModel ?? "");
            w.WriteBoolean("inference_mode", true);
            w.WriteEndObject();
        }
        return Encoding.UTF8.GetString(ms.ToArray());
    }
}
