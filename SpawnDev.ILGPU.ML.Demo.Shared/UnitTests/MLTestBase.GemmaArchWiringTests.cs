using System.IO;
using System.Linq;
using System.Text;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Regression guard for the two gemma3 forward-pass bugs found 2026-06-30 (gemma3:270m produced fluent
/// but factually-wrong output — "The capital of France is a place…", Paris at rank 143):
///  1. WEIGHTLESS V-NORM was wrongly applied to gemma3. It is a gemma4/gemma3n-only behavior; standard
///     Gemma 3 leaves V raw (llama.cpp gemma3.cpp). Normalizing V corrupted the attended values →
///     factual retrieval collapsed (Paris rank 143 → rank 0 once removed). Guard: UsesWeightlessVNorm.
///  2. SLIDING-WINDOW rope base. gemma3's GGUF omits sliding_window_pattern AND rope.freq_base_swa, so
///     llama.cpp's hardcoded defaults apply: a 5:1 local:global pattern (period 6) with local (SWA) layers
///     using rope base 10000, globals using 1e6. We ran every layer global at 1e6. Guard: GetLayerAttnConfig.
/// Both verified end-to-end on CUDA (gemma3 → "Paris", matches Ollama greedy).
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task GemmaArch_VNormGatedToGemma4_AndGemma3SwaRopePattern() => await RunTest(async accelerator =>
    {
        await Task.CompletedTask; // pure metadata / pure-function test; no GPU work.

        // ── (1) Weightless V-norm is gemma4 / gemma3n ONLY ──
        foreach (var a in new[] { "gemma4", "gemma4-assistant", "gemma3n" })
            if (!GGUFGraphBuilder.UsesWeightlessVNorm(a)) throw new Exception($"arch '{a}' SHOULD use weightless V-norm");
        foreach (var a in new[] { "gemma3", "gemma2", "gemma", "llama", "qwen2" })
            if (GGUFGraphBuilder.UsesWeightlessVNorm(a)) throw new Exception($"arch '{a}' must NOT use weightless V-norm (only Q/K are normed)");

        // ── (2) gemma3 SWA: 5:1 local:global (period 6), local rope 10000 / window 512, global 1e6 / window 0 ──
        var bytes = BuildMetadataOnlyGemma3GGUF(blockCount: 12, headCount: 4, headCountKv: 1,
            keyLength: 256, slidingWindow: 512, ropeFreqBase: 1_000_000f, embd: 640);
        var model = GGUFParser.Parse(bytes);
        for (int L = 0; L < 12; L++)
        {
            var cfg = GGUFGraphBuilder.GetLayerAttnConfig(model, L, nHeads: 4, defaultNKV: 1, defaultHeadDim: 160);
            bool expectGlobal = (L % 6) == 5; // layers 5, 11
            if (cfg.IsGlobal != expectGlobal)
                throw new Exception($"gemma3 layer {L}: IsGlobal={cfg.IsGlobal}, expected {expectGlobal} (5:1, every 6th global)");
            float wantBase = expectGlobal ? 1_000_000f : 10_000f;
            if (cfg.RopeBase != wantBase)
                throw new Exception($"gemma3 layer {L}: ropeBase={cfg.RopeBase}, expected {wantBase} (global=1e6, local/SWA=10000)");
            int wantWindow = expectGlobal ? 0 : 512;
            if (cfg.Window != wantWindow)
                throw new Exception($"gemma3 layer {L}: window={cfg.Window}, expected {wantWindow}");
            if (cfg.HeadDim != 256)
                throw new Exception($"gemma3 layer {L}: headDim={cfg.HeadDim}, expected 256 (key_length, NOT embd/nHeads=160)");
        }
        Console.WriteLine("[GemmaArch] V-norm gated to gemma4/3n; gemma3 SWA 5:1 (local 10000/w512, global 1e6/w0), headDim 256: OK");
    });

    /// <summary>A metadata-only GGUF (0 tensors) carrying just the gemma3 attention hparams GetLayerAttnConfig
    /// reads. Enough to exercise the SWA pattern / rope-base resolution without building real weights.</summary>
    private static byte[] BuildMetadataOnlyGemma3GGUF(int blockCount, int headCount, int headCountKv,
        int keyLength, int slidingWindow, float ropeFreqBase, int embd)
    {
        var meta = new (string Key, object Value)[]
        {
            ("general.architecture", "gemma3"),
            ("gemma3.block_count", (uint)blockCount),
            ("gemma3.embedding_length", (uint)embd),
            ("gemma3.attention.head_count", (uint)headCount),
            ("gemma3.attention.head_count_kv", (uint)headCountKv),
            ("gemma3.attention.key_length", (uint)keyLength),
            ("gemma3.attention.sliding_window", (uint)slidingWindow),
            ("gemma3.rope.freq_base", ropeFreqBase),
        };
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);
        bw.Write((byte)'G'); bw.Write((byte)'G'); bw.Write((byte)'U'); bw.Write((byte)'F');
        bw.Write((uint)3);            // version
        bw.Write((ulong)0);           // tensor count
        bw.Write((ulong)meta.Length); // metadata KV count
        void WriteStr(string s) { var u = Encoding.UTF8.GetBytes(s); bw.Write((ulong)u.Length); bw.Write(u); }
        foreach (var (key, value) in meta)
        {
            WriteStr(key);
            switch (value)
            {
                case string s: bw.Write((uint)8); WriteStr(s); break;  // GGUFValueType.String
                case uint u: bw.Write((uint)4); bw.Write(u); break;     // GGUFValueType.UInt32
                case float f: bw.Write((uint)6); bw.Write(f); break;    // GGUFValueType.Float32
                default: throw new ArgumentException($"unsupported metadata type {value.GetType()}");
            }
        }
        // Data section starts at the next 32-byte boundary (no tensors → just padding).
        while (ms.Position % 32 != 0) bw.Write((byte)0);
        bw.Flush();
        return ms.ToArray();
    }
}
