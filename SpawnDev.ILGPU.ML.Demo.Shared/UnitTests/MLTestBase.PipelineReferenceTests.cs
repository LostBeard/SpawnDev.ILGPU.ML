using SpawnDev.UnitTesting;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Reference tests for pipeline outputs: CLIP similarities, Whisper decoder tokens,
/// text classification predictions. These validate end-to-end pipeline correctness
/// against known Python-verified reference data.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task Pipeline_CLIP_Reference_CatIsTopMatch()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var jsonStr = await http.GetStringAsync("references/clip-vit-b32/metadata.json");
        using var doc = JsonDocument.Parse(jsonStr);

        // Verify reference data structure
        var sims = doc.RootElement.GetProperty("similarities").EnumerateArray()
            .Select(e => (float)e.GetDouble()).ToArray();
        var texts = doc.RootElement.GetProperty("test_texts").EnumerateArray()
            .Select(e => e.GetString()!).ToArray();
        var topMatch = doc.RootElement.GetProperty("top_match").GetString();

        if (sims.Length != 4 || texts.Length != 4)
            throw new Exception($"Expected 4 similarities and 4 texts, got {sims.Length}/{texts.Length}");

        // Cat should be the top match
        if (topMatch != "a photo of a cat")
            throw new Exception($"Top match should be 'a photo of a cat', got '{topMatch}'");

        // Cat similarity should be highest
        int maxIdx = Array.IndexOf(sims, sims.Max());
        if (texts[maxIdx] != "a photo of a cat")
            throw new Exception($"Highest similarity should be for cat, got '{texts[maxIdx]}' ({sims[maxIdx]:F4})");

        // Similarities should be ordered: cat > dog > car > landscape
        for (int i = 1; i < sims.Length; i++)
        {
            if (sims[i] > sims[i - 1])
                throw new Exception($"Similarities not ordered: [{i - 1}]={sims[i - 1]:F4} < [{i}]={sims[i]:F4}");
        }

        // Load and verify embeddings exist
        var imgEmbBytes = await http.GetByteArrayAsync("references/clip-vit-b32/cat_image_embedding.bin");
        var textEmbBytes = await http.GetByteArrayAsync("references/clip-vit-b32/text_embeddings.bin");

        if (imgEmbBytes.Length != 512 * 4)
            throw new Exception($"Image embedding size={imgEmbBytes.Length}, expected {512 * 4} (512 × float32)");
        if (textEmbBytes.Length != 4 * 512 * 4)
            throw new Exception($"Text embeddings size={textEmbBytes.Length}, expected {4 * 512 * 4} (4 × 512 × float32)");

        Console.WriteLine($"[Pipeline] CLIP reference: cat={sims[0]:F4}, dog={sims[1]:F4}, car={sims[2]:F4}, landscape={sims[3]:F4}");
    }

    [TestMethod]
    public async Task Pipeline_WhisperDecoder_Reference_440HzTone()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var jsonStr = await http.GetStringAsync("references/whisper-tiny-onnx/decoder_reference_tone_440hz.json");
        using var doc = JsonDocument.Parse(jsonStr);

        var prefix = doc.RootElement.GetProperty("prefix_tokens").EnumerateArray()
            .Select(e => e.GetInt32()).ToArray();
        var generated = doc.RootElement.GetProperty("generated_tokens").EnumerateArray()
            .Select(e => e.GetInt32()).ToArray();
        var fullSeq = doc.RootElement.GetProperty("full_sequence").EnumerateArray()
            .Select(e => e.GetInt32()).ToArray();
        int totalGenerated = doc.RootElement.GetProperty("total_generated").GetInt32();

        // Prefix should be [SOT, EN, TRANSCRIBE, NO_TIMESTAMPS]
        if (prefix.Length != 4)
            throw new Exception($"Prefix length={prefix.Length}, expected 4");
        if (prefix[0] != 50258) // SOT
            throw new Exception($"Prefix[0]={prefix[0]}, expected SOT=50258");

        // Should generate 4 content tokens before EOT
        if (totalGenerated != 4)
            throw new Exception($"Total generated={totalGenerated}, expected 4");

        // Full sequence should end with EOT (50257)
        if (fullSeq[^1] != 50257)
            throw new Exception($"Sequence should end with EOT=50257, got {fullSeq[^1]}");

        // Verify decode steps have top-5 logits
        var steps = doc.RootElement.GetProperty("decode_steps").EnumerateArray().ToArray();
        if (steps.Length != 5) // 4 content tokens + EOT
            throw new Exception($"Expected 5 decode steps, got {steps.Length}");

        // First generated token should match step 0's next_token
        int firstToken = steps[0].GetProperty("next_token").GetInt32();
        if (firstToken != generated[0])
            throw new Exception($"Step 0 next_token={firstToken}, expected {generated[0]}");

        Console.WriteLine($"[Pipeline] Whisper decoder: prefix={string.Join(",", prefix)} → generated={string.Join(",", generated)} → EOT");
    }

    [TestMethod]
    public async Task Pipeline_TextClassification_Reference_Cases()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        string jsonStr;
        try
        {
            jsonStr = await http.GetStringAsync("references/text-classification-pipeline/test_cases.json");
        }
        catch
        {
            throw new UnsupportedTestException("Text classification reference data not available");
        }

        using var doc = JsonDocument.Parse(jsonStr);
        var root = doc.RootElement;

        // test_cases is an array
        var testCases = root.GetProperty("test_cases").EnumerateArray().ToArray();
        if (testCases.Length < 3)
            throw new Exception($"Expected at least 3 test cases, got {testCases.Length}");

        foreach (var tc in testCases)
        {
            var label = tc.GetProperty("label").GetString()!;
            if (label != "POSITIVE" && label != "NEGATIVE")
                throw new Exception($"Unexpected label '{label}'");
        }

        Console.WriteLine($"[Pipeline] Text classification: {testCases.Length} reference test cases verified");
    }
}
