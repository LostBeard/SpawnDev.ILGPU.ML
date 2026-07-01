using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// End-to-end semantic-embedding test for the /embeddings demo. The page used to feed a WORD-HASH
/// (<c>word.GetHashCode() % 28000</c>) as token ids — garbage unrelated to the model vocab, so its
/// "search by meaning" claim was fake and there was no test. It now uses the real WordPiece tokenizer
/// (<see cref="TokenizerLoader"/> → <see cref="FeatureExtractionPipeline.EmbedAsync(string, LoadedTokenizer)"/>).
/// This proves the embeddings are actually semantic: a related sentence pair must score higher cosine
/// similarity than an unrelated pair, and identical text must be ~1.0.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task Embeddings_RealTokenizer_RelatedScoresHigherThanUnrelated() => await RunTest(async accelerator =>
    {
        using var http = CreateHuggingFaceHttpClient();
        var hf = new HuggingFaceClient(http);
        var repo = ModelHub.KnownModels.DistilBertSST2;

        var tokJson = await hf.DownloadFileAsync(repo, "tokenizer.json");
        var tok = TokenizerLoader.FromTokenizerJson(Encoding.UTF8.GetString(tokJson));
        if (tok.Tokenizer is not WordPieceTokenizer)
            throw new Exception($"embeddings tokenizer is {tok.Tokenizer.GetType().Name}, expected WordPieceTokenizer");

        var modelUrl = HuggingFaceClient.GetDownloadUrl(repo, "onnx/model.onnx");
        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http, modelUrl);
        using var session = InferenceSession.CreateFromFile(accelerator, modelBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["input_ids"] = new[] { 1, 128 },
                ["attention_mask"] = new[] { 1, 128 },
            });
        using var pipe = new FeatureExtractionPipeline(session, accelerator, maxLength: 128, hiddenSize: 768);

        const string anchor = "The cat sat on the warm mat by the fire.";
        const string related = "A kitten curled up on the soft rug near the hearth.";
        const string unrelated = "Quarterly interest rates rose as the bond market sold off.";

        float simRelated = await pipe.SimilarityAsync(anchor, related, tok);
        float simUnrelated = await pipe.SimilarityAsync(anchor, unrelated, tok);
        float simSelf = await pipe.SimilarityAsync(anchor, anchor, tok);
        Console.WriteLine($"[Embeddings] self={simSelf:F3} related={simRelated:F3} unrelated={simUnrelated:F3}");

        // Identical text → L2-normalized embeddings are the same vector → cosine ≈ 1.
        if (simSelf < 0.98f)
            throw new Exception($"Self-similarity {simSelf:F3} should be ~1.0 — embedding/pooling is wrong");
        // The semantic claim: the topically-related pair must out-score the unrelated pair.
        if (simRelated <= simUnrelated)
            throw new Exception($"Embeddings are not semantic: related {simRelated:F3} must exceed unrelated {simUnrelated:F3} " +
                "(real WordPiece tokenization should give meaningful similarity).");

        Console.WriteLine("[Embeddings] real-tokenizer embeddings are semantic (related > unrelated, self ≈ 1.0).");
    });
}
