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
        // 🔴 AN EMBEDDING MODEL, NOT A CLASSIFIER. This pointed at DistilBertSST2 until 2026-09-08, whose
        // graph declares exactly one output - `logits` [batch, 2]. FeatureExtractionPipeline read that as a
        // hidden state and zero-padded it to 768 dimensions, so every "embedding" here was a sentiment
        // vector and cosine similarity measured sentiment agreement: related -0.790, unrelated +0.842, the
        // same on all six backends.
        //
        // ⚠️ Self-similarity was 1.000 throughout and the test asserted it - the same two logits really do
        // come back for the same text, so the one check that looked like proof of correct pooling could
        // never have caught this. It takes a check that compares DIFFERENT inputs.
        //
        // all-MiniLM-L6-v2 declares last_hidden_state [batch, seq, 384] and is 86 MB rather than 257.
        var repo = ModelHub.KnownModels.AllMiniLmL6V2;
        const int HiddenSize = 384;

        var tokJson = await hf.DownloadFileAsync(repo, "tokenizer.json");
        var tok = TokenizerLoader.FromTokenizerJson(Encoding.UTF8.GetString(tokJson));
        if (tok.Tokenizer is not WordPieceTokenizer)
            throw new Exception($"embeddings tokenizer is {tok.Tokenizer.GetType().Name}, expected WordPieceTokenizer");

        var modelUrl = HuggingFaceClient.GetDownloadUrl(repo, "onnx/model.onnx");
        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http, modelUrl);
        using var session = InferenceSession.CreateFromFile(accelerator, modelBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                // ⚠️ ALL THREE. This model declares token_type_ids as well, and an input the session never
                // pins is an input the pipeline would be running against an unwritten buffer.
                ["input_ids"] = new[] { 1, 128 },
                ["attention_mask"] = new[] { 1, 128 },
                ["token_type_ids"] = new[] { 1, 128 },
            });
        using var pipe = new FeatureExtractionPipeline(session, accelerator, maxLength: 128, hiddenSize: HiddenSize);

        const string anchor = "The cat sat on the warm mat by the fire.";
        const string related = "A kitten curled up on the soft rug near the hearth.";
        const string unrelated = "Quarterly interest rates rose as the bond market sold off.";

        // 🔴 IS IT ACTUALLY AN EMBEDDING? A vector that is mostly zeros still normalises to a unit vector
        // and still gives self-similarity 1.000, so every check below can pass on one. This asserts the
        // shape of the thing being compared BEFORE comparing it - the check whose absence let two sentiment
        // logits masquerade as a 768-dimension embedding for months.
        var anchorEmbedding = (await pipe.EmbedAsync(anchor, tok)).Embedding;
        if (anchorEmbedding.Length != HiddenSize)
            throw new Exception($"embedding is {anchorEmbedding.Length} dims, expected {HiddenSize}");
        int nonZero = 0;
        foreach (var v in anchorEmbedding) if (v != 0f) nonZero++;
        Console.WriteLine($"[Embeddings] {nonZero}/{HiddenSize} dimensions are non-zero");
        if (nonZero < HiddenSize / 2)
            throw new Exception(
                $"only {nonZero} of {HiddenSize} embedding dimensions are non-zero - this is not a hidden "
              + "state. A classifier's logits read as an embedding look exactly like this: a couple of real "
              + "values padded with zeros, which still normalises to a confident unit vector.");

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
