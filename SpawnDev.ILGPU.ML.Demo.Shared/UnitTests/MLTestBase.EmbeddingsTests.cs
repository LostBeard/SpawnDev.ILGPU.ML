using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime;
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

        // 🔴 THE VALUES, NOT JUST THE ORDER. "related > unrelated" stayed green while the attention fusion
        // dropped this model's attention mask and swapped K's heads and sequence (2026-10-10): the 128-token
        // padding then dominated every embedding. Pinned to onnxruntime 1.26 on the same ONNX file, same
        // WordPiece ids, mean-pooled over the real tokens: related 0.5387, unrelated -0.0206.
        if (MathF.Abs(simRelated - 0.5387f) > 0.01f || MathF.Abs(simUnrelated - (-0.0206f)) > 0.01f)
            throw new Exception($"Embeddings differ from onnxruntime: related {simRelated:F4} (ort 0.5387), unrelated {simUnrelated:F4} (ort -0.0206)");

        // the device paths (pooled on the device into a caller's view; several texts into consecutive rows) = the embedding above, number for number:
        // the same kernels on the same inputs
        using var dest = accelerator.Allocate1D<float>(3 * HiddenSize);
        await pipe.EmbedAsync(anchor, tok, dest.View.SubView(0, HiddenSize));
        var viaView = await dest.View.SubView(0, HiddenSize).CopyToHostAsync();
        var texts = new[] { anchor, related, unrelated };
        var ids = new List<int[]>();
        foreach (var t in texts) { var e = tok.Encode(t); var l = new List<int> { tok.BosTokenId >= 0 ? tok.BosTokenId : 101 }; l.AddRange(e); l.Add(tok.EosTokenId >= 0 ? tok.EosTokenId : 102); ids.Add(l.ToArray()); }
        await pipe.EmbedBatchAsync(ids, dest.View);
        var batch = await dest.View.CopyToHostAsync();
        var relatedEmbedding = (await pipe.EmbedAsync(related, tok)).Embedding;
        bool viewSame = viaView.AsSpan().SequenceEqual(anchorEmbedding);
        bool batchSame = batch.AsSpan(0, HiddenSize).SequenceEqual(anchorEmbedding) && batch.AsSpan(HiddenSize, HiddenSize).SequenceEqual(relatedEmbedding);
        Console.WriteLine($"[Embeddings] into a device view: {(viewSame ? "= EmbedAsync" : "DIFFERS")}; batch rows: {(batchSame ? "= EmbedAsync" : "DIFFER")}");
        if (!viewSame || !batchSame) throw new Exception($"device-destination embeddings differ from EmbedAsync (view {viewSame}, batch {batchSame})");

        // run at the token count's bucket, not padded to 128 (BucketLengths, default on): the anchor is ~13 tokens -> 16. The vector must be the one the
        // 128-token run gives (padding is masked out of the attention and the pooling); the onnxruntime check above already ran bucketed.
        var anchorAgain = (await pipe.EmbedAsync(anchor, tok)).Embedding;
        int bucket = pipe.LastRunLength;
        pipe.BucketLengths = false;
        var padded128 = (await pipe.EmbedAsync(anchor, tok)).Embedding;
        int full = pipe.LastRunLength;
        pipe.BucketLengths = true;
        double dot = 0; for (int i = 0; i < HiddenSize; i++) dot += anchorAgain[i] * padded128[i];
        Console.WriteLine($"[Embeddings] bucketed run at {bucket} tokens vs padded to {full}: cosine {dot:F6}");
        if (session.SupportsShapeRecompile && (bucket != 16 || full != 128))
            throw new Exception($"the anchor ran at {bucket} tokens bucketed (expected 16) and {full} unbucketed (expected 128)");
        if (dot < 0.99999)
            throw new Exception($"the bucketed embedding differs from the 128-token one: cosine {dot:F6}");
        // the bucket edge (Data's ask): exactly 16 ids with [CLS] and [SEP] run at 16, 17 at 32, each the 128-token vector
        var words = tok.Encode("The lighthouse keeper watched the storm roll in over the grey water, counting the waves and the seconds between the flashes of light.");
        foreach (int total in new[] { 16, 17 })
        {
            var edge = new List<int> { tok.BosTokenId >= 0 ? tok.BosTokenId : 101 };
            edge.AddRange(words.Take(total - 2));
            edge.Add(tok.EosTokenId >= 0 ? tok.EosTokenId : 102);
            if (edge.Count != total) throw new Exception($"the edge text has {edge.Count} ids, wanted {total} (the sentence is too short)");
            var bucketed = (await pipe.EmbedAsync(edge.ToArray())).Embedding;
            int at = pipe.LastRunLength;
            pipe.BucketLengths = false;
            var full128 = (await pipe.EmbedAsync(edge.ToArray())).Embedding;
            pipe.BucketLengths = true;
            double c = 0; for (int i = 0; i < HiddenSize; i++) c += bucketed[i] * full128[i];
            int wantAt = total <= 16 ? 16 : 32;
            Console.WriteLine($"[Embeddings] {total} ids ran at {at}: cosine {c:F6} to the 128-token run");
            if (session.SupportsShapeRecompile && at != wantAt) throw new Exception($"{total} ids ran at {at}, expected {wantAt}");
            if (c < 0.99999) throw new Exception($"{total} ids at {at}: cosine {c:F6} to the 128-token run");
        }

        Console.WriteLine("[Embeddings] real-tokenizer embeddings are semantic (related > unrelated, self ≈ 1.0) and match onnxruntime.");
    });
}
