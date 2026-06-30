using System;
using System.Text;
using System.Threading.Tasks;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// End-to-end sentiment analysis on the real DistilBERT-SST2 model (Xenova/
/// distilbert-base-uncased-finetuned-sst-2-english). This is the first test that actually RUNS a
/// BERT-family classifier through our engine and checks the prediction is correct — proving the whole
/// chain: HuggingFace download → WordPiece tokenizer (via TokenizerLoader, model.type=="WordPiece") →
/// TextClassificationPipeline.ClassifyAsync(text, tokenizer) → GPU inference → POSITIVE/NEGATIVE.
///
/// Until WordPiece landed this could only ever be a repo-exists HEAD check; now it's a real assertion
/// on model output. If our InferenceSession is missing a DistilBERT op this test fails LOUD with the
/// op name (a Rule 2 library gap to fix at the source) rather than silently passing on garbage tokens.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task Sentiment_DistilBertSST2_ClassifiesPositiveAndNegative() => await RunTest(async accelerator =>
    {
        using var http = CreateHuggingFaceHttpClient();
        var hf = new HuggingFaceClient(http);
        var repo = ModelHub.KnownModels.DistilBertSST2;

        // Real WordPiece tokenizer straight from the model repo's tokenizer.json.
        var tokJsonBytes = await hf.DownloadFileAsync(repo, "tokenizer.json");
        var loaded = TokenizerLoader.FromTokenizerJson(Encoding.UTF8.GetString(tokJsonBytes));
        if (loaded.Tokenizer is not WordPieceTokenizer)
            throw new Exception($"DistilBERT tokenizer.json built a {loaded.Tokenizer.GetType().Name}, expected WordPieceTokenizer");

        // The fp32 ONNX classifier — chunked download (robust for the big file across browser backends).
        var modelUrl = HuggingFaceClient.GetDownloadUrl(repo, "onnx/model.onnx");
        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http, modelUrl);
        if (modelBytes.Length < 50_000_000)
            throw new Exception($"DistilBERT model.onnx is {modelBytes.Length:N0} bytes — too small, download likely failed");
        using var session = InferenceSession.CreateFromFile(accelerator, modelBytes);

        using var pipe = new TextClassificationPipeline(session, accelerator); // labels default NEGATIVE/POSITIVE

        var pos = await pipe.ClassifyAsync("I love this movie, it was absolutely fantastic!", loaded);
        var neg = await pipe.ClassifyAsync("This film was terrible, boring, and a complete waste of time.", loaded);

        Console.WriteLine($"[Sentiment] positive-text → {pos.TopLabel} {pos.TopConfidence:P1} (logits {string.Join(",", pos.Logits)})");
        Console.WriteLine($"[Sentiment] negative-text → {neg.TopLabel} {neg.TopConfidence:P1} (logits {string.Join(",", neg.Logits)})");

        if (pos.TopLabel != "POSITIVE")
            throw new Exception($"Positive text classified {pos.TopLabel} (conf {pos.TopConfidence:P1}) — expected POSITIVE");
        if (neg.TopLabel != "NEGATIVE")
            throw new Exception($"Negative text classified {neg.TopLabel} (conf {neg.TopConfidence:P1}) — expected NEGATIVE");
        // SST-2 is high-confidence on unambiguous text; a correct DistilBERT should be well above chance.
        if (pos.TopConfidence < 0.80f || neg.TopConfidence < 0.80f)
            throw new Exception($"Confidence too low (pos {pos.TopConfidence:P1}, neg {neg.TopConfidence:P1}) — " +
                "model ran but predictions look unreliable (tokenization or op mismatch?)");

        Console.WriteLine("[Sentiment] DistilBERT-SST2 correct on both polarities via real WordPiece tokenization.");
    });
}
