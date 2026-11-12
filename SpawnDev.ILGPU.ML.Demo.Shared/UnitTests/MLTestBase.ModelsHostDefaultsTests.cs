using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Ltx;
using SpawnDev.ILGPU.ML.Wan;
using SpawnDev.UnitTesting;
using System.Net;
using System.Net.Http.Headers;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Every file a pipeline's DEFAULT repo paths name must be served by the models host. Added 2026-11-11: <c>Wan22Paths.InRepo()</c> defaulted its text encoder to
/// <c>umt5_xxl_enc_bf16.safetensors</c>, which models.spawndev.com does not host (404), so a default repo load of Wan failed at the text encoder; SpawnGen found it.
/// Reads byte 0 of each file (a Range request: the host answers 206) so nothing large moves. A host that cannot be reached at all skips (no network), a missing file fails.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task ModelsHost_RepoDefaultFiles_AreServed() => await RunTest(async accelerator =>
    {
        var wan = Wan22Paths.InRepo();
        var ltx = LtxPaths.InRepo();
        var files = new List<(string Repo, string File)>
        {
            (Wan22Pipeline.DefaultRepo, wan.Dit), (Wan22Pipeline.DefaultRepo, wan.Vae), (Wan22Pipeline.DefaultRepo, wan.TextEncoder), (Wan22Pipeline.DefaultRepo, wan.Tokenizer),
            (LtxPipeline.DefaultRepo, ltx.Dit), (LtxPipeline.DefaultRepo, ltx.Vae), (LtxPipeline.DefaultRepo, ltx.VaeEncoder), (LtxPipeline.DefaultRepo, ltx.Upsampler),
            (LtxPipeline.DefaultRepo, ltx.TextEncoder), (LtxPipeline.DefaultRepo, ltx.Tokenizer),
        };
        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
        var missing = new List<string>(); var unreachable = new List<string>();
        int reached = 0;
        foreach (var (repo, file) in files)
        {
            string url = $"{ModelsHost.DefaultBaseUrl}/{repo}/resolve/main/{file}";
            using var req = new HttpRequestMessage(HttpMethod.Get, url);
            req.Headers.Range = new RangeHeaderValue(0, 0);
            HttpResponseMessage resp;
            try { resp = await http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead); }
            catch (HttpRequestException e) { unreachable.Add($"{repo}/{file}: {e.Message}"); continue; }
            catch (TaskCanceledException) { unreachable.Add($"{repo}/{file}: timed out"); continue; }
            using (resp)
            {
                reached++;
                if (resp.StatusCode != HttpStatusCode.PartialContent && resp.StatusCode != HttpStatusCode.OK) missing.Add($"{repo}/{file}: HTTP {(int)resp.StatusCode}");
            }
        }
        if (reached == 0) throw new UnsupportedTestException($"{ModelsHost.DefaultBaseUrl} could not be reached (no network)");
        missing.AddRange(unreachable);     // the host answered other files: one it did not answer is not "no network"
        if (missing.Count > 0) throw new Exception("default repo files the models host does not serve: " + string.Join("; ", missing));
        return $"{reached} of {files.Count} default files reached, all served (Wan text encoder {wan.TextEncoder}, LTX {ltx.TextEncoder})";
    });
}
