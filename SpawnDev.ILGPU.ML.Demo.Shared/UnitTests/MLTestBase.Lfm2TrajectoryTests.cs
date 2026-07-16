using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Cross-backend numerical gate for LFM2: one deterministic forward (fixed token ids - no tokenizer, no
/// sampling, no chat template) whose per-node residual trajectory is compared against a CUDA-captured
/// golden. Every backend runs the same graph, so any node whose RMS disagrees names a BACKEND CODEGEN
/// defect at the exact node - which no full-model text oracle can localize.
///
/// Why this exists (2026-07-16): LFM2 decoded coherently on CUDA and emitted token soup on WebGPU. The
/// shipped "WebGPU-verified" claim rested on a full-model E2E asserting Contains("Paris") - which passes
/// on garbage, because LFM2 emits "Paris" before it degenerates. ShortConv and the Q4_K M=1 GEMV both
/// pass their CPU-oracle kernel tests on WebGPU, so the defect lives elsewhere in the forward; this test
/// finds WHERE instead of guessing (Rule 4b). It also permanently guards every LFM2 op on every backend.
///
/// Golden: captured on RTX 4070 / CUDA at a 96-token prefill (Lfm2TrajIds) via
///   GGUF_CAPTURE=1 GGUF_CAPTURE_ALL=1 dotnet run --project Examples/04.GGUFTextGen.Console \
///     -- LFM2-1.2B-Q4_K_M.gguf "&lt;Lfm2TrajIds, comma-separated&gt;"
/// Both sides truncate each node at CaptureMaxElements=40000, so the RMS covers the same deterministic
/// prefix on every backend. Regenerate (and re-verify on a CPU/OpenCL lane) if the LFM2 graph, the ids,
/// the capture cap, or the GGUF file changes.
///
/// Run: PMT_EXCLUDE_CATEGORIES= PMT_FILTER=Lfm2Trajectory dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj
/// </summary>
public abstract partial class MLTestBase
{
    // 96 tokens: id 1 = LFM2's <|startoftext|> BOS, then a deterministic spread of valid ids. Length matters -
    // the demo prefills ~100 tokens (its system prompt alone is ~90), and a shape-dependent codegen defect is
    // invisible at seq=5. Must match the ids the golden was captured with:
    //   dotnet run --project Examples/04.GGUFTextGen.Console -- LFM2-1.2B-Q4_K_M.gguf "<these ids>"
    private static readonly int[] Lfm2TrajIds =
        new[] { 1 }.Concat(Enumerable.Range(0, 95).Select(i => (i * 37 + 100) % 60000 + 10)).ToArray();

    // CUDA golden: "<nodeKey>=<rms>" per line, in graph order.
    private const string Lfm2CudaGoldenRms = @"
000_Gather_embed_out=0.013827
001_RMSNormalization_blk.0_attn_norm=0.335266
002_MatMul_blk.0_sc_bcx=0.702913
003_ShortConv_blk.0_sc_y=0.038981
004_MatMul_blk.0_sc_out=0.055363
005_AddRMSNorm_blk.0_res1=0.052076
006_MatMul_blk.0_gate=0.103532
007_MatMul_blk.0_up=0.075153
008_SwiGLU_blk.0_ffn_act=0.005951
009_MatMul_blk.0_ffn_out=0.016920
010_Add_block_0_out=0.042255
011_RMSNormalization_blk.1_attn_norm=0.232039
012_MatMul_blk.1_sc_bcx=0.451906
013_ShortConv_blk.1_sc_y=0.009494
014_MatMul_blk.1_sc_out=0.015487
015_AddRMSNorm_blk.1_res1=0.037006
016_MatMul_blk.1_gate=0.134459
017_MatMul_blk.1_up=0.092070
018_SwiGLU_blk.1_ffn_act=0.006840
019_MatMul_blk.1_ffn_out=0.017082
020_Add_block_1_out=0.037017
021_RMSNormalization_blk.2_attn_norm=0.035650
022_MatMul_blk.2_q=0.059846
023_MatMul_blk.2_k=0.052751
024_MatMul_blk.2_v=0.028560
028_RMSNormalization_blk.2_q_qknorm=1.461288
029_RMSNormalization_blk.2_k_qknorm=1.500908
030_RoPE_blk.2_q_roped=1.461288
031_RoPE_blk.2_k_roped=1.500908
032_FusedAttention_blk.2_attn_val=0.014844
034_MatMul_blk.2_attn_out=0.015847
035_AddRMSNorm_blk.2_res1=0.036946
036_MatMul_blk.2_gate=0.125073
037_MatMul_blk.2_up=0.081384
038_SwiGLU_blk.2_ffn_act=0.005643
039_MatMul_blk.2_ffn_out=0.015435
040_Add_block_2_out=0.032244
041_RMSNormalization_blk.3_attn_norm=0.246262
042_MatMul_blk.3_sc_bcx=0.445943
043_ShortConv_blk.3_sc_y=0.007522
044_MatMul_blk.3_sc_out=0.008594
045_AddRMSNorm_blk.3_res1=0.032675
046_MatMul_blk.3_gate=0.115234
047_MatMul_blk.3_up=0.082949
048_SwiGLU_blk.3_ffn_act=0.004816
049_MatMul_blk.3_ffn_out=0.011348
050_Add_block_3_out=0.032166
051_RMSNormalization_blk.4_attn_norm=0.249953
052_MatMul_blk.4_sc_bcx=0.480481
053_ShortConv_blk.4_sc_y=0.008056
054_MatMul_blk.4_sc_out=0.009256
055_AddRMSNorm_blk.4_res1=0.033743
056_MatMul_blk.4_gate=0.126782
057_MatMul_blk.4_up=0.108696
058_SwiGLU_blk.4_ffn_act=0.007346
059_MatMul_blk.4_ffn_out=0.018149
060_Add_block_4_out=0.041698
061_RMSNormalization_blk.5_attn_norm=0.053532
062_MatMul_blk.5_q=0.085238
063_MatMul_blk.5_k=0.072210
064_MatMul_blk.5_v=0.047702
068_RMSNormalization_blk.5_q_qknorm=1.454080
069_RMSNormalization_blk.5_k_qknorm=1.603102
070_RoPE_blk.5_q_roped=1.454080
071_RoPE_blk.5_k_roped=1.603102
072_FusedAttention_blk.5_attn_val=0.032063
074_MatMul_blk.5_attn_out=0.027251
075_AddRMSNorm_blk.5_res1=0.048800
076_MatMul_blk.5_gate=0.135197
077_MatMul_blk.5_up=0.101024
078_SwiGLU_blk.5_ffn_act=0.007702
079_MatMul_blk.5_ffn_out=0.017172
080_Add_block_5_out=0.048691
081_RMSNormalization_blk.6_attn_norm=0.256322
082_MatMul_blk.6_sc_bcx=0.496458
083_ShortConv_blk.6_sc_y=0.013253
084_MatMul_blk.6_sc_out=0.015914
085_AddRMSNorm_blk.6_res1=0.049046
086_MatMul_blk.6_gate=0.146720
087_MatMul_blk.6_up=0.110227
088_SwiGLU_blk.6_ffn_act=0.010287
089_MatMul_blk.6_ffn_out=0.018843
090_Add_block_6_out=0.048864
091_RMSNormalization_blk.7_attn_norm=0.256288
092_MatMul_blk.7_sc_bcx=0.613377
093_ShortConv_blk.7_sc_y=0.012408
094_MatMul_blk.7_sc_out=0.013377
095_AddRMSNorm_blk.7_res1=0.050338
096_MatMul_blk.7_gate=0.162849
097_MatMul_blk.7_up=0.123066
098_SwiGLU_blk.7_ffn_act=0.098043
099_MatMul_blk.7_ffn_out=0.221497
100_Add_block_7_out=0.222293
101_RMSNormalization_blk.8_attn_norm=0.075969
102_MatMul_blk.8_q=0.113945
103_MatMul_blk.8_k=0.094005
104_MatMul_blk.8_v=0.064488
108_RMSNormalization_blk.8_q_qknorm=1.384486
109_RMSNormalization_blk.8_k_qknorm=1.482168
110_RoPE_blk.8_q_roped=1.384486
111_RoPE_blk.8_k_roped=1.482168
112_FusedAttention_blk.8_attn_val=0.032225
114_MatMul_blk.8_attn_out=0.024434
115_AddRMSNorm_blk.8_res1=0.224759
116_MatMul_blk.8_gate=0.149474
117_MatMul_blk.8_up=0.106674
118_SwiGLU_blk.8_ffn_act=0.008976
119_MatMul_blk.8_ffn_out=0.022185
120_Add_block_8_out=0.224928
121_RMSNormalization_blk.9_attn_norm=0.274236
122_MatMul_blk.9_sc_bcx=0.509554
123_ShortConv_blk.9_sc_y=0.016301
124_MatMul_blk.9_sc_out=0.022046
125_AddRMSNorm_blk.9_res1=0.240125
126_MatMul_blk.9_gate=0.197982
127_MatMul_blk.9_up=0.146987
128_SwiGLU_blk.9_ffn_act=0.015650
129_MatMul_blk.9_ffn_out=0.037054
130_Add_block_9_out=0.243833
131_RMSNormalization_blk.10_attn_norm=0.118810
132_MatMul_blk.10_q=0.177366
133_MatMul_blk.10_k=0.134313
134_MatMul_blk.10_v=0.092397
138_RMSNormalization_blk.10_q_qknorm=1.399312
139_RMSNormalization_blk.10_k_qknorm=1.575977
140_RoPE_blk.10_q_roped=1.399312
141_RoPE_blk.10_k_roped=1.575977
142_FusedAttention_blk.10_attn_val=0.042552
144_MatMul_blk.10_attn_out=0.038945
145_AddRMSNorm_blk.10_res1=0.249355
146_MatMul_blk.10_gate=0.257272
147_MatMul_blk.10_up=0.161628
148_SwiGLU_blk.10_ffn_act=0.025761
149_MatMul_blk.10_ffn_out=0.056966
150_Add_block_10_out=0.256190
151_RMSNormalization_blk.11_attn_norm=0.320958
152_MatMul_blk.11_sc_bcx=0.650350
153_ShortConv_blk.11_sc_y=0.037299
154_MatMul_blk.11_sc_out=0.048848
155_AddRMSNorm_blk.11_res1=0.269318
156_MatMul_blk.11_gate=0.293817
157_MatMul_blk.11_up=0.187581
158_SwiGLU_blk.11_ffn_act=0.025737
159_MatMul_blk.11_ffn_out=0.059474
160_Add_block_11_out=0.274277
161_RMSNormalization_blk.12_attn_norm=0.144591
162_MatMul_blk.12_q=0.198361
163_MatMul_blk.12_k=0.168824
164_MatMul_blk.12_v=0.191090
168_RMSNormalization_blk.12_q_qknorm=1.476124
169_RMSNormalization_blk.12_k_qknorm=1.604555
170_RoPE_blk.12_q_roped=1.476124
171_RoPE_blk.12_k_roped=1.604555
172_FusedAttention_blk.12_attn_val=0.080595
174_MatMul_blk.12_attn_out=0.079747
175_AddRMSNorm_blk.12_res1=0.289339
176_MatMul_blk.12_gate=0.407578
177_MatMul_blk.12_up=0.239834
178_SwiGLU_blk.12_ffn_act=0.045972
179_MatMul_blk.12_ffn_out=0.100717
180_Add_block_12_out=0.310214
181_RMSNormalization_blk.13_attn_norm=0.368347
182_MatMul_blk.13_sc_bcx=0.731480
183_ShortConv_blk.13_sc_y=0.056576
184_MatMul_blk.13_sc_out=0.068024
185_AddRMSNorm_blk.13_res1=0.326217
186_MatMul_blk.13_gate=0.509662
187_MatMul_blk.13_up=0.281288
188_SwiGLU_blk.13_ffn_act=0.059933
189_MatMul_blk.13_ffn_out=0.118405
190_Add_block_13_out=0.344304
191_RMSNormalization_blk.14_attn_norm=0.200752
192_MatMul_blk.14_q=0.267983
193_MatMul_blk.14_k=0.241787
194_MatMul_blk.14_v=0.477873
198_RMSNormalization_blk.14_q_qknorm=1.449299
199_RMSNormalization_blk.14_k_qknorm=1.376929
200_RoPE_blk.14_q_roped=1.449299
201_RoPE_blk.14_k_roped=1.376929
202_FusedAttention_blk.14_attn_val=0.165877
204_MatMul_blk.14_attn_out=0.173915
205_AddRMSNorm_blk.14_res1=0.402407
206_MatMul_blk.14_gate=0.583313
207_MatMul_blk.14_up=0.380507
208_SwiGLU_blk.14_ffn_act=0.147761
209_MatMul_blk.14_ffn_out=0.290851
210_Add_block_14_out=0.508633
211_RMSNormalization_blk.15_attn_norm=0.400113
212_MatMul_blk.15_sc_bcx=0.996720
213_ShortConv_blk.15_sc_y=0.163847
214_MatMul_blk.15_sc_out=0.179643
215_AddRMSNorm_blk.15_res1=0.498749
216_MatMul_blk.15_gate=0.647134
217_MatMul_blk.15_up=0.568800
218_SwiGLU_blk.15_ffn_act=0.239459
219_MatMul_blk.15_ffn_out=0.570894
220_Add_block_15_out=0.610461
221_RMSNormalization_final_norm_out=1.000448
222_MatMul_logits=2.073971
";

    /// <summary>
    /// LFM2 + the KV-PREFIX CACHE: asking the SAME question twice on one pipeline must give the SAME greedy
    /// answer. It did not.
    ///
    /// Prefix reuse (GgufGenerator, EnablePrefixCache=true BY DEFAULT - the demo runs it) reuses the longest
    /// common token prefix P with the resident sequence, sets the cursor to P and prefills only the suffix.
    /// That is valid for K/V, which is POSITION-ADDRESSED (row = absolute position, RoPE matches). It is NOT
    /// valid for the shortconv layers: ShortConvStateCache is a SHIFT REGISTER holding the history for exactly
    /// ONE cursor - the position the previous turn ENDED at. Any P other than that position makes the conv
    /// layers read a history describing different tokens. Asking the same question twice is the minimal
    /// trigger (P = promptLen-1, but the state sits at promptLen+responseLen); in the demo it fires whenever a
    /// new conversation, an edited system prompt, or a re-ask shares only a shorter prefix.
    ///
    /// Backend-agnostic (unlike the capture/replay defects) - this breaks CUDA too, and no existing test caught
    /// it because the identity gate DISABLES the prefix cache and qwen/gemma have no conv state.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy,HeavyCpu", RetryCount = 1)]
    public async Task Lfm2_PrefixCacheReuse_SamePromptTwice_IsIdentical()
        => await SamePromptTwiceIsIdentical("LiquidAI/LFM2-1.2B-GGUF", "LFM2-1.2B-Q4_K_M.gguf", "LFM2");

    /// <summary>CONTROL for the LFM2 case above: qwen2.5 has NO conv state, so if IT also returns a different
    /// answer to the same prompt twice on a backend, that backend has a generation-to-generation reset problem
    /// of its own and the conv state is exonerated there. Keeps me from attributing a lane's pre-existing bug
    /// to my change (and vice versa).</summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy,HeavyCpu", RetryCount = 1)]
    public async Task Qwen25_PrefixCacheReuse_SamePromptTwice_IsIdentical()
        => await SamePromptTwiceIsIdentical("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf", "qwen2.5");

    private async Task SamePromptTwiceIsIdentical(string repoId, string file, string tag) => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var client = new SpawnDev.WebTorrent.WebTorrentClient();
        bool prevPrefix = SpawnDev.ILGPU.ML.Pipelines.GgufGenerator.EnablePrefixCache;
        SpawnDev.ILGPU.ML.Pipelines.GgufGenerator.EnablePrefixCache = true;   // the demo's default
        try
        {
            var hub = new SpawnDev.ILGPU.ML.Hub.HubModelStream(client, http) { PrepareTimeout = TimeSpan.FromMinutes(8) };
            using var cts = new System.Threading.CancellationTokenSource(TimeSpan.FromMinutes(9));
            var model = await hub.OpenAsync(repoId, file, deselect: false, cts.Token);
            await using (model.Stream)
            using (var pipe = await SpawnDev.ILGPU.ML.Pipelines.GgufTextGenerationPipeline.CreateFromStreamAsync(
                accelerator, model.Stream, maxSeqLen: 4096, ct: cts.Token))
            {
                var msgs = new[] { ("user", "In two sentences, what is a chicken?") };
                var cfg = new SpawnDev.ILGPU.ML.Preprocessing.GenerationConfig { MaxNewTokens = 48, Strategy = "greedy" };
                var first = await pipe.GenerateAsync(msgs, config: cfg, ct: cts.Token);
                var second = await pipe.GenerateAsync(msgs, config: cfg, ct: cts.Token);   // hits prefix reuse
                Console.WriteLine($"[{tag}-prefix] reusedPrefix={pipe.LastReusedPrefix}");
                if (second != first)
                {
                    int at = 0; while (at < Math.Min(first.Length, second.Length) && first[at] == second[at]) at++;
                    throw new Exception(
                        $"[{tag}-prefix] SAME prompt, greedy, gave DIFFERENT answers - prefix-cache reuse (P="
                        + $"{pipe.LastReusedPrefix}) fed the shortconv layers a conv state belonging to a different "
                        + $"cursor. Diverges at char {at}.\n  first ='{first.Trim()}'\n  second='{second.Trim()}'");
                }
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network")
            || ex.Message.Contains("magnet") || ex.Message.Contains("preparing") || ex is TimeoutException)
        {
            throw new UnsupportedTestException($"[{tag}-prefix] hub/network unavailable: {ex.Message}");
        }
        finally
        {
            SpawnDev.ILGPU.ML.Pipelines.GgufGenerator.EnablePrefixCache = prevPrefix;
            await client.DisposeAsync();
        }
    });

    /// <summary>
    /// Is the LFM2 FORWARD itself reproducible on this backend? Runs the SAME fixed-id forward twice in one
    /// session and diffs the per-node trajectory against ITSELF (no golden, no decode, no sampling, no conv
    /// STATE - prefill runs at pastLen=0, where the conv zero-pads and never reads the state buffer).
    ///
    /// Why (2026-07-16): on WebGL, LFM2 answers the same prompt differently on a second generation and diverges
    /// at char 0 - the FIRST generated token, which comes from the prefill's last-position logits. Both runs do
    /// a full prefill, so no state can carry over: that points at the forward being non-reproducible, not at the
    /// conv-state cache. The qwen2.5 control passes on WebGL, so it is not backend-wide. Two hypotheses already
    /// died against evidence (Reset() disposing buffers; per-call params-buffer churn), so this stops guessing
    /// and names the FIRST node whose value changes between two identical forwards.
    ///
    /// RESULT (2026-07-16): **GREEN on every backend incl. WebGL** - the full-recompute forward IS bit-
    /// reproducible. So the WebGL bug is NOT forward nondeterminism (third hypothesis dead).
    /// ⚠️ MIND THE GAP this test does NOT cover: `RunAsync` is the FULL-RECOMPUTE path, which routes ShortConv
    /// through `ShortConvOperator` and **never touches `ShortConvStateCache`**. The pipeline's prefill is
    /// `RunDecodeStepAsync` at pastLen=0, which DOES run the state cache (zero-pads, then snapshots the tail).
    /// So the remaining suspect is the DECODE-path prefill / state update on WebGL, not the maths. The next
    /// probe should capture two decode-path prefills (EnableGGUFDecode + ResetGGUFDecode between) and bit-diff
    /// them - that needs the per-layer kvHeads/headDim geometry, so build it like
    /// `Lfm2Decode_IncrementalMatchesFullRecompute` in MLTestBase.ShortConvTests does.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy,HeavyCpu", RetryCount = 1)]
    public async Task Lfm2Trajectory_TwoIdenticalForwards_AreReproducible() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var client = new SpawnDev.WebTorrent.WebTorrentClient();
        try
        {
            var hub = new SpawnDev.ILGPU.ML.Hub.HubModelStream(client, http) { PrepareTimeout = TimeSpan.FromMinutes(8) };
            using var cts = new System.Threading.CancellationTokenSource(TimeSpan.FromMinutes(9));
            var model = await hub.OpenAsync("LiquidAI/LFM2-1.2B-GGUF", "LFM2-1.2B-Q4_K_M.gguf", deselect: false, cts.Token);

            await using (model.Stream)
            using (var session = await InferenceSession.CreateFromGGUFStreamAsync(accelerator, model.Stream, ct: cts.Token))
            {
                async Task<Dictionary<string, float[]>> ForwardCapture()
                {
                    GraphExecutor.CapturedOutputs = new();
                    GraphExecutor.CapturedNodeInfo = new();
                    GraphExecutor.CaptureMaxElements = 40000;
                    try
                    {
                        var idf = Lfm2TrajIds.Select(i => (float)i).ToArray();
                        using var inBuf = accelerator.Allocate1D(idf);
                        var input = new Tensor(inBuf.View, new[] { 1, Lfm2TrajIds.Length }, "input_ids");
                        await session.RunAsync(new Dictionary<string, Tensor> { ["input_ids"] = input });
                        await accelerator.SynchronizeAsync();
                        return new Dictionary<string, float[]>(GraphExecutor.CapturedOutputs!);
                    }
                    finally
                    {
                        GraphExecutor.CapturedOutputs = null;
                        GraphExecutor.CapturedNodeInfo = null;
                        GraphExecutor.CaptureMaxElements = 1024;
                    }
                }

                var a = await ForwardCapture();
                var b = await ForwardCapture();

                // BIT-exact is the right bar: identical inputs, identical kernels, same device, same order.
                foreach (var key in a.Keys)
                {
                    if (!b.TryGetValue(key, out var vb)) throw new Exception($"[LFM2-repro] node {key} missing from run B");
                    var va = a[key];
                    if (va.Length != vb.Length) throw new Exception($"[LFM2-repro] node {key} length {va.Length} vs {vb.Length}");
                    for (int i = 0; i < va.Length; i++)
                        if (va[i] != vb[i])
                            throw new Exception($"[{accelerator.AcceleratorType}] [LFM2-repro] the SAME forward run twice " +
                                $"gave DIFFERENT values - FIRST divergent node {key} at element {i}: {va[i]} vs {vb[i]}. " +
                                "No decode, no sampling, no conv state (prefill zero-pads) - the forward itself is " +
                                "not reproducible on this backend.");
                }
                Console.WriteLine($"[LFM2-repro] {accelerator.AcceleratorType}: {a.Count} nodes bit-identical across two forwards.");
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network")
            || ex.Message.Contains("magnet") || ex.Message.Contains("preparing") || ex is TimeoutException)
        {
            throw new UnsupportedTestException($"[LFM2-repro] hub/network unavailable: {ex.Message}");
        }
        finally { await client.DisposeAsync(); }
    });

    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy,HeavyCpu", RetryCount = 1)]
    public async Task Lfm2Trajectory_MatchesCudaGolden() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var client = new SpawnDev.WebTorrent.WebTorrentClient();
        try
        {
            var hub = new SpawnDev.ILGPU.ML.Hub.HubModelStream(client, http) { PrepareTimeout = TimeSpan.FromMinutes(8) };
            using var cts = new System.Threading.CancellationTokenSource(TimeSpan.FromMinutes(9));
            var model = await hub.OpenAsync("LiquidAI/LFM2-1.2B-GGUF", "LFM2-1.2B-Q4_K_M.gguf", deselect: false, cts.Token);

            await using (model.Stream)
            using (var session = await InferenceSession.CreateFromGGUFStreamAsync(accelerator, model.Stream, ct: cts.Token))
            {
                GraphExecutor.CapturedOutputs = new();
                GraphExecutor.CapturedNodeInfo = new();
                GraphExecutor.CaptureMaxElements = 40000;   // widest node: sc_bcx = 5 * 3*2048 = 30720
                Dictionary<string, float[]> caps;
                try
                {
                    var idf = Lfm2TrajIds.Select(i => (float)i).ToArray();
                    using var inBuf = accelerator.Allocate1D(idf);
                    var input = new Tensor(inBuf.View, new[] { 1, Lfm2TrajIds.Length }, "input_ids");
                    await session.RunAsync(new Dictionary<string, Tensor> { ["input_ids"] = input });
                    await accelerator.SynchronizeAsync();
                    caps = new Dictionary<string, float[]>(GraphExecutor.CapturedOutputs!);
                }
                finally
                {
                    GraphExecutor.CapturedOutputs = null;
                    GraphExecutor.CapturedNodeInfo = null;
                    GraphExecutor.CaptureMaxElements = 1024;
                }

                // Parse golden (graph order preserved).
                var golden = new List<(string Key, double Rms)>();
                foreach (var line in Lfm2CudaGoldenRms.Split('\n', StringSplitOptions.RemoveEmptyEntries))
                {
                    var s = line.Trim();
                    if (s.Length == 0) continue;
                    int eq = s.LastIndexOf('=');
                    golden.Add((s[..eq], double.Parse(s[(eq + 1)..], System.Globalization.CultureInfo.InvariantCulture)));
                }

                double RmsOf(float[] v)
                {
                    double s2 = 0;
                    for (int i = 0; i < v.Length; i++) s2 += (double)v[i] * v[i];
                    return v.Length > 0 ? Math.Sqrt(s2 / v.Length) : 0;
                }

                // 8% relative tolerance (abs floor 2e-4): absorbs Q4_K dequant + reduction-order differences
                // between backends, while a real codegen defect moves RMS by orders of magnitude.
                const double relTol = 0.08, absFloor = 2e-4;
                var diverged = new List<string>();
                int missing = 0;
                foreach (var (key, expect) in golden)
                {
                    if (!caps.TryGetValue(key, out var v)) { missing++; continue; }
                    double actual = RmsOf(v);
                    double allow = Math.Max(absFloor, Math.Abs(expect) * relTol);
                    if (Math.Abs(actual - expect) > allow || double.IsNaN(actual) || double.IsInfinity(actual))
                        diverged.Add($"{key}: cuda={expect:F6} {accelerator.AcceleratorType}={actual:F6} " +
                                     $"(delta={(actual - expect):+0.000000;-0.000000} allow=±{allow:F6})");
                }
                if (missing > golden.Count / 2)
                    throw new Exception($"[LFM2-TRAJ] only {golden.Count - missing}/{golden.Count} golden nodes " +
                        $"were captured - the graph changed; regenerate the golden. Captured keys={caps.Count}");

                if (diverged.Count > 0)
                {
                    // The FIRST divergence is the culprit; everything after it is downstream contamination.
                    var head = string.Join("\n  ", diverged.Take(8));
                    throw new Exception(
                        $"[LFM2-TRAJ] {accelerator.AcceleratorType}: {diverged.Count}/{golden.Count} nodes diverge " +
                        $"from the CUDA golden. FIRST (= the defect; later ones are downstream):\n  {head}");
                }
                Console.WriteLine($"[LFM2-TRAJ] {accelerator.AcceleratorType}: all {golden.Count} nodes match the CUDA golden.");
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network")
            || ex.Message.Contains("magnet") || ex.Message.Contains("preparing") || ex is TimeoutException)
        {
            throw new UnsupportedTestException($"[LFM2-TRAJ] hub/network unavailable: {ex.Message}");
        }
        finally { await client.DisposeAsync(); }
    });
}
