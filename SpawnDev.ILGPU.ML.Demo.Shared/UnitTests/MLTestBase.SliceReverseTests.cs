using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Gate for Slice with a NEGATIVE step - <c>x[..., ::-1]</c>.
///
/// <para>
/// WHY THIS FILE EXISTS: <c>SliceOperator.Execute</c> clamped <c>ends</c> into <c>[0, dim]</c> whatever the
/// sign of the step. ONNX clamps it into <c>[-1, dim-1]</c> for a negative step, because a reversed slice
/// legitimately ends one BEFORE index 0 - which is precisely what the <c>INT64_MIN</c> sentinel means. With
/// the end floored at 0, both copy loops ran <c>for (i = 2; i &lt; 0; i += -1)</c>: zero iterations, the
/// output buffer never written, ALL ZEROS, and no exception anywhere.
/// </para>
///
/// <para>
/// ⚠️ What made it invisible: <c>GraphCompiler</c> and the shape interpreter BOTH handled negative steps
/// correctly already. So the output tensor had exactly the right shape and was full of zeros, every
/// downstream shape check passed, and the model simply returned a wrong number. Found by running Silero
/// VAD against onnxruntime - its <c>adaptive_normalization</c> reverses a <c>[1,1,3]</c> axis twice, and
/// the detector produced a confident, plausible, wrong probability.
/// </para>
///
/// <para>
/// The fixtures (<c>tools/gen_slice_reference.py</c>) are built so the OLD code cannot pass them: data is a
/// graph INPUT rather than an initializer (a constant folds into the shape interpreter, which was never
/// broken), every expected value is non-zero so an all-zeros buffer cannot partially match, and each
/// reversed case is asymmetric so an engine that runs but fails to actually reverse also fails.
/// </para>
/// </summary>
public abstract partial class MLTestBase
{
    private async Task SliceCaseMatchesOnnxRuntime(string name)
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var json = await http.GetStringAsync($"references/slice/slice_{name}.json");
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        var shape = root.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var data = root.GetProperty("data").EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
        var expected = root.GetProperty("out").EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
        var expectedShape = root.GetProperty("out_shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var steps = root.GetProperty("steps").EnumerateArray().Select(e => e.GetInt64()).ToArray();

        var modelBytes = await http.GetByteArrayAsync($"references/slice/slice_{name}.onnx");

        await RunTest(async accelerator =>
        {
            using var session = InferenceSession.CreateFromFile(accelerator, modelBytes,
                inputShapes: new Dictionary<string, int[]> { ["data"] = shape });

            using var dataBuf = accelerator.Allocate1D(data);
            var outputs = await session.RunAsync(new Dictionary<string, Tensor>
            {
                ["data"] = new Tensor(dataBuf.View, shape),
            });

            if (!outputs.TryGetValue("out", out var got))
                throw new Exception($"Slice[{name}]: session produced no output named 'out'");

            if (!got.Shape.SequenceEqual(expectedShape))
                throw new Exception($"Slice[{name}]: shape [{string.Join(",", got.Shape)}] "
                                  + $"but onnxruntime gives [{string.Join(",", expectedShape)}]");

            int count = got.ElementCount;
            if (count != expected.Length)
                throw new Exception($"Slice[{name}]: {count} elements, expected {expected.Length}");

            using var host = accelerator.Allocate1D<float>(count);
            await host.View.CopyFromAsync(got.Data.SubView(0, count));
            await accelerator.SynchronizeAsync();
            var values = await host.CopyToHostAsync<float>(0, count);

            // Name the pre-fix failure rather than reporting a generic mismatch. Every expected value is
            // non-zero by construction, so an all-zeros buffer means the copy loop never ran at all.
            if (values.All(v => v == 0f))
                throw new Exception(
                    $"Slice[{name}] is ALL ZEROS with the correct shape [{string.Join(",", got.Shape)}] - "
                    + "the copy loop never executed. This is the exact failure this test exists for: a "
                    + "negative step with ends clamped to 0 gives `for (i = start; i < 0; i += -1)`.");

            double worst = 0; int worstAt = -1;
            for (int i = 0; i < expected.Length; i++)
            {
                double d = Math.Abs(values[i] - expected[i]);
                if (d > worst) { worst = d; worstAt = i; }
            }
            if (worst > 1e-5)
                throw new Exception($"Slice[{name}] steps=[{string.Join(",", steps)}]: max |d| {worst:E3} "
                                  + $"at {worstAt} (ORT {expected[worstAt]:F6} vs ours {values[worstAt]:F6})");

            Console.WriteLine($"[Slice] {name} steps=[{string.Join(",", steps)}] "
                            + $"[{string.Join(",", got.Shape)}] {expected.Length} values, max |d| {worst:E2}");
        });
    }

    [TestMethod(Timeout = 180000)]
    public async Task Slice_ReverseLastAxis_MatchesOnnxRuntime() => await SliceCaseMatchesOnnxRuntime("reverse_last_axis");

    [TestMethod(Timeout = 180000)]
    public async Task Slice_ReverseLongAxis_MatchesOnnxRuntime() => await SliceCaseMatchesOnnxRuntime("reverse_long_axis");

    [TestMethod(Timeout = 180000)]
    public async Task Slice_ReverseMiddleAxis_MatchesOnnxRuntime() => await SliceCaseMatchesOnnxRuntime("reverse_middle_axis");

    [TestMethod(Timeout = 180000)]
    public async Task Slice_ReverseStep2_MatchesOnnxRuntime() => await SliceCaseMatchesOnnxRuntime("reverse_step2");

    [TestMethod(Timeout = 180000)]
    public async Task Slice_ReversePartial_MatchesOnnxRuntime() => await SliceCaseMatchesOnnxRuntime("reverse_partial");

    [TestMethod(Timeout = 180000)]
    public async Task Slice_ReverseTwoAxes_MatchesOnnxRuntime() => await SliceCaseMatchesOnnxRuntime("reverse_two_axes");

    [TestMethod(Timeout = 180000)]
    public async Task Slice_ForwardBasic_MatchesOnnxRuntime() => await SliceCaseMatchesOnnxRuntime("forward_basic");

    [TestMethod(Timeout = 180000)]
    public async Task Slice_ForwardStep2_MatchesOnnxRuntime() => await SliceCaseMatchesOnnxRuntime("forward_step2");

    [TestMethod(Timeout = 180000)]
    public async Task Slice_ForwardToEnd_MatchesOnnxRuntime() => await SliceCaseMatchesOnnxRuntime("forward_to_end");
}
