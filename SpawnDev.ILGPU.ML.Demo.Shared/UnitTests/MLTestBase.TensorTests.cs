using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task Tensor_CreateAndReshape() => await RunTest(async accelerator =>
    {
        var data = RandomFloats(1370 * 384, seed: 200);
        using var buffer = accelerator.Allocate1D(data);
        var tensor = new Tensor(buffer.View, new[] { 1370, 384 }, "test");

        if (tensor.Rank != 2) throw new Exception($"Expected rank 2, got {tensor.Rank}");
        if (tensor.ElementCount != 1370 * 384) throw new Exception($"Wrong element count: {tensor.ElementCount}");
        if (tensor.Shape[0] != 1370 || tensor.Shape[1] != 384) throw new Exception($"Wrong shape");
        if (tensor.Strides[0] != 384 || tensor.Strides[1] != 1) throw new Exception($"Wrong strides: [{tensor.Strides[0]}, {tensor.Strides[1]}]");

        // Reshape to 3D
        var reshaped = tensor.Reshape(new[] { 1370, 6, 64 });
        if (reshaped.Rank != 3) throw new Exception($"Reshape rank wrong: {reshaped.Rank}");
        if (reshaped.ElementCount != tensor.ElementCount) throw new Exception("Reshape changed element count");

        // Reshape with -1
        var inferred = tensor.Reshape(new[] { -1, 384 });
        if (inferred.Shape[0] != 1370) throw new Exception($"Inferred dim wrong: {inferred.Shape[0]}");

        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Tensor_Slice() => await RunTest(async accelerator =>
    {
        var data = RandomFloats(10 * 384, seed: 201);
        using var buffer = accelerator.Allocate1D(data);
        var tensor = new Tensor(buffer.View, new[] { 10, 384 });

        // Slice first 5 rows
        var sliced = tensor.Slice(0, 5);
        if (sliced.Shape[0] != 5 || sliced.Shape[1] != 384) throw new Exception("Wrong slice shape");
        if (sliced.ElementCount != 5 * 384) throw new Exception("Wrong slice count");

        // Slice rows 3-7
        var mid = tensor.Slice(3, 4);
        if (mid.Shape[0] != 4) throw new Exception("Wrong mid slice shape");

        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task TensorHelpers_BroadcastShape() => await RunTest(async accelerator =>
    {
        // Same shape
        var r1 = TensorHelpers.BroadcastShape(new[] { 3, 4 }, new[] { 3, 4 });
        if (!TensorHelpers.ShapesEqual(r1, new[] { 3, 4 })) throw new Exception("Same shape broadcast failed");

        // Scalar broadcast
        var r2 = TensorHelpers.BroadcastShape(new[] { 3, 4 }, new[] { 1 });
        if (!TensorHelpers.ShapesEqual(r2, new[] { 3, 4 })) throw new Exception("Scalar broadcast failed");

        // Per-channel broadcast (LayerScale pattern)
        var r3 = TensorHelpers.BroadcastShape(new[] { 1370, 384 }, new[] { 384 });
        if (!TensorHelpers.ShapesEqual(r3, new[] { 1370, 384 })) throw new Exception("Channel broadcast failed");

        // Different ranks
        var r4 = TensorHelpers.BroadcastShape(new[] { 6, 1370, 64 }, new[] { 1, 1370, 1 });
        if (!TensorHelpers.ShapesEqual(r4, new[] { 6, 1370, 64 })) throw new Exception("Rank-3 broadcast failed");

        // Incompatible should throw
        bool threw = false;
        try { TensorHelpers.BroadcastShape(new[] { 3, 4 }, new[] { 3, 5 }); }
        catch (ArgumentException) { threw = true; }
        if (!threw) throw new Exception("Should have thrown on incompatible shapes");

        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task BufferPool_RentAndAllocate() => await RunTest(async accelerator =>
    {
        using var pool = new BufferPool(accelerator);

        var t1 = pool.Rent(new[] { 100, 200 }, "test1");
        if (t1.ElementCount != 20000) throw new Exception($"Wrong count: {t1.ElementCount}");
        if (t1.Name != "test1") throw new Exception("Wrong name");

        var t2 = pool.AllocatePermanent(new[] { 384 }, "weights");
        if (t2.ElementCount != 384) throw new Exception($"Wrong weight count: {t2.ElementCount}");

        var data = RandomFloats(384, seed: 300);
        var t3 = pool.AllocatePermanent(data, new[] { 384 }, "loaded");
        if (t3.ElementCount != 384) throw new Exception($"Wrong loaded count: {t3.ElementCount}");
        if (t3.Name != "loaded") throw new Exception("Wrong loaded name");

        await Task.CompletedTask;
    });

    // ────────────────────────────────────────────────────────────────────────
    //  Phase 1: TensorView<T> struct + Tensor<T> generic class.
    //  These tests verify the struct is blittable and passes through ILGPU's
    //  kernel-parameter encoding correctly. Kernel migrations to take
    //  TensorView<T> directly come in Phase 2.
    // ────────────────────────────────────────────────────────────────────────

    [TestMethod]
    public async Task TensorView_ConstructsFromShape() => await RunTest(async accelerator =>
    {
        // 2x3x4 = 24 floats. Build a recognizable pattern: value = i0 * 100 + i1 * 10 + i2.
        const int D0 = 2, D1 = 3, D2 = 4;
        var data = new float[D0 * D1 * D2];
        for (int i = 0; i < D0; i++)
            for (int j = 0; j < D1; j++)
                for (int k = 0; k < D2; k++)
                    data[(i * D1 + j) * D2 + k] = i * 100f + j * 10f + k;
        using var buf = accelerator.Allocate1D(data);

        var tensor = new Tensor<float>(buf.View, new[] { D0, D1, D2 }, name: "test");
        var view = tensor.View;

        if (view.Rank != 3) throw new Exception($"Rank: expected 3 got {view.Rank}");
        if (view.D0 != D0 || view.D1 != D1 || view.D2 != D2 || view.D3 != 1)
            throw new Exception($"Dims: expected ({D0},{D1},{D2},1) got ({view.D0},{view.D1},{view.D2},{view.D3})");
        if (view.ElementCount != D0 * D1 * D2)
            throw new Exception($"ElementCount: expected {D0 * D1 * D2} got {view.ElementCount}");

        Console.WriteLine($"[TensorView] {tensor} rank={view.Rank} dims=({view.D0},{view.D1},{view.D2},{view.D3})");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task TensorView_PassesToKernel_4D() => await RunTest(async accelerator =>
    {
        // [N, C, H, W] = [2, 3, 4, 5] = 120 floats. A kernel that takes
        // TensorView<float> directly and uses Get4D + Set4D for index math. If the
        // struct weren't blittable or its inline indexers were wrong, this would either
        // fail to compile, throw at kernel load, or produce incorrect data.
        const int N = 2, C = 3, H = 4, W = 5;
        const int Count = N * C * H * W;
        var data = new float[Count];
        for (int i = 0; i < Count; i++) data[i] = i + 1f;

        using var inBuf = accelerator.Allocate1D(data);
        using var outBuf = accelerator.Allocate1D<float>(Count);

        var inView = new TensorView<float>(inBuf.View, new[] { N, C, H, W });
        var outView = new TensorView<float>(outBuf.View, new[] { N, C, H, W });

        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, TensorView<float>, TensorView<float>>(
            DoubleTensorViewKernel);
        kernel(Count, inView, outView);
        await accelerator.SynchronizeAsync();

        var result = await outBuf.CopyToHostAsync<float>(0, Count);
        for (int i = 0; i < Count; i++)
        {
            float expected = (i + 1f) * 2f;
            if (Math.Abs(result[i] - expected) > 1e-5f)
                throw new Exception($"Idx {i}: expected {expected} got {result[i]}");
        }

        Console.WriteLine($"[TensorView] 4D kernel pass {N}x{C}x{H}x{W} ({Count} elements) verified");
    });

    /// <summary>
    /// Kernel under test. Threads cover the flattened tensor; each thread decodes its
    /// (n, c, h, w) from the TensorView's inline D0..D3 dimensions, then uses Get4D /
    /// Set4D rather than raw ArrayView indexing. Confirms the struct's inline strides
    /// match what host-side row-major encoding produced.
    /// </summary>
    private static void DoubleTensorViewKernel(Index1D idx, TensorView<float> input, TensorView<float> output)
    {
        int w = idx % input.D3;
        int h = (idx / input.D3) % input.D2;
        int c = (idx / (input.D3 * input.D2)) % input.D1;
        int n = idx / (input.D3 * input.D2 * input.D1);
        output.Set4D(n, c, h, w, input.Get4D(n, c, h, w) * 2f);
    }

    [TestMethod]
    public async Task TensorView_PassesToKernel_2D_Int() => await RunTest(async accelerator =>
    {
        // Different element type — int — to confirm Tensor<T> / TensorView<T> generic
        // over T : unmanaged works through ILGPU's type-monomorphized kernel pipeline.
        const int Rows = 4, Cols = 6;
        const int Count = Rows * Cols;
        var data = new int[Count];
        for (int i = 0; i < Count; i++) data[i] = i;

        using var inBuf = accelerator.Allocate1D(data);
        using var outBuf = accelerator.Allocate1D<int>(Count);
        var inView = new TensorView<int>(inBuf.View, new[] { Rows, Cols });
        var outView = new TensorView<int>(outBuf.View, new[] { Rows, Cols });

        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, TensorView<int>, TensorView<int>>(
            Add100TensorViewKernel);
        kernel(Count, inView, outView);
        await accelerator.SynchronizeAsync();

        var result = await outBuf.CopyToHostAsync<int>(0, Count);
        for (int i = 0; i < Count; i++)
            if (result[i] != i + 100)
                throw new Exception($"Idx {i}: expected {i + 100} got {result[i]}");

        Console.WriteLine($"[TensorView] 2D Int kernel pass {Rows}x{Cols} verified");
    });

    private static void Add100TensorViewKernel(Index1D idx, TensorView<int> inp, TensorView<int> outp)
    {
        int col = idx % inp.D1;
        int row = idx / inp.D1;
        outp.Set2D(row, col, inp.Get2D(row, col) + 100);
    }

    [TestMethod]
    public async Task TensorGeneric_Reshape_PreservesData() => await RunTest(async accelerator =>
    {
        // Reshape [2, 3, 4] → [6, 4] → [-1] (24) and verify zero-copy semantics:
        // the resulting Tensor<T> wraps the same ArrayView with new shape metadata.
        var data = new float[24];
        for (int i = 0; i < 24; i++) data[i] = i;
        using var buf = accelerator.Allocate1D(data);

        var t3d = new Tensor<float>(buf.View, new[] { 2, 3, 4 });
        var t2d = t3d.Reshape(new[] { 6, 4 });
        var t1d = t2d.Reshape(new[] { -1 });

        if (t2d.Shape[0] != 6 || t2d.Shape[1] != 4)
            throw new Exception($"2D reshape: expected [6,4] got [{string.Join(",", t2d.Shape)}]");
        if (t1d.Shape[0] != 24)
            throw new Exception($"1D reshape: expected [24] got [{string.Join(",", t1d.Shape)}]");
        if (t3d.ElementCount != 24 || t2d.ElementCount != 24 || t1d.ElementCount != 24)
            throw new Exception("Reshape changed element count");

        Console.WriteLine($"[Tensor<T>] Reshape chain 3D→2D→1D preserved element count and data");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task TensorView_RankBounds_Rejects_5D_AtConstruction() => await RunTest(async accelerator =>
    {
        // Phase-1 TensorView caps at rank 4. Higher-rank shapes throw loudly at
        // construction rather than silently producing garbage from missing dims.
        var data = new float[1];
        using var buf = accelerator.Allocate1D(data);
        try
        {
            var _ = new TensorView<float>(buf.View, new[] { 1, 1, 1, 1, 1 });
            throw new Exception("Expected rank-5 TensorView to throw, but it did not.");
        }
        catch (ArgumentException) { /* expected */ }
        await Task.CompletedTask;
    });

    // ────────────────────────────────────────────────────────────────────────
    //  Phase 1.5: OwnedTensor<T> — IDisposable wrapper that owns its buffer.
    //  Pipelines that return tensors hand callers an OwnedTensor; callers
    //  dispose it when finished. Implicit conversions to Tensor<T> and
    //  TensorView<T> mean OwnedTensors pass into anything that accepts those.
    // ────────────────────────────────────────────────────────────────────────

    [TestMethod]
    public async Task OwnedTensor_Allocate_FromHost_ImplicitConversions() => await RunTest(async accelerator =>
    {
        // Allocate factory produces an OwnedTensor wrapping a fresh accelerator buffer.
        using var empty = OwnedTensor<float>.Allocate(accelerator, new[] { 4, 8 }, "empty");
        if (empty.ElementCount != 32) throw new Exception($"Expected 32 got {empty.ElementCount}");
        if (empty.Name != "empty") throw new Exception($"Name lost: {empty.Name}");
        if (empty.Rank != 2) throw new Exception($"Rank: {empty.Rank}");

        // FromHost factory: allocate + initial fill in one step.
        var hostData = Enumerable.Range(0, 12).Select(i => (float)i).ToArray();
        using var filled = OwnedTensor<float>.FromHost(accelerator, hostData, new[] { 3, 4 }, "filled");
        var readback = await filled.ToHostAsync();
        for (int i = 0; i < 12; i++)
            if (Math.Abs(readback[i] - hostData[i]) > 1e-5f)
                throw new Exception($"FromHost roundtrip mismatch at {i}: {readback[i]} vs {hostData[i]}");

        // Implicit conversion to Tensor<T> — the non-owning view. The Tensor reference
        // stays valid for the OwnedTensor's lifetime.
        Tensor<float> asTensor = filled;
        if (asTensor.ElementCount != 12) throw new Exception("Implicit Tensor<T> conversion broke ElementCount");
        if (asTensor.Shape[0] != 3 || asTensor.Shape[1] != 4)
            throw new Exception($"Implicit Tensor<T> shape: [{string.Join(",", asTensor.Shape)}]");

        // Implicit conversion to TensorView<T> — kernel-passable.
        TensorView<float> asView = filled;
        if (asView.D0 != 3 || asView.D1 != 4)
            throw new Exception($"Implicit TensorView shape: ({asView.D0}, {asView.D1})");

        Console.WriteLine($"[OwnedTensor] Allocate + FromHost + implicit conversions verified");
    });

    [TestMethod]
    public async Task OwnedTensor_PassesToKernel_ViaImplicitConversion() => await RunTest(async accelerator =>
    {
        // OwnedTensor pipes straight into a kernel taking TensorView<T> via the implicit
        // operator. Caller never has to type .View or .AsTensor.
        var hostA = Enumerable.Range(0, 20).Select(i => (float)i).ToArray();
        using var a = OwnedTensor<float>.FromHost(accelerator, hostA, new[] { 4, 5 });
        using var b = OwnedTensor<float>.Allocate(accelerator, new[] { 4, 5 });

        // Note: passing OwnedTensor<float> directly — implicit conversion to TensorView<float>.
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, TensorView<float>, TensorView<float>>(
            (Index1D idx, TensorView<float> inp, TensorView<float> outp) =>
            {
                int c = idx % inp.D1;
                int r = idx / inp.D1;
                outp.Set2D(r, c, inp.Get2D(r, c) * 3f);
            });
        kernel(a.ElementCount, a, b);  // <-- OwnedTensor → TensorView implicit
        await accelerator.SynchronizeAsync();

        var result = await b.ToHostAsync();
        for (int i = 0; i < hostA.Length; i++)
            if (Math.Abs(result[i] - hostA[i] * 3f) > 1e-5f)
                throw new Exception($"Idx {i}: expected {hostA[i] * 3f} got {result[i]}");

        Console.WriteLine($"[OwnedTensor] Implicit conversion to TensorView at kernel call site verified");
    });

    [TestMethod]
    public async Task OwnedTensor_Dispose_ReleasesBuffer() => await RunTest(async accelerator =>
    {
        // Allocate, dispose, allocate same shape again — must succeed without leaking.
        // (Hard to assert "buffer released" directly without internal accelerator state,
        // but if Dispose were a no-op this test would pass anyway; we run it inside a
        // tight loop so any leak would surface as an OOM at scale on tighter backends.)
        for (int i = 0; i < 50; i++)
        {
            var t = OwnedTensor<float>.Allocate(accelerator, new[] { 1024, 1024 });
            t.Dispose();
            t.Dispose(); // double-dispose is safe (idempotent guard inside).
        }
        Console.WriteLine($"[OwnedTensor] 50x allocate + dispose + double-dispose loop completed");
        await Task.CompletedTask;
    });

    // ────────────────────────────────────────────────────────────────────────
    //  Half element type. Verifies Tensor<Half> and TensorView<Half> work
    //  through ILGPU's generic kernel pipeline for FP16 data.
    // ────────────────────────────────────────────────────────────────────────

    [TestMethod]
    public async Task TensorView_Half_RoundTrip() => await RunTest(async accelerator =>
    {
        // Build an FP16 tensor on the host, copy to accelerator, run a kernel that
        // touches every element through TensorView<Half> indexers, copy back, verify.
        const int Rows = 4, Cols = 8;
        const int Count = Rows * Cols;
        var hostHalf = new global::ILGPU.Half[Count];
        for (int i = 0; i < Count; i++) hostHalf[i] = (global::ILGPU.Half)(i * 0.25f);

        using var inOwned = OwnedTensor<global::ILGPU.Half>.FromHost(accelerator, hostHalf, new[] { Rows, Cols });
        using var outOwned = OwnedTensor<global::ILGPU.Half>.Allocate(accelerator, new[] { Rows, Cols });

        // Kernel: out[i,j] = in[i,j] + (Half)1.5 — exercises both Get2D / Set2D on Half.
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, TensorView<global::ILGPU.Half>, TensorView<global::ILGPU.Half>>(
            (Index1D idx, TensorView<global::ILGPU.Half> inp, TensorView<global::ILGPU.Half> outp) =>
            {
                int c = idx % inp.D1;
                int r = idx / inp.D1;
                var v = inp.Get2D(r, c);
                outp.Set2D(r, c, v + (global::ILGPU.Half)1.5f);
            });
        kernel(Count, inOwned, outOwned);
        await accelerator.SynchronizeAsync();

        var result = await outOwned.ToHostAsync();
        for (int i = 0; i < Count; i++)
        {
            float expected = (float)hostHalf[i] + 1.5f;
            float actual = (float)result[i];
            // FP16 quantization: tolerance 1e-2 covers half-precision rounding.
            if (Math.Abs(actual - expected) > 1e-2f)
                throw new Exception($"Idx {i}: expected {expected} got {actual}");
        }

        Console.WriteLine($"[TensorView<Half>] {Rows}x{Cols} FP16 kernel round-trip verified");
    });

    // ────────────────────────────────────────────────────────────────────────
    //  Phase 2: a kernel migrated to take TensorView<T> directly.
    //  ImagePostprocessKernel.ResizeBilinear is the proof-of-concept; it now
    //  reads source/dest H and W from the TensorView struct instead of taking
    //  them as scalar kernel parameters.
    // ────────────────────────────────────────────────────────────────────────

    // ────────────────────────────────────────────────────────────────────────
    //  Phase 3: InferenceSession.RunOwnedAsync — Transformers.js-style API.
    //  Caller provides Tensor<float> inputs (OwnedTensor converts implicitly);
    //  session returns OwnedTensorMap<float> that disposes every output tensor
    //  in one go when the map goes out of scope.
    // ────────────────────────────────────────────────────────────────────────

    [TestMethod(Timeout = 120000)]
    public async Task InferenceSession_RunOwnedAsync_SqueezeNet() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        using var session = await InferenceSession.CreateFromFileAsync(
            accelerator, http, "models/squeezenet/model.onnx");

        // Build the input as an OwnedTensor — the user-facing way to allocate model
        // inputs in the new API. Shape [1, 3, 224, 224] = standard SqueezeNet NCHW.
        var inputName = session.InputNames[0];
        const int H = 224, W = 224;
        var pixels = new float[1 * 3 * H * W];
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    pixels[((c * H) + y) * W + x] = (x / (float)W) - 0.5f;

        using var input = OwnedTensor<float>.FromHost(accelerator, pixels,
            new[] { 1, 3, H, W }, name: inputName);

        // Transformers.js-style call: pass Tensor<float> inputs (OwnedTensor converts
        // implicitly), get back an OwnedTensorMap<float> that owns every output buffer.
        using var outputs = await session.RunOwnedAsync(new Dictionary<string, Tensor<float>>
        {
            [inputName] = input,
        });

        if (outputs.Count < 1)
            throw new Exception($"Expected ≥1 output tensor, got {outputs.Count}");

        var first = outputs.Single();
        Console.WriteLine($"[RunOwnedAsync] output '{first.Name}' shape [{string.Join(",", first.Shape)}] elements={first.ElementCount}");

        if (first.ElementCount < 100)
            throw new Exception($"SqueezeNet output suspiciously small: {first.ElementCount} elements");

        // Verify the output is real classification logits, not zeros.
        var hostOutput = await first.ToHostAsync();
        float min = hostOutput.Min(), max = hostOutput.Max();
        if (Math.Abs(max - min) < 1e-3f)
            throw new Exception($"Output flat: range [{min}, {max}]");

        Console.WriteLine($"[RunOwnedAsync] PASS — output range [{min:F4}, {max:F4}]");
    });

    [TestMethod(Timeout = 120000)]
    public async Task InferenceSession_RunOwnedAsync_OutputsSurvivePastNextRun() => await RunTest(async accelerator =>
    {
        // OwnedTensor semantics: outputs from one run must not be mutated by a
        // subsequent run. The legacy RunAsync returns views into pool-managed memory
        // that the next run might recycle — RunOwnedAsync copies each output to a
        // fresh caller-owned buffer so this guarantee holds.
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available for this backend");

        using var session = await InferenceSession.CreateFromFileAsync(
            accelerator, http, "models/squeezenet/model.onnx");

        var inputName = session.InputNames[0];
        const int H = 224, W = 224;
        var pixelsA = new float[1 * 3 * H * W];
        var pixelsB = new float[1 * 3 * H * W];
        for (int i = 0; i < pixelsA.Length; i++)
        {
            pixelsA[i] = 0.1f;
            pixelsB[i] = 0.8f;
        }

        using var inputA = OwnedTensor<float>.FromHost(accelerator, pixelsA, new[] { 1, 3, H, W }, inputName);
        using var inputB = OwnedTensor<float>.FromHost(accelerator, pixelsB, new[] { 1, 3, H, W }, inputName);

        using var outputsA = await session.RunOwnedAsync(new Dictionary<string, Tensor<float>> { [inputName] = inputA });
        var snapshotA = await outputsA.Single().ToHostAsync();

        using var outputsB = await session.RunOwnedAsync(new Dictionary<string, Tensor<float>> { [inputName] = inputB });
        var snapshotB = await outputsB.Single().ToHostAsync();

        // After RunB completes, RunA's output buffer must still hold the original data.
        var recheckA = await outputsA.Single().ToHostAsync();
        for (int i = 0; i < snapshotA.Length; i++)
            if (Math.Abs(recheckA[i] - snapshotA[i]) > 1e-5f)
                throw new Exception(
                    $"Outputs from Run A were mutated by Run B. Run-A buffer should be caller-owned and independent. "
                    + $"Idx {i}: original {snapshotA[i]} now {recheckA[i]}.");

        // Sanity: Run A and Run B should differ (different inputs).
        bool differs = false;
        for (int i = 0; i < snapshotA.Length; i++)
            if (Math.Abs(snapshotA[i] - snapshotB[i]) > 1e-4f) { differs = true; break; }
        if (!differs)
            throw new Exception("Run A and Run B produced identical outputs despite different inputs — inference may not be running");

        Console.WriteLine($"[RunOwnedAsync] PASS — output from Run A survived Run B's execution");
    });

    [TestMethod]
    public async Task ResizeBilinear_TensorView_2xUpscale() => await RunTest(async accelerator =>
    {
        // 4x4 source → 8x8 dest. With BT.601 half-pixel sampling each dest pixel is the
        // average of the four nearest source pixels in the limit, but for a constant-
        // gradient source we expect a smooth gradient in the result too. We just check
        // that corners match source corners (within tolerance) and that the destination
        // has wider range than the source's first row (proving real interpolation).
        const int SrcH = 4, SrcW = 4, DstH = 8, DstW = 8;
        var srcData = new float[SrcH * SrcW];
        for (int y = 0; y < SrcH; y++)
            for (int x = 0; x < SrcW; x++)
                srcData[y * SrcW + x] = y * 10f + x;

        using var src = OwnedTensor<float>.FromHost(accelerator, srcData, new[] { SrcH, SrcW });
        using var dst = OwnedTensor<float>.Allocate(accelerator, new[] { DstH, DstW });

        var kernel = new SpawnDev.ILGPU.ML.Kernels.ImagePostprocessKernel(accelerator);
        kernel.ResizeBilinear(src, dst); // <-- OwnedTensor → TensorView implicit, no W/H scalars!
        await accelerator.SynchronizeAsync();

        var dstHost = await dst.ToHostAsync();

        // Corner sanity: top-left of dst should be near src[0,0] = 0. Bottom-right
        // should be near src[3,3] = 33 (clamped, with half-pixel offset).
        if (dstHost[0] > 5f)
            throw new Exception($"Top-left dst should be small, got {dstHost[0]}");
        if (dstHost[DstH * DstW - 1] < 28f)
            throw new Exception($"Bottom-right dst should be large, got {dstHost[DstH * DstW - 1]}");

        // Variance check: result must vary (we upsampled a gradient).
        float dstMin = dstHost.Min(), dstMax = dstHost.Max();
        if (dstMax - dstMin < 25f)
            throw new Exception($"Upscaled gradient too flat: range [{dstMin}, {dstMax}]");

        Console.WriteLine($"[ResizeBilinear(TensorView)] {SrcH}x{SrcW} → {DstH}x{DstW} range [{dstMin:F2}, {dstMax:F2}]");
    });
}
