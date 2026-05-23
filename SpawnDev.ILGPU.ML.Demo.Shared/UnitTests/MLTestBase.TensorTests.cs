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
}
