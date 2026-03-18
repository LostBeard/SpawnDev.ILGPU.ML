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
}
