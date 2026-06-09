using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// SD-Turbo regression: a UNet graph node fed ONE tensor to BOTH operands of a binary op (e.g.
    /// <c>x - x</c>). The graph executor hands the SAME <see cref="Tensor"/> instance to
    /// <c>ctx.Inputs[0]</c> and <c>ctx.Inputs[1]</c>, so the operator bound one GPU buffer to two
    /// <c>read_write</c> storage slots — which WebGPU/WebGL reject ("storage buffer aliasing detected
    /// in kernel 'Kernel__Sub...': binding 0 and binding 1 reference the same GPU buffer"). That crash
    /// is what made <c>SDTurbo_Generate_E2E</c> fail in diffusion. <c>Mul</c> already de-aliased identical
    /// operands; <c>Sub</c> and <c>Div</c> did not. This pins all four: feeding the same tensor to both
    /// operands must produce the correct element-wise result on every backend, not throw.
    /// </summary>
    [TestMethod]
    public async Task Operator_BinaryOps_SameTensorBothInputs_NoAliasingCrash() => await RunTest(async accelerator =>
    {
        var registry = new OperatorRegistry(accelerator);
        using var pool = new BufferPool(accelerator);

        // No zeros (Div x/x must be 1 everywhere); mixed signs to catch sign bugs.
        var xData = new float[] { 3f, -2f, 5f, 0.5f, 7f, -4f, 1.25f, -0.75f };
        int count = xData.Length;
        var x = pool.AllocatePermanent(xData, new[] { count });

        // Sub(x, x) = 0 — the actual SD-Turbo aliasing crash.
        var subOut = pool.Rent(new[] { count });
        registry.Resolve("Sub").Execute(new OnnxOpContext
        {
            Inputs = new[] { x, x }, Outputs = new[] { subOut },
            Attributes = new Dictionary<string, object>(), Pool = pool,
        });
        await accelerator.SynchronizeAsync();
        var sub = await subOut.Data.SubView(0, count).CopyToAsync(accelerator, count);
        AssertClose(new float[count], sub, 1e-5f, "Sub(x,x)=0: ");

        // Div(x, x) = 1.
        var divOut = pool.Rent(new[] { count });
        registry.Resolve("Div").Execute(new OnnxOpContext
        {
            Inputs = new[] { x, x }, Outputs = new[] { divOut },
            Attributes = new Dictionary<string, object>(), Pool = pool,
        });
        await accelerator.SynchronizeAsync();
        var div = await divOut.Data.SubView(0, count).CopyToAsync(accelerator, count);
        var ones = new float[count];
        for (int i = 0; i < count; i++) ones[i] = 1f;
        AssertClose(ones, div, 1e-5f, "Div(x,x)=1: ");

        // Mul(x, x) = x^2 (Mul already de-aliased; guard against regression in the shared helper).
        var mulOut = pool.Rent(new[] { count });
        registry.Resolve("Mul").Execute(new OnnxOpContext
        {
            Inputs = new[] { x, x }, Outputs = new[] { mulOut },
            Attributes = new Dictionary<string, object>(), Pool = pool,
        });
        await accelerator.SynchronizeAsync();
        var mul = await mulOut.Data.SubView(0, count).CopyToAsync(accelerator, count);
        var sq = new float[count];
        for (int i = 0; i < count; i++) sq[i] = xData[i] * xData[i];
        AssertClose(sq, mul, 1e-5f, "Mul(x,x)=x^2: ");

        // Add(x, x) = 2x (already safe via its two-step copy path; lock it in).
        var addOut = pool.Rent(new[] { count });
        registry.Resolve("Add").Execute(new OnnxOpContext
        {
            Inputs = new[] { x, x }, Outputs = new[] { addOut },
            Attributes = new Dictionary<string, object>(), Pool = pool,
        });
        await accelerator.SynchronizeAsync();
        var add = await addOut.Data.SubView(0, count).CopyToAsync(accelerator, count);
        var twice = new float[count];
        for (int i = 0; i < count; i++) twice[i] = xData[i] + xData[i];
        AssertClose(twice, add, 1e-5f, "Add(x,x)=2x: ");
    });
}

// Helper extension for reading tensor data (mirrors MLTestBase.OperatorTests.cs — file-scoped there too).
static file class BinaryOpAliasReadExtensions
{
    public static async Task<float[]> CopyToAsync(this ArrayView1D<float, Stride1D.Dense> view,
        Accelerator accelerator, int count)
    {
        using var temp = accelerator.Allocate1D<float>(count);
        var ew = new ElementWiseKernels(accelerator);
        ew.Scale(view, temp.View, count, 1f);
        await accelerator.SynchronizeAsync();
        return await temp.CopyToHostAsync<float>(0, count);
    }
}
