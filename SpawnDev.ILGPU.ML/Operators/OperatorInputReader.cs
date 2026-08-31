using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Operators;

/// <summary>
/// Reads an operator input as host floats, whether it is a compile-time constant or a live GPU tensor.
/// </summary>
/// <remarks>
/// <see cref="OnnxOpContext.TryGetInputValues"/> only ever returns COMPILE-TIME constants - it looks the
/// name up in <c>ConstantValues</c> and returns null for anything else. An operator that treats that null
/// as "cannot run" therefore cannot run at all on real data.
/// <para>
/// ⚠️ That is exactly how <c>LSTM</c>, <c>GRU</c> and <c>RNN</c> came to be advertised in
/// <c>BuiltinOpTypes</c> while producing NOTHING: each opened with
/// <c>if (wVals == null || rVals == null) return;</c> and <c>if (xVals == null) return;</c>, and X is the
/// runtime input, so every real inference returned having left its output buffer untouched. No exception,
/// no warning. Found by <c>tools/audit-operator-support.cs</c>.
/// </para>
/// <para>
/// The staging pattern here is the one <c>EinsumOperator</c> and <c>MoEOperator</c> already use:
/// GPU-&gt;GPU <c>CopyFrom</c> into a staging buffer (valid on every backend) then a readback. The sync
/// readback throws on WebGPU/WebGL/Wasm, which is why <c>ReadAsync</c> exists and why operators using this
/// must implement <c>ExecuteAsync</c> to be browser-safe.
/// </para>
/// </remarks>
internal static class OperatorInputReader
{
    /// <summary>
    /// Host values for input <paramref name="index"/>, or null if it is absent or unreadable
    /// synchronously (a browser backend - use <see cref="ReadAsync"/> there).
    /// </summary>
    public static float[]? Read(OperatorRegistry reg, OnnxOpContext ctx, int index)
    {
        if (index >= ctx.Inputs.Length) return null;
        var constVals = ctx.TryGetInputValues(index);
        if (constVals != null) return constVals;

        var tensor = ctx.Inputs[index];
        int count = tensor.ElementCount;
        if (count <= 0) return null;
        try
        {
            using var staging = reg.Accelerator.Allocate1D<float>(count);
            staging.View.SubView(0, count).CopyFrom(tensor.Data.SubView(0, count));
            reg.Accelerator.Synchronize();
            return staging.GetAsArray1D();
        }
        catch (NotSupportedException)
        {
            // Browser backends cannot read back synchronously. Returning null here is honest: the caller
            // must either use ReadAsync or fail loudly. Returning zeros would be the silent wrong answer
            // this whole class exists to remove.
            return null;
        }
    }

    /// <summary>Browser-safe host values for input <paramref name="index"/>, or null if absent.</summary>
    public static async Task<float[]?> ReadAsync(OperatorRegistry reg, OnnxOpContext ctx, int index)
    {
        if (index >= ctx.Inputs.Length) return null;
        var constVals = ctx.TryGetInputValues(index);
        if (constVals != null) return constVals;

        var tensor = ctx.Inputs[index];
        int count = tensor.ElementCount;
        if (count <= 0) return null;
        using var staging = reg.Accelerator.Allocate1D<float>(count);
        staging.View.SubView(0, count).CopyFrom(tensor.Data.SubView(0, count));
        return await staging.CopyToHostAsync<float>(0, count);
    }

    /// <summary>
    /// Host values for a STATIC input (a weight), cached by tensor name.
    /// </summary>
    /// <remarks>
    /// A recurrent layer re-reads W and R on every call, and a VAD runs ~31 times a second, so reading
    /// them back per frame is pure waste - they never change for the life of the session.
    /// <para>
    /// ⚠️ Cache ONLY genuinely static inputs. <c>initial_h</c> / <c>initial_c</c> look like
    /// weights and are NOT: Silero VAD passes its LSTM state in as graph inputs and expects the new state
    /// back each frame, so caching those would freeze the detector's memory at the first frame it saw.
    /// </para>
    /// </remarks>
    public static float[]? ReadCached(OperatorRegistry reg, OnnxOpContext ctx, int index,
        Dictionary<string, float[]> cache)
    {
        if (index >= ctx.Inputs.Length) return null;
        var name = index < ctx.InputNames.Length ? ctx.InputNames[index] : null;
        if (!string.IsNullOrEmpty(name) && cache.TryGetValue(name, out var hit)) return hit;

        var vals = Read(reg, ctx, index);
        if (vals != null && !string.IsNullOrEmpty(name)) cache[name] = vals;
        return vals;
    }

    /// <summary>Browser-safe <see cref="ReadCached"/>.</summary>
    public static async Task<float[]?> ReadCachedAsync(OperatorRegistry reg, OnnxOpContext ctx, int index,
        Dictionary<string, float[]> cache)
    {
        if (index >= ctx.Inputs.Length) return null;
        var name = index < ctx.InputNames.Length ? ctx.InputNames[index] : null;
        if (!string.IsNullOrEmpty(name) && cache.TryGetValue(name, out var hit)) return hit;

        var vals = await ReadAsync(reg, ctx, index);
        if (vals != null && !string.IsNullOrEmpty(name)) cache[name] = vals;
        return vals;
    }
}
