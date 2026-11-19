using ILGPU.Runtime;
using SpawnDev.UnitTesting;
using System.Reflection;
using System.Runtime.InteropServices;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// The generated kernels in SpawnDev.ILGPU.ML.Kernels cannot name SpawnDev.ILGPU.ML's public structs (that assembly sits above theirs), so they take layout twins and the
    /// host copies or reinterprets at the launch. A twin that drifts from its public struct (a field added, reordered, retyped) would hand a kernel the wrong bytes with no
    /// compile error: this pins every twin to its struct, size and each field's name, type and offset. Pure reflection, the same on every backend.
    /// </summary>
    [TestMethod]
    public async Task Kernel_LayoutTwins_MatchTheirPublicStructs() => await RunTest(async accelerator =>
    {
        await Task.CompletedTask;
        var pairs = new (Type Public, Type Twin)[]
        {
            (typeof(Training.BackwardKernels.AttnDims), typeof(Kernels.KernelAttnDims)),
        };
        var report = new List<string>();
        foreach (var (pub, twin) in pairs)
        {
            int sp = Marshal.SizeOf(pub), st = Marshal.SizeOf(twin);
            Ur_Expect(sp == st, $"{twin.Name}: {st} bytes, {pub.Name} {sp}");
            var fp = pub.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            var ft = twin.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            Ur_Expect(fp.Length == ft.Length, $"{twin.Name}: {ft.Length} fields, {pub.Name} {fp.Length}");
            foreach (var f in fp)
            {
                var g = ft.FirstOrDefault(x => x.Name == f.Name);
                Ur_Expect(g != null, $"{twin.Name} has no field {f.Name}");
                Ur_Expect(g!.FieldType == f.FieldType, $"{twin.Name}.{f.Name} is {g.FieldType.Name}, {pub.Name}.{f.Name} {f.FieldType.Name}");
                long op = Marshal.OffsetOf(pub, f.Name).ToInt64(), ot = Marshal.OffsetOf(twin, g.Name).ToInt64();
                Ur_Expect(op == ot, $"{twin.Name}.{f.Name} at offset {ot}, {pub.Name}.{f.Name} at {op}");
            }
            report.Add($"{twin.Name} == {pub.Name}: {st} bytes, {ft.Length} fields");
        }
        return string.Join("; ", report);
    });
}
