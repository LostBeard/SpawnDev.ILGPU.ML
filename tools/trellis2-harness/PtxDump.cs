using System.Reflection;
using ILGPU;
using ILGPU.Backends;
using ILGPU.Backends.EntryPoints;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using SpawnDev.ILGPU.ML.Trellis2;

namespace Trellis2Harness;

/// <summary>Dump the PTX ILGPU generates for one of the Trellis2 kernels and count local-memory traffic / registers (perf diagnosis).</summary>
internal static class PtxDump
{
    public static int Run(string typeName, string methodName, string outPath)
    {
        var (ctx, accel) = Gpu.Create();
        using var _c = ctx; using var _a = accel;
        if (accel is not CudaAccelerator cuda) { Console.WriteLine("CUDA only"); return 2; }
        var type = typeof(DiTKernels).Assembly.GetType("SpawnDev.ILGPU.ML.Trellis2." + typeName) ?? throw new ArgumentException("type " + typeName);
        var method = type.GetMethod(methodName, BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic) ?? throw new ArgumentException("method " + methodName);
        var entry = EntryPointDescription.FromExplicitlyGroupedKernel(method);
        var kernel = (ILGPU.Backends.PTX.PTXCompiledKernel)cuda.Backend.Compile(entry, default);
        File.WriteAllText(outPath, kernel.PTXAssembly);
        string ptx = kernel.PTXAssembly;
        int Count(string s) => ptx.Split('\n').Count(l => l.Contains(s));
        Console.WriteLine($"{ptx.Length} chars; lines {ptx.Split('\n').Length}; ld.local {Count("ld.local")}, st.local {Count("st.local")}, .local decls {Count(".local")}, mma {Count("mma.sync")}, ld.shared {Count("ld.shared")}, st.shared {Count("st.shared")}, bar.sync {Count("bar.sync")}, ld.global {Count("ld.global")}, cvt {Count("cvt.")}");
        Console.WriteLine($"wrote {outPath}");
        return 0;
    }
}
