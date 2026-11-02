#:project ../../SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML.csproj
#:property PublishAot=false
// Dumps the OpenCL C (or PTX) that ILGPU generates for one FusedDequantMatMul kernel, to see how its loads are emitted (e.g. whether AsAligned16 becomes a
// 128-bit load).   dotnet run dump_cl_source.cs -c Release -- <KernelMethodName> [opencl|cuda] > out.txt
using ILGPU;
using ILGPU.Backends;
using ILGPU.Backends.EntryPoints;
using ILGPU.Backends.OpenCL;
using ILGPU.Backends.PTX;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Kernels;
using System.Reflection;

string name = args.Length > 0 ? args[0] : "GemvQ4_KVecImpl";
string which = args.Length > 1 ? args[1] : "opencl";
using var context = MLContext.Create().ToContext();
using var acc = which == "cuda" ? context.GetCudaDevices()[0].CreateAccelerator(context) : context.GetCLDevices()[0].CreateAccelerator(context);
var method = typeof(FusedDequantMatMul).GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static) ?? throw new Exception($"no method {name}");
var entry = EntryPointDescription.FromExplicitlyGroupedKernel(method);
var backend = (Backend)typeof(Accelerator).GetProperty("Backend", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(acc)!;
var compiled = backend.Compile(entry, KernelSpecialization.Empty);
Console.WriteLine(compiled switch
{
    CLCompiledKernel cl => cl.Source,
    PTXCompiledKernel ptx => ptx.PTXAssembly,
    _ => compiled.GetType().Name,
});
