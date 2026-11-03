#:project ../../SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML.csproj
#:property PublishAot=false
// Dumps the OpenCL C (or PTX) ILGPU generates for one static kernel of an ML type, generic ones instantiated (the first type argument half|float, further ones
// by nested type name), to read how its views and sub-views are addressed:
//   dotnet run dump_cl_generic.cs -c Release -- <TypeName> <MethodName> [opencl|cuda] [half|float] [NestedType ...] > out.txt
using ILGPU;
using ILGPU.Backends;
using ILGPU.Backends.EntryPoints;
using ILGPU.Backends.OpenCL;
using ILGPU.Backends.PTX;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU.ML;
using System.Reflection;

var type = typeof(SpawnDev.ILGPU.ML.Pipelines.GgufBatchDecoder).Assembly.GetType(args[0]) ?? throw new Exception($"no type {args[0]}");
var method = type.GetMethod(args[1], BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Static) ?? throw new Exception($"no method {args[1]}");
string which = args.Length > 2 ? args[2] : "opencl";
if (method.IsGenericMethodDefinition)
{
    var targs = new List<Type> { args.Length > 3 && args[3] == "float" ? typeof(float) : typeof(global::ILGPU.Half) };
    for (int i = 4; i < args.Length && targs.Count < method.GetGenericArguments().Length; i++)
        targs.Add(type.GetNestedType(args[i], BindingFlags.NonPublic | BindingFlags.Public) ?? throw new Exception($"no nested type {args[i]}"));
    method = method.MakeGenericMethod(targs.ToArray());
}
using var context = MLContext.Create().ToContext();
using var acc = which == "cuda" ? context.GetCudaDevices()[0].CreateAccelerator(context) : context.GetCLDevices()[0].CreateAccelerator(context);
var entry = EntryPointDescription.FromExplicitlyGroupedKernel(method);
var backend = (Backend)typeof(Accelerator).GetProperty("Backend", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(acc)!;
var compiled = backend.Compile(entry, KernelSpecialization.Empty);
Console.WriteLine(compiled switch
{
    CLCompiledKernel cl => cl.Source,
    PTXCompiledKernel ptx => ptx.PTXAssembly,
    _ => compiled.GetType().Name,
});
