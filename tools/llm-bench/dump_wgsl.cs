#:project ../../SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML.csproj
#:property PublishAot=false
// Offline WGSL of one static kernel method of an ML type (no device, no browser), to validate with naga:
//   dotnet run dump_wgsl.cs -c Release -- <TypeName> <MethodName> <groupSize> out.wgsl     (TypeName: e.g. SpawnDev.ILGPU.ML.Pipelines.GgufBatchDecoder)
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using System.Reflection;

var type = typeof(SpawnDev.ILGPU.ML.Pipelines.GgufBatchDecoder).Assembly.GetType(args[0]) ?? throw new Exception($"no type {args[0]}");
var method = type.GetMethod(args[1], BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Static) ?? throw new Exception($"no method {args[1]}");
var g = ShaderCompiler.Generate(method, CapabilityProfiles.WebGPUFull, new KernelSpecialization(int.Parse(args[2]), null));
File.WriteAllText(args[3], g.Source);
Console.WriteLine($"{args[1]}: {g.Source.Length} chars -> {args[3]}");
