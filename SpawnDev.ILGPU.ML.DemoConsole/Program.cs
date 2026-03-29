using SpawnDev.UnitTesting;
using System.Reflection;

// Catch ILGPU assertion failures (CPU backend bounds checks) that would
// otherwise show "unknown hard error" dialogs and kill the process.
AppDomain.CurrentDomain.UnhandledException += (_, e) =>
{
    Console.Error.WriteLine($"FATAL: {e.ExceptionObject}");
    Environment.Exit(2);
};

try
{
    await ConsoleRunner.Run(args);
}
catch (Exception ex)
{
    Console.Error.WriteLine(ex);
    return 1;
}
return 0;
