using SpawnDev.UnitTesting;
using System.Reflection;
using System.Text.Json;

// Catch ILGPU assertion failures (CPU backend bounds checks) that would
// otherwise show "unknown hard error" dialogs and kill the process.
// Write a proper TEST: JSON line so PlaywrightMultiTest captures the error.
AppDomain.CurrentDomain.UnhandledException += (_, e) =>
{
    var errMsg = e.ExceptionObject?.ToString() ?? "Unknown fatal error";
    if (errMsg.Length > 500) errMsg = errMsg[..500];
    var testName = args.Length > 0 ? args[0] : "Unknown";
    var parts = testName.Split('.');
    var json = JsonSerializer.Serialize(new
    {
        TestName = testName,
        TestTypeName = parts.Length > 0 ? parts[0] : testName,
        TestMethodName = parts.Length > 1 ? parts[1] : testName,
        ResultText = "Error",
        Result = 1,
        State = 2,
        Duration = 0,
        Error = errMsg,
        StackTrace = ""
    });
    Console.WriteLine($"TEST: {json}");
    Console.Out.Flush();
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
