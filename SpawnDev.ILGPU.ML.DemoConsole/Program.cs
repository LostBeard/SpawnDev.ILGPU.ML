using SpawnDev.UnitTesting;
using System.Reflection;

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
