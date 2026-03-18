using Microsoft.Playwright;
using SpawnDev.UnitTesting;

namespace PlaywrightMultiTest
{
    [Parallelizable(ParallelScope.Self)]
    [TestFixture]
    public class Tests : PageTest
    {
        public static IEnumerable<TestCaseData> TestCases => ProjectRunner.Instance.TestCases;

        [OneTimeSetUp]
        public async Task StartApp()
        {
            await ProjectRunner.Instance.StartUp();
        }

        [Test, TestCaseSource(nameof(TestCases))]
        public async Task RunTest(ProjectTest test)
        {
            if (test.Project is TestableBlazorWasm blazorProj)
            {
                try
                {
                    await test.TestFunc(blazorProj.Page);
                    if (test.Result == TestResult.Unsupported)
                    {
                        Assert.Ignore(test.ResultMessage!);
                    }
                }
                catch (Exception ex)
                {
                    throw;
                }
            }
            else if (test.Project is TestableConsole consoleProj)
            {
                try
                {
                    await test.TestFunc(null!);
                    if (test.Result == TestResult.Unsupported)
                    {
                        Assert.Ignore(test.ResultMessage!);
                    }
                }
                catch (Exception ex)
                {
                    throw;
                }
            }
        }

        [OneTimeTearDown]
        public async Task StopApp()
        {
            await ProjectRunner.Instance.Shutdown();
        }
    }
}
