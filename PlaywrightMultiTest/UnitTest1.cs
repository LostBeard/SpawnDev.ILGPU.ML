using Microsoft.Playwright;
using SpawnDev.UnitTesting;
using System.Diagnostics;

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

        /// <summary>
        /// A passing test's result text, or null when it carries no report of its own.
        /// </summary>
        /// <remarks>
        /// Both layers substitute a placeholder when a test returned nothing:
        /// <c>UnitTestRunner</c> does <c>ResultText = test.Result.ToString()</c> ("Success") and
        /// <c>ProjectRunner</c> substitutes the literal "Success". Recording those would put a
        /// meaningless string on every one of thousands of passing rows, so they are filtered out here and
        /// only a test's OWN report survives.
        /// </remarks>
        private static string? NonTrivial(string? message)
        {
            if (string.IsNullOrWhiteSpace(message)) return null;
            var t = message.Trim();
            return t is "Success" or "Pass" or "Passed" or "None" or "-" ? null : t;
        }

        [Test, TestCaseSource(nameof(TestCases))]
        public async Task RunTest(ProjectTest test)
        {
            // Parallel path: the lane scheduler already ran this test in StartUp and
            // cached its outcome. Report it (keeps the NUnit trx + results JSON correct)
            // without re-running. Tests not in the cache (e.g. PMT-level integration
            // tests, or PMT_PARALLEL=off) fall through to the live path below.
            if (ProjectRunner.Instance.TryGetOutcome(test.Name, out var outcome))
            {
                var scheduledMessage = outcome.Status == "Pass" ? NonTrivial(outcome.Message) : outcome.Message;
                TestResultsWriter.RecordResult(test.Name, outcome.Status, scheduledMessage, outcome.DurationMs);
                if (outcome.Status == "Skip") Assert.Ignore(outcome.Message ?? "Skipped");
                if (outcome.Status == "Fail") Assert.Fail(outcome.Message ?? "Failed");
                if (scheduledMessage != null) Console.Error.WriteLine($"  REPORT {test.Name}: {scheduledMessage}");
                return; // Pass
            }

            var sw = Stopwatch.StartNew();
            // Whether this test already has its row in the results JSON. Every exit path below is
            // guarded by it, because the outcome is recorded BEFORE the NUnit call that reports it -
            // and those calls report by THROWING.
            var recorded = false;
            try
            {
                if (test.Project is TestableBlazorWasm blazorProj)
                {
                    await test.TestFunc(blazorProj.Page);
                    if (test.Result == TestResult.Unsupported)
                    {
                        sw.Stop();
                        TestResultsWriter.RecordResult(test.Name, "Skip", test.ResultMessage, sw.Elapsed.TotalMilliseconds);
                        recorded = true;
                        Assert.Ignore(test.ResultMessage!);
                    }
                }
                else if (test.Project is TestableConsole)
                {
                    await test.TestFunc(null!);
                    if (test.Result == TestResult.Unsupported)
                    {
                        sw.Stop();
                        TestResultsWriter.RecordResult(test.Name, "Skip", test.ResultMessage, sw.Elapsed.TotalMilliseconds);
                        recorded = true;
                        Assert.Ignore(test.ResultMessage!);
                    }
                }
                sw.Stop();
                // ⭐ Carry the PASSING test's own report through, instead of null. A test that returns a
                // string has it captured into UnitTest.ResultText by SpawnDev.UnitTesting and mapped to
                // ResultMessage here, which is how a diagnostic publishes its numbers WITHOUT throwing.
                // Recording null discarded exactly that, which is why diagnostics threw in the first place.
                var report = NonTrivial(test.ResultMessage);
                TestResultsWriter.RecordResult(test.Name, "Pass", report, sw.Elapsed.TotalMilliseconds);
                recorded = true;
                // Print it too. A number that only reaches the results JSON is a number nobody reads while
                // watching a sweep, and these are measurements (timings, node counts, NaN sweeps).
                // ⚠️ Console.Error, NOT TestContext.Progress: Progress does not reach dotnet test's
                // redirected stdout (MEASURED - zero REPORT lines in the log while every message was
                // correctly in the results JSON). stderr is the channel PMT already uses for its own
                // console diagnostics. Safe here: this is the NUnit testhost, not Blazor WASM, where
                // Console.Error would raise the framework error UI.
                if (report != null) Console.Error.WriteLine($"  REPORT {test.Name}: {report}");
            }
            // 🔴 NUnit REPORTS A NON-FAILURE BY THROWING. Assert.Ignore throws IgnoreException,
            // Assert.Pass throws SuccessException, Assert.Inconclusive throws InconclusiveException -
            // none of them is a failure, and all of them used to land in the general catch below and
            // record the SAME test a SECOND time as "Fail", carrying the skip reason as the error.
            //
            // ⚠️ MEASURED 2026-09-08 on a PMT_PARALLEL=off run: playwright-latest.json reported
            // failed 11 / total 129 for a run NUnit scored Failed 1 / Total 119 - one phantom failure
            // per skip, each a duplicate NAME with result "Fail" and the text "Skipped: ...". NUnit's
            // own trx was correct throughout, so only the artifact triage reads was wrong, and it
            // inflates with the skip count (the 2026-09-06 heavy sweep had 234 skips).
            catch (IgnoreException)
            {
                sw.Stop();
                if (!recorded) TestResultsWriter.RecordResult(test.Name, "Skip", test.ResultMessage, sw.Elapsed.TotalMilliseconds);
                throw;
            }
            catch (InconclusiveException ex)
            {
                sw.Stop();
                if (!recorded) TestResultsWriter.RecordResult(test.Name, "Skip", ex.Message, sw.Elapsed.TotalMilliseconds);
                throw;
            }
            catch (SuccessException)
            {
                sw.Stop();
                if (!recorded) TestResultsWriter.RecordResult(test.Name, "Pass", null, sw.Elapsed.TotalMilliseconds);
                throw;
            }
            catch (Exception ex)
            {
                sw.Stop();
                if (!recorded) TestResultsWriter.RecordResult(test.Name, "Fail", ex.Message, sw.Elapsed.TotalMilliseconds);
                throw;
            }
        }

        [OneTimeTearDown]
        public async Task StopApp()
        {
            TestResultsWriter.WriteFinalSummary();
            await ProjectRunner.Instance.Shutdown();
        }
    }
}
