using Microsoft.Playwright;
using Microsoft.Testing.Platform.Requests;
using PlaywrightMultiTest;
using SpawnDev.UnitTesting;

public class ProjectTest
{
    public TestableProject Project { get; }
    public string Name { get; }
    public string? TestClassName { get; }
    public string? TestTypeName { get; }
    public string? TestMethodName { get; }
    public string? TestPageUrl { get; }
    public TestResult Result { get; set; }
    public string? ResultMessage { get; set; }
    /// <summary>Category from [TestMethod(Category="...")], scraped from the row's
    /// data-test-category attribute. Used to skip slow categories in routine runs.</summary>
    public string Category { get; set; } = "";

    /// <summary>
    /// The page's .NET WASM runtime died during this test, so the page is unusable for every test after it.
    /// </summary>
    /// <remarks>
    /// An unhandled exception on a runtime callback (an async continuation, a finalizer, a JS interop
    /// resolve) does not fail one test - it EXITS THE WASM RUNTIME. Every later interop call then logs
    /// "Assert failed: .NET runtime already exited with N", the row never reaches Done, and the done-wait
    /// below burns its full timeout. At 600s per test over a 1,500-test browser lane that is a run which
    /// looks frozen for hours at near-zero CPU while it politely polls a corpse.
    ///
    /// It has to be OBSERVED rather than inferred, because a dead page is indistinguishable from a slow one
    /// through the DOM alone - both simply never reach Done. The console says so explicitly, and this method
    /// is already listening to it.
    ///
    /// The runner reads this and replaces the page, so one unhandled exception costs one test instead of
    /// the rest of the sweep. See ProjectRunner.RunLaneSequentialAsync.
    /// </remarks>
    public bool PageRuntimeDied { get; private set; }

    /// <summary>Console text proving the runtime exited - quoted verbatim into the failure message.</summary>
    private const string RuntimeExitedMarker = ".NET runtime already exited";
    public Func<IPage, Task> TestFunc { get; set; }
    public ProjectTest(TestableProject testableProject, string name)
    {
        Project = testableProject;
        Name = name;
        SetSuccess();
    }
    public ProjectTest(TestableProject testableProject, string typeName, string methodName, string testPage = "")
    {
        Project = testableProject;
        TestTypeName = typeName;
        TestMethodName = methodName;
        Name = $"{TestTypeName}.{TestMethodName}";
        TestClassName = $"{TestTypeName}-{TestMethodName}";
        TestPageUrl = testPage;
        TestFunc = RunTest;
    }
    public void SetSuccess()
    {
        Result = TestResult.Success;
        TestFunc = (page) => Task.CompletedTask;
    }
    public void SetError(string? err = null)
    {
        Result = TestResult.Error;
        TestFunc = async (page) =>
        {
            throw new Exception(string.IsNullOrWhiteSpace(err) ? "Failed" : err);
        };
    }
    public void SetDefault()
    {
        TestFunc = (page) => RunTest(page);
    }

    /// <summary>
    /// Checks if the Blazor error UI (#blazor-error-ui) is visible on the page.
    /// </summary>
    private static async Task<bool> IsBlazorErrorVisible(IPage page)
    {
        return await page.EvaluateAsync<bool>(
            "() => { var el = document.getElementById('blazor-error-ui'); return el != null && getComputedStyle(el).display !== 'none'; }");
    }

    /// <summary>
    /// Dismisses the Blazor error UI by reloading the page if it's visible.
    /// Returns the error text if one was found, null otherwise.
    /// </summary>
    private static async Task<string?> DismissBlazorErrorIfVisible(IPage page)
    {
        var visible = await IsBlazorErrorVisible(page);
        if (!visible) return null;
        var errorText = await page.EvaluateAsync<string>(
            "() => { var el = document.getElementById('blazor-error-ui'); return el ? el.innerText : ''; }");
        return string.IsNullOrWhiteSpace(errorText) ? "Blazor unhandled error (no message)" : errorText.Trim();
    }

    public async Task RunTest(IPage page)
    {
        try
        {
            var rowSelector = $"tr.{TestClassName}";

            // make sure we are on the test page this test is on
            if (page.Url != TestPageUrl)
            {
                await page.GotoAsync(TestPageUrl);

                // wait for test to load
                await page.WaitForSelectorAsync(rowSelector, new() { Timeout = 30000 });
            }

            // Check if Blazor error UI is already visible before this test
            var hadPreExistingBlazorError = await IsBlazorErrorVisible(page);

            // Capture console messages (errors + warnings) during the test
            var consoleErrors = new List<string>();
            var consoleWarnings = new List<string>();

            // ...and LOG-level messages, opt-in. Only error/warning were ever kept, so a test's own
            // Console.WriteLine (console.log in the browser) was silently DROPPED - which makes the
            // obvious way to instrument a browser test useless, and sends you diagnosing from byte
            // counts in exception messages instead. Off by default because a full sweep's log volume is
            // enormous; PMT_CONSOLE_LOG=<substring> keeps only matching lines, PMT_CONSOLE_LOG=1 (or *)
            // keeps everything for one scoped run.
            //   PMT_CONSOLE_LOG=[BufferPool] PMT_FILTER=SomeTest dotnet test PlaywrightMultiTest/...
            var logFilter = Environment.GetEnvironmentVariable("PMT_CONSOLE_LOG");
            var wantAllLogs = logFilter is "1" or "*";
            var captureLogs = !string.IsNullOrEmpty(logFilter);
            var consoleLogs = new List<string>();

            void OnConsole(object? sender, IConsoleMessage msg)
            {
                // The runtime announces its own death. Catch it here rather than waiting out the
                // done-timeout on a page that can no longer run anything - see PageRuntimeDied.
                if (msg.Type == "error" && msg.Text != null && msg.Text.Contains(RuntimeExitedMarker, StringComparison.Ordinal))
                    PageRuntimeDied = true;
                if (msg.Type == "error")
                    consoleErrors.Add(msg.Text);
                else if (msg.Type == "warning")
                    consoleWarnings.Add(msg.Text);
                else if (captureLogs && msg.Text != null
                         && (wantAllLogs || msg.Text.Contains(logFilter!, StringComparison.OrdinalIgnoreCase)))
                    consoleLogs.Add(msg.Text);
            }
            page.Console += OnConsole;

            // run the test
            var row = page.Locator(rowSelector);

            // find the button within THIS specific row
            var runButton = row.GetByRole(AriaRole.Button, new() { Name = "Run" });

            // wait for test button to be enabled
            await page.WaitForConditionAsync(async () =>
            {
                return await runButton.IsEnabledAsync();
            });

            // click the button to start the process for this row (button will be disabled)
            await runButton.ClickAsync();

            // Wait for THIS row to reach the Done state, keyed on the row's 'test-state-done'
            // class (set only when test.State == Done), and treat ANY page-query failure as
            // "not done yet" so the lane only advances on a POSITIVE Done observation.
            //
            // Two bugs this guards against, both of which let the lane advance while the test
            // was still running → tests ran CONCURRENTLY on the shared page → multiple heavy
            // models at once → GPU OOM (VK_ERROR_OUT_OF_DEVICE_MEMORY) + shared cached
            // accelerator disposed under sibling tests (ObjectDisposed cascade):
            //   1. The old check polled the Run button's *enabled* state — right after
            //      ClickAsync the button is briefly still enabled (Blazor has not re-rendered
            //      yet), so it returned true on the first poll and reported "done" instantly.
            //   2. Heavy tests saturate the single Blazor WASM main thread (per-node GPU→CPU
            //      readbacks + CPU-side verification), so an EvaluateAsync poll can time out
            //      and THROW. If that exception escapes the condition, WaitForConditionAsync
            //      propagates it and the lane advances. Catching it and returning false keeps
            //      us waiting until the page frees up and we actually observe Done.
            // querySelector returns null safely if the row is momentarily detached.
            // Done-wait cap. Default 600s, but a genuinely heavy browser test (e.g. a multi-GB model E2E whose
            // cold download alone exceeds 10 min) declares a larger [TestMethod(Timeout)] that this hard 600s
            // silently overrode. Allow raising it for a run via PMT_BROWSER_DONE_TIMEOUT_MS (parallel to the
            // console lane's PMT_CONSOLE_TIMEOUT_MS). Default unchanged so routine runs behave identically.
            int doneTimeoutMs = int.TryParse(Environment.GetEnvironmentVariable("PMT_BROWSER_DONE_TIMEOUT_MS"), out var dt) && dt > 0 ? dt : 600_000;
            await page.WaitForConditionAsync(async () =>
            {
                // A dead runtime can never reach Done, so stop waiting the moment it says so. Without this
                // the catch below - correctly, for a BUSY page - reads every failure as "not yet" and the
                // wait runs its full 600s against a page that will never answer.
                if (PageRuntimeDied) return true;
                try
                {
                    return await page.EvaluateAsync<bool>(
                        "sel => { const el = document.querySelector(sel); return el != null && el.classList.contains('test-state-done'); }",
                        rowSelector);
                }
                catch
                {
                    return false; // page busy / transient — keep waiting, never advance
                }
            }, doneTimeoutMs);

            // Stop capturing console
            page.Console -= OnConsole;

            // If the runtime died, decide honestly whether this test actually finished first. A test that
            // reached Done and then killed the runtime on its way out still has a real result worth
            // recording; one that never got there failed BECAUSE of it, and says so by name instead of
            // reporting a ten-minute timeout whose cause nobody can see.
            if (PageRuntimeDied)
            {
                bool reachedDone = false;
                try
                {
                    reachedDone = await page.EvaluateAsync<bool>(
                        "sel => { const el = document.querySelector(sel); return el != null && el.classList.contains('test-state-done'); }",
                        rowSelector);
                }
                catch { }

                if (!reachedDone)
                {
                    Result = TestResult.Error;
                    var firstFatal = consoleErrors.FirstOrDefault(e => e.Contains(RuntimeExitedMarker, StringComparison.Ordinal));
                    ResultMessage = "the page's .NET WASM runtime EXITED during this test, so it could never "
                                  + "report a result. An unhandled exception on a runtime callback (async "
                                  + "continuation, finalizer, or JS interop resolve) kills the runtime - look "
                                  + "for the FIRST 'Unhandled Exception' in the browser console, above the "
                                  + "flood of 'already exited' asserts. " + (firstFatal ?? "");
                    throw new Exception(ResultMessage);
                }
            }

            // Only flag a Blazor error if it appeared NEW during this test (wasn't already there)
            string? blazorError = null;
            if (!hadPreExistingBlazorError)
                blazorError = await DismissBlazorErrorIfVisible(page);

            // current state text
            var stateMessage = await row.Locator(".test-state").TextContentAsync();

            //  check for error  class
            var wasError = await row.EvaluateAsync<bool>("el => el.classList.contains('test-error')");

            //  check for error  class
            var unsupported = await row.EvaluateAsync<bool>("el => el.classList.contains('test-unsupported')");

            // Log console errors/warnings to stderr for diagnostics
            if (consoleErrors.Count > 0 || consoleWarnings.Count > 0 || consoleLogs.Count > 0)
            {
                Console.Error.WriteLine($"[{Name}] Console: {consoleErrors.Count} error(s), {consoleWarnings.Count} warning(s)"
                                        + (consoleLogs.Count > 0 ? $", {consoleLogs.Count} matched log(s)" : ""));
                foreach (var err in consoleErrors)
                    Console.Error.WriteLine($"  ERROR: {err}");
                foreach (var warn in consoleWarnings)
                    Console.Error.WriteLine($"  WARN: {warn}");
                foreach (var log in consoleLogs)
                    Console.Error.WriteLine($"  LOG: {log}");
            }

            ResultMessage = stateMessage;

            if (unsupported)
            {
                Result = TestResult.Unsupported;
                if (string.IsNullOrWhiteSpace(stateMessage))
                {
                    stateMessage = "Skipped";
                }
            }
            else if (wasError || blazorError != null)
            {
                Result = TestResult.Error;
                if (blazorError != null && !wasError)
                {
                    // Test reported success but Blazor framework threw an unhandled error
                    stateMessage = $"Blazor error during test: {blazorError}";
                }
                else if (string.IsNullOrWhiteSpace(stateMessage))
                {
                    stateMessage = "Failed";
                }
                throw new Exception(stateMessage);
            }
            else
            {
                Result = TestResult.Success;
                if (string.IsNullOrWhiteSpace(stateMessage))
                {
                    stateMessage = "Success";
                }
            }
        }
        catch (Exception ex)
        {
            throw new Exception($"Test {Name} failed with error: {ex.Message}");
        }
    }
}
