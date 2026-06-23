using Microsoft.Playwright;
using SpawnDev.UnitTesting;
using System.Diagnostics;
using System.Text.Json;

namespace PlaywrightMultiTest
{
    public class ProjectRunner
    {
        /// <summary>ML PMT HTTPS port (5451 = ILGPU). Separate origin so IndexedDB/localStorage do not collide.</summary>
        private const int MlHttpsPort = 5551;
        private static string MlBaseUrl => $"https://localhost:{MlHttpsPort}/";

        public static ProjectRunner Instance => GetRunner().GetAwaiter().GetResult()!;
        private static Task<ProjectRunner>? _projectRunner;
        // Cleared once per run so the browser console log ACCUMULATES across lanes/page-recreations (otherwise a
        // later lane wipes an earlier lane's lines, e.g. WebGPU [Benchmark] numbers lost when the Wasm lane starts).
        private static bool _browserConsoleLogCleared;
        public List<TestableProject> TestableProjects { get; } = new List<TestableProject>();

        /// <summary>
        /// Returns an initialized ProjectRunner singleton
        /// </summary>
        /// <returns></returns>
        static Task<ProjectRunner> GetRunner() => _projectRunner ??= new Func<Task<ProjectRunner>>(async () =>
        {
            var ret = new ProjectRunner();
            await ret.Init().ConfigureAwait(false);
            return ret;
        })();

        /// <summary>
        /// Private consturoctor to prevent external instantiation. The runner should only be created through the GetRunner property which ensures proper initialization.
        /// </summary>
        private ProjectRunner() { }

        private static async Task<int> RunDotnetAsync(string args, string workingDir, int timeoutMs = 300000)
        {
            LogStatus($"RunDotnetAsync: dotnet {args.Split(' ')[0]} (timeout={timeoutMs/1000}s)");
            var startInfo = new ProcessStartInfo("dotnet", args)
            {
                WorkingDirectory = workingDir,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            using var p = new Process();
            p.StartInfo = startInfo;
            p.EnableRaisingEvents = true;

            // Use event-based async reads to avoid pipe buffer deadlocks
            p.OutputDataReceived += (_, _) => { };
            p.ErrorDataReceived += (_, _) => { };

            var exitTcs = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            p.Exited += (_, _) => exitTcs.TrySetResult(true);

            p.Start();
            LogStatus($"RunDotnetAsync: started PID={p.Id}");
            p.BeginOutputReadLine();
            p.BeginErrorReadLine();

            // Wait for exit or timeout
            using var cts = new CancellationTokenSource(timeoutMs);
            using var reg = cts.Token.Register(() => exitTcs.TrySetResult(false));
            var exited = await exitTcs.Task.ConfigureAwait(false);

            if (exited)
            {
                // WaitForExit() with no args can hang if child processes still hold
                // redirected stream handles. Use a short timed wait instead.
                p.WaitForExit(5000);
                LogStatus($"RunDotnetAsync: done PID={p.Id} exit={p.ExitCode}");
                return p.ExitCode;
            }
            else
            {
                LogStatus($"RunDotnetAsync: TIMEOUT after {timeoutMs / 1000}s, killing PID={p.Id}...");
                try { p.Kill(entireProcessTree: true); } catch { }
                return -1;
            }
        }
        /// <summary>
        /// Async initialization method for the ProjectRunner. This is where you can perform any setup that needs to happen before tests are enumerated, such as reading configuration files, setting up logging, etc.
        /// </summary>
        /// <returns></returns>
        // Status file for diagnosing startup hangs
        private static readonly string StatusFile = Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "init_status.log");
        private static void LogStatus(string msg)
        {
            var tid = Environment.CurrentManagedThreadId;
            var isPool = Thread.CurrentThread.IsThreadPoolThread;
            var line = $"[{DateTime.Now:HH:mm:ss.fff}] [T{tid}{(isPool ? ",pool" : "")}] {msg}";
            Console.Error.WriteLine($"[PlaywrightMultiTest] {msg}");
            try { File.AppendAllText(StatusFile, line + "\n"); } catch { }
        }

        private async Task Init()
        {
            try { File.WriteAllText(StatusFile, ""); } catch { } // clear
            LogStatus("Init() started");

            string[] args = Environment.GetCommandLineArgs();
            // Support both --filter=VALUE and --filter VALUE formats
            var filter = args.LastOrDefault(o => o.StartsWith("--filter="))?.Substring(9);
            if (filter == null)
            {
                for (int i = 0; i < args.Length - 1; i++)
                {
                    if (args[i] == "--filter")
                    {
                        filter = args[i + 1];
                        break;
                    }
                }
            }
            // `dotnet test --filter` is consumed by the NUnit adapter, NOT passed to this
            // testhost as a process arg — so PMT enumerates the FULL set and NUnit selects
            // cases afterwards. That was fine when RunTest executed per case, but the parallel
            // scheduler runs the enumerated set up-front in StartUp, so an unscoped enumeration
            // would run everything. PMT_FILTER (substring match, this testhost CAN read it) lets
            // dev runs scope the scheduled set: e.g. `PMT_FILTER=VectorAddTest dotnet test ...`.
            filter ??= Environment.GetEnvironmentVariable("PMT_FILTER");
            if (!string.IsNullOrEmpty(filter)) LogStatus($"Test filter active: '{filter}' (substring match)");


            LogStatus("Discovering projects...");
            var projects = ProjectDiscovery.GetWorkspaceRoot();
            LogStatus($"Found {projects.Count()} projects");
            // add tests to _tests list based on the projects found. You can use the ProjectDetails to determine what kind of project it is and how to get the tests from it. For example, if it's a Blazor WASM project, you might want to start a Playwright instance and navigate to the app to get the tests. If it's a console app, you might want to run the exe with a specific argument to get the tests.
            foreach (var project in projects)
            {
                if (project.AppProjectType == ProjectType.BlazorWasm)
                {
                    var testableProject = new TestableBlazorWasm
                    {
                        ProjectDetails = project,
                    };
                    TestableProjects.Add(testableProject);

                    var buildTest = new ProjectTest(testableProject, $"Build {project.Name}");
                    testableProject.Tests.Add(buildTest);

                    var indexPath = Path.Combine(testableProject.ProjectDetails.WwwRoot, "index.html");

                    // build a publish version of the app for testing.
                    // -p:BuildInParallel=false + -maxcpucount:1 keep the publish single-threaded so
                    // MSBuild worker nodes do not crash with MSB4166 ("Child node exited prematurely")
                    // when other crew (Riker / Tuvok) are running their own PMT sweeps in parallel.
                    LogStatus($"Publishing {project.Name}...");
                    var pubResult = await RunDotnetAsync($"publish \"{project.CsprojPath}\" -c Release -p:BuildInParallel=false -maxcpucount:1", project.Directory).ConfigureAwait(false);
                    LogStatus($"Publish {project.Name}: exit={pubResult}");
                    if (pubResult != 0 || !File.Exists(indexPath))
                    {
                        // build failed
                        buildTest.SetError();
                        continue;
                    }

                    try
                    {
                        LogStatus("Installing Playwright browsers...");
                        var exitCode = Microsoft.Playwright.Program.Main(new[] { "install" });

                        if (exitCode != 0)
                        {
                            throw new Exception($"Playwright browser installation failed with exit code {exitCode}");
                        }

                        // start a static file server to serve the published output
                        // Fixed port so IndexedDB persists across runs (same origin = same IDB)
                        var baseUrl = MlBaseUrl;
                        testableProject.Server = new StaticFileServer(testableProject.ProjectDetails.WwwRoot, baseUrl);
                        // start https server to serve the Blazor WASM app
                        testableProject.Server.Start();
                        // Desktop DemoConsole subprocesses load models/test data via HTTP.
                        Environment.SetEnvironmentVariable("TEST_SERVER_URL", baseUrl);

                        // create a playwright browser, navigate to the app, and enumerate the tests
                        LogStatus("Creating Playwright instance...");
                        testableProject.Playwright = await Playwright.CreateAsync().ConfigureAwait(false);
                        // launch browser
                        // Use persistent context so IndexedDB, localStorage, and
                        // File System Access permissions survive across test runs.
                        // This enables ShaderDebugService's debug folder persistence.
                        var userDataDir = Path.Combine(Path.GetTempPath(), "SpawnDev.ILGPU.ML.PlaywrightProfile");
                        Directory.CreateDirectory(userDataDir);
                        LogStatus($"Launching Chromium (persistent profile: {userDataDir})...");
                        testableProject.BrowserContext = await testableProject.Playwright.Chromium.LaunchPersistentContextAsync(
                            userDataDir,
                            new BrowserTypeLaunchPersistentContextOptions
                            {
                                Headless = false,
                                Args = new[]
                                {
                                    "--enable-unsafe-webgpu",
                                    "--enable-features=Vulkan,WebGPUService,SkiaGraphite,FileSystemAccessPersistentPermission",
                                    "--ignore-gpu-blocklist",
                                    "--no-sandbox",
                                    // Auto-grant file system write permission (no prompt)
                                    "--disable-features=FileSystemAccessPermissionPrompt",
                                    "--allow-file-access-from-files"
                                }
                            }).ConfigureAwait(false);
                        testableProject.Browser = testableProject.BrowserContext.Browser;
                        // Grant all available permissions to avoid prompts
                        await testableProject.BrowserContext.GrantPermissionsAsync(
                            new[] { "clipboard-read", "clipboard-write" }).ConfigureAwait(false);
                        // Temporary: capture browser console output containing WGSL dumps to a log file
                        var wgslDumpDir = Path.Combine(project.Directory, "..", "PlaywrightMultiTest", "WGSLDumps");
                        Directory.CreateDirectory(wgslDumpDir);
                        var consoleLogPath = Path.Combine(wgslDumpDir, "browser_console.log");
                        // Clear ONCE per run (first browser setup) so subsequent lanes/page-recreations APPEND —
                        // otherwise a later lane (e.g. Wasm) wipes an earlier lane's lines (e.g. WebGPU benchmark
                        // numbers). Accumulating across lanes is what lets us read per-backend perf programmatically.
                        if (!_browserConsoleLogCleared) { File.WriteAllText(consoleLogPath, ""); _browserConsoleLogCleared = true; }
                        var wasmDumpChunks = new System.Collections.Generic.List<string>();

                        void HookPageConsole(Microsoft.Playwright.IPage page, string label)
                        {
                            page.Console += (_, msg) =>
                            {
                                var text = msg.Text;
                                // Capture Wasm binary dumps: collect base64 chunks and write to disk
                                if (text.StartsWith("[Wasm_DUMP]"))
                                {
                                    wasmDumpChunks.Add(text.Substring("[Wasm_DUMP]".Length));
                                }
                                else if (text.StartsWith("[Wasm_DUMP_END]") && wasmDumpChunks.Count > 0)
                                {
                                    try
                                    {
                                        var b64 = string.Join("", wasmDumpChunks);
                                        var bytes = Convert.FromBase64String(b64);
                                        var wasmPath = Path.Combine(wgslDumpDir, $"wasm_dump_{DateTime.Now:HHmmss}.wasm");
                                        File.WriteAllBytes(wasmPath, bytes);
                                        LogStatus($"Wasm binary dumped: {wasmPath} ({bytes.Length} bytes)");
                                    }
                                    catch (Exception ex) { LogStatus($"Wasm dump failed: {ex.Message}"); }
                                    wasmDumpChunks.Clear();
                                }
                                else if (text.StartsWith("[Wasm_DUMP_START]"))
                                {
                                    wasmDumpChunks.Clear();
                                }
                                // Log WGSL/Wasm traces, errors, and P2P-layer diagnostic lines so
                                // multi-popup WebRTC flows (P2P two-tab test) leave a trail for
                                // offline diagnosis.
                                if (text.Contains("WGSL") || text.Contains("@compute") || text.Contains("@workgroup_size") || text.Contains("WGSL_DUMP") || text.Contains("GLSL_DUMP") || text.Contains("[WasmWorker]") || text.Contains("[Wasm") || text.Contains("[Benchmark]") || text.Contains("CONV2D_TRACE") || text.Contains("TEX_UNIT") || text.Contains("PREPROCESS_TRACE") || text.Contains("LAYER_TRACE") || text.Contains("LOGITS_TRACE") || text.Contains("CPU_LOGITS") || text.Contains("DISP_TRACE") || text.Contains("TF_OFFSET") || text.Contains("[Peer]") || text.Contains("[RtcPeer]") || text.Contains("[sd_compute]") || text.Contains("[P2PCompute") || text.Contains("[P2P ") || text.Contains("[Torrent") || msg.Type == "error")
                                {
                                    try
                                    {
                                        File.AppendAllText(consoleLogPath, $"[{label}][{msg.Type}] {text}\n---END_MSG---\n");
                                    }
                                    catch { }
                                }
                            };
                        }

                        // Hook console on any popup/new page created by window.open so tests that
                        // drive multi-window flows (P2P two-popup test) capture diagnostics from
                        // every popup, not just the test-driver page.
                        testableProject.BrowserContext.Page += (_, newPage) =>
                        {
                            try { HookPageConsole(newPage, newPage.Url); } catch { }
                        };

                        // new page
                        testableProject.Page = await testableProject.BrowserContext.NewPageAsync().ConfigureAwait(false);
                        HookPageConsole(testableProject.Page, "main");

                        // go to the app's unit tests page.
                        var testPageUrl = new Uri(new Uri(baseUrl), testableProject.TestPage).ToString();
                        LogStatus($"Navigating to {testPageUrl}...");
                        await testableProject.Page.GotoAsync(testPageUrl).ConfigureAwait(false);
                        LogStatus("Page loaded, waiting for test table...");

                        // wait for tests to load
                        await testableProject.Page.WaitForSelectorAsync("table.unit-test-ready", new() { Timeout = 30000 }).ConfigureAwait(false);
                        LogStatus("Test table ready");

                        // Enumerate test rows via a single browser-side JS evaluation
                        // instead of one-IPC-per-row. With ~5000+ rows on the multi-
                        // backend ILGPU matrix the per-row round-trip pattern was
                        // burning multiple minutes of dead time after the page rendered
                        // but before the first test ran. Cribbed from Tuvok's
                        // tuvok-to-team-pmt-enumeration-speedup-2026-04-25.md (Codecs
                        // commit f16b27b). Same semantics, ~7000x fewer IPC calls.
                        var rowsJson = await testableProject.Page.EvaluateAsync<System.Text.Json.JsonElement>(@"() => {
                            const rows = document.querySelectorAll('table.unit-test-view tbody tr');
                            return Array.from(rows).map(r => ({
                                typeName: r.querySelector('.test-type-name')?.textContent ?? '',
                                methodName: r.querySelector('.test-method-name')?.textContent ?? '',
                                category: r.getAttribute('data-test-category') ?? ''
                            }));
                        }").ConfigureAwait(false);

                        var excludedCats = ExcludedCategories();
                    var wasmExcludedCats = WasmExcludedCategories();
                        int totalRows = rowsJson.GetArrayLength();
                        int excludedByCategory = 0;
                        for (int i = 0; i < totalRows; i++)
                        {
                            var row = rowsJson[i];
                            var typeName = row.GetProperty("typeName").GetString() ?? "";
                            var methodName = row.GetProperty("methodName").GetString() ?? "";
                            var category = row.TryGetProperty("category", out var catProp) ? (catProp.GetString() ?? "") : "";
                            var rowTest = new ProjectTest(testableProject, typeName, methodName, testPageUrl) { Category = category };

                            if (filter != null && !MatchesFilter(filter, rowTest))
                            {
                                continue;
                            }

                            // Default fast loop skips slow integration tests by category (e.g.
                            // "HeavyModel" — big-model end-to-end tests that run ~minutes via
                            // per-node shape readbacks). Applied REGARDLESS of PMT_FILTER, because
                            // the primary dev loop is a broad filter like PMT_FILTER=WebGPUTests —
                            // keeping the heavy tests off the single shared Chromium page there is
                            // exactly where the orphan-on-timeout overlap bug bites. To deliberately
                            // run a heavy test, clear the set (PMT_EXCLUDE_CATEGORIES=) and scope by
                            // name, or set PMT_EXCLUDE_CATEGORIES to a different category list.
                            // "WasmHeavy" is additionally skipped on Wasm-lane rows only (see
                            // WasmExcludedCategories).
                            if (IsCategoryExcluded(category, typeName, excludedCats, wasmExcludedCats))
                            {
                                excludedByCategory++;
                                continue;
                            }

                            testableProject.Tests.Add(rowTest);
                        }
                        LogStatus($"Browser tests enumerated: {testableProject.Tests.Count} tests"
                            + (excludedByCategory > 0 ? $" ({excludedByCategory} skipped by category [{string.Join(",", excludedCats)}])" : ""));

                    }
                    catch (Exception ex)
                    {
                        LogStatus($"Error initializing {project.Name}: {ex.Message}");
                    }
                }
                else if (project.AppProjectType == ProjectType.Exe)
                {
                    // enumerate tests by calling the console app. by default it will return a list of the tests in the exe

                    var testableProject = new TestableConsole
                    {
                        ProjectDetails = project,
                    };
                    TestableProjects.Add(testableProject);

                    var buildTest = new ProjectTest(testableProject, $"Build {project.Name}");
                    testableProject.Tests.Add(buildTest);

                    // build a publish version of the app for testing.
                    // Same MSBuild-worker resilience as the Blazor branch above.
                    LogStatus($"Publishing {project.Name}...");
                    var pubResult = await RunDotnetAsync($"publish \"{project.CsprojPath}\" -c Release -p:BuildInParallel=false -maxcpucount:1", project.Directory).ConfigureAwait(false);
                    LogStatus($"Publish {project.Name}: exit={pubResult}");
                    var publishedBinary = project.ExistingPublishBinary;
                    if (pubResult != 0 || string.IsNullOrEmpty(publishedBinary))
                    {
                        // build failed
                        buildTest.SetError();
                        continue;
                    }

                    // get list of tests by running the exe with a specific argument.
                    // PMT_EMIT_CATEGORY=1 opts the console enumeration into the
                    // "Type.Method|Category" line format (SpawnDev.UnitTesting 2.5.5+) so the
                    // desktop lanes can honor the same HeavyModel gate as the browser lane.
                    // The child process inherits this env var (ProcessRunner uses
                    // UseShellExecute=false without overriding EnvironmentVariables).
                    Environment.SetEnvironmentVariable("PMT_EMIT_CATEGORY", "1");
                    var excludedCats = ExcludedCategories();
                    var wasmExcludedCats = WasmExcludedCategories();
                    int excludedByCategory = 0;
                    LogStatus($"Enumerating tests from {Path.GetFileName(publishedBinary)}...");
                    var result = await ProcessRunner.Run(publishedBinary).ConfigureAwait(false);
                    LogStatus($"Enumeration done: exit={result.ExitCode}, lines={result.Text.Split('\n', StringSplitOptions.RemoveEmptyEntries).Length}");
                    var testList = result.Text.Split(new[] { '\n', '\r' }, StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
                    foreach (var test in testList)
                    {
                        // Lines are "Type.Method", or "Type.Method|Category" when the enumeration
                        // honored PMT_EMIT_CATEGORY. Strip the trailing "|Category" before the dot
                        // split (method/type names never contain '|').
                        var line = test;
                        var category = "";
                        var pipeIdx = line.IndexOf('|');
                        if (pipeIdx >= 0)
                        {
                            category = line.Substring(pipeIdx + 1);
                            line = line.Substring(0, pipeIdx);
                        }

                        // get test type name
                        var typeName = line.Split(".")[0];

                        // get test method name — skip lines where the method part is empty (e.g.
                        // console output from initialization code like "[LocalTracker] foo..." that
                        // ends with a dot but has no real method name after it)
                        var parts = line.Split(".");
                        var methodName = parts.Length >= 2 ? parts[1] : "";
                        if (string.IsNullOrWhiteSpace(methodName)) continue;

                        var rowTest = new ProjectTest(testableProject, typeName!, methodName!) { Category = category };
                        if (filter != null && !MatchesFilter(filter, rowTest))
                        {
                            continue;
                        }

                        // Same HeavyModel-style category gate as the browser branch, applied
                        // REGARDLESS of PMT_FILTER. Clear PMT_EXCLUDE_CATEGORIES to run them.
                        if (IsCategoryExcluded(category, typeName, excludedCats, wasmExcludedCats))
                        {
                            excludedByCategory++;
                            continue;
                        }

                        testableProject.Tests.Add(rowTest);

                        rowTest.TestFunc = async (page) =>
                        {
                            // Subprocess timeout: 10 minutes covers every test that ships with the
                            // demo today plus 1-2 retries inside the subprocess. The slower P2P /
                            // WebRTC paths take 135-260s standalone (LargeBuffer_1MB / 10MB) and
                            // their TestMethod attributes specify 180s / 240s timeouts with up to
                            // RetryCount=2 - PMT's outer timeout has to cover the worst-case retry
                            // budget (3 * 240s = 720s). The TEST method's own [TestMethod
                            // (Timeout=...)] attribute still bounds the in-test work; PMT's outer
                            // timeout only fires if the subprocess itself wedges past the test's
                            // own timeout + retry budget. The previous 120s hardcoded value
                            // pre-empted every WebRTC test that did real peer discovery, even when
                            // the test's own attribute granted 180s+. Bumping to 10 minutes
                            // restores the contract: "PMT respects test-method timeouts."
                            var result = await ProcessRunner.Run(publishedBinary, rowTest.Name, timeout: ConsoleTestTimeoutMs()).ConfigureAwait(false);
                            var resultLines = result.Text.Split(new[] { '\n', '\r' }, StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
                            var testResltTest = resultLines.LastOrDefault(o => o.StartsWith("TEST: "))?.Substring(6);
                            var unitTest = testResltTest != null ? JsonSerializer.Deserialize<UnitTest>(testResltTest) : null;
                            if (unitTest == null)
                            {
                                throw new Exception("Test run failed");
                            }
                            var stateMessage = unitTest.ResultText;
                            rowTest.Result = unitTest.Result;

                            if (rowTest.Result == TestResult.Unsupported)
                            {
                                if (string.IsNullOrWhiteSpace(stateMessage))
                                {
                                    stateMessage = "Skipped";
                                }
                            }
                            else if (rowTest.Result == TestResult.Error)
                            {
                                // Use the actual error details from the test runner, not just
                                // the result enum name ("Error"). unitTest.Error contains the
                                // real exception message and stack trace.
                                var errorDetail = !string.IsNullOrWhiteSpace(unitTest.Error)
                                    ? unitTest.Error
                                    : stateMessage;
                                if (string.IsNullOrWhiteSpace(errorDetail))
                                {
                                    errorDetail = "Failed";
                                }
                                rowTest.ResultMessage = errorDetail;
                                throw new Exception(errorDetail);
                            }
                            else
                            {
                                if (string.IsNullOrWhiteSpace(stateMessage))
                                {
                                    stateMessage = "Success";
                                }
                            }

                            rowTest.ResultMessage = stateMessage;
                            rowTest.Result = unitTest.Result;
                            var nmtt = true;
                        };
                    }

                    LogStatus($"Console tests enumerated for {project.Name}: {testableProject.Tests.Count} tests"
                        + (excludedByCategory > 0 ? $" ({excludedByCategory} skipped by category [{string.Join(",", excludedCats)}])" : ""));

                    var nmt11 = true;
                }
            }
            LogStatus($"Init() complete. Total projects={TestableProjects.Count}, " +
                $"total tests={TestableProjects.Sum(p => p.Tests.Count)}");
            var nmt = true;
        }
        IEnumerable<TestCaseData>? _TestCases;
        public IEnumerable<TestCaseData> TestCases => _TestCases ??= GetPlaywrightTasks();

        /// <summary>
        /// Returns all the tests that are found. This is called before StartUp, so you should not rely on any services or infrastructure being available when this is called. You can return any tests you want to run here, and they will be run by the test runner.
        /// </summary>
        /// <returns></returns>
        public IEnumerable<TestCaseData> GetPlaywrightTasks()
        {
            Debug.WriteLine("GetPlaywrightTasks()");
            foreach (var testableProject in TestableProjects)
            {
                foreach (var test in testableProject.Tests)
                {
                    var testCaseData = new TestCaseData(test).SetName(test.Name).SetCategory(test.TestTypeName ?? test.Name);
                    yield return testCaseData;
                }
            }
            var nmt = true;
            // Playwright-level P2P integration tests (multi-tab real-WebRTC, ~120s timeouts).
            // GATED OFF — P2P backend is on hold (core-6 focus), consistent with the P2P unit
            // tests gated out of the Demo/DemoConsole registrations. Uncomment to re-enable
            // alongside the P2PLogicTests registration blocks in the Program.cs files.
            // The P2PSwarm* helper methods below are retained for that re-enable.
            // foreach (var testableProject in TestableProjects)
            // {
            //     if (testableProject is TestableBlazorWasm)
            //     {
            //         var proj = testableProject;
            //         yield return new TestCaseData(new ProjectTest(proj, "P2PSwarm", "TwoTab_PeerDiscovery")
            //         {
            //             TestFunc = async (page) => await P2PSwarmTwoTabTest(page, (TestableBlazorWasm)proj),
            //         }).SetName("P2PSwarm.TwoTab_PeerDiscovery").SetCategory("P2PSwarm");
            //
            //         yield return new TestCaseData(new ProjectTest(proj, "P2PSwarm", "PeerStability_NoCascadeAfterHandshake")
            //         {
            //             TestFunc = async (page) => await P2PSwarmStabilityTest(page, (TestableBlazorWasm)proj),
            //         }).SetName("P2PSwarm.PeerStability_NoCascadeAfterHandshake").SetCategory("P2PSwarm");
            //
            //         yield return new TestCaseData(new ProjectTest(proj, "P2PSwarm", "RenderMandelbrot_PaintsCanvas")
            //         {
            //             TestFunc = async (page) => await P2PSwarmRenderMandelbrotTest(page, (TestableBlazorWasm)proj),
            //         }).SetName("P2PSwarm.RenderMandelbrot_PaintsCanvas").SetCategory("P2PSwarm");
            //     }
            // }
        }

        /// <summary>
        /// Playwright-level P2P test: drives two tabs via CDP - creates a swarm in tab 1,
        /// joins from tab 2 via the auto-shared join link, verifies peer count goes up,
        /// then closes tab 2 to verify peer dropout.
        ///
        /// Reads ComputeSwarm state via the demo's PublishToHarness mechanism (same hook
        /// WasmP2PBrowserTests.ComputeSwarm_Benchmark_RoundTrips_BetweenTwoPopups uses).
        /// The demo writes <c>computeSwarmState_&lt;testId&gt;</c> to its own window when
        /// <c>?testId=</c> is in the URL; Playwright reads it via Page.EvaluateAsync.
        /// No DOM scraping, no fragile XPath, no fake "tracker can't relay same-context"
        /// excuse - if peer discovery breaks, this test FAILS.
        /// </summary>
        private static async Task P2PSwarmTwoTabTest(IPage page1, TestableBlazorWasm blazorProj)
        {
            var baseUrl = MlBaseUrl.TrimEnd('/');
            var coordTestId = $"twotab-coord-{Guid.NewGuid():N}".Substring(0, 24);
            var workerTestId = $"twotab-work-{Guid.NewGuid():N}".Substring(0, 24);

            // Tab 1: coordinator. autoCreate=true triggers the create-swarm flow on load.
            var coordPage = await blazorProj.BrowserContext.NewPageAsync();
            coordPage.Console += (_, msg) => LogStatus($"[Coord {msg.Type}] {msg.Text}");

            var coordUrl = $"{baseUrl}/compute?testId={Uri.EscapeDataString(coordTestId)}&autoCreate=true";
            await coordPage.GotoAsync(coordUrl);

            // PublishToHarness writes the state as a JSON STRING via SpawnDev.BlazorJS's
            // serializing JS.Set, so we parse on read. Helper isolates the parse.
            const string parseStateJs = "(k) => { var v = globalThis[k]; return typeof v === 'string' ? JSON.parse(v) : v; }";

            // Wait for the demo to publish a join link to its own window.
            string? joinLink = await WaitForJsValueAsync(coordPage,
                $"() => {{ var s = ({parseStateJs})('computeSwarmState_{coordTestId}'); return s ? s.joinLink : null; }}",
                timeoutSeconds: 60,
                label: $"coordinator joinLink for {coordTestId}");

            if (string.IsNullOrEmpty(joinLink) || !joinLink.StartsWith("http"))
                throw new Exception($"Coordinator did not publish a usable joinLink: '{joinLink}'");
            LogStatus($"[P2P TwoTab] Join URL: {joinLink[..Math.Min(80, joinLink.Length)]}...");

            // Tab 2: worker. autojoin=1 skips the consent dialog.
            var separator = joinLink.Contains('?') ? "&" : "?";
            var workerUrl = $"{joinLink}{separator}autojoin=1&testId={Uri.EscapeDataString(workerTestId)}";

            var page2 = await blazorProj.BrowserContext.NewPageAsync();
            page2.Console += (_, msg) => LogStatus($"[Worker {msg.Type}] {msg.Text}");
            try
            {
                await page2.GotoAsync(workerUrl);
                LogStatus("[P2P TwoTab] Worker tab opened, waiting for peer discovery...");

                // Wait for coordinator to see at least one peer. Real WebRTC peer
                // discovery + DTLS over hub.spawndev.com can take up to ~60-80s on
                // a cold tracker connection. Logged per poll so we can see progress.
                var peerCountJs = $"() => {{ var s = ({parseStateJs})('computeSwarmState_{coordTestId}'); return s ? s.peerCount : null; }}";
                int lastLogged = -1;
                var peerCount = await WaitForJsValueAsync(coordPage, peerCountJs,
                    timeoutSeconds: 120,
                    label: "coordinator.peerCount >= 1",
                    predicate: v =>
                    {
                        var n = AsInt(v);
                        if (n.HasValue && n.Value != lastLogged)
                        {
                            LogStatus($"[P2P TwoTab] coord.peerCount={n.Value}");
                            lastLogged = n.Value;
                        }
                        return n.HasValue && n.Value >= 1;
                    });

                var pc = AsInt(peerCount) ?? -1;
                if (pc < 1)
                    throw new Exception($"Coordinator never saw a peer (final peerCount={peerCount}). " +
                        "Check that BrowserContext.NewPageAsync produces distinct WebTorrent PeerIds " +
                        "and the tracker is reachable.");

                LogStatus($"[P2P TwoTab] Peer connected ✓ (coord.peerCount={pc})");

                // Diagnostic: dump peerIds so a phantom-registration bug can be diagnosed
                // when peerCount > 1 (one tab, but the coordinator sees multiple peers).
                var peerIdsJs = $"() => {{ var s = ({parseStateJs})('computeSwarmState_{coordTestId}'); return s ? JSON.stringify(s.peerIds) : '[]'; }}";
                var peerIdsConnected = await coordPage.EvaluateAsync<string>(peerIdsJs);
                LogStatus($"[P2P TwoTab] peerIds at connected: {peerIdsConnected}");

                // Close worker tab to verify dropout.
                await page2.CloseAsync();
                page2 = null;

                LogStatus("[P2P TwoTab] Worker closed, waiting for peer dropout...");
                // BRRTC-DIAG: probe each poller's state across the dropout window
                try
                {
                    var pcCount = await coordPage.EvaluateAsync<int?>("() => globalThis.__brrtc_pc_count");
                    LogStatus($"[P2P TwoTab][BRRTC-DIAG] pc_count={pcCount}");
                    for (int probe = 0; probe < 18; probe++)
                    {
                        await Task.Delay(5000);
                        var states = await coordPage.EvaluateAsync<string?>(
                            "() => { var n = globalThis.__brrtc_pc_count || 0; var arr = []; for (var i = 1; i <= n; i++) { arr.push(globalThis['__brrtc_pc_' + i] || '?'); } return arr.join(' || '); }");
                        var synthStates = await coordPage.EvaluateAsync<string?>(
                            "() => { var n = globalThis.__brrtc_pc_count || 0; var arr = []; for (var i = 1; i <= n; i++) { var s = globalThis['__brrtc_synth_' + i]; if (s) arr.push('#' + i + ':' + s); } return arr.join(' | '); }");
                        var bridgeOnClose = await coordPage.EvaluateAsync<string?>("() => globalThis.__bridge_wire_onclose");
                        var bridgeSchedule = await coordPage.EvaluateAsync<string?>("() => globalThis.__bridge_schedule_unreg");
                        var bridgeUnreg = await coordPage.EvaluateAsync<string?>("() => globalThis.__bridge_unregister_fired");
                        var bridgeShort = await coordPage.EvaluateAsync<string?>("() => globalThis.__bridge_short_circuit");
                        var bridgeWireset = await coordPage.EvaluateAsync<string?>("() => globalThis.__bridge_wireset_dump");
                        var pcNow = await coordPage.EvaluateAsync<object?>(peerCountJs);
                        LogStatus($"[P2P TwoTab][BRRTC-DIAG] t+{(probe+1)*5}s peerCount={pcNow} states={states}");
                        if (!string.IsNullOrEmpty(synthStates))
                            LogStatus($"[P2P TwoTab][BRRTC-DIAG] t+{(probe+1)*5}s synth={synthStates}");
                        if (!string.IsNullOrEmpty(bridgeOnClose))
                            LogStatus($"[P2P TwoTab][BRRTC-DIAG] t+{(probe+1)*5}s bridge_onclose={bridgeOnClose}");
                        if (!string.IsNullOrEmpty(bridgeSchedule))
                            LogStatus($"[P2P TwoTab][BRRTC-DIAG] t+{(probe+1)*5}s bridge_schedule={bridgeSchedule}");
                        if (!string.IsNullOrEmpty(bridgeUnreg))
                            LogStatus($"[P2P TwoTab][BRRTC-DIAG] t+{(probe+1)*5}s bridge_unregister={bridgeUnreg}");
                        if (!string.IsNullOrEmpty(bridgeShort))
                            LogStatus($"[P2P TwoTab][BRRTC-DIAG] t+{(probe+1)*5}s bridge_short={bridgeShort}");
                        if (!string.IsNullOrEmpty(bridgeWireset))
                            LogStatus($"[P2P TwoTab][BRRTC-DIAG] t+{(probe+1)*5}s bridge_wireset={bridgeWireset}");
                    }
                } catch (Exception ex) { LogStatus($"[P2P TwoTab][BRRTC-DIAG] probe error: {ex.Message}"); }
                var lastDropLogged = -1;
                var droppedCount = await WaitForJsValueAsync(coordPage, peerCountJs,
                    timeoutSeconds: 90,
                    label: "coordinator.peerCount returns to 0",
                    predicate: v =>
                    {
                        var n = AsInt(v);
                        if (n.HasValue && n.Value != lastDropLogged)
                        {
                            LogStatus($"[P2P TwoTab] coord.peerCount (post-drop)={n.Value}");
                            lastDropLogged = n.Value;
                        }
                        return AsInt(v) == 0;
                    });

                LogStatus($"[P2P TwoTab] Coordinator peer count after dropout: {droppedCount}");
                LogStatus("[P2P TwoTab] Two-tab peer discovery + dropout: COMPLETE ✓");
            }
            finally
            {
                if (page2 != null)
                {
                    try { await page2.CloseAsync(); } catch { }
                }
                try { await coordPage.CloseAsync(); } catch { }
            }
        }

        /// <summary>
        /// Live-demo regression test: opens coord + worker tabs through the demo's
        /// own `P2PCompute.CreateSwarmAsync` / `JoinSwarmAsync` flow (matching what
        /// the deployed RenderMandelbrot demo does), waits for the BT handshake to
        /// complete, then keeps both tabs open and asserts the connection STAYS
        /// alive without an SCTP cascade.
        ///
        /// Catches the rc.5 dedup-via-dispose cascade trigger that the original
        /// `TwoTab_PeerDiscovery` test missed: TwoTab waited for `peerCount &gt;= 1`
        /// (passes the moment the first peer connects), then immediately closed
        /// the worker tab and waited for `peerCount = 0`. A cascade-induced drop
        /// during the open window is indistinguishable from the worker-close drop
        /// in that flow, so the test passed for the wrong reason.
        ///
        /// This test fails on TWO conditions:
        ///   1. `peerCount` drops below 1 while BOTH tabs are still open (cascade
        ///      took out the connection without anyone closing).
        ///   2. Either tab's browser console emits a string matching the cascade
        ///      signature (`sctp-failure` / `User-Initiated Abort` /
        ///      `sctpCauseCode=12` / `[CH-ERROR-DIAG]`).
        ///
        /// The stability-window length (60s) was chosen because Captain's live repro
        /// 2026-05-03 fired the cascade within ~5-10s of BT handshake completion.
        /// 60s gives generous headroom while staying inside PMT's per-test timeout.
        /// </summary>
        private static async Task P2PSwarmStabilityTest(IPage page1, TestableBlazorWasm blazorProj)
        {
            var baseUrl = MlBaseUrl.TrimEnd('/');
            var coordTestId = $"stability-coord-{Guid.NewGuid():N}".Substring(0, 26);
            var workerTestId = $"stability-work-{Guid.NewGuid():N}".Substring(0, 26);

            // Capture cascade-signature messages from BOTH tabs into a shared list
            // and assert it's empty at the end. Match against the strings RtcPeer's
            // CH-ERROR-DIAG handler emits when Chromium fires the cascade.
            var cascadeMatches = new System.Collections.Concurrent.ConcurrentBag<string>();
            void ScanForCascade(string source, string text)
            {
                if (string.IsNullOrEmpty(text)) return;
                if (text.Contains("sctp-failure", StringComparison.OrdinalIgnoreCase)
                    || text.Contains("User-Initiated Abort", StringComparison.OrdinalIgnoreCase)
                    || text.Contains("sctpCauseCode=12", StringComparison.OrdinalIgnoreCase)
                    || text.Contains("[CH-ERROR-DIAG]", StringComparison.OrdinalIgnoreCase))
                {
                    cascadeMatches.Add($"[{source}] {text}");
                }
            }

            var coordPage = await blazorProj.BrowserContext.NewPageAsync();
            coordPage.Console += (_, msg) =>
            {
                LogStatus($"[Stability/Coord {msg.Type}] {msg.Text}");
                ScanForCascade("Coord", msg.Text);
            };

            var coordUrl = $"{baseUrl}/compute?testId={Uri.EscapeDataString(coordTestId)}&autoCreate=true";
            await coordPage.GotoAsync(coordUrl);

            const string parseStateJs = "(k) => { var v = globalThis[k]; return typeof v === 'string' ? JSON.parse(v) : v; }";

            string? joinLink = await WaitForJsValueAsync(coordPage,
                $"() => {{ var s = ({parseStateJs})('computeSwarmState_{coordTestId}'); return s ? s.joinLink : null; }}",
                timeoutSeconds: 60,
                label: $"coordinator joinLink for {coordTestId}");

            if (string.IsNullOrEmpty(joinLink) || !joinLink.StartsWith("http"))
                throw new Exception($"Coordinator did not publish a usable joinLink: '{joinLink}'");
            LogStatus($"[P2P Stability] Join URL: {joinLink[..Math.Min(80, joinLink.Length)]}...");

            var separator = joinLink.Contains('?') ? "&" : "?";
            var workerUrl = $"{joinLink}{separator}autojoin=1&testId={Uri.EscapeDataString(workerTestId)}";

            var page2 = await blazorProj.BrowserContext.NewPageAsync();
            page2.Console += (_, msg) =>
            {
                LogStatus($"[Stability/Worker {msg.Type}] {msg.Text}");
                ScanForCascade("Worker", msg.Text);
            };

            try
            {
                await page2.GotoAsync(workerUrl);
                LogStatus("[P2P Stability] Worker tab opened, waiting for peer discovery...");

                var peerCountJs = $"() => {{ var s = ({parseStateJs})('computeSwarmState_{coordTestId}'); return s ? s.peerCount : null; }}";
                int lastLogged = -1;
                await WaitForJsValueAsync(coordPage, peerCountJs,
                    timeoutSeconds: 120,
                    label: "coordinator.peerCount >= 1 (initial connect)",
                    predicate: v =>
                    {
                        var n = AsInt(v);
                        if (n.HasValue && n.Value != lastLogged)
                        {
                            LogStatus($"[P2P Stability] coord.peerCount={n.Value}");
                            lastLogged = n.Value;
                        }
                        return n.HasValue && n.Value >= 1;
                    });

                LogStatus("[P2P Stability] Peer connected. Holding both tabs open for 60s, asserting peerCount stays >= 1 with no SCTP cascade...");

                // Stability window: poll every 2s for 60s. peerCount must stay >= 1
                // the entire time. Closing either tab is FORBIDDEN - this is the
                // exact "sit there until they threw an error" scenario from Captain's
                // 2026-05-03 live repro.
                const int stabilitySeconds = 60;
                const int pollIntervalMs = 2000;
                int polls = stabilitySeconds * 1000 / pollIntervalMs;
                int minPeerCount = int.MaxValue;
                int firstDropAtPollIndex = -1;
                for (int i = 0; i < polls; i++)
                {
                    await Task.Delay(pollIntervalMs);
                    var raw = await coordPage.EvaluateAsync<object?>(peerCountJs);
                    var n = AsInt(raw);
                    if (n.HasValue)
                    {
                        if (n.Value < minPeerCount) minPeerCount = n.Value;
                        if (n.Value < 1 && firstDropAtPollIndex < 0)
                        {
                            firstDropAtPollIndex = i;
                            LogStatus($"[P2P Stability] FAIL: coord.peerCount dropped to {n.Value} at t+{(i + 1) * pollIntervalMs / 1000}s while both tabs still open");
                            // Keep polling so we can see what else happens, but the test will fail at the end.
                        }
                        if ((i + 1) % 5 == 0)
                            LogStatus($"[P2P Stability] t+{(i + 1) * pollIntervalMs / 1000}s peerCount={n.Value} (min={minPeerCount})");
                    }
                }

                LogStatus($"[P2P Stability] 60s window complete. minPeerCount={minPeerCount} cascadeMatches={cascadeMatches.Count}");

                if (firstDropAtPollIndex >= 0)
                {
                    var dropTimeSec = (firstDropAtPollIndex + 1) * pollIntervalMs / 1000;
                    throw new Exception(
                        $"P2PSwarm.PeerStability: coord.peerCount dropped below 1 (to {minPeerCount}) at t+{dropTimeSec}s while both tabs were still open. " +
                        $"This is the cascade-trigger regression. cascadeMatches={cascadeMatches.Count}");
                }

                if (cascadeMatches.Count > 0)
                {
                    var sample = string.Join(" | ", cascadeMatches.Take(3));
                    throw new Exception(
                        $"P2PSwarm.PeerStability: detected {cascadeMatches.Count} cascade-signature console message(s) in 60s window even though peerCount stayed >= 1. " +
                        $"Sample: {sample}");
                }

                LogStatus("[P2P Stability] PASS: peerCount stayed >= 1 for 60s with no SCTP cascade signatures observed.");
            }
            finally
            {
                if (page2 != null)
                {
                    try { await page2.CloseAsync(); } catch { }
                }
                try { await coordPage.CloseAsync(); } catch { }
            }
        }

        /// <summary>
        /// Live-demo regression test for the canvas paint pipeline: opens coord +
        /// worker through the demo's own swarm-create / swarm-join flow, waits for
        /// the BT handshake, clicks the Mandelbrot Render button on the coord, and
        /// asserts the canvas contains real (non-uniform) pixel data after the
        /// dispatch completes.
        ///
        /// Catches the 2026-05-03 blank-canvas regression: dispatch succeeded
        /// (timer ticked, "Rendered in 908ms" UI text shown) but the canvas stayed
        /// the dark-blue page background because the manual `PutImageBytes` path
        /// was wired to an `ElementReference` the JS-side `getElementById` could
        /// not resolve. The fix replaced that path with the canonical
        /// `CanvasRendererFactory` pattern; this test guards that flip from
        /// regressing.
        ///
        /// Failure conditions:
        ///   1. Dispatch never reports a timing (UI never shows "Rendered in").
        ///   2. The canvas is still uniform (max-min channel range &lt; 8) after
        ///      the dispatch — the cardinal "blank canvas" symptom.
        /// </summary>
        private static async Task P2PSwarmRenderMandelbrotTest(IPage page1, TestableBlazorWasm blazorProj)
        {
            var baseUrl = MlBaseUrl.TrimEnd('/');
            var coordTestId = $"render-coord-{Guid.NewGuid():N}".Substring(0, 24);
            var workerTestId = $"render-work-{Guid.NewGuid():N}".Substring(0, 24);

            var coordPage = await blazorProj.BrowserContext.NewPageAsync();
            coordPage.Console += (_, msg) => LogStatus($"[Render/Coord {msg.Type}] {msg.Text}");

            var coordUrl = $"{baseUrl}/compute?testId={Uri.EscapeDataString(coordTestId)}&autoCreate=true";
            await coordPage.GotoAsync(coordUrl);

            const string parseStateJs = "(k) => { var v = globalThis[k]; return typeof v === 'string' ? JSON.parse(v) : v; }";

            string? joinLink = await WaitForJsValueAsync(coordPage,
                $"() => {{ var s = ({parseStateJs})('computeSwarmState_{coordTestId}'); return s ? s.joinLink : null; }}",
                timeoutSeconds: 60,
                label: $"coordinator joinLink for {coordTestId}");

            if (string.IsNullOrEmpty(joinLink) || !joinLink.StartsWith("http"))
                throw new Exception($"Coordinator did not publish a usable joinLink: '{joinLink}'");

            var separator = joinLink.Contains('?') ? "&" : "?";
            var workerUrl = $"{joinLink}{separator}autojoin=1&testId={Uri.EscapeDataString(workerTestId)}";

            IPage? page2 = await blazorProj.BrowserContext.NewPageAsync();
            page2.Console += (_, msg) => LogStatus($"[Render/Worker {msg.Type}] {msg.Text}");

            try
            {
                await page2.GotoAsync(workerUrl);

                var peerCountJs = $"() => {{ var s = ({parseStateJs})('computeSwarmState_{coordTestId}'); return s ? s.peerCount : null; }}";
                await WaitForJsValueAsync(coordPage, peerCountJs,
                    timeoutSeconds: 120,
                    label: "coordinator.peerCount >= 1 for render dispatch",
                    predicate: v => { var n = AsInt(v); return n.HasValue && n.Value >= 1; });

                LogStatus("[P2P Render] Peer connected. Clicking Mandelbrot Render button...");

                // The Render button is the one whose handler binds to the
                // _mandelbrotRunning gating field. There's a single 'Render' button
                // on /compute scoped to the Mandelbrot panel.
                await coordPage.GetByText("Render", new() { Exact = true }).First.ClickAsync();

                // Wait up to 60s for the dispatch to finish + UI to publish the
                // _mandelbrotTimeMs timer. The demo writes timing into the same
                // computeSwarmState publish hook (extended in the harness flow).
                var renderedTextLocator = coordPage.Locator("text=Rendered in").First;
                await renderedTextLocator.WaitForAsync(new() { Timeout = 60_000 });
                LogStatus("[P2P Render] Dispatch finished, UI shows 'Rendered in ...'. Sampling canvas...");

                // Sample the Mandelbrot canvas — locate it by its parent containing
                // 'Distributed Mandelbrot' text rather than querySelector('canvas')
                // which would return whichever canvas is first in the DOM (n-body /
                // qr / others). A blank canvas yields range 0; even a faint
                // Mandelbrot has a dark body + bright outer ring, so range >= 8 is
                // conservative.
                var sampleJs = @"
                    () => {
                        var canvases = Array.from(document.querySelectorAll('canvas'));
                        var canvasInfo = canvases.map(c => ({ w: c.width, h: c.height, parentText: (c.parentElement && c.parentElement.innerText || '').slice(0,80), display: getComputedStyle(c).display }));
                        var c = canvases.find(c => {
                            var p = c.parentElement;
                            while (p) {
                                if ((p.innerText || '').includes('Distributed Mandelbrot')) return true;
                                p = p.parentElement;
                            }
                            return false;
                        });
                        if (!c) return { err: 'no Mandelbrot canvas found', allCanvases: canvasInfo };
                        var ctx = c.getContext('2d');
                        if (!ctx) return { err: 'no 2d ctx', canvasInfo: canvasInfo };
                        var w = c.width, h = c.height;
                        if (!w || !h) return { err: 'canvas has zero size: ' + w + 'x' + h };
                        var img = ctx.getImageData(0, 0, w, h);
                        var d = img.data;
                        var minR=255,minG=255,minB=255,maxR=0,maxG=0,maxB=0,nonZero=0;
                        var sampleStride = 64; // sample every 64th pixel
                        for (var i = 0; i < d.length; i += 4 * sampleStride) {
                            var r = d[i], g = d[i+1], b = d[i+2];
                            if (r|g|b) nonZero++;
                            if (r<minR) minR=r; if (r>maxR) maxR=r;
                            if (g<minG) minG=g; if (g>maxG) maxG=g;
                            if (b<minB) minB=b; if (b>maxB) maxB=b;
                        }
                        return { w:w, h:h, minR:minR, maxR:maxR, minG:minG, maxG:maxG, minB:minB, maxB:maxB, nonZero:nonZero, canvasCount: canvases.length };
                    }";

                var sampleResult = await coordPage.EvaluateAsync<JsonElement>(sampleJs);
                LogStatus($"[P2P Render] Canvas sample: {sampleResult}");

                if (sampleResult.TryGetProperty("err", out var errProp))
                    throw new Exception($"Canvas sample failed: {errProp.GetString()}");

                int rangeR = sampleResult.GetProperty("maxR").GetInt32() - sampleResult.GetProperty("minR").GetInt32();
                int rangeG = sampleResult.GetProperty("maxG").GetInt32() - sampleResult.GetProperty("minG").GetInt32();
                int rangeB = sampleResult.GetProperty("maxB").GetInt32() - sampleResult.GetProperty("minB").GetInt32();
                int maxRange = Math.Max(rangeR, Math.Max(rangeG, rangeB));

                if (maxRange < 8)
                    throw new Exception($"Canvas paint regression: blank canvas detected (max channel range={maxRange}). " +
                                        $"R={rangeR}, G={rangeG}, B={rangeB}. The Mandelbrot dispatch reported a timing " +
                                        $"but the canvas pixels are uniform — CanvasRendererFactory paint pipeline broken.");

                LogStatus($"[P2P Render] PASS: canvas painted with non-uniform pixels (max channel range={maxRange}).");
            }
            finally
            {
                if (page2 != null)
                {
                    try { await page2.CloseAsync(); } catch { }
                }
                try { await coordPage.CloseAsync(); } catch { }
            }
        }

        // Polls a JS expression on the page until it returns a non-null value (or one that
        // satisfies the optional predicate). Throws on timeout - no fake skip-on-timeout.
        // Returns the last value seen on the page when the predicate matches.
        private static async Task<object?> WaitForJsValueAsync(IPage page, string jsExpr,
            int timeoutSeconds, string label, Func<object?, bool>? predicate = null)
        {
            var deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
            object? lastValue = null;
            while (DateTime.UtcNow < deadline)
            {
                try
                {
                    var raw = await page.EvaluateAsync<object?>(jsExpr);
                    lastValue = raw is JsonElement je
                        ? (je.ValueKind == JsonValueKind.Number ? (object)je.GetInt32()
                           : je.ValueKind == JsonValueKind.String ? je.GetString()
                           : je.ValueKind == JsonValueKind.True ? true
                           : je.ValueKind == JsonValueKind.False ? false
                           : je.ValueKind == JsonValueKind.Null ? null
                           : raw)
                        : raw;
                    if (predicate != null)
                    {
                        if (predicate(lastValue)) return lastValue;
                    }
                    else if (lastValue != null)
                    {
                        return lastValue;
                    }
                }
                catch { /* page navigating or not yet ready */ }
                await Task.Delay(500);
            }
            throw new Exception($"Timeout ({timeoutSeconds}s) waiting for {label}. Last value: {lastValue}");
        }

        // String overload helper - typed wait for a JS expression that returns a string.
        private static async Task<string?> WaitForJsValueAsync(IPage page, string jsExpr,
            int timeoutSeconds, string label)
        {
            var raw = await WaitForJsValueAsync(page, jsExpr, timeoutSeconds, label, v => v is string s && !string.IsNullOrEmpty(s));
            return raw as string;
        }

        // Coerces whatever shape Page.EvaluateAsync returns for a JS number (JsonElement,
        // long, int, double) into a nullable int. Returns null for non-numeric / null.
        private static int? AsInt(object? v) => v switch
        {
            int i => i,
            long l => (int)l,
            double d => (int)d,
            JsonElement je when je.ValueKind == JsonValueKind.Number => je.GetInt32(),
            _ => null,
        };

        // ───────────────────────── Parallel lane scheduler ─────────────────────────
        // The full sweep used to run strictly sequentially: NUnit's [Parallelizable
        // (ParallelScope.Self)] runs the TestCaseSource cases one at a time, and every
        // browser test clicked rows on a single shared Chromium page. This scheduler
        // moves execution into OneTimeSetUp (StartUp) and runs backend "lanes" in
        // parallel, then NUnit's RunTest just reports the cached per-test outcome.
        //
        // Lane model (v1):
        //   Phase A (parallel): browser non-Wasm rows (sequential on the one page) ‖
        //     CPU subprocesses (cap N) ‖ CUDA subprocesses (cap 1, GPU) ‖
        //     OpenCL subprocesses (cap 1, GPU).
        //   Phase B (isolated): Wasm rows alone — Wasm sort kernels spawn
        //     hardwareConcurrency pure-spin barrier workers that STARVE under CPU
        //     oversubscription, so Wasm never overlaps any other CPU-heavy lane.
        // Set PMT_PARALLEL=off to fall back to the original sequential per-case path.

        public sealed record ScheduledOutcome(string Status, string? Message, double DurationMs);
        private readonly System.Collections.Concurrent.ConcurrentDictionary<string, ScheduledOutcome> _outcomes = new();
        public bool ParallelEnabled { get; private set; }
        public bool TryGetOutcome(string name, out ScheduledOutcome outcome) => _outcomes.TryGetValue(name, out outcome!);

        private static int EnvInt(string name, int dflt)
            => int.TryParse(Environment.GetEnvironmentVariable(name), out var v) && v > 0 ? v : dflt;

        // Per-lane concurrency caps (env-overridable). GPU-bound lanes default to 1 to
        // avoid OOM/contention with the browser WebGPU lane sharing the same card.
        private static int CapFor(string lane) => lane switch
        {
            "cpu" => EnvInt("PMT_CPU_PARALLELISM", 4),
            "cuda" => EnvInt("PMT_CUDA_PARALLELISM", 1),
            "opencl" => EnvInt("PMT_OPENCL_PARALLELISM", 1),
            _ => EnvInt("PMT_DESKTOP_PARALLELISM", 4),
        };

        private static bool IsWasm(ProjectTest t) => (t.TestTypeName ?? "").Contains("Wasm", StringComparison.OrdinalIgnoreCase);

        // Substring, case-insensitive match against a test's full name / type / method.
        private static bool MatchesFilter(string filter, ProjectTest t) =>
            (t.Name?.Contains(filter, StringComparison.OrdinalIgnoreCase) ?? false)
            || (t.TestTypeName?.Contains(filter, StringComparison.OrdinalIgnoreCase) ?? false)
            || (t.TestMethodName?.Contains(filter, StringComparison.OrdinalIgnoreCase) ?? false);

        // Test categories ([TestMethod(Category="...")]) skipped by default in routine runs.
        // "HeavyModel" = slow big-model end-to-end tests (minutes each via per-node shape
        // readbacks). Override via env PMT_EXCLUDE_CATEGORIES (comma-separated; empty string
        // = exclude nothing / run everything).
        // A test's Category may itself be comma-separated ("HeavyCpu,WasmHeavy") - it is
        // excluded if ANY of its categories is in the active exclusion set.
        private static readonly string[] DefaultExcludedCategories = { "HeavyModel" };
        private static string[] ExcludedCategories()
        {
            var env = Environment.GetEnvironmentVariable("PMT_EXCLUDE_CATEGORIES");
            if (env == null) return DefaultExcludedCategories;
            return env.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        }

        // LANE-SCOPED exclusions: categories skipped ONLY on the Wasm lane (test TypeName
        // contains "Wasm"). "WasmHeavy" = tests that exceed the Wasm lane's interpreted-IL
        // time budget but PASS within budget on every other backend (Captain's 2026-06-12
        // timeout-policy decision: keep full coverage on the fast lanes, skip-by-default on
        // Wasm, run deliberately via PMT_EXCLUDE_CATEGORIES_WASM= + a name filter).
        private static readonly string[] DefaultWasmExcludedCategories = { "WasmHeavy" };
        private static string[] WasmExcludedCategories()
        {
            var env = Environment.GetEnvironmentVariable("PMT_EXCLUDE_CATEGORIES_WASM");
            if (env == null) return DefaultWasmExcludedCategories;
            return env.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        }

        /// <summary>True when the test's (comma-separable) category list intersects the
        /// global exclusion set, or - for Wasm-lane rows - the Wasm-scoped exclusion set.</summary>
        private static bool IsCategoryExcluded(string? category, string? typeName,
            string[] excludedCats, string[] wasmExcludedCats)
        {
            if (string.IsNullOrEmpty(category)) return false;
            bool isWasmRow = typeName?.Contains("Wasm", StringComparison.OrdinalIgnoreCase) ?? false;
            foreach (var cat in category.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
            {
                if (excludedCats.Contains(cat, StringComparer.OrdinalIgnoreCase)) return true;
                if (isWasmRow && wasmExcludedCats.Contains(cat, StringComparer.OrdinalIgnoreCase)) return true;
            }
            return false;
        }

        // Per-console-test hard cap (subprocess kill, ms). Default 10 min. A genuinely heavy test whose
        // own [TestMethod(Timeout)] exceeds this (e.g. a multi-GB model E2E whose 2.5GB download alone
        // needs >10 min) would otherwise be killed by this outer cap before its own timeout — raise it
        // via PMT_CONSOLE_TIMEOUT_MS for that run. Default unchanged so routine runs behave identically.
        private static int ConsoleTestTimeoutMs()
        {
            var env = Environment.GetEnvironmentVariable("PMT_CONSOLE_TIMEOUT_MS");
            return int.TryParse(env, out var ms) && ms > 0 ? ms : 600_000;
        }

        private static string DesktopLaneOf(string? typeName) => (typeName ?? "") switch
        {
            "CudaTests" or "CuRandTests" or "NvJpegTests" => "cuda",
            "OpenCLTests" => "opencl",
            _ => "cpu", // CPUTests + capability/logic test classes (fast, CPU-bound)
        };

        /// <summary>
        /// Called after tests are enumerated, before NUnit runs the cases. When parallel
        /// scheduling is enabled this runs the entire sweep in lanes and caches outcomes;
        /// RunTest then reports them. With PMT_PARALLEL=off this is a no-op and RunTest
        /// executes each test live (original sequential behaviour).
        /// </summary>
        public async Task StartUp()
        {
            Debug.WriteLine("StartUp()");
            ParallelEnabled = !string.Equals(Environment.GetEnvironmentVariable("PMT_PARALLEL"), "off", StringComparison.OrdinalIgnoreCase);
            if (!ParallelEnabled)
            {
                LogStatus("Parallel scheduler DISABLED (PMT_PARALLEL=off) — NUnit runs cases sequentially.");
                return;
            }
            await RunScheduledAsync().ConfigureAwait(false);
        }

        private async Task RunScheduledAsync()
        {
            var swAll = Stopwatch.StartNew();
            var blazor = TestableProjects.OfType<TestableBlazorWasm>().FirstOrDefault();

            // Trivial/build tests (no TestTypeName) — record their preset result instantly.
            foreach (var proj in TestableProjects)
                foreach (var t in proj.Tests.Where(t => t.TestTypeName == null))
                    await ExecuteAndCaptureAsync(t, null).ConfigureAwait(false);

            // ── Phase A: non-Wasm browser lane ‖ desktop lanes ──────────────────────
            var phaseA = new List<Task>();

            foreach (var console in TestableProjects.OfType<TestableConsole>())
            {
                foreach (var laneGroup in console.Tests
                             .Where(t => t.TestTypeName != null)
                             .GroupBy(t => DesktopLaneOf(t.TestTypeName)))
                {
                    var tests = laneGroup.ToList();
                    var cap = CapFor(laneGroup.Key);

                    // CPU lane only: the ILGPU CPU accelerator already saturates ALL cores per
                    // process, so running several compute-heavy CPU tests concurrently is pure
                    // oversubscription (no throughput gain) and inflates each test's wall-clock
                    // past its [TestMethod(Timeout=...)] — the failure mode that made every
                    // real-model CPU test (style transfer etc.) "fail" with a timeout while the
                    // exact same compute passed in isolation. So split: HeavyCpu-tagged tests run
                    // serialized (cap=1, and alone — after the light burst drains) while light
                    // tests keep the lane cap. The cuda/opencl lanes are already cap=1 and the
                    // browser lane is single-page-sequential, so only the cpu lane needs this.
                    if (laneGroup.Key == "cpu")
                    {
                        var heavy = tests.Where(IsCpuHeavy).ToList();
                        var light = tests.Where(t => !IsCpuHeavy(t)).ToList();
                        int heavyCap = EnvInt("PMT_CPU_HEAVY_PARALLELISM", 1);
                        LogStatus($"Phase A desktop lane 'cpu': {light.Count} light (cap={cap}) + {heavy.Count} HeavyCpu (cap={heavyCap}, serialized after light)");
                        phaseA.Add(RunCpuLaneAsync(light, cap, heavy, heavyCap));
                    }
                    else
                    {
                        LogStatus($"Phase A desktop lane '{laneGroup.Key}': {tests.Count} tests, cap={cap}");
                        phaseA.Add(RunLaneConcurrentAsync(tests, cap, null));
                    }
                }
            }

            if (blazor?.Page != null)
            {
                var nonWasm = blazor.Tests.Where(t => t.TestTypeName != null && !IsWasm(t)).ToList();
                LogStatus($"Phase A browser non-Wasm lane: {nonWasm.Count} tests (sequential on shared page)");
                phaseA.Add(RunLaneSequentialAsync(nonWasm, blazor.Page));
            }

            await Task.WhenAll(phaseA).ConfigureAwait(false);
            LogStatus($"Phase A complete in {swAll.Elapsed:hh\\:mm\\:ss}. Starting Phase B (Wasm, isolated)...");

            // ── Phase B: Wasm lane alone (no other CPU-heavy lane running) ──────────
            if (blazor?.Page != null)
            {
                var wasm = blazor.Tests.Where(t => t.TestTypeName != null && IsWasm(t)).ToList();
                var nonWasmBrowser = blazor.Tests
                    .Where(t => t.TestTypeName != null && !IsWasm(t))
                    .ToList();
                if (wasm.Count > 0
                    && nonWasmBrowser.Any(t => t.TestTypeName == "WebGPUTests"))
                {
                    var reloadUrl = wasm[0].TestPageUrl ?? nonWasmBrowser[0].TestPageUrl;
                    if (!string.IsNullOrEmpty(reloadUrl))
                        blazor.Page = await RecreateTestPageAsync(blazor, reloadUrl).ConfigureAwait(false);
                }

                LogStatus($"Phase B Wasm lane: {wasm.Count} tests (isolated, sequential)");
                await RunLaneSequentialAsync(wasm, blazor.Page).ConfigureAwait(false);
            }

            LogStatus($"All scheduled tests complete in {swAll.Elapsed:hh\\:mm\\:ss} ({_outcomes.Count} outcomes cached).");
        }

        private async Task RunLaneSequentialAsync(List<ProjectTest> tests, IPage page)
        {
            string? lastType = null;
            foreach (var t in tests)
            {
                // WebGPU leaves browser GPU state that breaks subsequent WebGL rows on the
                // same page (TensorView_Half_RoundTrip: Wasm/WebGL pass in isolation).
                if (lastType == "WebGPUTests"
                    && t.TestTypeName != null
                    && t.TestTypeName != "WebGPUTests"
                    && !string.IsNullOrEmpty(t.TestPageUrl)
                    && page.Context != null)
                {
                    var blazor = TestableProjects.OfType<TestableBlazorWasm>().FirstOrDefault();
                    if (blazor != null)
                        page = await RecreateTestPageAsync(blazor, t.TestPageUrl).ConfigureAwait(false);
                    else
                        await ReloadTestPageAsync(page, t.TestPageUrl).ConfigureAwait(false);
                }

                await ExecuteAndCaptureAsync(t, page).ConfigureAwait(false);
                lastType = t.TestTypeName;
            }
        }

        private async Task ReloadTestPageAsync(IPage page, string testPageUrl)
        {
            LogStatus("Reloading Blazor test page to reset browser GPU state after WebGPU...");
            await page.GotoAsync(testPageUrl).ConfigureAwait(false);
            await page.WaitForSelectorAsync("table.unit-test-ready", new() { Timeout = 30000 })
                .ConfigureAwait(false);
        }

        /// <summary>
        /// WebGPU can leave the browser in a state where a plain reload is not enough for
        /// the next backend (Wasm). Swap to a fresh page in the same context.
        /// </summary>
        private static async Task<IPage> RecreateTestPageAsync(TestableBlazorWasm blazor, string testPageUrl)
        {
            LogStatus("Recreating browser page after WebGPU (fresh page for Wasm lane)...");
            var oldPage = blazor.Page;
            var newPage = await blazor.BrowserContext!.NewPageAsync().ConfigureAwait(false);
            blazor.Page = newPage;
            if (oldPage != null)
            {
                try { await oldPage.CloseAsync().ConfigureAwait(false); } catch { }
            }

            await newPage.GotoAsync(testPageUrl).ConfigureAwait(false);
            await newPage.WaitForSelectorAsync("table.unit-test-ready", new() { Timeout = 30000 })
                .ConfigureAwait(false);
            return newPage;
        }

        // Categories whose tests must run serialized on the CPU lane (one all-core CPU-accelerator
        // process at a time). NOT an exclusion list — HeavyCpu tests run normally on every other
        // lane (browser/WebGPU/WebGL/Wasm/CUDA/OpenCL), where they are fast; they are only slow on
        // the CPU backend. Distinct from DefaultExcludedCategories ("HeavyModel" = skipped
        // everywhere). Override the set via PMT_CPU_HEAVY_CATEGORIES (comma-separated).
        private static readonly string[] DefaultCpuHeavyCategories = { "HeavyCpu" };
        private static string[] CpuHeavyCategories()
        {
            var env = Environment.GetEnvironmentVariable("PMT_CPU_HEAVY_CATEGORIES");
            if (env == null) return DefaultCpuHeavyCategories;
            return env.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        }
        private static bool IsCpuHeavy(ProjectTest t) =>
            !string.IsNullOrEmpty(t.Category)
            // A test's Category may be comma-separated ("HeavyCpu,WasmHeavy") - match any part.
            && t.Category.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
                .Any(c => CpuHeavyCategories().Contains(c, StringComparer.OrdinalIgnoreCase));

        /// <summary>
        /// CPU desktop lane: run the light tests at the lane cap (parallelism hides per-process
        /// JIT/startup), then run the HeavyCpu tests serialized so each compute-bound, all-core
        /// CPU test runs with ZERO competition from sibling CPU tests. A cap=1 PMT run proved a
        /// heavy CPU style model passes well within its 120s budget when it does not contend with
        /// other heavy CPU processes (the GPU/browser lanes running concurrently are fine). Light
        /// runs first so the quick wins land before the serialized heavy pole.
        /// </summary>
        private async Task RunCpuLaneAsync(List<ProjectTest> light, int lightCap, List<ProjectTest> heavy, int heavyCap)
        {
            if (light.Count > 0) await RunLaneConcurrentAsync(light, lightCap, null).ConfigureAwait(false);
            if (heavy.Count > 0) await RunLaneConcurrentAsync(heavy, heavyCap, null).ConfigureAwait(false);
        }

        private async Task RunLaneConcurrentAsync(List<ProjectTest> tests, int cap, IPage? page)
        {
            using var sem = new SemaphoreSlim(cap, cap);
            var tasks = tests.Select(async t =>
            {
                await sem.WaitAsync().ConfigureAwait(false);
                try { await ExecuteAndCaptureAsync(t, page!).ConfigureAwait(false); }
                finally { sem.Release(); }
            });
            await Task.WhenAll(tasks).ConfigureAwait(false);
        }

        /// <summary>
        /// Runs one test via its existing TestFunc (browser = click row on the given page;
        /// desktop = spawn subprocess) and records the outcome. Mirrors the verdict logic
        /// in UnitTest1.RunTest so cached + live paths agree.
        /// </summary>
        private async Task ExecuteAndCaptureAsync(ProjectTest test, IPage? page)
        {
            var sw = Stopwatch.StartNew();
            try
            {
                await test.TestFunc(page!).ConfigureAwait(false);
                sw.Stop();
                var status = test.Result == TestResult.Unsupported ? "Skip" : "Pass";
                _outcomes[test.Name] = new ScheduledOutcome(status, test.ResultMessage, sw.Elapsed.TotalMilliseconds);
            }
            catch (Exception ex)
            {
                sw.Stop();
                _outcomes[test.Name] = new ScheduledOutcome("Fail", ex.Message, sw.Elapsed.TotalMilliseconds);
            }
        }

        /// <summary>
        /// This is called after tests have ran. You can use this to stop any services or infrastructure started in StartUp.
        /// </summary>
        /// <returns></returns>
        public async Task Shutdown()
        {
            Debug.WriteLine("Shutdown()");
            foreach (var testableProject in TestableProjects)
            {
                if (testableProject is TestableBlazorWasm blazorProj)
                {
                    try { if (blazorProj.Page != null) await blazorProj.Page.CloseAsync().ConfigureAwait(false); } catch { }
                    try { if (blazorProj.BrowserContext != null) await blazorProj.BrowserContext.CloseAsync().ConfigureAwait(false); } catch { }
                    try { if (blazorProj.Browser != null) await blazorProj.Browser.CloseAsync().ConfigureAwait(false); } catch { }
                    try { blazorProj.Playwright?.Dispose(); } catch { }
                    try { if (blazorProj.Server != null) await blazorProj.Server.Stop().ConfigureAwait(false); } catch { }
                }
                else if (testableProject is TestableConsole consoleProj)
                {
                    // do any cleanup needed for console projects
                }
            }
        }
    }
}