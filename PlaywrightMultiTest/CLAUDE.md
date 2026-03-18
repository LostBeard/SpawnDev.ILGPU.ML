# PlaywrightMultiTest

Unified NUnit + Playwright test runner. Runs ALL tests (desktop + browser) via `dotnet test`.

## Running Tests

```bash
# All tests with timestamped results
timestamp=$(date +%Y%m%d_%H%M%S) && dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj --logger "trx;LogFileName=results_${timestamp}.trx" --results-directory PlaywrightMultiTest/TestResults

# Filter to specific tests
dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj --filter "FullyQualifiedName~WasmTests.KernelTest"
```

## How It Works
- `ProjectDiscovery` scans for `<PlaywrightMultiTest>` element in `.csproj` files
- **Blazor WASM**: publishes app, starts HTTPS static file server, launches Chromium (with `--enable-unsafe-webgpu`), navigates to test page, enumerates tests from DOM
- **Console/Exe**: publishes app, runs binary as subprocess for each test
- Tests surfaced as NUnit `TestCaseSource` — standard NUnit `--filter` works

## Key Constraints
- **Blazor WASM publish** takes under 2 minutes — anything longer means it's hung
- **Process hang fix**: uses event-based async reads + `WaitForExit(5000)` with timeout
- **Blazor error detection**: checks `#blazor-error-ui` before/after each test
- **Console capture**: browser console errors/warnings via Playwright `page.Console` event
- **Never start duplicate test processes** — log timing, track carefully
