@echo off
rem ============================================================================
rem  Scoped PMT gate launcher.   %1 = PMT_FILTER value    %2 = log basename
rem                              %3 = PMT_CONSOLE_LOG substring (OPTIONAL - browser console lines are
rem                                   otherwise summarised away and never reach the log)
rem
rem  Usage (launch OUTSIDE the agent shell's job object, per the global Rule 5b,
rem  so the sweep survives the session that started it):
rem
rem    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
rem      CommandLine='cmd.exe /c ""<repo>\tools\run-scoped-gate.cmd" "DA3Small" "pmt-da3""'}
rem
rem  Start-Process inherits the shell's job object and dies with it. WMI parents to
rem  WmiPrvSE and still lands in SessionId 1, so non-headless Chromium works.
rem
rem  🔴 THE TRAP THIS FILE EXISTS TO ENCODE:
rem     PlaywrightMultiTest/CLAUDE.md documents the HeavyModel escape hatch as the
rem     BASH prefix assignment `PMT_EXCLUDE_CATEGORIES= dotnet test ...`, which sets
rem     the variable to an EMPTY STRING. In cmd.exe, `set VAR=` DELETES the variable
rem     instead. ProjectRunner.ExcludedCategories() does:
rem
rem         var env = Environment.GetEnvironmentVariable("PMT_EXCLUDE_CATEGORIES");
rem         if (env == null) return DefaultExcludedCategories;   // { "HeavyModel" }
rem
rem     ...so a deleted variable RESTORES the HeavyModel exclusion and the "heavy"
rem     run silently schedules zero heavy tests and reports a fast, trivial pass.
rem     A non-null sentinel that names no real category is what actually clears it.
rem     Verified 2026-09-08: with `=none` a scoped DA3Small run scheduled and ran
rem     the HeavyModel rows on all six backends.
rem
rem  ⚠️ Confirm scoping by TEST COUNT, never by a green tally. The env values are
rem     echoed into the log below so an empty one is visible.
rem ============================================================================
setlocal
if "%~1"=="" echo ERROR: arg 1 must be the PMT_FILTER value & exit /b 2
if "%~2"=="" echo ERROR: arg 2 must be the log basename & exit /b 2
cd /d "%~dp0.."
set "PMT_EXCLUDE_CATEGORIES=none"
set "PMT_FILTER=%~1"
set "PMT_PARALLEL=off"
rem  ⚠️ ARG 3 (optional) = PMT_CONSOLE_LOG substring. PMT summarises browser console output to
rem  "Console: N error(s), M warning(s)", so a Console.WriteLine benchmark or diagnostic line printed by a
rem  WebGPU/WebGL test DOES NOT REACH THE LOG AT ALL. That is how a diagnostic can run every sweep and have
rem  its verdict discarded - the ZipVoice ELIDE A/B computes which half of capture/replay is at fault and
rem  prints it as [Benchmark] ZipVoiceFidelity ..., and nobody had ever read it.
rem  Pass a substring ("ZipVoiceFidelity"), or 1 / * for everything.
rem  ⚠️ Left UNSET when arg 3 is absent - deliberately. Unlike PMT_EXCLUDE_CATEGORIES, whose null case
rem  restores a DEFAULT (see the header), PMT_CONSOLE_LOG's null case is simply "no console capture", so
rem  not setting it is the correct no-op here.
if not "%~3"=="" set "PMT_CONSOLE_LOG=%~3"
if "%GATE_LOG_DIR%"=="" set "GATE_LOG_DIR=%TEMP%"
set "LOG=%GATE_LOG_DIR%\%~2.log"
echo === PMT scoped gate start %DATE% %TIME% === > "%LOG%"
echo PMT_EXCLUDE_CATEGORIES=[%PMT_EXCLUDE_CATEGORIES%] >> "%LOG%"
echo PMT_FILTER=[%PMT_FILTER%] >> "%LOG%"
echo PMT_PARALLEL=[%PMT_PARALLEL%] >> "%LOG%"
echo PMT_CONSOLE_LOG=[%PMT_CONSOLE_LOG%] >> "%LOG%"
echo LOG=[%LOG%] >> "%LOG%"
dotnet test PlaywrightMultiTest\PlaywrightMultiTest.csproj -c Release >> "%LOG%" 2>&1
echo PMT_EXITCODE=%ERRORLEVEL% >> "%LOG%"
echo === PMT scoped gate end %DATE% %TIME% === >> "%LOG%"
