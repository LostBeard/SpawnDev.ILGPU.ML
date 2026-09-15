@echo off
rem ============================================================================
rem  FULL PMT sweep launcher.   %1 = log basename (OPTIONAL, default "pmt-full")
rem
rem  The sibling of run-scoped-gate.cmd, for a change that cannot be scoped: the
rem  shape interpreter, the graph compiler, the executor's node loop, the buffer
rem  pool - anything every model goes through.
rem
rem  ⚠️ HeavyModel stays EXCLUDED here (the default). That is deliberate: this is
rem     the ~500-test regression gate, and the heavy end-to-end models are a
rem     SEPARATE, scoped, SEQUENTIAL run - concurrent heavy WebGPU + CUDA +
rem     OpenCL on the one GPU causes D3D12 DEVICE_REMOVED. Do BOTH before
rem     calling an interpreter/executor change verified.
rem
rem  Launch OUTSIDE the agent shell's job object (global Rule 5b):
rem    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
rem      CommandLine='cmd.exe /c ""<repo>\tools\run-full-gate.cmd" "pmt-full""'}
rem ============================================================================
setlocal
cd /d "%~dp0.."
if "%~1"=="" (set "BASE=pmt-full") else (set "BASE=%~1")
set "PMT_PARALLEL=off"
if "%GATE_LOG_DIR%"=="" set "GATE_LOG_DIR=%TEMP%"
set "LOG=%GATE_LOG_DIR%\%BASE%.log"
set "TRX=%BASE%.trx"
echo === PMT full sweep start %DATE% %TIME% === > "%LOG%"
echo PMT_FILTER=[%PMT_FILTER%]  ^(empty = every test^) >> "%LOG%"
echo PMT_EXCLUDE_CATEGORIES=[%PMT_EXCLUDE_CATEGORIES%]  ^(empty = default, HeavyModel excluded^) >> "%LOG%"
echo LOG=[%LOG%] >> "%LOG%"
dotnet test PlaywrightMultiTest\PlaywrightMultiTest.csproj -c Release --logger "trx;LogFileName=%TRX%" --results-directory PlaywrightMultiTest\TestResults >> "%LOG%" 2>&1
echo PMT_EXITCODE=%ERRORLEVEL% >> "%LOG%"

rem  Attribute the failures before anyone reads the count. On 2026-09-15 "Failed: 348" was 346 pieces of
rem  wreckage from ONE device hang plus 2 real defects, and the raw number reads as a broad regression.
dotnet run "%~dp0gate-summary.cs" -- "%LOG%" >> "%LOG%" 2>&1

echo === PMT full sweep end %DATE% %TIME% === >> "%LOG%"
