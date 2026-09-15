@echo off
rem ============================================================================
rem  Overnight gate: the FULL regression sweep, then the HeavyModel end-to-end
rem  models, sequentially. %1 = log basename prefix (default "pmt-overnight").
rem
rem  Why chained rather than two launches: the full sweep excludes HeavyModel by
rem  design (they are minutes each) and the heavy rows must not run concurrently
rem  with it - concurrent heavy WebGPU + CUDA + OpenCL on the one GPU causes
rem  D3D12 DEVICE_REMOVED. Running them back to back uses a long idle window and
rem  still never overlaps them.
rem
rem  ⚠️ Builds FIRST, deliberately. A sweep whose assemblies were rebuilt underneath
rem     it reports a green that belongs to no particular commit - which is exactly
rem     what happened on 2026-09-14 and cost that run its meaning.
rem
rem  Launch OUTSIDE the agent shell's job object (global Rule 5b):
rem    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
rem      CommandLine='cmd.exe /c ""<repo>\tools\run-overnight-gate.cmd" "pmt-overnight""'}
rem ============================================================================
setlocal
cd /d "%~dp0.."
if "%~1"=="" (set "BASE=pmt-overnight") else (set "BASE=%~1")
if "%GATE_LOG_DIR%"=="" set "GATE_LOG_DIR=%TEMP%"

echo === overnight gate start %DATE% %TIME% === > "%GATE_LOG_DIR%\%BASE%-build.log"
git rev-parse HEAD >> "%GATE_LOG_DIR%\%BASE%-build.log" 2>&1
dotnet build -c Release --nologo >> "%GATE_LOG_DIR%\%BASE%-build.log" 2>&1
echo BUILD_EXITCODE=%ERRORLEVEL% >> "%GATE_LOG_DIR%\%BASE%-build.log"
if not "%ERRORLEVEL%"=="0" (
  echo BUILD FAILED - neither sweep was run >> "%GATE_LOG_DIR%\%BASE%-build.log"
  exit /b 1
)

call "%~dp0run-full-gate.cmd" "%BASE%-full"
call "%~dp0run-scoped-gate.cmd" "Pipeline_Kokoro,Pipeline_Whisper,DA3Small" "%BASE%-heavy"
echo === overnight gate end %DATE% %TIME% === >> "%GATE_LOG_DIR%\%BASE%-build.log"
