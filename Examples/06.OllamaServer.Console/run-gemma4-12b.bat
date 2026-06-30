@echo off
REM ============================================================================
REM  Start the SpawnDev.ILGPU.ML Ollama-replacement server and launch the Pi
REM  agentic CLI connected to it (Pi runs on OUR native-GPU engine, not a cloud).
REM
REM  Pi reaches us through its built-in `pi-ollama` provider: it speaks the Ollama
REM  API (/api/tags, /api/show, /api/chat) and we point it at our server with
REM  OLLAMA_HOST. Pi is lighter-weight than Claude CLI (much smaller turn-1 prompt),
REM  so it's the recommended front end while the big-context prefill perf work lands.
REM
REM  Just double-click this file. A separate window opens for the server; leave it
REM  running. Close either window to stop.
REM ============================================================================
setlocal

REM ---- Which cached model to use. Any name from:  dotnet run -- --list ----
set "MODEL=gemma4:12b"

REM ---- Port: Pi's ollama provider is configured (in ~/.pi/agent/models.json) to reach
REM      http://127.0.0.1:11434/v1, so we run ON 11434. (Don't run real Ollama at the same time.)
set "PORT=11434"

REM ---- Interactive bounds ----
REM   NUM_CTX: KV-cache size. A large (12B+) model plus a big context OOMs a 12GB card — e.g. gemma4:12b
REM            (7.6GB) + a 16k+ cache approaches 12GB, pegging the GPU with ZERO token throughput. Keep
REM            7B/8B models at 32768; cap big models to 4096 (verified to fit gemma4:12b on a 4070 with
REM            headroom). Raise the 4096 only if you have the VRAM.
REM   MAX_OUTPUT: caps generated tokens. A file-writing tool call carries the whole file IN the call, so a
REM            small cap truncates it mid-content and drops trailing required args (path) -> "validation
REM            failed". 2048 holds a typical file-write call; raise it if you write large files.
set "OLLAMA_NUM_CTX=32768"
echo %MODEL%| findstr /I "12b 13b 14b 24b 27b 30b 32b 70b gemma4" >nul && set "OLLAMA_NUM_CTX=4096"
set "OLLAMA_MAX_OUTPUT=2048"

echo.
echo   SpawnDev.ILGPU.ML  -  Pi on your own GPU
echo   model: %MODEL%   server: http://localhost:%PORT%
echo.
echo   Opening the server in a new window (leave it running)...

set "OLLAMA_PORT=%PORT%"
REM Pre-load the model at server startup so it is GPU-resident before Pi's first request.
set "OLLAMA_PRELOAD=%MODEL%"
start "SpawnDev.ILGPU.ML Server (%MODEL%)" cmd /k dotnet run --project "%~dp0OllamaServer.Console.csproj" -c Release

echo   Waiting for the server to load %MODEL% onto the GPU (first load of a multi-GB model takes ~10-40s)...
:waitready
timeout /t 2 /nobreak >nul
curl -s -o nul -m 2 http://localhost:%PORT%/api/version
if errorlevel 1 goto waitready
echo   Server ready (model resident on the GPU).
