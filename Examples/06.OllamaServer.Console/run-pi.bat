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
set "MODEL=qwen2.5-coder:7b-instruct-q4_K_M"

REM ---- Port: Pi's ollama provider is configured (in ~/.pi/agent/models.json) to reach
REM      http://127.0.0.1:11434/v1, so we run ON 11434. (Don't run real Ollama at the same time.)
set "PORT=11434"

REM ---- Interactive bounds ----
REM   NUM_CTX: KV-cache size. A large (12B+) model plus a big context OOMs a 12GB card — e.g. gemma4:12b
REM            (7.6GB) + a 16k+ cache approaches 12GB, pegging the GPU with ZERO token throughput. Keep
REM            7B/8B models at 32768; cap big models to 4096 (verified to fit gemma4:12b on a 4070 with
REM            headroom). Raise the 4096 only if you have the VRAM. MAX_OUTPUT caps a verbose answer.
set "OLLAMA_NUM_CTX=32768"
echo %MODEL%| findstr /I "12b 13b 14b 24b 27b 30b 32b 70b gemma4" >nul && set "OLLAMA_NUM_CTX=4096"
set "OLLAMA_MAX_OUTPUT=1024"

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

REM ---- Point Pi's ollama provider at our server ----
REM   pi-ollama defaults to http://127.0.0.1:11434; OLLAMA_HOST overrides it to our port.
set "OLLAMA_HOST=http://127.0.0.1:%PORT%"
set "OLLAMA_BASE_URL=http://127.0.0.1:%PORT%"

echo.
echo   Launching Pi (ollama provider -^> our server). If Pi shows a stale model list, run /ollama-refresh.
echo.
REM Model id uses Pi's "ollama/" provider prefix (matches its enabledModels). Verified: returns "Paris".
REM
REM -nc (--no-context-files): SKIP auto-loading AGENTS.md + CLAUDE.md. Those are ~20K tokens of AI-agent
REM   META-RULES (not codebase docs), and a ~20K-token prompt means a multi-MINUTE turn-1 prefill on the
REM   current engine (large-context prefill is the open perf frontier) - i.e. "Pi never answers". Skipping
REM   them keeps the prompt small so Pi responds fast. Drop -nc once large-context prefill is fast, or add a
REM   small codebase-specific AGENTS.md. Pass a prompt/context explicitly with @file when you need it.
pi -nc --model "ollama/%MODEL%"

echo.
echo   Pi exited. The server is still running in its own window - close it to stop.
pause
endlocal
