@echo off
REM ============================================================================
REM  Start the SpawnDev.ILGPU.ML Ollama-replacement server and launch Claude CLI
REM  connected to it (Claude CLI runs on OUR native-GPU engine, not Anthropic's).
REM
REM  Just double-click this file. A separate window opens for the server; leave it
REM  running. Close either window to stop.
REM ============================================================================
setlocal

REM ---- Which cached model to use. Any name from:  dotnet run -- --list ----
REM     qwen2.5-coder is the coding model. For a bigger one try qwen2.5-coder:14b.
set "MODEL=qwen2.5-coder:latest"

REM ---- Port (11435 avoids a clash with a running real Ollama on 11434) ----
set "PORT=11435"

echo.
echo   SpawnDev.ILGPU.ML  -  Claude CLI on your own GPU
echo   model: %MODEL%   server: http://localhost:%PORT%
echo.
echo   Opening the server in a new window (leave it running)...

set "OLLAMA_PORT=%PORT%"
start "SpawnDev.ILGPU.ML Server (%MODEL%)" cmd /k dotnet run --project "%~dp0OllamaServer.Console.csproj" -c Release

echo   Waiting ~10s for the server to come up...
timeout /t 10 /nobreak >nul

REM ---- Point Claude CLI at our server (it speaks the Anthropic Messages API) ----
set "ANTHROPIC_BASE_URL=http://localhost:%PORT%"
set "ANTHROPIC_AUTH_TOKEN=local-spawndev"
set "ANTHROPIC_MODEL=%MODEL%"
set "ANTHROPIC_CUSTOM_MODEL_OPTION=%MODEL%"
set "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME=Local %MODEL%"
set "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1"

echo.
echo   Launching Claude CLI...
echo   (your FIRST message loads the model onto the GPU - give it ~15-20s)
echo.
claude

echo.
echo   Claude CLI exited. The server is still running in its own window - close it to stop.
pause
endlocal
