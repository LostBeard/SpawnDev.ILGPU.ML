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
REM     qwen2.5-coder 7B = the FAST interactive pick (~90ms/tok). gemma4:12b is slower (12B) and has a
REM     large-context decode bug being fixed separately.
set "MODEL=qwen2.5-coder:7b-instruct-q4_K_M"

REM ---- Port (11435 avoids a clash with a running real Ollama on 11434) ----
set "PORT=11435"

REM ---- Interactive bounds ----
REM   NUM_CTX: FULL context — no truncation. Claude Code re-sends ~14.5K tokens (CLAUDE.md + its system prompt
REM            + tools) every turn, where the STATIC head is identical turn-over-turn. KV-PREFIX CACHING reuses
REM            that head: turn 1 prefills the whole prompt, turns 2+ reuse the bit-identical cached prefix and
REM            prefill ONLY the new suffix → fast AND full-context (true Ollama parity — Ollama does not
REM            truncate either). We set the model's full context so the prompt is NEVER tail-truncated (which
REM            would drop the reusable static head AND shift RoPE positions, disabling reuse). Turn-1 prefill
REM            cost is the next perf frontier (faster dequant-GEMM/attention + a persistent prefix), NOT a
REM            reason to drop context. See claude-cli-perf.log for per-turn reused/prefilled token counts.
REM   MAX_OUTPUT: hard cap on generated tokens so a verbose answer can't run for minutes.
set "OLLAMA_NUM_CTX=32768"
set "OLLAMA_MAX_OUTPUT=1024"

echo.
echo   SpawnDev.ILGPU.ML  -  Claude CLI on your own GPU
echo   model: %MODEL%   server: http://localhost:%PORT%
echo.
echo   Opening the server in a new window (leave it running)...

set "OLLAMA_PORT=%PORT%"
REM Pre-load the model at server startup so it is GPU-resident BEFORE Claude CLI's concurrent startup burst
REM hits it. A lazy first-request load held the single generation gate, and Claude's other startup requests
REM (title + warmup) canceled while waiting (the OperationCanceledException in the request log). /api/version
REM only answers once the model is loaded, so we poll it below instead of guessing a fixed wait.
set "OLLAMA_PRELOAD=%MODEL%"
start "SpawnDev.ILGPU.ML Server (%MODEL%)" cmd /k dotnet run --project "%~dp0OllamaServer.Console.csproj" -c Release

echo   Waiting for the server to load %MODEL% onto the GPU (first load of a multi-GB model takes ~10-40s)...
:waitready
timeout /t 2 /nobreak >nul
curl -s -o nul -m 2 http://localhost:%PORT%/api/version
if errorlevel 1 goto waitready
echo   Server ready (model resident on the GPU).

REM ---- Point Claude CLI at our server (it speaks the Anthropic Messages API) ----
set "ANTHROPIC_BASE_URL=http://localhost:%PORT%"
set "ANTHROPIC_AUTH_TOKEN=local-spawndev"
set "ANTHROPIC_MODEL=%MODEL%"
set "ANTHROPIC_CUSTOM_MODEL_OPTION=%MODEL%"
set "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME=Local %MODEL%"
set "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1"

echo.
echo   Launching Claude CLI (model already resident - first message responds right away)...
echo.
claude

echo.
echo   Claude CLI exited. The server is still running in its own window - close it to stop.
pause
endlocal
