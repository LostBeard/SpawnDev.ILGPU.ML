@echo off
@REM Publish SpawnDev.Phonemizer to the LOCAL SpawnDevPackages feed.
@REM Finds the most recent Release-build .nupkg under the library project's bin\Release,
@REM registers it with `nuget add` (per-package / per-version hierarchy only).
@REM Do NOT copy .nupkg to the feed root and do NOT use `dotnet nuget push` against this feed.
set projectPath=%~dp0SpawnDev.Phonemizer
set releaseFolder=%projectPath%\bin\Release
set feedRoot=D:\users\SpawnDevPackages

@echo:
@echo === SpawnDev.Phonemizer (LOCAL FEED) ===

FOR /F "eol=| delims=" %%I IN ('DIR "%releaseFolder%\*.nupkg" /A-D /B /O-D /TW 2^>nul') DO SET "NewestFile=%%I" & GOTO FoundFile
ECHO No *.nupkg file found
GOTO :EOF

:FoundFile
ECHO Latest *.nupkg file is:
ECHO %NewestFile%

nuget add "%releaseFolder%\%NewestFile%" -source "%feedRoot%"
pause
