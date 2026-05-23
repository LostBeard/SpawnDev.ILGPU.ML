@echo off
@REM Publish SpawnDev.ILGPU.ML to the LOCAL SpawnDevPackages feed.
@REM Finds the most recent Release-build .nupkg under the library project's bin\Release,
@REM registers it with `nuget add` (per-package subdirectory layout), AND drops a flat
@REM top-level copy. Some consumers scan the flat layout directly; without the top-level
@REM copy they will pin to a stale rc.
set projectPath=%~dp0SpawnDev.ILGPU.ML
set releaseFolder=%projectPath%\bin\Release
set feedRoot=D:\users\SpawnDevPackages

@echo:
@echo === SpawnDev.ILGPU.ML (LOCAL FEED) ===

FOR /F "eol=| delims=" %%I IN ('DIR "%releaseFolder%\*.nupkg" /A-D /B /O-D /TW 2^>nul') DO SET "NewestFile=%%I" & GOTO FoundFile
ECHO No *.nupkg file found
GOTO :EOF

:FoundFile
ECHO Latest *.nupkg file is:
ECHO %NewestFile%

nuget add "%releaseFolder%\%NewestFile%" -source "%feedRoot%"
@REM Top-level flat copy for consumers that don't walk the per-package subdir layout.
copy /Y "%releaseFolder%\%NewestFile%" "%feedRoot%\%NewestFile%"
