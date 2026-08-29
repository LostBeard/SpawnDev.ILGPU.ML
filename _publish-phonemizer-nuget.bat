@echo off
@REM Publish SpawnDev.Phonemizer to NuGet.org.
@REM Captain authority required per push (see D:\users\tj\Projects\CLAUDE.md NuGet
@REM Package Publishing -> Official Publish section). Never run without an explicit
@REM "go" from Captain in the same session.
@REM
@REM ORDER MATTERS: SpawnDev.ILGPU.ML declares a dependency on this package, so this one
@REM must land on nuget.org FIRST. Pushing ILGPU.ML while this is unpublished gives every
@REM consumer NU1101 on restore.
set projectPath=%~dp0SpawnDev.Phonemizer
set releaseFolder=%projectPath%\bin\Release

@echo:
@echo === SpawnDev.Phonemizer (NUGET.ORG) ===

FOR /F "eol=| delims=" %%I IN ('DIR "%releaseFolder%\*.nupkg" /A-D /B /O-D /TW 2^>nul') DO SET "NewestFile=%%I" & GOTO FoundFile
ECHO No *.nupkg file found
GOTO :EOF

:FoundFile
ECHO Latest *.nupkg file is:
ECHO %NewestFile%

@REM Non-interactive push. API key resolution order:
@REM   1. NUGET_API_KEY environment variable, if set.
@REM   2. Stored credential in %AppData%\Roaming\NuGet\NuGet.Config, set once via:
@REM        nuget setapikey YOUR_KEY -Source https://api.nuget.org/v3/index.json
@IF NOT "%NUGET_API_KEY%" == "" (
    dotnet nuget push "%releaseFolder%\%NewestFile%" --api-key %NUGET_API_KEY% --source https://api.nuget.org/v3/index.json
) else (
    dotnet nuget push "%releaseFolder%\%NewestFile%" --source https://api.nuget.org/v3/index.json
)
