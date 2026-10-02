@echo off
rem Copies SigmaFpRaw.ofx.bundle into the OFX plug-in folder DaVinci Resolve loads.
setlocal
set "SRC=%~dp0SigmaFpRaw.ofx.bundle"
set "DEST=%CommonProgramFiles%\OFX\Plugins"

if not exist "%SRC%\Contents\Win64\SigmaFpRaw.ofx" (
    echo SigmaFpRaw.ofx.bundle was not found next to this script.
    echo Unzip the whole download first, then run this script from the unzipped folder.
    pause
    exit /b 1
)

net session >nul 2>&1
if errorlevel 1 (
    echo Asking for administrator rights...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

tasklist /FI "IMAGENAME eq Resolve.exe" 2>nul | find /I "Resolve.exe" >nul
if not errorlevel 1 (
    echo DaVinci Resolve is running. Close it, then run this script again.
    pause
    exit /b 1
)

if not exist "%DEST%" mkdir "%DEST%"
if exist "%DEST%\SigmaFpRaw.ofx.bundle" rmdir /S /Q "%DEST%\SigmaFpRaw.ofx.bundle"
xcopy "%SRC%" "%DEST%\SigmaFpRaw.ofx.bundle\" /E /I /Y /Q >nul
if errorlevel 1 (
    echo The copy failed.
    pause
    exit /b 1
)

echo.
echo Sigma fp RAW is installed in:
echo   %DEST%\SigmaFpRaw.ofx.bundle
echo Start DaVinci Resolve: Color page ^> Effects ^> Sigma fp ^> Sigma fp RAW.
pause
