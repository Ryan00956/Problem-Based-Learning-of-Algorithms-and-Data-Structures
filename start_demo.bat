@echo off
setlocal

cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_frontend.ps1" %*
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Startup failed. Keep this window open and read the message above.
    pause
    exit /b %EXIT_CODE%
)

echo.
echo Backend has stopped.
pause
