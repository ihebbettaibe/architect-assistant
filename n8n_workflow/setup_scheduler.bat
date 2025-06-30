@echo off
REM Production Tecnocasa Automation - Windows Task Scheduler Setup
REM This script creates a scheduled task to run the automation daily at 2:00 AM

echo Setting up Tecnocasa Production Automation...
echo.

REM Get current directory
set SCRIPT_DIR=%~dp0
set PYTHON_SCRIPT=%SCRIPT_DIR%production_automation.py
set LOG_FILE=%SCRIPT_DIR%logs\scheduler_setup.log

REM Create logs directory if it doesn't exist
if not exist "%SCRIPT_DIR%logs" mkdir "%SCRIPT_DIR%logs"

echo Script Directory: %SCRIPT_DIR%
echo Python Script: %PYTHON_SCRIPT%
echo Log File: %LOG_FILE%
echo.

REM Check if Python script exists
if not exist "%PYTHON_SCRIPT%" (
    echo ERROR: production_automation.py not found in %SCRIPT_DIR%
    echo Please make sure the script is in the correct location.
    pause
    exit /b 1
)

REM Find Python executable
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python or add it to your PATH
    pause
    exit /b 1
)

REM Get Python executable path
for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)"') do set PYTHON_EXE=%%i

echo Python Executable: %PYTHON_EXE%
echo.

REM Create the scheduled task
echo Creating Windows Scheduled Task...

schtasks /create ^
    /tn "Tecnocasa Production Automation" ^
    /tr "\"%PYTHON_EXE%\" \"%PYTHON_SCRIPT%\"" ^
    /sc daily ^
    /st 02:00 ^
    /sd %date% ^
    /ru SYSTEM ^
    /f

if errorlevel 1 (
    echo ERROR: Failed to create scheduled task
    echo You may need to run this script as Administrator
    pause
    exit /b 1
)

echo.
echo SUCCESS: Scheduled task created successfully!
echo.
echo Task Details:
echo - Name: Tecnocasa Production Automation
echo - Schedule: Daily at 2:00 AM
echo - Script: %PYTHON_SCRIPT%
echo - Python: %PYTHON_EXE%
echo.

REM Display the created task
echo Verifying task creation...
schtasks /query /tn "Tecnocasa Production Automation" /fo table

echo.
echo Setup complete! The automation will run daily at 2:00 AM.
echo.
echo To manage the scheduled task:
echo - View: schtasks /query /tn "Tecnocasa Production Automation"
echo - Run now: schtasks /run /tn "Tecnocasa Production Automation"
echo - Delete: schtasks /delete /tn "Tecnocasa Production Automation" /f
echo.
echo Press any key to exit...
pause >nul
