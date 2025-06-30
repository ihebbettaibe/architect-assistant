@echo off
REM Manual Run Script for Production Tecnocasa Automation
REM Use this to run the automation immediately for testing

echo ================================================================================
echo             Tecnocasa Production Automation - Manual Run
echo ================================================================================
echo.

REM Get current directory and set up paths
set SCRIPT_DIR=%~dp0
set PYTHON_SCRIPT=%SCRIPT_DIR%production_automation.py
set LOG_DIR=%SCRIPT_DIR%logs

REM Create directories if they don't exist
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%SCRIPT_DIR%tecnocasa_html" mkdir "%SCRIPT_DIR%tecnocasa_html"
if not exist "%SCRIPT_DIR%tecnocasa_data" mkdir "%SCRIPT_DIR%tecnocasa_data"
if not exist "%SCRIPT_DIR%reports" mkdir "%SCRIPT_DIR%reports"
if not exist "%SCRIPT_DIR%daily_snapshots" mkdir "%SCRIPT_DIR%daily_snapshots"

echo Directories: 
echo - Script: %SCRIPT_DIR%
echo - Logs: %LOG_DIR%
echo.

REM Check if Python script exists
if not exist "%PYTHON_SCRIPT%" (
    echo ERROR: production_automation.py not found!
    echo Please make sure the script is in the same directory as this batch file.
    echo.
    pause
    exit /b 1
)

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python and add it to your PATH.
    echo.
    pause
    exit /b 1
)

REM Display Python version
echo Python Version:
python --version
echo.

REM Install required packages if needed
echo Checking and installing required packages...
pip install requests beautifulsoup4 pandas couchdb schedule

echo.
echo Starting Tecnocasa Production Automation...
echo ================================================================================
echo.

REM Run the automation
cd /d "%SCRIPT_DIR%"
python "%PYTHON_SCRIPT%"

echo.
echo ================================================================================
echo                         Automation Complete
echo ================================================================================
echo.

REM Display results summary
echo Results can be found in:
echo - Logs: %LOG_DIR%
echo - Data: %SCRIPT_DIR%tecnocasa_data\
echo - Reports: %SCRIPT_DIR%reports\
echo - HTML Files: %SCRIPT_DIR%tecnocasa_html\
echo - Snapshots: %SCRIPT_DIR%daily_snapshots\
echo.

echo Press any key to exit...
pause >nul
