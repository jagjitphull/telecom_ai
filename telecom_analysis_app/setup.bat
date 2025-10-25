@echo off
REM Telecom Log Analysis Setup Script for Windows

setlocal enabledelayedexpansion

color 0A
cls

echo.
echo ╔════════════════════════════════════════╗
echo ║  Telecom Log Analysis - Setup Script   ║
echo ╚════════════════════════════════════════╝
echo.

REM Check Python
echo [1/5] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VER=%%i
echo [OK] Python %PYTHON_VER% found

REM Install dependencies
echo.
echo [2/5] Installing Python dependencies...
python -m pip install -q -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed

REM Check Ollama
echo.
echo [3/5] Checking Ollama...
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Ollama not found
    echo Please install Ollama from https://ollama.ai/
    echo.
) else (
    echo [OK] Ollama found
)

REM Create directories
echo.
echo [4/5] Creating directories...
if not exist "uploads" mkdir uploads
if not exist "reports" mkdir reports
echo [OK] Directories created

REM Complete
echo.
echo ╔════════════════════════════════════════╗
echo ║  Setup Complete!                       ║
echo ╚════════════════════════════════════════╝
echo.
echo Quick Start:
echo 1. Start Ollama:
echo    - Open Command Prompt or PowerShell
echo    - Run: ollama serve
echo.
echo 2. Start the Flask app:
echo    - Open another Command Prompt
echo    - Run: python app.py
echo.
echo 3. Open browser: http://localhost:5000
echo.
echo Documentation: Read README.md for details
echo.
pause
