@echo off
REM Startup script for AI Trading Platform
REM Launches all required services

echo ========================================
echo AI TRADING PLATFORM - STARTUP
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    pause
    exit /b 1
)

echo [1/4] Starting API Server...
start "API Server" cmd /k "cd /d %~dp0..\\.. && python backend\\scripts\\start_api.py"
timeout /t 3 /nobreak >nul

echo [2/4] Starting Market Stream...
start "Market Stream" cmd /k "cd /d %~dp0..\\.. && python backend\\scripts\\start_market_stream.py"
timeout /t 3 /nobreak >nul

echo [3/4] Starting Execution Loop...
start "Execution Loop" cmd /k "cd /d %~dp0..\\.. && python backend\\scripts\\start_execution_loop.py"
timeout /t 3 /nobreak >nul

echo [4/4] Opening Web UI...
timeout /t 5 /nobreak >nul
start http://localhost:8000

echo.
echo ========================================
echo ALL SERVICES STARTED!
echo ========================================
echo.
echo Services running:
echo   - API Server:      http://localhost:8000
echo   - Market Stream:   Receiving live data
echo   - Execution Loop:  Evaluating signals
echo.
echo Press any key to open diagnostics...
pause >nul

echo.
echo Running signal diagnostics...
python backend\\scripts\\diagnose_signals.py
pause
