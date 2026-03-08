@echo off
title Alivai Synapse Server
cd /d "%~dp0"

REM Stop existing Alivai bridge process to avoid port 8000 bind errors
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -match 'hff_bridge.py|bridge_server.py') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

REM Ensure Open WebUI container is running
docker start open-webui >nul 2>&1

REM Open WebUI in default browser
start "" "http://localhost:3000"

REM Start the bridge (keeps this window open with logs)
call .venv\Scripts\python.exe hff_bridge.py

if errorlevel 1 (
	echo.
	echo Failed to start Alivai bridge. Check if port 8000 is in use by another app.
	pause
)
