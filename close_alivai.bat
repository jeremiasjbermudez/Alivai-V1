@echo off
title Close Alivai
cd /d "%~dp0"

REM Stop Alivai bridge processes only
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -match 'hff_bridge.py|bridge_server.py') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

REM Stop Open WebUI container
docker stop open-webui >nul 2>&1

REM Stop Ollama process (frees model RAM)
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Process -Name ollama -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue"

echo Alivai services have been stopped.
timeout /t 2 >nul
