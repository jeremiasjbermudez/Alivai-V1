@echo off
REM Start the HFF server (hff_bridge.py) using the project venv
start cmd.exe /k "cd /d c:\Users\Alivai\Documents\Alivai_V1 && .venv\Scripts\python.exe hff_bridge.py"
REM Give the server a moment to start
timeout /t 2 >nul
REM Open dashboard via the server so /v1/hff/pulse works
start "" "http://localhost:8000/dashboard"
REM Open command window for logs in project directory
start cmd.exe /k "cd /d c:\Users\Alivai\Documents\Alivai_V1"
