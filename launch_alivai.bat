@echo off
title Alivai — Synapse Server
cd /d "%~dp0"
start "" "http://localhost:3000"
call .venv\Scripts\python.exe hff_bridge.py
