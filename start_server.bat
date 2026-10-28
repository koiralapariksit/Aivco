@echo off
title AIVCO Argon Edition Server
cd /d "%~dp0"

echo ========================================================
echo        STARTING AIVCO ARGON EDITION SERVER...
echo ========================================================
echo Access URL: http://127.0.0.1:8000
echo.

start http://127.0.0.1:8000

.\.venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --reload
pause

