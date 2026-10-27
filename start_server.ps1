# AIVCO Argon Edition - Quick Launch Script
Set-Location -Path $PSScriptRoot

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "       STARTING AIVCO ARGON EDITION SERVER...           " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Access URL: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host ""

Start-Process "http://127.0.0.1:8000"

& ".\.venv\Scripts\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload

