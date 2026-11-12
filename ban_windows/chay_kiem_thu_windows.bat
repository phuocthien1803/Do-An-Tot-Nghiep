@echo off
REM ======================================================================
REM  BAN WINDOWS - chay toan bo kiem thu
REM  Chi dung tren may Windows (CPU). Ban Mac: xem ban_mac\.
REM ======================================================================
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set "PY=%USERPROFILE%\anaconda3\envs\second-eye\python.exe"
if not exist "%PY%" set "PY=python"
cd /d "%~dp0.."
set PYTHONPATH=.
"%PY%" -m unittest discover -s tests -p "test_*.py" %*
endlocal
