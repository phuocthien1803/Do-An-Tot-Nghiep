@echo off
REM ======================================================================
REM  BAN WINDOWS - cai thu vien TUY CHON (hoi bang giong noi, phim tat toan he thong) - can mang
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
set HF_HUB_OFFLINE=0
set TRANSFORMERS_OFFLINE=0
"%PY%" -m pip install -r requirements-tuy-chon.txt
"%PY%" scripts/download_models.py --asr small
endlocal
