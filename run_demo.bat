@echo off
REM Double-click to run the CUDA RSA demo from Windows. The program runs inside WSL (Ubuntu).
REM Optional arguments, e.g.:  run_demo.bat verify 1000
REM Build it first inside WSL:  bash build_wsl.sh   (or: make ARCH=sm_XX with a normal CUDA install)
cd /d "%~dp0"
set ARGS=%*
if "%ARGS%"=="" set ARGS=demo 8
wsl -e bash -lc "cd \"$(wslpath '%~dp0.')\" && ./rsa_cuda %ARGS%"
echo.
pause
