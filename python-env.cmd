@echo off
setlocal
set "PROJECT_BASE="
for /f "usebackq tokens=1,2,*" %%A in ("%~dp0.venv\pyvenv.cfg") do if "%%A"=="executable" set "PROJECT_BASE=%%C"
if not defined PROJECT_BASE exit /b 2
set "__PYVENV_LAUNCHER__=%~dp0.venv\Scripts\python.exe"
"%PROJECT_BASE%" %*
exit /b %errorlevel%
