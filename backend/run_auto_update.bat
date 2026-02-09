@echo off
REM Run auto_update_and_push.py using the repository venv
REM Place this file in the backend/ folder and schedule it with Task Scheduler

SET SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Activate virtual environment (Windows)
if exist "%SCRIPT_DIR%venv\Scripts\activate.bat" (
  call "%SCRIPT_DIR%venv\Scripts\activate.bat"
) else (
  echo Virtual environment activate script not found. Attempting to run with system python.
)

python "%SCRIPT_DIR%auto_update_and_push.py"

REM Exit with the same code so Task Scheduler sees failures
exit /b %ERRORLEVEL%
