@echo off
REM ============================================================================
REM  Windows launcher -> runs the WSL pipeline script (run_all.sh).
REM  cmd cannot cd into \\wsl.localhost (UNC), so we cd inside WSL instead.
REM ============================================================================
set PROJECT=/home/vigoroth/projects/deel_learning_master
echo Launching pipeline inside WSL...
wsl -d Ubuntu-20.04 bash -lic "cd '%PROJECT%' && ./run_all.sh"
if errorlevel 1 (
    echo.
    echo [FAILED] WSL pipeline exited with error %errorlevel%.
    pause
    exit /b %errorlevel%
)
echo.
echo Done. See results\ for outputs.
pause
