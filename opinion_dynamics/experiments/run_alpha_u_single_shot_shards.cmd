@echo off
setlocal EnableDelayedExpansion

set "PYTHON_CMD=%~1"
if "%PYTHON_CMD%"=="" set "PYTHON_CMD=python"

set "TORCH_THREADS=%~2"
if "%TORCH_THREADS%"=="" set "TORCH_THREADS=2"

set "POLL_SECONDS=%~3"
if "%POLL_SECONDS%"=="" set "POLL_SECONDS=60"

set "NOTEBOOK=experiment_2026_09_06_alpha_u_single_shot.ipynb"
set "RESULTS_DIR=results\alpha_u_ss"
set "LOG_DIR=logs_alpha_u_ss"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo Starting alpha-u single-shot benchmark: 8 shards, %TORCH_THREADS% Torch threads/shard.
for /L %%S in (0,1,7) do (
    echo   starting shard %%S
    start "" /B cmd /C "set BROAD_FULL_SHARD_ID=%%S&& set BROAD_FULL_TORCH_THREADS=%TORCH_THREADS%&& set BROAD_FULL_DEVICE=cpu&& set MPLBACKEND=Agg&& %PYTHON_CMD% run_ipynb_cells.py %NOTEBOOK% > %LOG_DIR%\shard_%%S.log 2>&1"
)

:wait_loop
set /A DONE=0
for /L %%S in (0,1,7) do (
    if exist "%RESULTS_DIR%\SHARD_%%S_COMPLETE.json" set /A DONE+=1
)
echo Completed shards: !DONE!/8
if !DONE! LSS 8 (
    timeout /T %POLL_SECONDS% /NOBREAK >NUL
    goto wait_loop
)

echo All worker shards completed. Running check/merge pass...
%PYTHON_CMD% run_ipynb_cells.py %NOTEBOOK%
if errorlevel 1 exit /B 1

echo Alpha-u single-shot benchmark complete.
endlocal
