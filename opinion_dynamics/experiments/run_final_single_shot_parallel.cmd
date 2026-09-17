@echo off
setlocal EnableExtensions

set PYTHON_CMD=%~1
if "%PYTHON_CMD%"=="" set PYTHON_CMD=python

set TORCH_THREADS=%~2
if "%TORCH_THREADS%"=="" set TORCH_THREADS=2

set POLL_SECONDS=%~3
if "%POLL_SECONDS%"=="" set POLL_SECONDS=60

set NOTEBOOK=experiment_2026_09_17_broad_all_dynamics_multitopology_multiseed_unbounded_lambda1.ipynb
set LOG_DIR=logs_final_single_shot
set DONE_DIR=%LOG_DIR%\_launcher_done

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%DONE_DIR%" mkdir "%DONE_DIR%"
del /q "%DONE_DIR%\shard_*.exit" >nul 2>&1

echo Launching FINAL single-shot benchmark: 8 shards, %TORCH_THREADS% Torch threads/shard.
echo Notebook: %NOTEBOOK%
echo Poll interval: %POLL_SECONDS% seconds.

for /L %%S in (0,1,7) do (
  start "" /B cmd /C "set BROAD_FULL_SHARD_ID=%%S&& set BROAD_FULL_TORCH_THREADS=%TORCH_THREADS%&& set BROAD_FULL_DEVICE=cpu&& set MPLBACKEND=Agg&& %PYTHON_CMD% run_ipynb_cells.py %NOTEBOOK% > %LOG_DIR%\shard_%%S.log 2>&1 && (> %DONE_DIR%\shard_%%S.exit echo 0) || (> %DONE_DIR%\shard_%%S.exit echo 1)"
)

:WAIT
set DONE_COUNT=0
for /L %%S in (0,1,7) do (
  if exist "%DONE_DIR%\shard_%%S.exit" set /A DONE_COUNT+=1
)
echo [%TIME%] completed shards: %DONE_COUNT%/8
if not "%DONE_COUNT%"=="8" (
  timeout /t %POLL_SECONDS% /nobreak >nul
  goto WAIT
)

set FAILED=0
for /L %%S in (0,1,7) do (
  set /p RC=<"%DONE_DIR%\shard_%%S.exit"
  call if not "%%RC%%"=="0" set FAILED=1
)
if "%FAILED%"=="1" (
  echo One or more shards failed. Inspect %LOG_DIR%\shard_*.log
  exit /b 1
)

echo All shards finished. Running final cache validation/merge in non-worker mode...
set BROAD_FULL_SHARD_ID=
set BROAD_FULL_TORCH_THREADS=
set BROAD_FULL_DEVICE=
set MPLBACKEND=Agg
%PYTHON_CMD% run_ipynb_cells.py %NOTEBOOK% > %LOG_DIR%\final_validation.log 2>&1
if errorlevel 1 (
  echo Final validation/merge failed. Inspect %LOG_DIR%\final_validation.log
  exit /b 1
)

type "%LOG_DIR%\final_validation.log"
echo FINAL single-shot benchmark complete.
exit /b 0
