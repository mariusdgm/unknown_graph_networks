@echo off
setlocal EnableExtensions

set PYTHON_CMD=%~1
if "%PYTHON_CMD%"=="" set PYTHON_CMD=python
set TORCH_THREADS=%~2
if "%TORCH_THREADS%"=="" set TORCH_THREADS=2
set POLL_SECONDS=%~3
if "%POLL_SECONDS%"=="" set POLL_SECONDS=60
set NUM_SHARDS=%~4
if "%NUM_SHARDS%"=="" set NUM_SHARDS=8

set /A LAST_SHARD=%NUM_SHARDS%-1
if %LAST_SHARD% LSS 0 exit /b 2

set NOTEBOOK=experiment_2026_09_29_abundant_full_regularization_sweep.ipynb
set LOG_DIR=logs_abundant_full_regularization_sweep
set DONE_DIR=%LOG_DIR%\_launcher_done
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%DONE_DIR%" mkdir "%DONE_DIR%"
del /q "%DONE_DIR%\shard_*.exit" >nul 2>&1

echo Launching ABUNDANT full regularization sweep.
echo   seed: 0
echo   variants: Gini + entropy + theta_l2; no-reg reference reused from finalized run
echo   exact 2500-pair datasets reused
echo   shards: %NUM_SHARDS%

for /L %%S in (0,1,%LAST_SHARD%) do (
  start "" /B cmd /C "set FULL_REG_AB_SHARD_ID=%%S&& set FULL_REG_AB_NUM_SHARDS=%NUM_SHARDS%&& set FULL_REG_AB_TORCH_THREADS=%TORCH_THREADS%&& set MPLBACKEND=Agg&& %PYTHON_CMD% run_ipynb_cells.py %NOTEBOOK% > %LOG_DIR%\shard_%%S.log 2>&1 && (> %DONE_DIR%\shard_%%S.exit echo 0) || (> %DONE_DIR%\shard_%%S.exit echo 1)"
)

:WAIT
set DONE_COUNT=0
for /L %%S in (0,1,%LAST_SHARD%) do (
  if exist "%DONE_DIR%\shard_%%S.exit" set /A DONE_COUNT+=1
)
echo [%TIME%] completed shards: %DONE_COUNT%/%NUM_SHARDS%
if not "%DONE_COUNT%"=="%NUM_SHARDS%" (
  timeout /t %POLL_SECONDS% /nobreak >nul
  goto WAIT
)

set FAILED=0
for /L %%S in (0,1,%LAST_SHARD%) do (
  set /p RC=<"%DONE_DIR%\shard_%%S.exit"
  call if not "%%RC%%"=="0" set FAILED=1
)
if "%FAILED%"=="1" (
  echo One or more shards failed. Inspect %LOG_DIR%\shard_*.log
  exit /b 1
)

echo All shards finished. Running validation/merge...
set FULL_REG_AB_SHARD_ID=
set FULL_REG_AB_NUM_SHARDS=%NUM_SHARDS%
set FULL_REG_AB_TORCH_THREADS=
set MPLBACKEND=Agg
%PYTHON_CMD% run_ipynb_cells.py %NOTEBOOK% > %LOG_DIR%\final_validation.log 2>&1
if errorlevel 1 (
  echo Validation/merge failed. Inspect %LOG_DIR%\final_validation.log
  exit /b 1
)
type "%LOG_DIR%\final_validation.log"
echo Abundant full regularization sweep complete.
exit /b 0
