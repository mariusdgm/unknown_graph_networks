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

set NOTEBOOK=experiment_2026_09_29_abundant_gini_joint_sweep.ipynb
set LOG_DIR=logs_joint_gini_abundant_sweep
set DONE_DIR=%LOG_DIR%\_launcher_done
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%DONE_DIR%" mkdir "%DONE_DIR%"
del /q "%DONE_DIR%\shard_*.exit" >nul 2>&1

echo Launching joint Gini ABUNDANT sweep.
echo   lambdas: 1e-6, 3e-6, 1e-5, 3e-5
echo   learning seed: 0
echo   exact final 2500-pair datasets are reused
echo   old valid sweep points are reused when present
echo   shards: %NUM_SHARDS%

for /L %%S in (0,1,%LAST_SHARD%) do (
  start "" /B cmd /C "set GINI_JOINT_AB_SHARD_ID=%%S&& set GINI_JOINT_AB_NUM_SHARDS=%NUM_SHARDS%&& set GINI_JOINT_AB_TORCH_THREADS=%TORCH_THREADS%&& set MPLBACKEND=Agg&& %PYTHON_CMD% run_ipynb_cells.py %NOTEBOOK% > %LOG_DIR%\shard_%%S.log 2>&1 && (> %DONE_DIR%\shard_%%S.exit echo 0) || (> %DONE_DIR%\shard_%%S.exit echo 1)"
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
set GINI_JOINT_AB_SHARD_ID=
set GINI_JOINT_AB_NUM_SHARDS=%NUM_SHARDS%
set GINI_JOINT_AB_TORCH_THREADS=
set MPLBACKEND=Agg
%PYTHON_CMD% run_ipynb_cells.py %NOTEBOOK% > %LOG_DIR%\final_validation.log 2>&1
if errorlevel 1 (
  echo Validation/merge failed. Inspect %LOG_DIR%\final_validation.log
  exit /b 1
)
type "%LOG_DIR%\final_validation.log"
echo Joint Gini abundant sweep complete.
exit /b 0
