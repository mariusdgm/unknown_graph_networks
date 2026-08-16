@echo off
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0"

set "PYTHON_CMD=%~1"
if "%PYTHON_CMD%"=="" set "PYTHON_CMD=python"
set "THREADS_PER_SHARD=%~2"
if "%THREADS_PER_SHARD%"=="" set "THREADS_PER_SHARD=3"
set "POLL_SECONDS=%~3"
if "%POLL_SECONDS%"=="" set "POLL_SECONDS=60"

set "NOTEBOOK=experiment_2026_08_15_fj_structured_topology_discovery.ipynb"
set "RUNNER=run_ipynb_cells.py"
set "LOG_DIR=fj_structured_topology_discovery_logs"

if not exist "%RUNNER%" (
  echo ERROR: %RUNNER% not found.
  popd
  exit /b 1
)
if not exist "%NOTEBOOK%" (
  echo ERROR: %NOTEBOOK% not found.
  popd
  exit /b 1
)
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for %%S in (0 1 2) do (
  del /q "%LOG_DIR%\shard_%%S.done" 2>nul
  del /q "%LOG_DIR%\shard_%%S.failed" 2>nul
  del /q "%LOG_DIR%\shard_%%S.log" 2>nul
)

echo Starting three FJ shards...

for %%S in (0 1 2) do (
  start "FJ-shard-%%S" /b cmd /v:on /c ^
    "set FJ_DISCOVERY_SHARD_ID=%%S&& set FJ_DISCOVERY_TORCH_THREADS=%THREADS_PER_SHARD%&& set MPLBACKEND=Agg&& %PYTHON_CMD% %RUNNER% %NOTEBOOK% > %LOG_DIR%\shard_%%S.log 2>&1 && (echo success> %LOG_DIR%\shard_%%S.done) || (echo failed> %LOG_DIR%\shard_%%S.failed)"
)

:poll
set /a DONE_COUNT=0
set /a FAILED_COUNT=0
for %%S in (0 1 2) do (
  if exist "%LOG_DIR%\shard_%%S.done" set /a DONE_COUNT+=1
  if exist "%LOG_DIR%\shard_%%S.failed" set /a FAILED_COUNT+=1
)
echo [%date% %time%] completed=!DONE_COUNT!/3 failed=!FAILED_COUNT!/3
if !FAILED_COUNT! GTR 0 goto failed
if !DONE_COUNT! EQU 3 goto success
timeout /t %POLL_SECONDS% /nobreak >nul
goto poll

:failed
echo One or more shards failed. Inspect "%LOG_DIR%".
popd
exit /b 1

:success
echo All three shards completed successfully.
echo Open experiment_2026_08_15_fj_structured_topology_discovery.ipynb normally and Run All with the shard variable unset.
popd
exit /b 0
