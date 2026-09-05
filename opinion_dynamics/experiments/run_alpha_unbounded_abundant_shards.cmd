@echo off
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0"

set "PYTHON_CMD=%~1"
if "%PYTHON_CMD%"=="" set "PYTHON_CMD=python"

set "THREADS_PER_SHARD=%~2"
if "%THREADS_PER_SHARD%"=="" set "THREADS_PER_SHARD=2"

set "POLL_SECONDS=%~3"
if "%POLL_SECONDS%"=="" set "POLL_SECONDS=60"

set "NUM_SHARDS=8"
set "NOTEBOOK=experiment_2026_09_05_alpha_unbounded_abundant_all_dynamics_multitopology_multiseed.ipynb"
set "RUNNER=run_ipynb_cells.py"
set "LOG_DIR=abundant_all_dynamics_multitopology_logs_alpha_unbounded"

if not exist "%RUNNER%" (
    echo ERROR: %RUNNER% not found next to this launcher.
    popd
    exit /b 1
)
if not exist "%NOTEBOOK%" (
    echo ERROR: %NOTEBOOK% not found next to this launcher.
    popd
    exit /b 1
)
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for %%S in (0 1 2 3 4 5 6 7) do (
    del /q "%LOG_DIR%\shard_%%S.done" 2>nul
    del /q "%LOG_DIR%\shard_%%S.failed" 2>nul
    del /q "%LOG_DIR%\shard_%%S.log" 2>nul
)

echo Starting eight abundant-data broad-experiment shards...
echo.
echo Design:
echo   exact 195-environment registry from the final broad single-shot study
echo   100 prior-data resets x 5 collection campaigns per environment
echo   one frozen-model fit per learning seed after data collection
echo   3 learned seeds per environment = 585 learned evaluations
echo   same 20-campaign evaluation and baselines as single-shot
echo.
echo Python command: %PYTHON_CMD%
echo Threads per shard: %THREADS_PER_SHARD%
echo Poll interval: %POLL_SECONDS% seconds
echo.

for %%S in (0 1 2 3 4 5 6 7) do (
    start "abundant-full-shard-%%S" /b cmd /v:on /c ^
        "set ABUNDANT_FULL_SHARD_ID=%%S&& set ABUNDANT_FULL_TORCH_THREADS=%THREADS_PER_SHARD%&& set ABUNDANT_FULL_DEVICE=cpu&& set MPLBACKEND=Agg&& %PYTHON_CMD% %RUNNER% %NOTEBOOK% > %LOG_DIR%\shard_%%S.log 2>&1 && (echo success> %LOG_DIR%\shard_%%S.done) || (echo failed> %LOG_DIR%\shard_%%S.failed)"
)

:poll
set /a DONE_COUNT=0
set /a FAILED_COUNT=0
for %%S in (0 1 2 3 4 5 6 7) do (
    if exist "%LOG_DIR%\shard_%%S.done" set /a DONE_COUNT+=1
    if exist "%LOG_DIR%\shard_%%S.failed" set /a FAILED_COUNT+=1
)

echo [%date% %time%] completed=!DONE_COUNT!/%NUM_SHARDS% failed=!FAILED_COUNT!/%NUM_SHARDS%
if !FAILED_COUNT! GTR 0 goto failed
if !DONE_COUNT! EQU %NUM_SHARDS% goto success

timeout /t %POLL_SECONDS% /nobreak >nul
goto poll

:success
echo.
echo All abundant-data shards completed successfully.
echo Open the notebook normally once afterward to validate/merge combined_summary.csv.
popd
exit /b 0

:failed
echo.
echo One or more abundant-data shards failed. Check %LOG_DIR%\shard_*.log.
echo Re-running this launcher is safe: completed per-environment caches are reused.
popd
exit /b 1
