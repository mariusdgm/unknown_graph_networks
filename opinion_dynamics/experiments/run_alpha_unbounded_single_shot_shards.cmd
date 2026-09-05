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
set "NOTEBOOK=experiment_2026_09_05_alpha_unbounded_broad_all_dynamics_multitopology_multiseed.ipynb"
set "RUNNER=run_ipynb_cells.py"
set "LOG_DIR=broad_all_dynamics_multitopology_logs_alpha_unbounded"

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

echo Starting eight broad-experiment shards...
echo.
echo Design:
echo   150 unstructured environments across all six dynamics
echo   45 structured mechanism environments
echo   195 distinct environments total
echo   3 learned seeds per environment
echo   585 learned single-shot runs
echo   no explicit random exploration
echo.
echo Python command: %PYTHON_CMD%
echo Threads per shard: %THREADS_PER_SHARD%
echo Poll interval: %POLL_SECONDS% seconds
echo.

for %%S in (0 1 2 3 4 5 6 7) do (
    start "broad-full-shard-%%S" /b cmd /v:on /c ^
        "set BROAD_FULL_SHARD_ID=%%S&& set BROAD_FULL_TORCH_THREADS=%THREADS_PER_SHARD%&& set BROAD_FULL_DEVICE=cpu&& set MPLBACKEND=Agg&& %PYTHON_CMD% %RUNNER% %NOTEBOOK% > %LOG_DIR%\shard_%%S.log 2>&1 && (echo success> %LOG_DIR%\shard_%%S.done) || (echo failed> %LOG_DIR%\shard_%%S.failed)"
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

:failed
echo.
echo At least one shard failed.
echo Inspect "%LOG_DIR%\shard_X.log".
echo Completed cache entries are persistent; rerunning resumes the experiment.
popd
exit /b 1

:success
echo.
echo All eight shards completed successfully.
echo.
echo Next:
echo   1. Open experiment_2026_09_05_alpha_unbounded_broad_all_dynamics_multitopology_multiseed.ipynb
echo      normally and Run All once to verify zero missing cache entries.
echo   2. Open analyze_2026_08_17_broad_all_dynamics_multitopology.ipynb
echo      and Run All.
echo   3. Change PLOT_STYLE between "paper_bw" and "explore" whenever needed.
echo      This never retrains the models.
echo.
popd
exit /b 0
