@echo off
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0"

set "PYTHON_CMD=%~1"
if "%PYTHON_CMD%"=="" set "PYTHON_CMD=python"

set "THREADS_PER_SHARD=%~2"
if "%THREADS_PER_SHARD%"=="" set "THREADS_PER_SHARD=2"

set "POLL_SECONDS=%~3"
if "%POLL_SECONDS%"=="" set "POLL_SECONDS=60"

set "NUM_SHARDS=6"
set "NOTEBOOK=experiment_2026_08_17_structured_mechanism_full_multiseed.ipynb"
set "RUNNER=run_ipynb_cells.py"
set "LOG_DIR=structured_full_experiment_logs"

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

for %%S in (0 1 2 3 4 5) do (
    del /q "%LOG_DIR%\shard_%%S.done" 2>nul
    del /q "%LOG_DIR%\shard_%%S.failed" 2>nul
    del /q "%LOG_DIR%\shard_%%S.log" 2>nul
)

echo Starting six structured full-experiment shards...
echo Notebook: %NOTEBOOK%
echo Python command: %PYTHON_CMD%
echo Threads per shard: %THREADS_PER_SHARD%
echo Poll interval: %POLL_SECONDS% seconds
echo.
echo Design:
echo   45 distinct environments
echo   3 learning seeds per environment
echo   135 learned online runs total
echo   no explicit random exploration
echo.

for %%S in (0 1 2 3 4 5) do (
    start "structured-full-shard-%%S" /b cmd /v:on /c ^
        "set STRUCTURED_FULL_SHARD_ID=%%S&& set STRUCTURED_FULL_TORCH_THREADS=%THREADS_PER_SHARD%&& set STRUCTURED_FULL_DEVICE=cpu&& set MPLBACKEND=Agg&& %PYTHON_CMD% %RUNNER% %NOTEBOOK% > %LOG_DIR%\shard_%%S.log 2>&1 && (echo success> %LOG_DIR%\shard_%%S.done) || (echo failed> %LOG_DIR%\shard_%%S.failed)"
)

:poll
set /a DONE_COUNT=0
set /a FAILED_COUNT=0

for %%S in (0 1 2 3 4 5) do (
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
echo Inspect:
echo   %LOG_DIR%\shard_0.log
echo   ...
echo   %LOG_DIR%\shard_5.log
echo.
echo Completed caches are persistent; rerunning the launcher resumes them.
popd
exit /b 1

:success
echo.
echo All six shards completed successfully.
echo.
echo Next:
echo   Open %NOTEBOOK% normally in VS Code/Jupyter and Run All.
echo   Make sure STRUCTURED_FULL_SHARD_ID is NOT set in that notebook process.
echo   Analysis mode will validate the complete cache and generate CSVs/plots.
echo.
popd
exit /b 0
