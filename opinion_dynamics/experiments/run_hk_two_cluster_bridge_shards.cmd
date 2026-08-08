@echo off
setlocal EnableExtensions EnableDelayedExpansion

pushd "%~dp0"

set "PYTHON_CMD=%~1"
if "%PYTHON_CMD%"=="" set "PYTHON_CMD=python"

set "THREADS_PER_SHARD=%~2"
if "%THREADS_PER_SHARD%"=="" set "THREADS_PER_SHARD=3"

set "POLL_SECONDS=%~3"
if "%POLL_SECONDS%"=="" set "POLL_SECONDS=60"

set "NOTEBOOK=experiment_2026_08_08_hk_two_cluster_latent_bridge.ipynb"
set "LOG_DIR=hk_two_cluster_bridge_logs"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for %%S in (0 1 2) do (
    del /q "%LOG_DIR%\shard_%%S.done" 2>nul
    del /q "%LOG_DIR%\shard_%%S.failed" 2>nul
    del /q "%LOG_DIR%\shard_%%S.log" 2>nul
)

echo Starting three HK latent-bridge shards...
echo Python command: %PYTHON_CMD%
echo Threads per shard: %THREADS_PER_SHARD%
echo Poll interval: %POLL_SECONDS% seconds
echo.

for %%S in (0 1 2) do (
    start "hk-bridge-shard-%%S" /b cmd /v:on /c ^
        "set HK_CLUSTER_SHARD_ID=%%S&& set HK_CLUSTER_TORCH_THREADS=%THREADS_PER_SHARD%&& set HK_CLUSTER_DEVICE=cpu&& %PYTHON_CMD% -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output executed_hk_two_cluster_bridge_shard_%%S_of_3.ipynb %NOTEBOOK% > %LOG_DIR%\shard_%%S.log 2>&1 && (echo success> %LOG_DIR%\shard_%%S.done) || (echo failed> %LOG_DIR%\shard_%%S.failed)"
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
echo.
echo At least one shard failed.
echo Inspect "%LOG_DIR%" for details.
popd
exit /b 1

:success
echo.
echo All three shards completed successfully.
echo Open the original notebook WITHOUT HK_CLUSTER_SHARD_ID and run it
echo to load the cache, validate all 25 paired trials, and regenerate plots.
popd
exit /b 0
