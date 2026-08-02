@echo off
setlocal EnableExtensions EnableDelayedExpansion

if /I "%~1"=="__worker__" goto worker

set "ROOT=%~dp0"
set "NOTEBOOK_NAME=experiment_2026_08_01_lambda_mix_sweep_all_dynamics_parallel.ipynb"
set "NOTEBOOK=%ROOT%%NOTEBOOK_NAME%"
set "LOG_DIR=%ROOT%lambda_mix_shard_logs"

set "PYTHON_CMD=%~1"
if not defined PYTHON_CMD set "PYTHON_CMD=python"

set "THREADS_PER_SHARD=%~2"
if not defined THREADS_PER_SHARD set "THREADS_PER_SHARD=4"

set "POLL_SECONDS=%~3"
if not defined POLL_SECONDS set "POLL_SECONDS=60"

if not exist "%NOTEBOOK%" (
    echo ERROR: Notebook not found:
    echo %NOTEBOOK%
    exit /b 1
)

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo.
echo Launching 3 lambda-mix shards
echo Notebook: %NOTEBOOK%
echo Python: %PYTHON_CMD%
echo Threads per shard: %THREADS_PER_SHARD%
echo Poll interval: %POLL_SECONDS% seconds
echo Logs: %LOG_DIR%
echo.

for %%S in (0 1 2) do (
    del /q "%LOG_DIR%\lambda_mix_shard_%%S_of_3.done" 2>nul
    del /q "%LOG_DIR%\lambda_mix_shard_%%S_of_3.failed" 2>nul
    del /q "%LOG_DIR%\lambda_mix_shard_%%S_of_3.log" 2>nul
)

for %%S in (0 1 2) do (
    echo Starting shard %%S...
    start "lambda-mix shard %%S" /b cmd /d /c call "%~f0" __worker__ %%S "%PYTHON_CMD%" %THREADS_PER_SHARD%
)

echo.
echo All shards launched. Waiting for completion...
echo.

:wait_loop
set /a DONE_COUNT=0
set /a FAIL_COUNT=0

for %%S in (0 1 2) do (
    if exist "%LOG_DIR%\lambda_mix_shard_%%S_of_3.done" set /a DONE_COUNT+=1
    if exist "%LOG_DIR%\lambda_mix_shard_%%S_of_3.failed" set /a FAIL_COUNT+=1
)

<nul set /p "=Completed: !DONE_COUNT!/3   Failed: !FAIL_COUNT!/3   "
echo %time%

if !FAIL_COUNT! GTR 0 goto failed
if !DONE_COUNT! EQU 3 goto success

timeout /t %POLL_SECONDS% /nobreak >nul
goto wait_loop

:success
echo.
echo All three shards completed successfully.
echo Logs are in:
echo   %LOG_DIR%
echo.
echo Next: run merge_lambda_mix_shards.ipynb
exit /b 0

:failed
echo.
echo At least one shard failed.
echo Inspect:
for %%S in (0 1 2) do (
    if exist "%LOG_DIR%\lambda_mix_shard_%%S_of_3.failed" (
        echo   %LOG_DIR%\lambda_mix_shard_%%S_of_3.log
    )
)
echo.
echo Successful cached trials are preserved.
exit /b 1

:worker
set "SHARD_ID=%~2"
set "PYTHON_CMD=%~3"
set "THREADS_PER_SHARD=%~4"
set "ROOT=%~dp0"
set "NOTEBOOK_NAME=experiment_2026_08_01_lambda_mix_sweep_all_dynamics_parallel.ipynb"
set "OUTPUT_NAME=executed_lambda_mix_shard_%SHARD_ID%_of_3.ipynb"
set "LOG_DIR=%ROOT%lambda_mix_shard_logs"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set "LOG_PATH=%LOG_DIR%\lambda_mix_shard_%SHARD_ID%_of_3.log"
set "DONE_PATH=%LOG_DIR%\lambda_mix_shard_%SHARD_ID%_of_3.done"
set "FAIL_PATH=%LOG_DIR%\lambda_mix_shard_%SHARD_ID%_of_3.failed"

set "LAMBDA_MIX_SHARD_ID=%SHARD_ID%"
set "LAMBDA_MIX_DEVICE=cpu"
set "LAMBDA_MIX_TORCH_THREADS=%THREADS_PER_SHARD%"
set "OMP_NUM_THREADS=%THREADS_PER_SHARD%"
set "MKL_NUM_THREADS=%THREADS_PER_SHARD%"
set "OPENBLAS_NUM_THREADS=%THREADS_PER_SHARD%"
set "NUMEXPR_NUM_THREADS=%THREADS_PER_SHARD%"

echo Shard %SHARD_ID% started at %date% %time% > "%LOG_PATH%"
echo Python command: %PYTHON_CMD% >> "%LOG_PATH%"
echo Threads: %THREADS_PER_SHARD% >> "%LOG_PATH%"
echo. >> "%LOG_PATH%"

pushd "%ROOT%"

call %PYTHON_CMD% -m nbconvert ^
    --to notebook ^
    --execute "%NOTEBOOK_NAME%" ^
    --output "%OUTPUT_NAME%" ^
    --ExecutePreprocessor.timeout=-1 >> "%LOG_PATH%" 2>&1

set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" (
    echo FAILED with exit code %EXIT_CODE% at %date% %time% >> "%LOG_PATH%"
    > "%FAIL_PATH%" echo exit_code=%EXIT_CODE%
    exit /b %EXIT_CODE%
)

echo Completed successfully at %date% %time% >> "%LOG_PATH%"
> "%DONE_PATH%" echo success
exit /b 0
