@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "PYTHON_CMD=%~1"
if "%PYTHON_CMD%"=="" set "PYTHON_CMD=python"

set "TORCH_THREADS=%~2"
if "%TORCH_THREADS%"=="" set "TORCH_THREADS=2"

set "POLL_SECONDS=%~3"
if "%POLL_SECONDS%"=="" set "POLL_SECONDS=60"

set "NOTEBOOK=experiment_2026_09_06_alpha_u_single_shot.ipynb"
set "LOG_DIR=logs_alpha_u_ss"
set "DONE_DIR=%LOG_DIR%\_launcher_done"

if exist "%LOG_DIR%" rmdir /S /Q "%LOG_DIR%"
mkdir "%DONE_DIR%"

for /F %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set "START_EPOCH=%%T"

call :STATUS "Starting alpha-u single-shot benchmark: 8 shards, %TORCH_THREADS% Torch threads/shard"

for /L %%S in (0,1,7) do (
    call :STATUS "Launching shard %%S"
    start "" /B cmd /C "set BROAD_FULL_SHARD_ID=%%S&& set BROAD_FULL_TORCH_THREADS=%TORCH_THREADS%&& set BROAD_FULL_DEVICE=cpu&& set MPLBACKEND=Agg&& %PYTHON_CMD% run_ipynb_cells.py %NOTEBOOK% > %LOG_DIR%\shard_%%S.log 2>&1 && (> %DONE_DIR%\shard_%%S.exit echo 0) || (> %DONE_DIR%\shard_%%S.exit echo 1)"
)

:WAIT_LOOP
set /A DONE=0
set /A FAILED=0

for /L %%S in (0,1,7) do (
    if exist "%DONE_DIR%\shard_%%S.exit" (
        set /A DONE+=1
        set "EXIT_CODE="
        set /P EXIT_CODE=<"%DONE_DIR%\shard_%%S.exit"
        if not "!EXIT_CODE!"=="0" set /A FAILED+=1
    )
)

call :STATUS "Completed shards: !DONE!/8; failed: !FAILED!/8"

if !DONE! LSS 8 (
    timeout /T %POLL_SECONDS% /NOBREAK >NUL
    goto WAIT_LOOP
)

if !FAILED! GTR 0 (
    call :STATUS "One or more shards failed. Check %LOG_DIR%\shard_*.log"
    exit /B 1
)

call :STATUS "All worker shards completed; starting final validation/merge pass"
%PYTHON_CMD% run_ipynb_cells.py %NOTEBOOK%
if errorlevel 1 (
    call :STATUS "Final validation/merge pass FAILED"
    exit /B 1
)

call :STATUS "alpha-u single-shot benchmark complete"
exit /B 0

:STATUS
for /F %%T in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set "NOW_EPOCH=%%T"
set /A ELAPSED_SEC=NOW_EPOCH-START_EPOCH
for /F "delims=" %%E in ('powershell -NoProfile -Command "([TimeSpan]::FromSeconds(!ELAPSED_SEC!)).ToString()"') do set "ELAPSED=%%E"
for /F "delims=" %%D in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'"') do set "STAMP=%%D"
echo [!STAMP!] %~1 ^| elapsed !ELAPSED!
exit /B 0
