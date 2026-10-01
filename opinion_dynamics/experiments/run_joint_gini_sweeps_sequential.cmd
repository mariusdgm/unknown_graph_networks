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

call run_joint_gini_single_shot_sweep_parallel.cmd %PYTHON_CMD% %TORCH_THREADS% %POLL_SECONDS% %NUM_SHARDS%
if errorlevel 1 exit /b 1

call run_joint_gini_abundant_sweep_parallel.cmd %PYTHON_CMD% %TORCH_THREADS% %POLL_SECONDS% %NUM_SHARDS%
if errorlevel 1 exit /b 1

echo Both joint Gini sweeps completed.
exit /b 0
