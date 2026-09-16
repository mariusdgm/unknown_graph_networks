@echo off
setlocal EnableExtensions
set PYTHON_CMD=%~1
if "%PYTHON_CMD%"=="" set PYTHON_CMD=python
set TORCH_THREADS=%~2
if "%TORCH_THREADS%"=="" set TORCH_THREADS=2
set POLL_SECONDS=%~3
if "%POLL_SECONDS%"=="" set POLL_SECONDS=60

call run_hk_epsilon_robustness_bounded.cmd %PYTHON_CMD% %TORCH_THREADS% %POLL_SECONDS%
if errorlevel 1 exit /b 1

call run_hk_epsilon_robustness_unbounded.cmd %PYTHON_CMD% %TORCH_THREADS% %POLL_SECONDS%
if errorlevel 1 exit /b 1

echo.
echo Both HK epsilon robustness variants completed successfully.
exit /b 0
