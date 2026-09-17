@echo off
setlocal EnableExtensions

set PYTHON_CMD=%~1
if "%PYTHON_CMD%"=="" set PYTHON_CMD=python

set NOTEBOOK=analyze_final_math_simplification_before_after.ipynb
set LOG_DIR=logs_final_comparison

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set MPLBACKEND=Agg

echo Running final before/after comparison...
echo Notebook: %NOTEBOOK%
%PYTHON_CMD% run_ipynb_cells.py %NOTEBOOK% > "%LOG_DIR%\comparison.log" 2>&1
if errorlevel 1 (
  echo Comparison failed. Inspect %LOG_DIR%\comparison.log
  type "%LOG_DIR%\comparison.log"
  exit /b 1
)

type "%LOG_DIR%\comparison.log"
echo Final comparison complete.
exit /b 0
