@echo off
setlocal EnableExtensions
set PYTHON_CMD=%~1
if "%PYTHON_CMD%"=="" set PYTHON_CMD=python
set MPLBACKEND=Agg
%PYTHON_CMD% run_ipynb_cells.py analyze_2026_09_30_final_unregularized_results_to_paper.ipynb
exit /b %ERRORLEVEL%
