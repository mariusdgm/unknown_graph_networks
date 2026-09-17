@echo off
setlocal

set "PYTHON_EXE=%~1"
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

echo ============================================================
echo FINAL PAPER RESULTS - CENTRAL ANALYSIS
echo Python: %PYTHON_EXE%
echo ============================================================
echo.
echo This is analysis only. No model training will be run.
echo.

"%PYTHON_EXE%" run_ipynb_cells.py analyze_final_paper_all_results.ipynb
if errorlevel 1 (
    echo.
    echo ERROR: central paper analysis failed.
    popd
    exit /b 1
)

echo.
echo ============================================================
echo CENTRAL PAPER ANALYSIS COMPLETE
echo.
echo Open:
echo   analyze_final_paper_all_results.ipynb
echo.
echo Result folders:
echo   results\experiment_2026_09_17_broad_all_dynamics_multitopology_multiseed_unbounded_lambda1
echo   results\experiment_2026_09_17_abundant_all_dynamics_multitopology_multiseed_unbounded_lambda1
echo ============================================================

popd
endlocal
exit /b 0
