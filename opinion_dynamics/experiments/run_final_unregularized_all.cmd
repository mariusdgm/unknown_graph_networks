@echo off
setlocal EnableExtensions

call run_final_unregularized_single_shot_parallel.cmd %1 %2 %3
if errorlevel 1 exit /b 1

call run_final_unregularized_abundant_parallel.cmd %1 %2 %3
if errorlevel 1 exit /b 1

echo Both final unregularized experiments completed.
echo Next run: analyze_2026_09_30_final_unregularized_results_to_paper.ipynb
exit /b 0
