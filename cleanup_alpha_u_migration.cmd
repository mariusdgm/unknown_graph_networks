@echo off
setlocal

echo Removing temporary alpha-ablation migration files...

for %%F in (
    patch_alpha_unbounded_experiments.py
    patch_alpha_unbounded_experiments_v2.py
    patch_alpha_unbounded_experiments_v3.py
    fix_alpha_cache_writer.py
) do (
    if exist "%%F" del /Q "%%F"
)

pushd opinion_dynamics\experiments

for %%F in (
    compare_alpha_ablation.py
    previous_alpha_bounded_reference_scores.json
    SETUP_AND_RUN.txt
    experiment_2026_09_05_alpha_unbounded_broad_all_dynamics_multitopology_multiseed.ipynb
    experiment_2026_09_05_alpha_unbounded_abundant_all_dynamics_multitopology_multiseed.ipynb
    run_alpha_unbounded_single_shot_shards.cmd
    run_alpha_unbounded_abundant_shards.cmd
) do (
    if exist "%%F" del /Q "%%F"
)

if exist "logs_alpha_u_ss" rmdir /S /Q "logs_alpha_u_ss"
if exist "logs_alpha_u_ab" rmdir /S /Q "logs_alpha_u_ab"
if exist "broad_all_dynamics_multitopology_logs_alpha_unbounded" rmdir /S /Q "broad_all_dynamics_multitopology_logs_alpha_unbounded"
if exist "abundant_all_dynamics_multitopology_logs_alpha_unbounded" rmdir /S /Q "abundant_all_dynamics_multitopology_logs_alpha_unbounded"

if exist "results\_trial_cache\alpha_unbounded_broad_all_dynamics_multitopology_multiseed" rmdir /S /Q "results\_trial_cache\alpha_unbounded_broad_all_dynamics_multitopology_multiseed"
if exist "results\_trial_cache\alpha_unbounded_abundant_all_dynamics_multitopology_multiseed" rmdir /S /Q "results\_trial_cache\alpha_unbounded_abundant_all_dynamics_multitopology_multiseed"
if exist "results\experiment_2026_09_05_alpha_unbounded_broad_all_dynamics_multitopology_multiseed" rmdir /S /Q "results\experiment_2026_09_05_alpha_unbounded_broad_all_dynamics_multitopology_multiseed"
if exist "results\experiment_2026_09_05_alpha_unbounded_abundant_all_dynamics_multitopology_multiseed" rmdir /S /Q "results\experiment_2026_09_05_alpha_unbounded_abundant_all_dynamics_multitopology_multiseed"

rem Also clear any partial results from the new short names before a fresh run.
if exist "results\_trial_cache\alpha_u_ss" rmdir /S /Q "results\_trial_cache\alpha_u_ss"
if exist "results\_trial_cache\alpha_u_ab" rmdir /S /Q "results\_trial_cache\alpha_u_ab"
if exist "results\alpha_u_ss" rmdir /S /Q "results\alpha_u_ss"
if exist "results\alpha_u_ab" rmdir /S /Q "results\alpha_u_ab"

popd

echo.
echo Cleanup complete.
echo KEEP opinion_dynamics\identify_nonlinear_unbounded.py
echo KEEP the original August benchmark notebooks and launchers.
endlocal
