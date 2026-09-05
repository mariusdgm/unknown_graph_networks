from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DYNAMICS_MAP = {
    "laplacian": "Linear",
    "linear": "Linear",
    "coca": "COCA",
    "hegselmannkrause": "HK",
    "hk": "HK",
    "friedkinjohnsen": "FJ",
    "fj": "FJ",
    "nonlinearinfluence": "Nonlinear",
    "nonlinear": "Nonlinear",
    "repulsion": "Repulsion",
}

SCENARIO_MAP = {
    "unstructured_random": "generic",
    "generic": "generic",
    "structured_mechanism": "structured",
    "structured": "structured",
}


def choose_value_column(df: pd.DataFrame) -> str:
    for col in ["mean_end", "final_mean", "learned_final_mean"]:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not find a performance column. Expected one of "
        "mean_end, final_mean, learned_final_mean."
    )


def normalize_policy(value: str) -> str:
    s = str(value).lower()
    if "learned" in s:
        return "learned"
    if "true_graph" in s or "true graph" in s:
        return "true_graph"
    if "uniform" in s:
        return "uniform"
    if "no_control" in s or "no control" in s:
        return "no_control"
    return s


def summarize_new(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    value_col = choose_value_column(df)

    required = {"scenario_class", "dynamics", "policy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    work = df.copy()
    work["benchmark"] = work["scenario_class"].map(
        lambda x: SCENARIO_MAP.get(str(x).lower(), str(x))
    )
    work["dynamics_label"] = work["dynamics"].map(
        lambda x: DYNAMICS_MAP.get(str(x).lower(), str(x))
    )
    work["policy_norm"] = work["policy"].map(normalize_policy)

    return (
        work.groupby(
            ["benchmark", "dynamics_label", "policy_norm"],
            as_index=False,
        )[value_col]
        .mean()
        .rename(columns={value_col: "new_mean"})
    )


def old_rows(reference: dict, regime: str) -> pd.DataFrame:
    rows = []
    for benchmark, dyns in reference[regime].items():
        for dynamics, vals in dyns.items():
            for policy in ["no_control", "uniform", "learned", "true_graph"]:
                rows.append(
                    {
                        "benchmark": benchmark,
                        "dynamics_label": dynamics,
                        "policy_norm": policy,
                        "old_mean": vals[policy],
                    }
                )
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("regime", choices=["single_shot", "abundant"])
    p.add_argument("combined_summary", type=Path)
    p.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).with_name("previous_alpha_bounded_reference_scores.json"),
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    new = summarize_new(args.combined_summary)
    old = old_rows(reference, args.regime)

    merged = old.merge(
        new,
        on=["benchmark", "dynamics_label", "policy_norm"],
        how="left",
        validate="one_to_one",
    )
    merged["new_minus_old"] = merged["new_mean"] - merged["old_mean"]

    # Add the most useful control-effect comparison.
    pivot = merged.pivot_table(
        index=["benchmark", "dynamics_label"],
        columns="policy_norm",
        values=["old_mean", "new_mean"],
        aggfunc="first",
    )
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    pivot["old_learned_minus_uniform"] = (
        pivot["old_mean_learned"] - pivot["old_mean_uniform"]
    )
    pivot["new_learned_minus_uniform"] = (
        pivot["new_mean_learned"] - pivot["new_mean_uniform"]
    )
    pivot["effect_change"] = (
        pivot["new_learned_minus_uniform"]
        - pivot["old_learned_minus_uniform"]
    )

    cols = [
        "benchmark",
        "dynamics_label",
        "old_mean_learned",
        "new_mean_learned",
        "old_learned_minus_uniform",
        "new_learned_minus_uniform",
        "effect_change",
    ]
    print(pivot[cols].round(5).to_string(index=False))

    out = args.out or args.combined_summary.with_name(
        f"alpha_unbounded_vs_bounded_{args.regime}.csv"
    )
    pivot[cols].to_csv(out, index=False)
    print("\nWrote:", out)


if __name__ == "__main__":
    main()
