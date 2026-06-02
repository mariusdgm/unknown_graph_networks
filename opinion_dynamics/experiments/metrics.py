"""Metrics and data-need helpers for experiment notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx


def graph_sanity(A):
    A = np.asarray(A, dtype=float)
    diag_max = float(np.max(np.abs(np.diag(A))))
    rs = A.sum(axis=1)
    asym = float(np.linalg.norm(A - A.T) / (np.linalg.norm(A) + 1e-12))
    edges = int(np.sum(A > 1e-12))
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    sccs = list(nx.strongly_connected_components(G))
    sink_sizes = []
    for S in sccs:
        is_sink = True
        for u in S:
            for v in G.successors(u):
                if v not in S:
                    is_sink = False
                    break
            if not is_sink:
                break
        if is_sink:
            sink_sizes.append(len(S))
    return dict(
        diag_max=diag_max,
        row_sum_min=float(rs.min()),
        row_sum_mean=float(rs.mean()),
        row_sum_max=float(rs.max()),
        asym=asym,
        edges=edges,
        sink_sizes=sink_sizes,
        has_singleton_sink=any(sz == 1 for sz in sink_sizes),
    )


# =========================================================
# Validation learning-curve / data-need helpers
# =========================================================

_VALIDATION_METRICS_DEFAULT = [
    "mean_gap_to_oracle_end",
    "mean_gain_vs_uniform_end",
    "mean_gain_vs_noc_end",
    "A_MAE_final",
    "v_L1_final",
]



def add_trial_count_column(df_in: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with human-readable 1-based trial counts."""
    out = df_in.copy()
    if "trial_count" not in out.columns:
        out["trial_count"] = out["repeat"].astype(int) + 1
    return out


def aggregate_learning_curve(
    df_in: pd.DataFrame,
    *,
    metrics: list[str] | None = None,
    group_cols=("dynamics", "trial_count"),
) -> pd.DataFrame:
    """
    Aggregate per-seed/per-repeat validation logs into mean/std/sem curves.
    """
    dfc = add_trial_count_column(df_in)
    if metrics is None:
        metrics = [m for m in _VALIDATION_METRICS_DEFAULT if m in dfc.columns]
    metrics = [m for m in metrics if m in dfc.columns]
    if not metrics:
        raise ValueError("No requested metrics are present in the dataframe.")

    agg = (
        dfc.groupby(list(group_cols))[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Flatten MultiIndex columns.
    flat_cols = []
    for c in agg.columns:
        if isinstance(c, tuple):
            flat_cols.append(c[0] if c[1] == "" else f"{c[0]}_{c[1]}")
        else:
            flat_cols.append(c)
    agg.columns = flat_cols

    for m in metrics:
        count_col = f"{m}_count"
        std_col = f"{m}_std"
        if count_col in agg.columns and std_col in agg.columns:
            agg[f"{m}_sem"] = agg[std_col] / np.sqrt(np.maximum(agg[count_col], 1))

    return agg


def _criterion_mask(values, op: str, threshold: float):
    values = np.asarray(values, dtype=float)
    if op == "<=":
        return values <= threshold
    if op == "<":
        return values < threshold
    if op == ">=":
        return values >= threshold
    if op == ">":
        return values > threshold
    if op in ("==", "="):
        return values == threshold
    raise ValueError(f"Unsupported criterion operator: {op!r}")


def estimate_trials_needed(
    df_in: pd.DataFrame,
    criteria: dict[str, tuple[str, float]],
    *,
    group_cols=("dynamics",),
    seed_col="seed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate first trial count satisfying all criteria.

    Returns:
      per_seed: first satisfying trial count for each seed/dynamics.
      summary:  mean/median/p90/count summary across seeds.
    """
    dfc = add_trial_count_column(df_in)
    missing = [m for m in criteria if m not in dfc.columns]
    if missing:
        raise ValueError(f"Criteria metrics missing from dataframe: {missing}")

    rows = []
    gb_cols = list(group_cols) + [seed_col]
    for keys, sub in dfc.sort_values("trial_count").groupby(gb_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(gb_cols, keys))

        ok = np.ones(len(sub), dtype=bool)
        for metric, (op, threshold) in criteria.items():
            ok &= _criterion_mask(sub[metric].to_numpy(dtype=float), op, float(threshold))

        if ok.any():
            first = sub.loc[ok].iloc[0]
            key_dict.update(
                trials_needed=int(first["trial_count"]),
                train_pairs_needed=int(first["train_pairs_total"]),
                reached=True,
            )
            for metric in criteria:
                key_dict[f"{metric}_at_hit"] = float(first[metric])
        else:
            last = sub.iloc[-1]
            key_dict.update(
                trials_needed=np.nan,
                train_pairs_needed=np.nan,
                reached=False,
            )
            for metric in criteria:
                key_dict[f"{metric}_at_last"] = float(last[metric])

        rows.append(key_dict)

    per_seed = pd.DataFrame(rows)

    summary_rows = []
    for keys, sub in per_seed.groupby(list(group_cols)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        hit = sub[sub["reached"]].copy()
        row["seeds_total"] = int(len(sub))
        row["seeds_reached"] = int(hit.shape[0])
        row["hit_rate"] = float(hit.shape[0] / max(len(sub), 1))
        if len(hit):
            row["trials_needed_mean"] = float(hit["trials_needed"].mean())
            row["trials_needed_median"] = float(hit["trials_needed"].median())
            row["trials_needed_p90"] = float(hit["trials_needed"].quantile(0.90))
            row["pairs_needed_median"] = float(hit["train_pairs_needed"].median())
        else:
            row["trials_needed_mean"] = np.nan
            row["trials_needed_median"] = np.nan
            row["trials_needed_p90"] = np.nan
            row["pairs_needed_median"] = np.nan
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    return per_seed, summary


