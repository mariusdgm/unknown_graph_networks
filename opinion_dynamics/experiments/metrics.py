import numpy as np
import pandas as pd
import networkx as nx
import torch
try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = print


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


def transition_pairs_from_intermediate_list(intermediate_states_list):
    xs, ys = [], []
    for inter in intermediate_states_list:
        if inter is None:
            continue
        arr = np.asarray(inter, dtype=float)
        if arr.shape[0] < 2:
            continue
        xs.append(arr[:-1])
        ys.append(arr[1:])
    if not xs:
        raise ValueError("No valid intermediate transition pairs found.")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def model_prediction_mae_on_arrays(model, X, Y, *, device="cpu"):
    model = model.to(device).eval()
    X_t = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
    Y_t = torch.tensor(np.asarray(Y), dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model.predict_next(X_t)
        return float((pred - Y_t).abs().mean().item())


def model_prediction_mae_on_intermediates(model, intermediate_states_list, *, device="cpu"):
    X, Y = transition_pairs_from_intermediate_list(intermediate_states_list)
    return model_prediction_mae_on_arrays(model, X, Y, device=device)


def identity_prediction_mae_on_arrays(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    return float(np.mean(np.abs(Y - X)))


def identity_prediction_mae_on_intermediates(intermediate_states_list):
    X, Y = transition_pairs_from_intermediate_list(intermediate_states_list)
    return identity_prediction_mae_on_arrays(X, Y)


def safe_ratio(numer, denom, *, eps=1e-12):
    denom = float(denom)
    if abs(denom) <= eps:
        return float("nan")
    return float(numer) / denom


# Backwards-compatible private name used by current notebooks.
_safe_ratio = safe_ratio


def support_jaccard(u, v, *, eps=1e-12):
    su = set(np.flatnonzero(np.abs(np.asarray(u, dtype=float)) > eps).tolist())
    sv = set(np.flatnonzero(np.abs(np.asarray(v, dtype=float)) > eps).tolist())
    if not su and not sv:
        return 1.0
    return len(su & sv) / max(len(su | sv), 1)


_support_jaccard = support_jaccard


def topk_overlap_from_actions(u, v, *, eps=1e-12):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    ku = int(np.sum(np.abs(u) > eps))
    kv = int(np.sum(np.abs(v) > eps))
    k = max(ku, kv, 1)
    top_u = set(np.argsort(-np.abs(u))[:k].tolist())
    top_v = set(np.argsort(-np.abs(v))[:k].tolist())
    return len(top_u & top_v) / k


def action_alignment_metrics(actions_learn, actions_oracle, *, eps=1e-12):
    A = np.asarray(actions_learn, dtype=float)
    B = np.asarray(actions_oracle, dtype=float)
    T = min(A.shape[0], B.shape[0])
    if T == 0:
        return dict(
            action_l1_avg=float("nan"),
            action_linf_avg=float("nan"),
            action_support_jaccard_avg=float("nan"),
            action_topk_overlap_avg=float("nan"),
            action_top1_match_avg=float("nan"),
        )
    A = A[:T]
    B = B[:T]
    l1 = np.sum(np.abs(A - B), axis=1)
    linf = np.max(np.abs(A - B), axis=1)
    jac = np.array([support_jaccard(a, b, eps=eps) for a, b in zip(A, B)], dtype=float)
    topk = np.array([topk_overlap_from_actions(a, b, eps=eps) for a, b in zip(A, B)], dtype=float)
    top1 = np.array([int(np.argmax(np.abs(a)) == np.argmax(np.abs(b))) for a, b in zip(A, B)], dtype=float)
    return dict(
        action_l1_avg=float(l1.mean()),
        action_l1_first=float(l1[0]),
        action_linf_avg=float(linf.mean()),
        action_support_jaccard_avg=float(jac.mean()),
        action_support_jaccard_first=float(jac[0]),
        action_topk_overlap_avg=float(topk.mean()),
        action_topk_overlap_first=float(topk[0]),
        action_top1_match_avg=float(top1.mean()),
        action_top1_match_first=float(top1[0]),
    )


def effective_centrality_alignment_metrics(effective_centralities, v_true):
    v_true = np.asarray(v_true, dtype=float).reshape(-1)
    if not effective_centralities:
        return dict(v_eff_L1_avg=float("nan"), v_eff_L1_first=float("nan"), v_eff_topk_overlap_first=float("nan"))
    vals = [np.asarray(v, dtype=float).reshape(-1) for v in effective_centralities]
    l1 = np.array([np.sum(np.abs(v - v_true)) for v in vals], dtype=float)
    k = max(1, int(np.ceil(len(v_true) / 3)))
    true_top = set(np.argsort(-v_true)[:k].tolist())
    overlaps = []
    for v in vals:
        top = set(np.argsort(-v)[:k].tolist())
        overlaps.append(len(top & true_top) / k)
    overlaps = np.asarray(overlaps, dtype=float)
    return dict(
        v_eff_L1_avg=float(l1.mean()),
        v_eff_L1_first=float(l1[0]),
        v_eff_L1_last=float(l1[-1]),
        v_eff_topk_overlap_avg=float(overlaps.mean()),
        v_eff_topk_overlap_first=float(overlaps[0]),
        v_eff_topk_overlap_last=float(overlaps[-1]),
    )


def add_trial_count(df_in):
    df = df_in.copy()
    if "trial_count" not in df.columns:
        df["trial_count"] = df["repeat"].astype(int) + 1
    return df


def amount_data_text_diagnostics(df_in):
    df = add_trial_count(df_in)
    print("=== Amount-of-data rows ===")
    print("rows:", len(df))
    print("columns:", list(df.columns))

    core = [
        "val_model_over_identity",
        "val_improvement_over_identity",
        "one_step_val_mae",
        "val_identity_mae",
        "mean_gap_to_oracle_end",
        "mean_gain_vs_uniform_end",
        "action_l1_avg",
        "action_support_jaccard_avg",
        "v_eff_L1_avg",
        "A_MAE_final",
        "v_L1_final",
        "time_fit_inner",
        "fit_steps_run",
        "fit_mae_after",
        "fit_A_l1_change",
        "fit_alpha_l2_change",
    ]
    available = [c for c in core if c in df.columns]
    if available:
        print("\n=== Mean by dynamics at selected trial counts ===")
        selected = df[df["trial_count"].isin([1, 2, 3, 5, 10, 20, 50, 100])]
        if len(selected):
            display(
                selected.groupby(["dynamics", "trial_count"])[available]
                .mean()
                .round(4)
            )

        print("\n=== First vs last checkpoint by dynamics ===")
        for dyn, sub in df.sort_values(["seed", "trial_count"]).groupby("dynamics"):
            first = sub[sub["trial_count"] == sub["trial_count"].min()]
            last = sub[sub["trial_count"] == sub["trial_count"].max()]
            print(f"\n{dyn}")
            for col in available:
                a = first[col].mean()
                b = last[col].mean()
                print(f"  {col}: {a:.5g} -> {b:.5g} | delta={b-a:+.5g}")

    if "fit_stop_reason" in df.columns:
        print("\n=== Fit stop reasons ===")
        print(pd.crosstab([df["dynamics"], df["trial_count"]], df["fit_stop_reason"]).tail(20))

    print(
        "\nInterpretation guide:\n"
        "- If val_model_over_identity is below 1, the model beats x_next=x.\n"
        "- If it is already low at trial 1, the one-step dynamics are learned very early or the validation task is easy.\n"
        "- If action metrics plateau while prediction improves, the top-k ranking policy has saturated.\n"
        "- If oracle barely beats flat/deviation-only baselines, adjacency knowledge is not strongly needed by this environment setting.\n"
        "- If fit_steps_run is tiny or fit_stop_reason=mae_stop at trial 1, tighten mae_stop/check frequency for the data-need experiment."
    )

# =============================================================================
# Source-of-truth notebook learning-curve and policy diagnostics
# =============================================================================
import torch

def model_prediction_mae_on_arrays(model, X, Y, *, device: str = "cpu") -> float:
    """
    One-step prediction MAE on explicit transition arrays.

    This is a dynamics/validation metric, not a graph-recovery metric. It is useful
    when A itself is not identifiable, because it asks whether the learned model
    predicts the next state on held-out states.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.size == 0 or Y.size == 0:
        return float("nan")
    if X.ndim != 2 or Y.ndim != 2 or X.shape != Y.shape:
        raise ValueError(f"X and Y must be matching 2D arrays, got {X.shape} and {Y.shape}")

    mdl = model.to(device).eval()
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32, device=device)
        yt = torch.tensor(Y, dtype=torch.float32, device=device)
        yp = mdl.predict_next(xt)
        return float((yp - yt).abs().mean().item())


def transition_pairs_from_intermediate_list(intermediate_states_list) -> tuple[np.ndarray, np.ndarray]:
    """Collect one-step transition pairs from a list of env info['intermediate_states'] arrays."""
    xs, ys = [], []
    for inter in intermediate_states_list:
        if inter is None:
            continue
        inter = np.asarray(inter, dtype=float)
        if inter.ndim != 2 or inter.shape[0] < 2:
            continue
        x_i, y_i = pairs_from_intermediate(inter)
        xs.append(np.asarray(x_i, dtype=float))
        ys.append(np.asarray(y_i, dtype=float))
    if not xs:
        return np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=float)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def model_prediction_mae_on_intermediates(model, intermediate_states_list, *, device: str = "cpu") -> float:
    """One-step prediction MAE on transition pairs extracted from intermediate trajectories."""
    X, Y = transition_pairs_from_intermediate_list(intermediate_states_list)
    if X.size == 0:
        return float("nan")
    return model_prediction_mae_on_arrays(model, X, Y, device=device)


def _support_jaccard(u: np.ndarray, v: np.ndarray, *, eps: float = 1e-12) -> float:
    su = np.asarray(u, dtype=float) > eps
    sv = np.asarray(v, dtype=float) > eps
    union = su | sv
    if not union.any():
        return 1.0
    return float((su & sv).sum() / union.sum())


def action_alignment_metrics(actions_learn, actions_oracle, *, eps: float = 1e-12) -> dict[str, float]:
    """
    Policy/action agreement between learned and oracle over the same rollout.

    This is often more diagnostic than raw A error: if the learned model picks the
    same controlled nodes and similar control magnitudes, it can perform well even
    when A_hat is not equal to A_true.
    """
    ul = np.asarray(actions_learn, dtype=float)
    uo = np.asarray(actions_oracle, dtype=float)
    T = min(ul.shape[0], uo.shape[0]) if ul.ndim == 2 and uo.ndim == 2 else 0
    if T == 0:
        return dict(
            action_l1_avg=float("nan"),
            action_l1_end=float("nan"),
            action_linf_max=float("nan"),
            action_cosine_avg=float("nan"),
            action_support_jaccard_avg=float("nan"),
            action_top1_match_avg=float("nan"),
            action_budget_gap_avg=float("nan"),
        )

    ul = ul[:T]
    uo = uo[:T]
    diff = np.abs(ul - uo)
    l1 = diff.sum(axis=1)
    linf = diff.max(axis=1)

    cos_vals = []
    jac_vals = []
    top1_vals = []
    for a, b in zip(ul, uo):
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= eps and nb <= eps:
            cos_vals.append(1.0)
        elif na <= eps or nb <= eps:
            cos_vals.append(0.0)
        else:
            cos_vals.append(float(np.dot(a, b) / (na * nb)))
        jac_vals.append(_support_jaccard(a, b, eps=eps))
        top1_vals.append(float(int(np.argmax(a) == np.argmax(b))))

    budget_gap = np.abs(ul.sum(axis=1) - uo.sum(axis=1))
    return dict(
        action_l1_avg=float(np.mean(l1)),
        action_l1_end=float(l1[-1]),
        action_linf_max=float(np.max(linf)),
        action_cosine_avg=float(np.mean(cos_vals)),
        action_support_jaccard_avg=float(np.mean(jac_vals)),
        action_top1_match_avg=float(np.mean(top1_vals)),
        action_budget_gap_avg=float(np.mean(budget_gap)),
    )


def effective_centrality_alignment_metrics(effective_centralities, v_true) -> dict[str, float]:
    """
    Compare the learned state-dependent centrality v_eff(x_k) to the true-v oracle.

    This is more relevant than v_hat_final for state-dependent models because the
    controller actually uses v_eff(x_k), not raw centrality of the static A_hat.
    """
    v_true = np.asarray(v_true, dtype=float).reshape(-1)
    vals = []
    for v in effective_centralities:
        v = np.asarray(v, dtype=float).reshape(-1)
        if v.shape == v_true.shape:
            vals.append(float(np.sum(np.abs(v - v_true))))
    if not vals:
        return dict(v_eff_L1_avg=float("nan"), v_eff_L1_campaign0=float("nan"), v_eff_L1_end=float("nan"))
    return dict(
        v_eff_L1_avg=float(np.mean(vals)),
        v_eff_L1_campaign0=float(vals[0]),
        v_eff_L1_end=float(vals[-1]),
    )


def _safe_fraction_of_oracle(learn: float, baseline: float, oracle: float, *, eps: float = 1e-12) -> float:
    """
    Fraction of oracle-over-baseline improvement captured by learned.

    1.0 means learned matches oracle, 0.0 means it matches the baseline, >1.0 means
    it beats the oracle under this scalar objective. Returns NaN if oracle and
    baseline are indistinguishable.
    """
    denom = float(oracle) - float(baseline)
    if abs(denom) <= eps:
        return float("nan")
    return float((float(learn) - float(baseline)) / denom)

# Backwards-compatible public aliases.
support_jaccard = _support_jaccard
safe_ratio = _safe_fraction_of_oracle

# =========================================================
# Validation learning-curve / data-need helpers
# =========================================================

# Policy-centered metrics should drive the "how much data is enough?" question.
# Raw graph-recovery metrics are still available, but for nonlinear/state-dependent
# dynamics they can be non-identifiable and misleading.
POLICY_LEARNING_METRICS = [
    "mean_gap_to_oracle_end",
    "mean_auc_gap_to_oracle",
    "mean_gain_vs_uniform_end",
    "mean_gain_vs_noc_end",
    "policy_frac_oracle_vs_uniform_end",
    "policy_frac_oracle_vs_uniform_auc",
    "one_step_val_mae",
    "action_l1_avg",
    "action_support_jaccard_avg",
    "v_eff_L1_avg",
]

GRAPH_DIAGNOSTIC_METRICS = [
    "A_MAE_final",
    "A_Fro_final",
    "v_L1_final",
]

_VALIDATION_METRICS_DEFAULT = POLICY_LEARNING_METRICS



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

# Alias used by earlier refactors.
add_trial_count = add_trial_count_column
