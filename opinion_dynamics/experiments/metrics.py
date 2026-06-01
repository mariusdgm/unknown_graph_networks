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


# Backwards-compatible name used by earlier plotting helpers.
def add_trial_count_column(df_in):
    return add_trial_count(df_in)


def aggregate_learning_curve(df_in, metrics):
    df = add_trial_count(df_in)
    agg_parts = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        g = df.groupby(["dynamics", "trial_count"])[metric]
        part = g.agg(["mean", "std", "count"]).reset_index()
        part["sem"] = part["std"] / np.sqrt(part["count"].clip(lower=1))
        part = part.rename(columns={
            "mean": f"{metric}_mean",
            "std": f"{metric}_std",
            "count": f"{metric}_count",
            "sem": f"{metric}_sem",
        })
        agg_parts.append(part)
    if not agg_parts:
        return pd.DataFrame(columns=["dynamics", "trial_count"])
    out = agg_parts[0]
    for part in agg_parts[1:]:
        out = out.merge(part, on=["dynamics", "trial_count"], how="outer")
    return out.sort_values(["dynamics", "trial_count"]).reset_index(drop=True)
