"""Plotting helpers for the network opinion experiments.

The mean/baseline plot intentionally has no forced y-limits and keeps the
legend outside the axes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from opinion_dynamics.experiments.rollouts import apply_impulse_control


def show_matrix_with_cell_grid(
    A,
    title,
    *,
    figsize=(8, 6),
    vmin=None,
    vmax=None,
    grid_color="#C8C8C8",
    grid_alpha=0.35,
    grid_lw=0.6,
    show_ticks=True,
):
    A = np.asarray(A)
    nrows, ncols = A.shape

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(A, aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color=grid_color, linestyle="-", linewidth=grid_lw, alpha=grid_alpha)
    ax.tick_params(which="minor", bottom=False, left=False)

    if show_ticks:
        ax.set_xlabel("j")
        ax.set_ylabel("i")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    plt.show()


def plot_impulse_node_trajectories(inter_states, inter_times, title, ylim=(0, 1)):
    inter_states = np.asarray(inter_states)
    inter_times = np.asarray(inter_times)

    t = inter_times.reshape(-1)
    x = inter_states.reshape(-1, inter_states.shape[-1])

    plt.figure(figsize=(10, 4))
    for i in range(x.shape[1]):
        plt.plot(t, x[:, i], linewidth=1)
    plt.xlabel("time")
    plt.ylabel("opinion $x_i$")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.ylim(*ylim)
    plt.show()


def concat_intermediate(inter_list, time_list, *, dt=None, campaign_gap=None):
    xs_all, ts_all = [], []
    offset = 0.0

    for xs, ts in zip(inter_list, time_list):
        if xs is None or ts is None:
            continue
        xs = np.asarray(xs, dtype=float)
        ts = np.asarray(ts, dtype=float).reshape(-1)
        ts_shift = ts + offset
        xs_all.append(xs)
        ts_all.append(ts_shift)

        if campaign_gap is not None:
            offset = ts_shift[-1] + float(campaign_gap)
        elif dt is not None:
            offset = ts_shift[-1] + float(dt)
        else:
            offset = ts_shift[-1] + 1.0

    if not xs_all:
        raise RuntimeError("No valid intermediate traces to concatenate.")

    return np.concatenate(xs_all, axis=0), np.concatenate(ts_all, axis=0)


def legend_outside(ax, *, loc="center left", anchor=(1.02, 0.5), ncol=1, frameon=True):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    return ax.legend(
        handles, labels,
        loc=loc,
        bbox_to_anchor=anchor,
        ncol=ncol,
        frameon=frameon,
        borderaxespad=0.0,
    )


_legend_outside = legend_outside


def plot_mean_baseline_comparison(mean_series, title):
    """Mean/baseline comparison with no forced y-limits and an outside legend."""
    mean_series = {k: np.asarray(v, dtype=float) for k, v in mean_series.items()}
    first = next(iter(mean_series.values()))
    x = np.arange(first.shape[0])

    style_cycle = [
        dict(marker="o", linestyle="-", linewidth=2.5, markersize=5.2, zorder=5),
        dict(marker="s", linestyle="--", linewidth=2.1, markersize=5.0, zorder=4),
        dict(marker="^", linestyle="-.", linewidth=2.1, markersize=5.0, zorder=3),
        dict(marker="D", linestyle=":", linewidth=2.3, markersize=4.8, zorder=2),
        dict(marker="x", linestyle=(0, (3, 1, 1, 1)), linewidth=2.0, markersize=5.4, zorder=1),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    for (label, y), style in zip(mean_series.items(), style_cycle):
        ax.plot(x, y, label=label, **style)

    ax.set_xlabel("Campaign boundary k")
    ax.set_ylabel("mean opinion")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.margins(x=0.03)
    legend_outside(
        ax,
        loc="upper center",
        anchor=(0.5, -0.28),
        ncol=min(3, max(1, len(mean_series))),
        frameon=True,
    )
    fig.subplots_adjust(bottom=0.34)
    plt.show()
    return fig, ax

def build_augmented_campaign_trajectory(
    states_boundary: np.ndarray,
    actions: np.ndarray,
    intermediate_states_list,
    *,
    desired_opinion: float,
    dt: float,
    eps_t: float = 1e-6,
):
    """
    Build a fine-grained trajectory that explicitly includes, for each campaign:
      1) boundary state x_k
      2) immediate post-impulse state x_k^ctrl
      3) propagation substeps returned in intermediate_states_list

    This makes jumps caused by the campaign impulse explicit and avoids the
    misleading appearance that different policies "start" from different points.
    """
    states_boundary = np.asarray(states_boundary, dtype=float)
    actions = np.asarray(actions, dtype=float)

    if states_boundary.ndim != 2:
        raise ValueError("states_boundary must have shape (K+1, N).")

    X_parts = [states_boundary[0][None, :].copy()]
    T_parts = [np.array([0.0], dtype=float)]

    current_time = 0.0
    num_campaigns = min(len(actions), len(intermediate_states_list), max(states_boundary.shape[0] - 1, 0))

    for k in range(num_campaigns):
        x_pre = np.asarray(states_boundary[k], dtype=float)
        u_k = np.asarray(actions[k], dtype=float)
        x_ctrl = apply_impulse_control(x_pre, u_k, desired_opinion=float(desired_opinion))

        # Explicit immediate post-impulse point.
        X_parts.append(x_ctrl[None, :].copy())
        T_parts.append(np.array([current_time + eps_t], dtype=float))

        inter = intermediate_states_list[k]
        if inter is not None:
            inter_arr = np.asarray(inter, dtype=float)

            # If the env already included x_ctrl as the first intermediate state,
            # drop that duplicate because we just inserted it explicitly.
            if inter_arr.ndim == 2 and inter_arr.shape[0] > 0:
                if np.max(np.abs(inter_arr[0] - x_ctrl)) <= 1e-8:
                    inter_arr = inter_arr[1:]

            if inter_arr.ndim == 2 and inter_arr.shape[0] > 0:
                t_inter = current_time + eps_t + dt * np.arange(1, inter_arr.shape[0] + 1, dtype=float)
                X_parts.append(inter_arr)
                T_parts.append(t_inter)

        # Advance to the next campaign boundary.
        current_time += float(dt)

    X = np.vstack(X_parts)
    T = np.concatenate(T_parts)
    return X, T


def plot_amount_metric(df, metric, *, ylabel=None, title=None, by="dynamics", x="trial_count"):
    if metric not in df.columns:
        print(f"Skipping {metric}: not in DataFrame")
        return None
    dfx = df.copy()
    if x not in dfx.columns:
        if x == "trial_count" and "repeat" in dfx.columns:
            dfx[x] = dfx["repeat"].astype(int) + 1
        else:
            raise KeyError(f"x={x!r} not found in DataFrame")
    g = dfx.groupby([by, x])[metric]
    mean = g.mean().reset_index()
    std = g.std().reset_index().rename(columns={metric: "std"})
    m = mean.merge(std, on=[by, x], how="left")

    fig, ax = plt.subplots(figsize=(10, 4))
    for key, sub in m.groupby(by):
        sub = sub.sort_values(x)
        xv = sub[x].to_numpy(dtype=float)
        yv = sub[metric].to_numpy(dtype=float)
        sv = sub["std"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(xv, yv, marker="o", label=str(key))
        ax.fill_between(xv, yv - sv, yv + sv, alpha=0.15)
    ax.set_xlabel("training trials / random no-control rollouts")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or metric)
    ax.grid(True, alpha=0.3)
    legend_outside(ax, loc="center left", anchor=(1.02, 0.5), frameon=True)
    fig.subplots_adjust(right=0.78)
    plt.show()
    return fig, ax


def plot_learning_curve_metric(
    df_in,
    metric: str,
    *,
    ylabel: str | None = None,
    title: str | None = None,
    show_seed_traces: bool = True,
    show_sem_band: bool = True,
):
    from opinion_dynamics.experiments.metrics import add_trial_count, aggregate_learning_curve

    dfc = add_trial_count(df_in)
    if metric not in dfc.columns:
        raise ValueError(f"Metric {metric!r} not found. Available columns include: {list(dfc.columns)[:20]}...")
    agg = aggregate_learning_curve(dfc, metrics=[metric])
    dyns = list(dfc["dynamics"].drop_duplicates())

    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    for dyn in dyns:
        sub_dyn = dfc[dfc["dynamics"] == dyn].sort_values("trial_count")
        if show_seed_traces:
            for _seed, sub_seed in sub_dyn.groupby("seed"):
                ax.plot(sub_seed["trial_count"], sub_seed[metric], linewidth=0.8, alpha=0.18)

        sub_agg = agg[agg["dynamics"] == dyn].sort_values("trial_count")
        xv = sub_agg["trial_count"].to_numpy(dtype=float)
        yv = sub_agg[f"{metric}_mean"].to_numpy(dtype=float)
        ax.plot(xv, yv, marker="o", linewidth=2.3, label=dyn)
        if show_sem_band and f"{metric}_sem" in sub_agg.columns:
            sem = sub_agg[f"{metric}_sem"].fillna(0.0).to_numpy(dtype=float)
            ax.fill_between(xv, yv - sem, yv + sem, alpha=0.12)

    ax.set_xlabel("no-control training trials")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"Validation learning curve: {metric}")
    ax.grid(True, alpha=0.3)
    legend_outside(ax, loc="upper left", anchor=(1.02, 1.0), frameon=False)
    fig.subplots_adjust(right=0.78)
    plt.show()
    return fig, ax
