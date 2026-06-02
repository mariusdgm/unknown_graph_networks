"""Plot helpers for experiment notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from opinion_dynamics.experiments.rollouts import apply_impulse_control
from opinion_dynamics.experiments.metrics import aggregate_learning_curve
from opinion_dynamics.experiments.metrics import add_trial_count_column


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
    ax.grid(
        which="minor",
        color=grid_color,
        linestyle="-",
        linewidth=grid_lw,
        alpha=grid_alpha,
    )
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

    # Supports both flattened (T,N)/(T,) and stacked (K,M,N)/(K,M) inputs.
    if inter_states.ndim == 2:
        x = inter_states
        t = inter_times.reshape(-1)
    else:
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
        ts = np.asarray(ts, dtype=float)
        if ts.ndim != 1:
            ts = ts.reshape(-1)

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


def build_augmented_campaign_trajectory(
    states_boundary: np.ndarray,
    actions: np.ndarray,
    intermediate_states_list,
    *,
    desired_opinion: float,
    dt: float,
    t_campaign: float | None = None,
    eps_t: float = 1e-6,
    atol: float = 1e-8,
):
    """
    Build a fine-grained trajectory that explicitly includes, for each campaign:
      1) boundary/pre-impulse state x_k at campaign time k * t_campaign
      2) immediate post-impulse state x_k^ctrl at k * t_campaign + eps_t
      3) propagation substeps at their physical substep times
      4) next campaign boundary x_{k+1} at (k+1) * t_campaign when needed

    Important:
      dt is the substep size (env.t_s).
      t_campaign is the campaign duration (env.t_campaign).

    The previous version advanced campaign starts by dt. With defaults
    t_campaign=2.0 and t_s=0.5, that compressed/overlapped campaign times
    by a factor of 4 and made later points appear at the wrong x-axis locations.
    """
    states_boundary = np.asarray(states_boundary, dtype=float)
    actions = np.asarray(actions, dtype=float)

    if states_boundary.ndim != 2:
        raise ValueError("states_boundary must have shape (K+1, N).")
    if actions.ndim != 2:
        raise ValueError("actions must have shape (K, N).")

    dt = float(dt)
    campaign_duration = float(dt if t_campaign is None else t_campaign)
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if campaign_duration <= 0:
        raise ValueError("t_campaign/campaign_duration must be positive.")

    X_parts = []
    T_parts = []

    def _append_state(x, t):
        """Append unless it is an exact duplicate of the previous plotted point."""
        x = np.asarray(x, dtype=float)
        t = float(t)
        if X_parts:
            x_prev = X_parts[-1][-1]
            t_prev = T_parts[-1][-1]
            if abs(t - t_prev) <= atol and np.max(np.abs(x - x_prev)) <= atol:
                return
        X_parts.append(x[None, :].copy())
        T_parts.append(np.array([t], dtype=float))

    num_campaigns = min(
        len(actions),
        len(intermediate_states_list),
        max(states_boundary.shape[0] - 1, 0),
    )

    # Common initial state.
    _append_state(states_boundary[0], 0.0)

    for k in range(num_campaigns):
        campaign_start = k * campaign_duration
        campaign_end = campaign_start + campaign_duration

        x_pre = np.asarray(states_boundary[k], dtype=float)
        u_k = np.asarray(actions[k], dtype=float)
        x_ctrl = apply_impulse_control(
            x_pre, u_k, desired_opinion=float(desired_opinion)
        )

        # Explicit campaign boundary/pre-impulse point.
        _append_state(x_pre, campaign_start)

        # Explicit immediate post-impulse point.
        _append_state(x_ctrl, campaign_start + eps_t)

        inter = intermediate_states_list[k]
        if inter is not None:
            inter_arr = np.asarray(inter, dtype=float)

            if inter_arr.ndim != 2:
                raise ValueError(
                    f"intermediate_states_list[{k}] must have shape (M, N); "
                    f"got shape {inter_arr.shape}."
                )

            if inter_arr.shape[0] > 0:
                local_t = dt * np.arange(inter_arr.shape[0], dtype=float)

                # Common env convention: inter[0] is the immediate post-impulse
                # state. We already inserted it at campaign_start + eps_t.
                if np.max(np.abs(inter_arr[0] - x_ctrl)) <= atol:
                    inter_arr = inter_arr[1:]
                    local_t = local_t[1:]

                # Defensive handling for envs that include the pre-impulse state.
                if (
                    inter_arr.shape[0] > 0
                    and np.max(np.abs(inter_arr[0] - x_pre)) <= atol
                ):
                    inter_arr = inter_arr[1:]
                    local_t = local_t[1:]

                for x_sub, tau in zip(inter_arr, local_t):
                    _append_state(x_sub, campaign_start + float(tau))

        # Ensure the boundary state exists at the exact campaign end. This makes
        # the time grid correct even if info["intermediate_states"] is missing or
        # uses a slightly different convention.
        _append_state(states_boundary[k + 1], campaign_end)

    X = np.vstack(X_parts)
    T = np.concatenate(T_parts)

    if np.any(np.diff(T) < -10 * atol):
        bad = np.where(np.diff(T) < -10 * atol)[0][:5]
        raise RuntimeError(
            f"Non-monotone augmented time vector near indices {bad}: {T[bad]} -> {T[bad + 1]}"
        )

    return X, T


def plot_learning_curve_metric(
    df_in: pd.DataFrame,
    metric: str,
    *,
    ylabel: str | None = None,
    title: str | None = None,
    show_seed_traces: bool = True,
    show_sem_band: bool = True,
):
    """
    Plot validation metric vs number of no-control training trials.

    Uses light per-seed traces plus a thicker mean curve. No in-plot legend.
    """
    dfc = add_trial_count_column(df_in)
    if metric not in dfc.columns:
        raise ValueError(
            f"Metric {metric!r} not found. Available columns include: {list(dfc.columns)[:20]}..."
        )

    agg = aggregate_learning_curve(dfc, metrics=[metric])
    dyns = list(dfc["dynamics"].drop_duplicates())

    fig, ax = plt.subplots(figsize=(10.5, 4.0))

    for dyn in dyns:
        sub_dyn = dfc[dfc["dynamics"] == dyn].sort_values("trial_count")
        if show_seed_traces:
            for _seed, sub_seed in sub_dyn.groupby("seed"):
                ax.plot(
                    sub_seed["trial_count"],
                    sub_seed[metric],
                    linewidth=0.8,
                    alpha=0.18,
                )

        sub_agg = agg[agg["dynamics"] == dyn].sort_values("trial_count")
        x = sub_agg["trial_count"].to_numpy(dtype=float)
        y = sub_agg[f"{metric}_mean"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.3, label=dyn)

        if show_sem_band and f"{metric}_sem" in sub_agg.columns:
            sem = sub_agg[f"{metric}_sem"].fillna(0.0).to_numpy(dtype=float)
            ax.fill_between(x, y - sem, y + sem, alpha=0.12)

    ax.set_xlabel("no-control training trials")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"Validation learning curve: {metric}")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False)
    fig.subplots_adjust(right=0.78)
    plt.show()


def plot_trials_needed_summary(summary_df: pd.DataFrame):
    """Simple visual summary of median trials needed by dynamics."""
    if summary_df.empty:
        print("No summary rows to plot.")
        return
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    x = np.arange(summary_df.shape[0])
    ax.bar(x, summary_df["trials_needed_median"].to_numpy(dtype=float))
    ax.set_xticks(x)
    ax.set_xticklabels(
        summary_df["dynamics"].astype(str).tolist(), rotation=20, ha="right"
    )
    ax.set_ylabel("median trials needed")
    ax.set_title("Data need estimate from validation criteria")
    ax.grid(True, axis="y", alpha=0.3)
    plt.show()


def _legend_outside(ax, *, loc="center left", anchor=(1.02, 0.5), ncol=1, frameon=True):
    """Put the legend outside the axes so it does not cover the data."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    return ax.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=anchor,
        ncol=ncol,
        frameon=frameon,
        borderaxespad=0.0,
    )


def plot_mean_baseline_comparison(mean_series, title, *args, **kwargs):
    """
    Clear mean/baseline comparison plot.

    The legend is kept, but moved outside the axes.  Overlapping curves are
    distinguished using line style + marker shape, so the legend directly maps
    visual style to baseline name.
    """
    mean_series = {k: np.asarray(v, dtype=float) for k, v in mean_series.items()}
    first = next(iter(mean_series.values()))
    x = np.arange(first.shape[0])

    style_cycle = [
        dict(marker="o", linestyle="-", linewidth=2.5, markersize=5.2, zorder=5),
        dict(marker="s", linestyle="--", linewidth=2.1, markersize=5.0, zorder=4),
        dict(marker="^", linestyle="-.", linewidth=2.1, markersize=5.0, zorder=3),
        dict(marker="D", linestyle=":", linewidth=2.3, markersize=4.8, zorder=2),
        dict(
            marker="x",
            linestyle=(0, (3, 1, 1, 1)),
            linewidth=2.0,
            markersize=5.4,
            zorder=1,
        ),
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

    # Keep the legend visible, but outside the data region.
    _legend_outside(
        ax,
        loc="upper center",
        anchor=(0.5, -0.28),
        ncol=min(3, max(1, len(mean_series))),
        frameon=True,
    )
    fig.subplots_adjust(bottom=0.34)
    plt.show()
