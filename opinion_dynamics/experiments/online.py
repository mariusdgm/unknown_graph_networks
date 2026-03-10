import contextlib
import io

import numpy as np
import pandas as pd

from rl_envs_forge.envs.network_graph.graph_utils import compute_laplacian, compute_eigenvector_centrality

from ..identify import GraphIdentifierEnv, pairs_from_intermediate, train_graph_identifier
from ..utils.env_setup import EnvironmentFactory
from ..baseline import centrality_based_continuous_control

from rollouts import rollout_with_v


def run_single_paper_experiment_per_campaign_budget_on_env(
    env,
    *,
    B_campaign=1.0,
    num_campaigns_total=5,
    lr=1e-3,
    l2_lambda=0.0,
    device="cpu",
    update_A_each_campaign=True,
    suppress_fit_logs=True,
):
    """
    Your online experiment, but takes an env instance (so we can control the seed externally).
    Returns: env, states (K+1,N), actions (K,N), A_hats (list)
    """
    N = env.num_agents
    ubar_vec = np.asarray(env.max_u, dtype=float)

    states, actions, rewards = [], [], []
    A_hats, v_hats = [], []
    buf_x, buf_y = [], []

    x, _ = env.reset()
    states.append(x.copy())

    # Campaign 0: zero control (data collection)
    u0 = np.zeros(N, dtype=float)
    x1, r0, done, trunc, info0 = env.step(u0)
    actions.append(u0.copy())
    rewards.append(float(r0))
    states.append(x1.copy())

    inter0 = info0.get("intermediate_states", None)
    if inter0 is None:
        raise RuntimeError("env.step did not return info['intermediate_states']")
    Xp, Yp = pairs_from_intermediate(inter0)
    buf_x.append(Xp)
    buf_y.append(Yp)

    gi = GraphIdentifierEnv(N=N, s=env.t_s, l2_lambda=l2_lambda, zero_diag=True)

    def _fit_current():
        X = np.concatenate(buf_x, axis=0)
        Y = np.concatenate(buf_y, axis=0)
        if suppress_fit_logs:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                A_hat = train_graph_identifier(gi, X, Y, lr=lr, device=device)
        else:
            A_hat = train_graph_identifier(gi, X, Y, lr=lr, device=device)
        return A_hat

    # initial fit after campaign 0
    A_hat = _fit_current()
    A_hats.append(A_hat)
    v_hat = compute_eigenvector_centrality(compute_laplacian(A_hat))
    v_hats.append(v_hat)

    # Campaigns 1.. (online control + re-fit)
    for k in range(1, num_campaigns_total):
        if done or trunc:
            break

        beta_k = min(float(B_campaign), float(ubar_vec.sum()))
        uk, _ = centrality_based_continuous_control(env, beta_k, v=v_hat)

        x_next, r, done, trunc, info_k = env.step(uk)
        actions.append(uk.copy())
        rewards.append(float(r))
        states.append(x_next.copy())

        inter = info_k.get("intermediate_states", None)
        if inter is not None:
            Xp, Yp = pairs_from_intermediate(inter)
            buf_x.append(Xp)
            buf_y.append(Yp)

        if update_A_each_campaign:
            A_hat = _fit_current()
            v_hat = compute_eigenvector_centrality(compute_laplacian(A_hat))
            A_hats.append(A_hat)
            v_hats.append(v_hat)

    return {
        "env": env,
        "states": np.asarray(states),
        "actions": np.asarray(actions),
        "rewards": np.asarray(rewards),
        "A_hats": A_hats,
        "v_hats": v_hats,
    }


def run_multi_seed_experiment(
    *,
    seeds=range(10),
    B_campaign=1.0,
    num_campaigns_total=5,
    lr=1e-3,
    l2_lambda=0.0,
    device="cpu",
    update_A_each_campaign=True,
    suppress_fit_logs=True,
):
    env_factory = EnvironmentFactory()
    rows = []

    for seed in seeds:
        env = env_factory.get_randomized_env(seed=int(seed))

        # run learned-online on THIS env
        out = run_single_paper_experiment_per_campaign_budget_on_env(
            env,
            B_campaign=B_campaign,
            num_campaigns_total=num_campaigns_total,
            lr=lr,
            l2_lambda=l2_lambda,
            device=device,
            update_A_each_campaign=update_A_each_campaign,
            suppress_fit_logs=suppress_fit_logs,
        )

        env0 = out["env"]
        states_learn = out["states"]
        A_hat_final = np.asarray(out["A_hats"][-1], dtype=float)

        # true A/v on this env
        A_true = np.asarray(env0.connectivity_matrix, dtype=float)
        v_true = compute_eigenvector_centrality(compute_laplacian(A_true))
        v_hat_final = compute_eigenvector_centrality(compute_laplacian(A_hat_final))

        # oracle & no-control on same graph, same x0
        x0 = states_learn[0].copy()
        K_total = states_learn.shape[0] - 1
        states_or = rollout_with_v(env0, x0, K_total, B_campaign, v_true)
        states_nc = rollout_with_v(env0, x0, K_total, B_campaign, None)

        # trajectory metrics (boundary-level)
        T = min(states_learn.shape[0], states_or.shape[0], states_nc.shape[0])
        mean_le = states_learn[:T].mean(axis=1)
        mean_or = states_or[:T].mean(axis=1)
        mean_nc = states_nc[:T].mean(axis=1)

        vx_le = states_learn[:T] @ v_true
        vx_or = states_or[:T] @ v_true
        vx_nc = states_nc[:T] @ v_true

        mean_err = np.abs(mean_le - mean_or)
        vx_err = np.abs(vx_le - vx_or)

        rows.append(
            dict(
                seed=int(seed),
                N=int(env0.num_agents),
                # identification
                v_L1_final=float(np.sum(np.abs(v_hat_final - v_true))),
                A_Fro_final=float(np.linalg.norm(A_hat_final - A_true, ord="fro")),
                A_MAE_final=float(np.mean(np.abs(A_hat_final - A_true))),
                # trajectories
                mean_oracle_end=float(mean_or[-1]),
                mean_learn_end=float(mean_le[-1]),
                mean_noc_end=float(mean_nc[-1]),
                mean_gap_to_oracle_end=float(np.abs(mean_le[-1] - mean_or[-1])),
                mean_err_avg=float(mean_err.mean()),
                mean_err_max=float(mean_err.max()),
                vx_gap_to_oracle_end=float(np.abs(vx_le[-1] - vx_or[-1])),
                vx_err_avg=float(vx_err.mean()),
                vx_err_max=float(vx_err.max()),
            )
        )

    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
