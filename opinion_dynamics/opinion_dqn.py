import os, sys


def get_dir_n_levels_up(path, n):
    # Go up n levels from the given path
    for _ in range(n):
        path = os.path.dirname(path)
    return path


proj_root = get_dir_n_levels_up(os.path.abspath(__file__), 2)
sys.path.append(proj_root)


import datetime
import torch
import math
import numpy as np
from collections import defaultdict, deque

from pathlib import Path
from typing import Dict


import torch.optim as optim
import torch.nn.functional as F


from opinion_dynamics.replay_buffer import ReplayBuffer
from opinion_dynamics.models import OpinionNet, OpinionNetCommonAB
from opinion_dynamics.utils.my_logging import setup_logger
from opinion_dynamics.utils.generic import replace_keys, seed_everything
from opinion_dynamics.utils.env_setup import EnvironmentFactory


class EarlyStop(Exception):
    pass


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

def robust_quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    """
    Compute the q-quantile over all elements of x.
    Works on older PyTorch without torch.quantile by falling back to kthvalue.
    """
    x = x.reshape(-1)
    q = float(q)
    if hasattr(torch, "quantile"):
        return torch.quantile(x, q)
    # Fallback: kthvalue on the flattened vector (dim=0)
    n = x.numel()
    # k should be 1..n; use ceil so q=0.98 picks the 98th percentile
    k = max(1, min(n, int(math.ceil(q * n))))
    return x.kthvalue(k, dim=0).values  # .values for newer torch; .[0] for older


class AgentDQN:
    def __init__(
        self,
        experiment_output_folder=None,
        experiment_name=None,
        resume_training_path=None,
        save_checkpoints=True,
        logger=None,
        config={},
    ) -> None:
        """A DQN agent implementation.

        Args:

            experiment_output_folder (str, optional): Path to the folder where the training outputs will be saved.
                                                         Defaults to None.
            experiment_name (str, optional): A string describing the experiment being run. Defaults to None.
            resume_training_path (str, optional): Path to the folder where the outputs of a previous training
                                                    session can be found. Defaults to None.
            save_checkpoints (bool, optional): Whether to save the outputs of the training. Defaults to True.
            logger (logger, optional): Necessary Logger instance. Defaults to None.
            config (Dict, optional): Settings of the agent relevant to the models and training.
                                    If none is provided in the input, the agent will automatically build the default settings.
                                    Defaults to {}.
            enable_tensorboard_logging (bool, optional): Specifies if logs should also be made using tensorboard.
                                                        Defaults to True.
        """

        # assign environments
        self.save_checkpoints = save_checkpoints
        self.logger = logger or setup_logger("dqn")
        self.logger.info(
            f"asserts_enabled={sys.flags.optimize == 0}, optimize={sys.flags.optimize}, PYTHONOPTIMIZE={os.getenv('PYTHONOPTIMIZE')}"
        )

        # set up path names
        self.experiment_output_folder = experiment_output_folder
        self.experiment_name = experiment_name

        self.model_file_folder = (
            "model_checkpoints"  # models will be saved at each epoch
        )
        self.model_checkpoint_file_basename = "mck"

        if self.experiment_output_folder and self.experiment_name:
            self.replay_buffer_file = os.path.join(
                self.experiment_output_folder, f"{self.experiment_name}_replay_buffer"
            )
            self.train_stats_file = os.path.join(
                self.experiment_output_folder, f"{self.experiment_name}_train_stats"
            )

        self.config = config
        if self.config:
            self.config = replace_keys(self.config, "args_", "args")

        self._load_config_settings(self.config)
        self._init_models(self.config)  # init policy, target and optim

        # Set initial values related to training and monitoring
        self.t = 0  # frame nr
        self.episodes = 0  # episode nr
        self.policy_model_update_counter = 0

        self.log_stride = 5_000  # change to taste (e.g., 5_000 if debugging)

        # Metrics to track training progress
        self._last_entropy = None
        self._last_frac_over_cap = None
        self._last_rel_target_drift = None

        self._tgt_clip_ema = None
        self._tgt_clip_alpha = 0.1

        self.reset_training_episode_tracker()

        self.training_stats = []
        self.validation_stats = []

        self._init_early_stopping_vars()
        
        # check that all paths were provided and that the files can be found
        if resume_training_path:
            self.load_training_state(resume_training_path)


    def _should_log(self):
        return (self.t % self.log_stride) == 0 and self.t > 0

    def _make_model_checkpoint_file_path(self, experiment_output_folder, epoch_cnt=0):
        """Dynamically build the path where to save the model checkpoint."""
        return os.path.join(
            experiment_output_folder,
            self.model_file_folder,
            f"{self.model_checkpoint_file_basename}_{epoch_cnt}",
        )

    def load_models_at(
        self,
        checkpoint: int | str,
        resume_training_path: str | None = None,
        eval_mode: bool = True,
    ) -> bool:
        """
        Load ONLY the model weights from a specific checkpoint.

        Modes:
        - Mode A (index): checkpoint is an int, and resume_training_path is provided.
                            The path is resolved via self._make_model_checkpoint_file_path(resume_training_path, checkpoint).
        - Mode B (full path): checkpoint is a str that points directly to the checkpoint file.

        Args:
            checkpoint (int | str): Epoch/step index (int) or full checkpoint file path (str).
            resume_training_path (str | None): Base directory used when `checkpoint` is an int.
            eval_mode (bool): If True, put loaded models into eval().

        Returns:
            bool: True if the models were loaded successfully, False otherwise.

        Raises:
            FileNotFoundError: If the resolved checkpoint file does not exist.
            ValueError: If arguments are inconsistent (e.g., int checkpoint without a base path).
        """
        # Resolve checkpoint path
        if isinstance(checkpoint, int):
            if resume_training_path is None:
                raise ValueError(
                    "When `checkpoint` is an int, `resume_training_path` must be provided."
                )
            if checkpoint < 0:
                raise ValueError("`checkpoint` index must be non-negative.")
            ckpt_path = self._make_model_checkpoint_file_path(
                resume_training_path, checkpoint
            )
        elif isinstance(checkpoint, str):
            # Accept absolute or relative full file path
            ckpt_path = os.path.abspath(checkpoint)
        else:
            raise ValueError(
                "`checkpoint` must be an int (index) or str (full file path)."
            )

        # Validate existence
        if not os.path.exists(ckpt_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        try:
            # Load just the models (no buffer, no stats)
            self.load_models(ckpt_path)

            # Optional: set eval() to avoid train-time layers affecting evaluation
            if eval_mode:
                for attr in ("policy_model", "target_model"):
                    module = getattr(self, attr, None)
                    if hasattr(module, "eval"):
                        module.eval()

            self.logger.info(f"Loaded model weights from checkpoint: {ckpt_path}")
            return True

        except Exception as e:
            # Keep training loop alive; caller can decide what to do next
            self.logger.exception(f"Failed to load models from {ckpt_path}: {e}")
            return False

    def load_training_state(self, resume_training_path: str):
        """In order to resume training the following files are needed:
        - ReplayBuffer file
        - Training stats file
        - Model weights file (found as the last checkpoint in the models subfolder)
        Args:
            resume_training_path (str): path to where the files needed to resume training can be found

        Raises:
            FileNotFoundError: Raised if a required file was not found.
        """

        ### build the file paths
        resume_files = {}

        resume_files["replay_buffer_file"] = os.path.join(
            resume_training_path, f"{self.experiment_name}_replay_buffer"
        )
        resume_files["train_stats_file"] = os.path.join(
            resume_training_path, f"{self.experiment_name}_train_stats"
        )

        # check that the file paths exist
        for file in resume_files:
            if not os.path.exists(resume_files[file]):
                self.logger.info(
                    f"Could not find the file {resume_files[file]} for {file} either because a wrong path was given, or because no training was done for this experiment."
                )
                return False

        # read through the stats file to find what was the epoch for the last recorded state
        self.load_training_stats(resume_files["train_stats_file"])
        self.replay_buffer.load(resume_files["replay_buffer_file"])

        epoch_cnt = len(self.training_stats)

        resume_files["checkpoint_model_file"] = self._make_model_checkpoint_file_path(
            resume_training_path, epoch_cnt
        )
        if not os.path.exists(resume_files["checkpoint_model_file"]):
            raise FileNotFoundError(
                f"Could not find the file {resume_files['checkpoint_model_file']} for 'checkpoint_model_file'."
            )

        self.load_models(resume_files["checkpoint_model_file"])

        self.logger.info(
            f"Loaded previous training status from the following files: {str(resume_files)}"
        )

    def _load_config_settings(self, config={}):
        """
        Load the settings from config.
        If config was not provided, then default values are used.
        """
        agent_params = config.get("agent_params", {}).get("args", {})

        # setup training configuration
        self.train_step_cnt = agent_params.get("train_step_cnt", 200_000)
        self.validation_enabled = agent_params.get("validation_enabled", True)
        self.validation_step_cnt = agent_params.get("validation_step_cnt", 3_000)
        self.validation_epsilon = agent_params.get("validation_epsilon", 0.0)

        self.replay_start_size = agent_params.get("replay_start_size", 5_000)

        self.batch_size = agent_params.get("batch_size", 32)
        self.training_freq = agent_params.get("training_freq", 4)
        self.target_model_update_freq = agent_params.get(
            "target_model_update_freq", 100
        )
        self.target_soft_tau = agent_params.get("target_soft_tau", 0.005)
        self.grad_norm_clip = agent_params.get("grad_norm_clip", 5.0)
        self.gamma = agent_params.get("gamma", 0.99)
        # self.loss_function = agent_params.get("loss_fcn", "mse_loss")

        self.action_w_noise_amplitude = agent_params.get(
            "action_w_noise_amplitude", 0.3
        )
        self.action_w_noise_eps_floor = agent_params.get(
            "action_w_noise_eps_floor", 0.0
        )
        self.betas = agent_params.get("betas", [0, 0.5, 1])

        eps_settings = agent_params.get(
            "epsilon", {"start": 1.0, "end": 0.01, "decay": 250_000}
        )
        self.epsilon_by_frame = self._get_linear_decay_function(
            start=eps_settings["start"],
            end=eps_settings["end"],
            decay=eps_settings["decay"],
            eps_decay_start=self.replay_start_size,
        )
        
        self.use_hard_target_updates = agent_params.get("use_hard_target_updates", False)
        self.hard_target_update_every = agent_params.get("hard_target_update_every", 10_000)
        # guard: if hard updates on, we will skip soft updates
        if self.use_hard_target_updates:
            self.logger.info(f"[CFG] Using HARD target updates every {self.hard_target_update_every} policy updates.")
        else:
            self.logger.info(f"[CFG] Using SOFT updates with tau={self.target_soft_tau}.")

        # ---- NEW: optional LR scheduler (cosine) ----
        sched_cfg = agent_params.get("lr_scheduler", None)
        self._lr_sched_cfg = sched_cfg if isinstance(sched_cfg, dict) else None

        self._read_and_init_envs()  # sets up in_features etc...

        buffer_settings = config.get(
            "replay_buffer", {"max_size": 100_000, "n_step": 0}
        )
        self.replay_buffer = ReplayBuffer(
            max_size=buffer_settings.get("max_size", 100_000),
            state_dim=self.in_features,
            action_dim=self.train_env.action_space.shape[0],
            n_step=buffer_settings.get("n_step", 0),
            betas=self.betas,
        )

        self.logger.info("Loaded configuration settings.")

    def _init_early_stopping_vars(self):
        # ---- Early stop: rolling metrics & thresholds tuned for your failure mode ----
        W = 10  # was 8; slightly smoother rolling stats (log points, not steps)
        self._es_win = {
            "H": deque(maxlen=W),  # action entropy
            "frac_cap": deque(maxlen=W),
            "td_p95": deque(maxlen=W),  # p95(|TD|)
            "clamp_pct": deque(maxlen=W),
            "tgt_drift": deque(maxlen=W),  # ||target-source||/||source||
            "max_q": deque(maxlen=W),
        }

        # Derived entropy baseline ~ uniform
        self._H_uniform = math.log(self.num_actions) if self.num_actions > 0 else 0.0

        self._es_cfg = {
            # action collapse
            "min_entropy_frac": 0.22,  # was 0.15  → trip sooner if entropy too low
            "high_frac_cap": 0.65,  # was 0.70  → treat saturation as “bad” earlier
            # TD/target instability
            "clamp_pct_high": 0.50,
            "td_p95_jump": 1.6,
            "tgt_drift_high": 0.22,
            # patience in log points (not steps/epochs)
            "pat_points": 5,  # was 4     → require a few points, still quick
            # NEW hard tripwires (absolute guards)
            "max_q_abs": 1500,  # stop if mean max-Q goes way out of range
            "A_floor_ratio": 0.90,  # if ~all agents sit at A_min, treat as degeneracy
            "no_prog_mult": 0.95,  # entropy median must improve by ≥5% across halves
        }

        self._nonfinite_counter = 0
        self.nonfinite_patience = 3

    def _es_update(
        self,
        *,
        H=None,
        frac_cap=None,
        td_p95=None,
        clamp_pct=None,
        tgt_drift=None,
        max_q=None,
    ):
        if H is not None:
            self._es_win["H"].append(float(H))
        if frac_cap is not None:
            self._es_win["frac_cap"].append(float(frac_cap))
        if td_p95 is not None:
            self._es_win["td_p95"].append(float(td_p95))
        if clamp_pct is not None:
            self._es_win["clamp_pct"].append(float(clamp_pct))
        if tgt_drift is not None:
            self._es_win["tgt_drift"].append(float(tgt_drift))
        if max_q is not None:
            self._es_win["max_q"].append(float(max_q))

    def _median(self, arr):
        if not arr:
            return None
        v = sorted(arr)
        n = len(v)
        return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])

    def _es_trend_ratio(self, arr):
        """Ratio of median(last half) / median(first half); returns None if insufficient data."""
        if len(arr) < max(2, self._es_cfg["pat_points"]):
            return None
        mid = len(arr) // 2
        first = self._median(list(arr)[:mid])
        last = self._median(list(arr)[mid:])
        if first is None or last is None or first <= 1e-12:
            return None
        return last / first

    def _early_stop_maybe(self):
        need = self._es_cfg["pat_points"]
        if any(len(self._es_win[k]) < need for k in self._es_win):
            return

        H_u = max(self._H_uniform, 1.0)
        H_med = self._median(self._es_win["H"])
        frac_med = self._median(self._es_win["frac_cap"])
        clamp_med = self._median(self._es_win["clamp_pct"])  # here: A at floor
        drift_med = self._median(self._es_win["tgt_drift"])
        td_growth = self._es_trend_ratio(self._es_win["td_p95"])

        # Hard tripwire 1: Q explosion
        max_q_med = self._median(self._es_win["max_q"])
        if (max_q_med is not None) and (max_q_med > self._es_cfg["max_q_abs"]):
            self.logger.error(
                f"[EARLY STOP] Q explosion: median max_q={max_q_med:.3e} > {self._es_cfg['max_q_abs']}"
            )
            raise EarlyStop("Q explosion.")

        # Hard tripwire 2: A stuck at floor
        if (clamp_med is not None) and (clamp_med >= self._es_cfg["A_floor_ratio"]):
            self.logger.error(
                f"[EARLY STOP] A at floor: clamp_med={clamp_med:.2f} >= {self._es_cfg['A_floor_ratio']}"
            )
            raise EarlyStop("A_diag collapsed to floor.")

        # Composite signals
        S_action_collapse = (
            (H_med is not None)
            and (H_med < self._es_cfg["min_entropy_frac"] * H_u)
            and (frac_med is not None)
            and (frac_med > self._es_cfg["high_frac_cap"])
        )
        S_target_instab = (
            (clamp_med is not None)
            and (clamp_med > 0.60)  # if you want a second, stricter gate
            and (td_growth is not None)
            and (td_growth > self._es_cfg["td_p95_jump"])
        )
        S_drift_spike = (drift_med is not None) and (
            drift_med > self._es_cfg["tgt_drift_high"]
        )

        # Soft "no progress": entropy should trend UP across halves by >=5%
        H_trend = self._es_trend_ratio(self._es_win["H"])
        S_no_progress = (H_trend is not None) and (
            H_trend < self._es_cfg["no_prog_mult"]
        )

        # Trip if (1) any hard tripwire, or (2) at least two soft collapse signals, or (3) entropy keeps degrading
        if (
            sum([S_action_collapse, S_target_instab, S_drift_spike]) >= 2
            or S_no_progress
        ):
            msg = (
                "[EARLY STOP] Likely irrecoverable collapse:\n"
                f"- action_collapse={S_action_collapse} (H_med={None if H_med is None else round(H_med,3)}, frac_cap_med={None if frac_med is None else round(frac_med,3)})\n"
                f"- target_instab={S_target_instab} (A_floor_med={None if clamp_med is None else round(clamp_med,3)}, td_growth={None if td_growth is None else round(td_growth,3)})\n"
                f"- drift_spike={S_drift_spike} (drift_med={None if drift_med is None else round(drift_med,3)})\n"
                f"- no_progress={S_no_progress} (H_trend={None if H_trend is None else round(H_trend,3)})"
            )
            self.logger.error(msg)
            raise EarlyStop("Collapse criteria satisfied.")

    def _get_exp_decay_function(self, start: float, end: float, decay: float):
        return lambda x: end + (start - end) * np.exp(-1.0 * x / decay)

    def _get_linear_decay_function(
        self, start: float, end: float, decay: float, eps_decay_start: float = None
    ):
        """Return a function that enables getting the value of epsilon at step x.

        Args:
            start (float): start value of the epsilon function (x=0)
            end (float): end value of the epsilon function (x=decay)
            decay (float): how many steps to reach the end value
            eps_decay_start(float, optional): after how many frames to actually start decaying,
                                            uses self.replay_start_size by default

        Returns:
            function: function to compute the epsillon based on current frame counter
        """
        if not eps_decay_start:
            eps_decay_start = self.replay_start_size

        return lambda x: max(
            end, min(start, start - (start - end) * ((x - eps_decay_start) / decay))
        )

    def _init_models(self, config):
        """Instantiate the policy and target networks.

        Args:
            config (Dict): Settings with parameters for the models

        Raises:
            ValueError: The configuration contains an estimator name that the agent does not
                        know to instantiate.
        """
        estimator_settings = config.get("estimator", {"model": "Conv_QNET", "args": {}})

        if estimator_settings["model"] == "OpinionNet":
            self.policy_model = OpinionNet(
                self.in_features,
                nr_betas=len(self.betas),
                **estimator_settings["args"],
            )
            self.target_model = OpinionNet(
                self.in_features,
                nr_betas=len(self.betas),
                **estimator_settings["args"],
            )

        elif estimator_settings["model"] == "OpinionNetCommonAB":
            self.policy_model = OpinionNetCommonAB(
                self.in_features,
                nr_betas=len(self.betas),
                **estimator_settings["args"],
            )
            self.target_model = OpinionNetCommonAB(
                self.in_features,
                nr_betas=len(self.betas),
                **estimator_settings["args"],
            )

        else:
            estimator_name = estimator_settings["model"]
            raise ValueError(f"Could not setup estimator. Tried with: {estimator_name}")

        self.policy_model.train()
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()
      
        optimizer_settings = config.get("optim", {"name": "Adam", "args": {}})
        self.optimizer = optim.Adam(
            self.policy_model.parameters(), **optimizer_settings["args"]
        )
        
        self.lr_scheduler = None
        if self._lr_sched_cfg:
            try:
                name = str(self._lr_sched_cfg.get("name", "cosine")).lower()
                if name == "cosine":
                    T_max = int(self._lr_sched_cfg.get("T_max", 100_000))
                    eta_min = float(self._lr_sched_cfg.get("eta_min", 0.0))
                    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer, T_max=T_max, eta_min=eta_min
                    )
                    self.logger.info(f"[CFG] LR scheduler: CosineAnnealingLR(T_max={T_max}, eta_min={eta_min})")
                elif name in ("cosine_wr", "cosinewarmrestarts", "cosine_awr"):
                    T_0 = int(self._lr_sched_cfg.get("T_0", 300_000))
                    T_mult = int(self._lr_sched_cfg.get("T_mult", 2))
                    eta_min = float(self._lr_sched_cfg.get("eta_min", 0.0))
                    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        self.optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
                    )
                    self.logger.info(f"[CFG] LR scheduler: CosineAnnealingWarmRestarts(T_0={T_0}, T_mult={T_mult}, eta_min={eta_min})")
                else:
                    self.logger.warning(f"[CFG] Unknown lr_scheduler name={name}; scheduler disabled.")
            except Exception as e:
                self.logger.warning(f"[CFG] Failed to init lr scheduler: {e}")
                self.lr_scheduler = None

        self._log_model_and_optim_summaries()

        self.logger.info("Initialized networks and optimizer.")

    def _log_model_and_optim_summaries(self):
        """Log the key hyperparameters of the policy model and optimizer at init time."""
        try:
            pm = self.policy_model
            # Model summary (works for OpinionNet and OpinionNetCommonAB)
            model_info = {
                "model_class": pm.__class__.__name__,
                "nr_agents": getattr(pm, "nr_agents", None),
                "nr_betas": getattr(pm, "nr_betas", None),
                "lin_hidden_size": getattr(pm, "lin_hidden_size", None),
                "softplus_beta": getattr(pm, "softplus_beta", None),
                "wstar_eps": getattr(pm, "wstar_eps", None),
                "A_min": getattr(pm, "A_min", None),
                "A_max": getattr(pm, "A_max", None),
                "b_tanh_scale": getattr(pm, "b_tanh_scale", None),
                "c_tanh_scale": getattr(pm, "c_tanh_scale", None),
            }

            # Optimizer summary (first param group)
            og = self.optimizer.param_groups[0] if self.optimizer.param_groups else {}
            optim_info = {
                "optim_class": self.optimizer.__class__.__name__,
                "lr": og.get("lr", None),
                "betas": og.get("betas", None),
                "eps": og.get("eps", None),
                "weight_decay": og.get("weight_decay", None),
            }

            # Count params
            total_params = sum(p.numel() for p in pm.parameters())
            trainable_params = sum(
                p.numel() for p in pm.parameters() if p.requires_grad
            )

            self.logger.info(
                "[INIT] Model config: %s | Optimizer: %s | params: total=%d, trainable=%d",
                str(model_info),
                str(optim_info),
                total_params,
                trainable_params,
            )
        except Exception as e:
            self.logger.warning(f"[INIT] Failed to log model/optim summary: {e}")

    @torch.no_grad()
    def _soft_update(self, tau):
        # measure drift BEFORE update (occasionally)
        if self._should_log():
            try:
                num, den = 0.0, 1e-12
                for tp, sp in zip(
                    self.target_model.parameters(), self.policy_model.parameters()
                ):
                    num += (tp.data - sp.data).pow(2).sum().item()
                    den += sp.data.pow(2).sum().item()
                rel_dist = (num**0.5) / (den**0.5)
                self._last_rel_target_drift = rel_dist
                self._es_update(tgt_drift=rel_dist)
                self.logger.info(
                    f"target_drift@t={self.t} | ||t-s||/||s||={rel_dist:.3e} | tau={tau}"
                )
            except Exception as e:
                self.logger.debug(f"[drift-log-skip] {e}")

        # EMA for parameters
        for tp, sp in zip(
            self.target_model.parameters(), self.policy_model.parameters()
        ):
            tp.data.lerp_(sp.data, tau)
        for tb, sb in zip(self.target_model.buffers(), self.policy_model.buffers()):
            tb.copy_(sb)

    def _make_train_env(self):
        return self.env_factory.get_randomized_env()

    def _make_validation_env(self):
        total_versions = len(self.env_factory.validation_versions)

        # Determine how many times each version has been used
        usage_counts = [
            (version, self.validation_env_counters.get(version, 0))
            for version in range(total_versions)
        ]

        # Find the version(s) with the minimum usage
        min_usage = min(count for _, count in usage_counts)
        least_used_versions = [v for v, count in usage_counts if count == min_usage]

        # Break ties by choosing the smallest version index
        chosen_version = min(least_used_versions)

        env = self.env_factory.get_validation_env(version=chosen_version)
        self.validation_env_counters[chosen_version] += 1
        return env

    def _read_and_init_envs(self):
        """Read dimensions of the input and output of the simulation environment"""
        self.env_factory = EnvironmentFactory()
        self.validation_env_counters = defaultdict(int)

        self.train_env = self._make_train_env()
        self.validation_env = self._make_validation_env()

        self.train_env_s, _ = self.train_env.reset(randomize_opinions=True)
        self.val_env_s, _ = self.validation_env.reset()

        self.in_features = self.train_env.observation_space.shape[0]
        self.num_actions = self.train_env.action_space.shape[0]

    def load_models(self, models_load_file):
        checkpoint = torch.load(
            models_load_file, map_location=device, weights_only=False
        )
        self.policy_model.load_state_dict(checkpoint["policy_model_state_dict"])
        self.policy_model.train()
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.target_model.eval()
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        try:
            sched_state = checkpoint.get("lr_scheduler_state_dict", None)
            if (self.lr_scheduler is not None) and (sched_state is not None):
                self.lr_scheduler.load_state_dict(sched_state)
                # Optional: log where we are in the cycle
                lr_now = self.optimizer.param_groups[0]["lr"]
                self.logger.info(f"[RESUME] Restored LR scheduler state; current lr={lr_now:.6g}")
            else:
                if self.lr_scheduler is not None:
                    self.logger.info("[RESUME] No scheduler state in checkpoint; scheduler continues from init.")
                # else: no scheduler configured → nothing to restore
        except Exception as e:
            self.logger.warning(f"[RESUME] Failed to load scheduler state: {e}")
        
    def load_training_stats(self, training_stats_file):
        checkpoint = torch.load(
            training_stats_file, map_location=device, weights_only=False
        )

        self.t = checkpoint["frame"]
        self.episodes = checkpoint["episode"]
        self.policy_model_update_counter = checkpoint["policy_model_update_counter"]

        self.training_stats = checkpoint["training_stats"]
        self.validation_stats = checkpoint["validation_stats"]

        self._unpack_es_state(checkpoint.get("es_state", None))
        
    def save_checkpoint(self):
        if not (self.experiment_output_folder and self.experiment_name):
            self.logger.warning("Skipping checkpoint: missing experiment paths.")
            return

        self.logger.info(f"Saving checkpoint at t = {self.t} ...")
        self.save_model()
        self.save_training_status()
        self.replay_buffer.save(self.replay_buffer_file)
        self.logger.info(f"Checkpoint saved at t = {self.t}")

    def save_model(self):
        model_file = self._make_model_checkpoint_file_path(
            self.experiment_output_folder, len(self.training_stats)
        )
        Path(os.path.dirname(model_file)).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_model_state_dict": self.policy_model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": (self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None),
            },
            model_file,
        )
        self.logger.debug(f"Models saved at t = {self.t}")

    def save_training_status(self):
        status_dict = {
            "frame": self.t,
            "episode": self.episodes,
            "policy_model_update_counter": self.policy_model_update_counter,
            "training_stats": self.training_stats,
            "validation_stats": self.validation_stats,
            "es_state": self._pack_es_state(),
        }

        torch.save(
            status_dict,
            self.train_stats_file,
        )

        self.logger.debug(f"Training status saved at t = {self.t}")

    def _pack_es_state(self) -> dict:
        """Serialize logging/early-stop internals for checkpoint."""
        return {
            "win": {k: list(v) for k, v in self._es_win.items()},           # deques → lists
            "cfg": dict(self._es_cfg),                                       # thresholds (if you tweak later)
            "H_uniform": self._H_uniform,
            "last_entropy": self._last_entropy,
            "last_frac_over_cap": self._last_frac_over_cap,
            "last_rel_target_drift": self._last_rel_target_drift,
            "nonfinite_counter": self._nonfinite_counter,
            "tgt_clip_ema": self._tgt_clip_ema,
            "tgt_clip_alpha": self._tgt_clip_alpha,
            "validation_env_counters": dict(self.validation_env_counters),   # which val. version to pick next
        }
        
    def _unpack_es_state(self, es: dict | None) -> None:
        """Restore logging/early-stop internals from checkpoint."""
        if not es:
            return
        # Rebuild rolling windows with a consistent maxlen (keep your W=10)
        W = 10
        win = es.get("win", {})
        self._es_win = {k: deque(win.get(k, []), maxlen=W) for k in ("H","frac_cap","td_p95","clamp_pct","tgt_drift","max_q")}
        # Restore thresholds/baselines
        self._es_cfg.update(es.get("cfg", {}))
        self._H_uniform = es.get("H_uniform", self._H_uniform)
        # Restore last logged values
        self._last_entropy = es.get("last_entropy", None)
        self._last_frac_over_cap = es.get("last_frac_over_cap", None)
        self._last_rel_target_drift = es.get("last_rel_target_drift", None)
        # Restore counters/ema
        self._nonfinite_counter = es.get("nonfinite_counter", 0)
        self._tgt_clip_ema = es.get("tgt_clip_ema", None)
        self._tgt_clip_alpha = es.get("tgt_clip_alpha", self._tgt_clip_alpha)
        # Validation version usage so we keep cycling evenly
        if "validation_env_counters" in es:
            self.validation_env_counters = defaultdict(int, es["validation_env_counters"])
        
    def select_action(
        self,
        state: torch.Tensor,
        epsilon: float = None,
        random_action: bool = False,
        action_noise: bool = False,
    ):
        """
        Returns:
            u: np.ndarray (B, N)        -- continuous action (allocations)
            beta_idx: np.ndarray (B,)   -- chosen beta indices
            w_full: np.ndarray (B, J, N)-- logits for all betas (zeros except chosen row)
            q_scalar: float             -- mean max-Q for logging
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        device = next(self.policy_model.parameters()).device
        dtype = torch.float32

        state = state.to(device=device, dtype=dtype)

        with torch.no_grad():
            # === Forward pass ===
            abc_model = self.policy_model(state)
            A_diag, b, c = (
                abc_model["A_diag"],
                abc_model["b"],
                abc_model["c"],
            )  # (B,J,N), (B,J,N), (B,J)
            B, J, N = A_diag.shape

            assert b.shape == (B, J, N), f"b shape mismatch: {b.shape}"
            assert c.shape == (B, J), f"c shape mismatch: {c.shape}"

            # logits (pre-softmax) per beta/agent
            w_star = self.policy_model.compute_w_star(A_diag, b)  # (B,J,N)
            q_values = self.policy_model.compute_q_values(w_star, A_diag, b, c)  # (B,J)

            assert w_star.shape == (B, J, N), f"w_star shape mismatch: {w_star.shape}"
            assert q_values.shape == (
                B,
                J,
            ), f"q_values shape mismatch: {q_values.shape}"

            eps = 0.0 if epsilon is None else float(epsilon)
            if action_noise:
                # recommended: base 0.4, floor 0.10–0.15
                sigma = self.action_w_noise_amplitude
                floor = self.action_w_noise_eps_floor
                noise_amplitude = sigma * max(eps, floor)
            else:
                noise_amplitude = 0.0
                
            if self._should_log():
                if action_noise:
                    self._log_debug(f"eff_noise_amp={noise_amplitude:.4f} (sigma={sigma}, eps={eps:.3f}, floor={floor})")
                else:
                    self._log_debug(f"eff_noise_amp={noise_amplitude:.4f} (action_noise=False, eps={eps:.3f})")
            
            # Prepare betas on correct device/dtype
            betas_t = torch.tensor(self.betas, device=device, dtype=dtype)  # (J,)

            # === RANDOM ACTION BRANCH ===
            take_random = random_action or (
                epsilon is not None and np.random.rand() < epsilon
            )
            if take_random:
                rand_idx = torch.randint(
                    low=0, high=J, size=(B,), dtype=torch.long, device=device
                )
                rand_idx_exp = (
                    rand_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, N)
                )  # (B,1,N)

                # chosen logits
                w_rand = w_star.gather(1, rand_idx_exp).squeeze(1)  # (B,N)
                assert w_rand.shape == (B, N), f"w_rand shape: {w_rand.shape}"

                # (optional) exploration noise on logits
                if noise_amplitude > 0.0:
                    w_rand = self.policy_model.apply_action_noise(
                        w_rand, noise_amplitude
                    )

                # build ws_to_store: zeros except chosen β row
                ws_to_store = torch.zeros_like(w_star)  # (B,J,N)
                ws_to_store.scatter_(1, rand_idx_exp, w_rand.unsqueeze(1))

                # turn logits into allocation
                rand_beta_values = (
                    betas_t.index_select(0, rand_idx).unsqueeze(1).expand(-1, N)
                )  # (B,N)
                u = self.policy_model.compute_action_from_w(
                    w_rand, rand_beta_values
                )  # (B,N)

                # quick sanity: stored row shouldn't look like probs
                if self._should_log():
                    rowsums = w_rand.sum(dim=1)
                    nonneg = (w_rand >= -1e-8).all().item()
                    looks_prob = (
                        torch.allclose(
                            rowsums, torch.ones_like(rowsums), rtol=1e-2, atol=1e-2
                        )
                        and nonneg
                    )
                    if looks_prob:
                        self.logger.warning(
                            "[SANITY] Stored w looks probability-like; expected logits."
                        )

                q_rand = q_values.gather(1, rand_idx.unsqueeze(1)).squeeze(1)  # (B,)
                q_rand_mean = q_rand.mean(dim=0)  # scalar

                return (
                    u.cpu().numpy(),
                    rand_idx.cpu().numpy().astype(np.int64),
                    ws_to_store.cpu().numpy(),
                    q_rand_mean.item(),
                )

            # === GREEDY ACTION BRANCH ===
            max_q, beta_idx = q_values.max(dim=1)  # (B,)
            beta_idx_exp = (
                beta_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, N)
            )  # (B,1,N)

            # chosen logits
            optimal_w = w_star.gather(1, beta_idx_exp).squeeze(1)  # (B,N)

            # (optional) exploration noise on greedy path
            if noise_amplitude > 0.0:
                optimal_w = self.policy_model.apply_action_noise(
                    optimal_w, noise_amplitude
                )

            # store logits: zeros except chosen β row
            ws_to_store = torch.zeros_like(w_star)  # (B,J,N)
            ws_to_store.scatter_(1, beta_idx_exp, optimal_w.unsqueeze(1))

            beta_values = (
                betas_t.index_select(0, beta_idx).unsqueeze(1).expand(-1, N)
            )  # (B,N)
            u = self.policy_model.compute_action_from_w(optimal_w, beta_values)  # (B,N)

            # sanity: stored row shouldn't look like probs
            if self._should_log():
                rowsums = optimal_w.sum(dim=1)
                nonneg = (optimal_w >= -1e-8).all().item()
                looks_prob = (
                    torch.allclose(
                        rowsums, torch.ones_like(rowsums), rtol=1e-2, atol=1e-2
                    )
                    and nonneg
                )
                if looks_prob:
                    self.logger.warning(
                        "[SANITY] Stored w looks probability-like; expected logits."
                    )

            assert u.shape == (B, N), f"u shape: {u.shape}"

            return (
                u.cpu().numpy(),
                beta_idx.cpu().numpy().astype(np.int64),
                ws_to_store.cpu().numpy(),
                max_q.mean().item(),
            )

    def model_learn(self, sample, debug=True):
        """TD learning with Double DQN targets, Huber loss, grad clipping, and soft target updates."""
        # Unpack & move to device
        states, (beta_indices, ws), rewards, next_states, dones = sample
        device = next(self.policy_model.parameters()).device
        B = len(states)

        states = states.to(device=device, dtype=torch.float32)
        next_states = next_states.to(device=device, dtype=torch.float32)
        beta_indices = beta_indices.to(device=device, dtype=torch.long).view(-1)
        ws = ws.to(device=device, dtype=torch.float32)  # (B,J,N)
        rewards = rewards.to(device=device, dtype=torch.float32).view(B, 1)
        dones = dones.to(device=device, dtype=torch.float32).view(B, 1)

        # Contract checks for ws
        # (keeps replay buffer dumb, but training strict)
        # If this trips, the bug is upstream (select_action / append).
        self._assert_ws_contract(ws, beta_indices)
        self._warn_if_ws_looks_probabilities(ws, beta_indices)

        # Online Q(s,β) for stored ws
        q_vals, A, b, c = self._online_q_for_ws(states, ws)
        q_sa = q_vals.gather(1, beta_indices.unsqueeze(1))  # (B,1)

        # Double DQN target
        with torch.no_grad():
            max_next_q = self._double_dqn_target(next_states)  # (B,1)
            target = rewards + self.gamma * (1.0 - dones) * max_next_q  # (B,1)

        # Log batch stats (compact)
        self._maybe(lambda: self._log_learn_batch_stats(q_sa, target, A, b, c, ws, beta_indices))
        
        loss = F.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad()
        if not torch.isfinite(loss):
            self.logger.error(
                f"[EARLY STOP] Non-finite loss at t={self.t}: {float(loss.item()) if loss.numel()==1 else 'tensor'}"
            )
            raise EarlyStop("Non-finite loss encountered.")
        loss.backward()

        # Non-finite gradients quick check
        bad_grad = False
        for p in self.policy_model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad_grad = True
                break
        if bad_grad:
            self._nonfinite_counter += 1
            if self._nonfinite_counter > self.nonfinite_patience:
                raise EarlyStop("Repeated non-finite gradients.")
        else:
            self._nonfinite_counter = 0

        # Grad norms (only on log step)
        grad_norm_pre = None
        self._maybe(lambda: globals().__setitem__("__gnp", self._grad_norm()))
        grad_norm_pre = globals().pop("__gnp", None)

        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(), self.grad_norm_clip
        )
        self.optimizer.step()

        if self.lr_scheduler is not None:
            try:
                self.lr_scheduler.step()
            except Exception as e:
                self._log_debug(f"[lr-sched-skip] {e}")
                
        # Post grad norm + optim log (only on log step)
        self._maybe(lambda: globals().__setitem__("__gnpost", self._grad_norm()))
        grad_norm_post = globals().pop("__gnpost", None)
        self._maybe(lambda: self._log_optim_step(loss, grad_norm_pre, grad_norm_post))

        # Soft target update
        if not self.use_hard_target_updates:
            self._soft_update(tau=self.target_soft_tau)

        # Param sanity check (rare)
        def _param_sanity():
            for name, p in self.policy_model.named_parameters():
                if torch.isnan(p).any() or torch.isinf(p).any():
                    self.logger.info(f"[WARN] non-finite param in {name} @ t={self.t}")
                    break

        self._maybe(_param_sanity)

        return float(loss.item())

    # ---------- Q(s, β) online for stored ws ----------
    def _online_q_for_ws(self, states: torch.Tensor, ws: torch.Tensor):
        """
        Returns: q_sa (B,1), (A,b,c) for logging
        """
        self.policy_model.train()
        abc = self.policy_model(states)
        A, b, c = abc["A_diag"], abc["b"], abc["c"]  # (B,J,N), (B,J,N), (B,J)
        q_vals = self.policy_model.compute_q_values(ws, A, b, c)  # (B,J)
        return q_vals, A, b, c

    # ---------- Double DQN target ----------
    @torch.no_grad()
    def _double_dqn_target(self, next_states: torch.Tensor) -> torch.Tensor:
        abc_no = self.policy_model(next_states)
        A_do, b_o, c_o = abc_no["A_diag"], abc_no["b"], abc_no["c"]
        w_star_o = self.policy_model.compute_w_star(A_do, b_o)
        q_next_online = self.policy_model.compute_q_values(w_star_o, A_do, b_o, c_o)
        next_beta_idx = q_next_online.argmax(dim=1, keepdim=True)  # (B,1)

        abc_nt = self.target_model(next_states)
        A_dt, b_t, c_t = abc_nt["A_diag"], abc_nt["b"], abc_nt["c"]
        w_star_t = self.target_model.compute_w_star(A_dt, b_t)
        q_next_tgt = self.target_model.compute_q_values(w_star_t, A_dt, b_t, c_t)

        if self._should_log():
            try:
                q_on_sel = q_next_online.gather(1, next_beta_idx)  # (B,1)
                q_tg_sel = q_next_tgt.gather(1, next_beta_idx)     # (B,1)
                q_gap = (q_on_sel - q_tg_sel).mean().item()
                self._log_info(f"target_gap@t={self.t} | mean(Q_on(s',β*)-Q_tg(s',β*))={q_gap:.3g}")
            except Exception as e:
                self._log_debug(f"[gap-log-skip] {e}")
            
        return q_next_tgt.gather(1, next_beta_idx)  # (B,1)

    # ---------- Batch stat logging ----------
    def _log_learn_batch_stats(self, q_sa, target, A_diag, b, c, ws, beta_indices):
        with torch.no_grad():
            td = target - q_sa                  # (B,1)
            td_abs = td.abs()
            td_m  = td.mean().item()
            td_p95 = robust_quantile(td_abs, 0.95).item()
            td_p50 = robust_quantile(td_abs, 0.50).item()
            td_sign = (td > 0).float().mean().item()

            qsa_m = q_sa.mean().item()
            tgt_m = target.mean().item()
            q_scale = q_sa.abs().mean().item()

            A_min_v = A_diag.min().item()
            A_max_v = A_diag.max().item()
            b_mean = b.abs().mean().item()
            c_mean = c.abs().mean().item()

            # clamp percentage: fraction of A at its configured floor
            clamp_pct = None
            try:
                A_min_cfg = float(getattr(self.policy_model, "A_min", None))
                if not np.isnan(A_min_cfg):
                    eps = 1e-8
                    clamp_pct = ((A_diag <= (A_min_cfg + eps)).float().mean()).item()
            except Exception:
                clamp_pct = None

            # Select per-sample chosen β row for ws, A, b, c
            B, J, N = ws.shape
            beta_idx_exp = beta_indices.view(-1, 1, 1).expand(-1, 1, N)  # (B,1,N)
            ws_sel = ws.gather(1, beta_idx_exp).squeeze(1)               # (B,N)
            A_sel  = A_diag.gather(1, beta_idx_exp).squeeze(1)           # (B,N)
            b_sel  = b.gather(1, beta_idx_exp).squeeze(1)                # (B,N)
            c_sel  = c.gather(1, beta_indices.view(-1,1)).squeeze(1)     # (B,)

            quad = 0.5 * (A_sel * (ws_sel ** 2)).sum(dim=1)              # (B,)
            lin  = (b_sel * ws_sel).sum(dim=1)                           # (B,)
            q_decomp_mean = (c_sel - quad + lin).mean().item()
            quad_m = quad.mean().item()
            lin_m  = lin.mean().item()

            # value at w* for the chosen β (diagnostic only)
            q_star = c_sel + 0.5 * ((b_sel ** 2) / (A_sel + 1e-6)).sum(dim=1)
            q_star_m = q_star.mean().item()

        self._es_update(td_p95=td_p95, clamp_pct=(0.0 if clamp_pct is None else clamp_pct))

        self._log_info(
            f"learn@t={self.t} | q_sa={qsa_m:.3g} | tgt={tgt_m:.3g} | |q|_mean={q_scale:.3g} "
            f"| td={td_m:.3g} | |td|_p50={td_p50:.3g} | |td|_p95={td_p95:.3g} | td>0={td_sign:.2f} "
            f"| A[min,max]=[{A_min_v:.3g},{A_max_v:.3g}] | clamp_pct={('NA' if clamp_pct is None else f'{clamp_pct:.2f}')} "
            f"| |b|_mean={b_mean:.3g} | |c|_mean={c_mean:.3g} "
            f"| q_decomp_mean={q_decomp_mean:.3g} | quad_m={quad_m:.3g} | lin_m={lin_m:.3g} | q*_m={q_star_m:.3g}"
        )
        
    # ---------- Grad norm + optim log ----------
    def _grad_norm(self) -> float:
        total_sq = 0.0
        for p in self.policy_model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                total_sq += g.pow(2).sum().item()
        return total_sq**0.5

    def _log_optim_step(self, loss, grad_norm_pre, grad_norm_post):
        try:
            lr_val = self.optimizer.param_groups[0].get("lr", None)
        except Exception:
            lr_val = None
        try:
            with torch.no_grad():
                s = 0.0
                for p in self.policy_model.parameters():
                    s += p.data.pow(2).sum().item()
                param_norm = s**0.5
        except Exception:
            param_norm = float("nan")
        self._log_info(
            "optim@t=%d | loss=%.4g | grad||pre=%.3g | grad||post=%.3g | param||=%.3g | lr=%s"
            % (
                self.t,
                float(loss.item()),
                (grad_norm_pre if grad_norm_pre is not None else float("nan")),
                (grad_norm_post if grad_norm_post is not None else float("nan")),
                (param_norm if param_norm is not None else float("nan")),
                (f"{lr_val:.3g}" if isinstance(lr_val, float) else str(lr_val)),
            )
        )

    def _assert_ws_contract(self, ws: torch.Tensor, beta_indices: torch.Tensor):
        B, J, N = ws.shape
        row_sums = ws.abs().sum(dim=2)  # (B, J)
        nonzero_rows = (row_sums > 1e-12).sum(dim=1)  # (B,)
        if not torch.all(nonzero_rows == 1):
            bad_idx = (nonzero_rows != 1).nonzero(as_tuple=False).flatten()
            msg = f"[ws-contract] expected exactly one nonzero β row, got counts={nonzero_rows[bad_idx[:5]].tolist()} ..."
            self.logger.error(msg)
            raise AssertionError(msg)

        chosen_row_sum = row_sums.gather(1, beta_indices.view(-1, 1)).squeeze(1)
        if not torch.all(chosen_row_sum > 1e-12):
            msg = "[ws-contract] chosen β row appears zero for some samples."
            self.logger.error(msg)
            raise AssertionError(msg)

    def _warn_if_ws_looks_probabilities(
        self, ws: torch.Tensor, beta_indices: torch.Tensor
    ):
        with torch.no_grad():
            B, J, N = ws.shape
            rows = ws.gather(1, beta_indices.view(-1, 1, 1).expand(-1, 1, N)).squeeze(
                1
            )  # (B,N)
            rowsums = rows.sum(dim=1)
            nonneg = (rows >= -1e-8).all(dim=1)
            looks_prob = (
                torch.isclose(rowsums, torch.ones_like(rowsums), rtol=1e-2, atol=1e-2)
                & nonneg
            )
            if looks_prob.any():
                self._log_warn(
                    f"[SANITY] ws chosen rows look probability-like; expected logits."
                )

    def train(self, train_epochs: int) -> True:
        """The main call for the training loop of the DQN Agent.

        Args:
            train_epochs (int): Represent the number of epochs we want to train for.
                            Note: if the training is resumed, then the number of training epochs that will be done is
                            as many as is needed to reach the train_epochs number.
        """
        if not self.training_stats:
            self.logger.info(f"Starting training session at: {self.t}")
        else:
            epochs_left_to_train = train_epochs - len(self.training_stats)
            self.logger.info(
                f"Resuming training session at: {self.t} ({epochs_left_to_train} epochs left)"
            )
            train_epochs = epochs_left_to_train

        for epoch in range(train_epochs):
            start_time = datetime.datetime.now()

            ep_train_stats = self.train_epoch()
            self.display_training_epoch_info(ep_train_stats)
            self.training_stats.append(ep_train_stats)

            if self.validation_enabled:
                ep_validation_stats = self.validate_epoch()
                self.display_validation_epoch_info(ep_validation_stats)
                self.validation_stats.append(ep_validation_stats)

            if self.save_checkpoints:
                self.save_checkpoint()

            end_time = datetime.datetime.now()
            epoch_time = end_time - start_time

            self.logger.info(f"Epoch {epoch} completed in {epoch_time}")
            self.logger.info("\n")

        self.logger.info(
            f"Ended training session after {train_epochs} epochs at t = {self.t}"
        )

        return True

    def train_epoch(self) -> Dict:
        """Do a single training epoch.

        Returns:
            Dict: dictionary containing the statistics of the training epoch.
        """
        self.logger.info(f"Starting training epoch at t = {self.t}")
        epoch_t = 0
        policy_trained_times = 0
        target_trained_times = 0

        epoch_episode_rewards = []
        epoch_episode_discounted_rewards = []
        epoch_episode_nr_frames = []
        epoch_losses = []
        epoch_max_qs = []

        start_time = datetime.datetime.now()
        while epoch_t < self.train_step_cnt:
            (
                is_terminated,
                truncated,
                epoch_t,
                current_episode_reward,
                current_episode_discounted_reward,
                ep_frames,
                ep_policy_trained_times,
                ep_target_trained_times,
                ep_losses,
                ep_max_qs,
            ) = self.train_episode(epoch_t, self.train_step_cnt)

            policy_trained_times += ep_policy_trained_times
            target_trained_times += ep_target_trained_times

            if is_terminated or truncated:
                # we only want to append these stats if the episode was completed,
                # otherwise it means it was stopped due to the agent nr of frames criterion
                epoch_episode_rewards.append(current_episode_reward)
                epoch_episode_discounted_rewards.append(
                    current_episode_discounted_reward
                )
                epoch_episode_nr_frames.append(ep_frames)
                epoch_losses.extend(ep_losses)
                epoch_max_qs.extend(ep_max_qs)

                self.episodes += 1
                self.reset_training_episode_tracker()

        end_time = datetime.datetime.now()
        epoch_time = end_time - start_time

        epoch_stats = self.compute_training_epoch_stats(
            epoch_episode_rewards,
            epoch_episode_discounted_rewards,
            epoch_episode_nr_frames,
            policy_trained_times,
            target_trained_times,
            epoch_losses,
            epoch_max_qs,
            epoch_time,
        )
        
        if self.lr_scheduler is not None:
            try:
                lr_now = self.optimizer.param_groups[0]["lr"]
                self.logger.info(f"[LR] end-of-epoch lr={lr_now:.6g}")
            except Exception:
                pass

        return epoch_stats

    # ---------- Minimal logging helpers (no-ops when not due to log) ----------
    def _log_enabled(self) -> bool:
        return self._should_log()

    def _log_info(self, msg: str) -> None:
        if self._log_enabled():
            self.logger.info(msg)

    def _log_debug(self, msg: str) -> None:
        if self._log_enabled():
            self.logger.debug(msg)

    def _log_warn(self, msg: str) -> None:
        if self._log_enabled():
            self.logger.warning(msg)

    def _maybe(self, fn) -> None:
        """Run fn() only on log steps; swallow exceptions to avoid training stalls."""
        if not self._log_enabled():
            return
        try:
            fn()
        except Exception as e:
            self.logger.debug(f"[log-skip] {e}")

    def train_episode(self, epoch_t: int, train_frames: int):
        """Do a single training episode.

        Args:
            epoch_t (int): The total number of frames seen in this epoch, relevant for early stopping of
                            the training episode.
            train_frames (int): How many frames we want to limit the training epoch to

        Returns:
            Tuple[bool, int, float, int, int, int, list, list]: Information relevant to this training episode. Some variables are stored in
                                                            the class so that the training episode can resume in the following epoch.
        """
        policy_trained_times = 0
        target_trained_times = 0

        is_terminated = False
        truncated = False
        while (not is_terminated) and (not truncated) and (epoch_t < train_frames):
            self.logger.debug(f"State (s) shape before step: {self.train_env_s.shape}")

            action, beta_idx, w, max_q = self.select_action(
                torch.tensor(self.train_env_s, dtype=torch.float32),
                epsilon=self.epsilon_by_frame(self.t),
                action_noise=True,
            )
            action = np.asarray(action, dtype=np.float32)  # (1, N)
            action = action[0]  # -> (N,) even when N==1

            if self._should_log():
                try:
                    # action is (N,)
                    act = torch.tensor(action, dtype=torch.float32)
                    frac_over_cap = (act > 0.4).float().mean().item()
                    topk_vals, _ = torch.topk(act, k=min(3, act.numel()))
                    # Pull chosen w "logits" for entropy proxy
                    bidx = int(np.asarray(beta_idx).reshape(-1)[0])
                    w_tensor = torch.tensor(w, dtype=torch.float32)  # (1, J, N)
                    w_chosen = w_tensor[0, bidx, :]  # (N,)
                    with torch.no_grad():
                        p = torch.softmax(w_chosen, dim=-1)
                        H = -(p * p.clamp_min(1e-8).log()).sum().item()
                    self.logger.info(
                        f"t={self.t} | eps={self.epsilon_by_frame(self.t):.4f} | beta_idx={bidx} "
                        f"| action_entropy={H:.3f} | frac_u>0.4={frac_over_cap:.3f} "
                        f"| top3_u={topk_vals.tolist()} | max_q={max_q:.3g}"
                    )

                    # use declared env cap if available, else 0.4 fallback
                    cap = float(getattr(self.train_env, "max_u", 0.4))
                    frac_over_cap_env = (act > cap).float().mean().item()
                    overshoot_L1 = (act - cap).clamp_min(0).sum().item()

                    # logit distribution diagnostics (before softmax*beta)
                    H_w = H  # entropy of softmax(w) already computed as H
                    HHI = (p**2).sum().item()  # 1 means all mass on one node
                    w_l2 = w_chosen.norm(2).item()
                    p_max = p.max().item()
                    argmax_u = int(torch.argmax(act).item())

                    self.logger.info(
                        f"t={self.t} | cap={cap:.3g} | frac_u>cap={frac_over_cap_env:.3f} | overshoot_L1={overshoot_L1:.3g} "
                        f"| HHI={HHI:.3f} | H_w={H_w:.3f} | ||w||2={w_l2:.3g} | p_max={p_max:.3f} | argmax_u={argmax_u}"
                    )
                    # feed early-stop windows (entropy already fed); track cap usage as well
                    self._es_update(frac_cap=frac_over_cap_env)

                    self._last_entropy = H
                    self._last_frac_over_cap = frac_over_cap
                    self._es_update(H=H, frac_cap=frac_over_cap, max_q=max_q)

                    try:
                        self._early_stop_maybe()
                    except EarlyStop:
                        raise
                    except Exception as e:
                        self.logger.debug(f"[es-check-skip] {e}")

                except Exception as e:
                    self.logger.debug(f"[act-log-skip] {e}")

            self.logger.debug(f"State shape: {self.train_env_s.shape}")
            self.logger.debug(f"Action shape: {action.shape}")
            self.logger.debug(f"Beta index: {beta_idx}")

            if self._should_log():
                if not np.isfinite(action).all():
                    self.logger.info(
                        f"[WARN] non-finite action at t={self.t}: {action}"
                    )

            assert action.ndim == 1, f"expected (N,), got {action.shape}"
            if not np.isfinite(action).all():
                self.logger.error(
                    f"[EARLY STOP] Non-finite action at t={self.t}: {action}"
                )
                raise EarlyStop("Non-finite action encountered.")
            s_prime, reward, is_terminated, truncated, info = self.train_env.step(
                action
            )

            self.logger.debug(f"State (s') shape after step: {s_prime.shape}")

            w = np.asarray(w, dtype=np.float32)
            if w.ndim == 3 and w.shape[0] == 1:  # actor returns (1, J, N)
                w = w[0]  # -> (J, N)
            assert w.ndim == 2 and w.shape == (
                len(self.betas),
                self.num_actions,
            ), f"expected w=(J,N)=({len(self.betas)},{self.num_actions}), got {w.shape}"

            beta_idx = int(np.asarray(beta_idx).reshape(-1)[0])
            done_flag = bool(is_terminated or truncated)

            if self._should_log():
                try:
                    w_t = torch.tensor(w, dtype=torch.float32)  # (J,N)
                    row_sums = w_t.sum(dim=1)
                    nonneg = (w_t >= -1e-8).all().item()
                    looks_prob = (
                        torch.allclose(
                            row_sums, torch.ones_like(row_sums), rtol=1e-2, atol=1e-2
                        )
                        and nonneg
                    )
                    if looks_prob:
                        self.logger.warning(
                            "[RB-SANITY] 'w' looks like normalized probabilities (u) instead of logits. "
                            "Learning expects pre-softmax logits."
                        )
                except Exception:
                    pass

            self.replay_buffer.append(
                self.train_env_s, (beta_idx, w), reward, s_prime, done_flag
            )
            self.max_qs.append(max_q)

            if self._should_log():
                try:
                    self.logger.info(
                        f"buffer@t={self.t} | size={len(self.replay_buffer)} | batch={self.batch_size} "
                        f"| train_freq={self.training_freq} | gamma={self.gamma}"
                    )
                except Exception as e:
                    self.logger.debug(f"[buffer-log-skip] {e}")

            # Train policy model
            if (
                self.t > self.replay_start_size
                and len(self.replay_buffer) >= self.batch_size
            ):
                if self.t % self.training_freq == 0:
                    sample = self.replay_buffer.sample(self.batch_size)
                    loss_val = self.model_learn(sample)
                    self.losses.append(loss_val)
                    self.policy_model_update_counter += 1
                    policy_trained_times += 1

                if self.use_hard_target_updates:
                    if (
                        self.policy_model_update_counter > 0
                        and (self.policy_model_update_counter % self.hard_target_update_every) == 0
                    ):
                        self.target_model.load_state_dict(self.policy_model.state_dict())
                        target_trained_times += 1
                        if self._should_log():
                            self.logger.info(f"[HARD-TGT] Synced target at policy_update={self.policy_model_update_counter}")

            self.current_episode_reward += reward
            self.current_episode_discounted_reward += self.discount_factor * reward
            self.discount_factor *= self.gamma
            self.t += 1
            epoch_t += 1
            self.ep_frames += 1

            self.train_env_s = s_prime

        return (
            is_terminated,
            truncated,
            epoch_t,
            self.current_episode_reward,
            self.current_episode_discounted_reward,
            self.ep_frames,
            policy_trained_times,
            target_trained_times,
            self.losses,
            self.max_qs,
        )

    def reset_training_episode_tracker(self):
        """Resets the environment and the variables that keep track of the training episode."""
        self.current_episode_reward = 0.0
        self.current_episode_discounted_reward = 0.0
        self.discount_factor = 1.0

        self.ep_frames = 0
        self.losses = []
        self.max_qs = []

        self.train_env_s, _ = self.train_env.reset(randomize_opinions=True)

    def display_training_epoch_info(self, stats):
        extra = (
            f" | Entropy(last)={None if self._last_entropy is None else round(self._last_entropy,3)}"
            f" | frac_u>0.4(last)={None if self._last_frac_over_cap is None else round(self._last_frac_over_cap,3)}"
            f" | tgt_drift(last)={None if self._last_rel_target_drift is None else f'{self._last_rel_target_drift:.2e}'}"
        )

        self.logger.info(
            "TRAINING STATS"
            + " | Frames seen: "
            + str(self.t)
            + " | Episode: "
            + str(self.episodes)
            + " | Max reward: "
            + str(stats["episode_rewards"]["max"])
            + " | Avg reward: "
            + str(stats["episode_rewards"]["mean"])
            + " | Avg frames (episode): "
            + str(stats["episode_frames"]["mean"])
            + " | Avg max Q: "
            + str(stats["episode_max_qs"]["mean"])
            + " | Epsilon: "
            + str(self.epsilon_by_frame(self.t))
            + " | Train epoch time: "
            + str(stats["epoch_time"])
            + extra
        )

    def compute_training_epoch_stats(
        self,
        episode_rewards,
        episode_discounted_rewards,
        episode_nr_frames,
        policy_trained_times,
        target_trained_times,
        ep_losses,
        ep_max_qs,
        epoch_time,
    ) -> Dict:
        """Computes the statistics of the current training epoch.

        Args:
            episode_rewards (List): list contraining the final reward of each episode in the current epoch.
            episode_discounted_rewards (List): list contraining the final discounted reward of each episode in the current epoch.
            episode_nr_frames (List): list contraining the final number of frames of each episode in the current epoch.
            policy_trained_times (int): Number representing how many times the policy network was updated.
            target_trained_times (int): Number representing how many times the target network was updated.
            ep_losses (List): list contraining losses from the current epoch.
            ep_max_qs (List): list contraining maximum Q values from the current epoch.
            epoch_time (float): How much time the epoch took to compute in seconds.

        Returns:
            Dict: Dictionary with the relevant statistics
        """
        stats = {}

        stats["frame_stamp"] = self.t
        stats["greedy_epsilon"] = self.epsilon_by_frame(self.t)

        stats["episode_rewards"] = self.get_vector_stats(episode_rewards)
        stats["episode_discounted_rewards"] = self.get_vector_stats(
            episode_discounted_rewards
        )
        stats["episode_frames"] = self.get_vector_stats(episode_nr_frames)
        stats["episode_losses"] = self.get_vector_stats(ep_losses)
        stats["episode_max_qs"] = self.get_vector_stats(ep_max_qs)

        stats["policy_trained_times"] = policy_trained_times
        stats["target_trained_times"] = target_trained_times
        stats["epoch_time"] = epoch_time

        return stats

    def get_vector_stats(self, vector):
        """Compute statistics for a list of values, handling None gracefully."""
        stats = {}

        # Filter out None values
        clean_vector = [v for v in vector if v is not None]

        if len(clean_vector) > 0:
            stats["min"] = np.nanmin(clean_vector)
            stats["max"] = np.nanmax(clean_vector)
            stats["mean"] = np.nanmean(clean_vector)
            stats["median"] = np.nanmedian(clean_vector)
            stats["std"] = np.nanstd(clean_vector)
        else:
            stats["min"] = None
            stats["max"] = None
            stats["mean"] = None
            stats["median"] = None
            stats["std"] = None

        return stats

    def validate_epoch(self):
        self.logger.info(f"Starting validation epoch at t = {self.t}")

        epoch_episode_rewards = []
        epoch_episode_discounted_rewards = []
        epoch_episode_nr_frames = []
        epoch_max_qs = []
        validation_t = 0

        start_time = datetime.datetime.now()

        while validation_t < self.validation_step_cnt:
            (
                current_episode_reward,
                current_episode_discounted_reward,
                ep_frames,
                ep_max_qs,
            ) = self.validate_episode()

            validation_t += ep_frames

            epoch_episode_rewards.append(current_episode_reward)
            epoch_episode_discounted_rewards.append(current_episode_discounted_reward)
            epoch_episode_nr_frames.append(ep_frames)
            epoch_max_qs.extend(ep_max_qs)

        end_time = datetime.datetime.now()
        epoch_time = end_time - start_time

        epoch_stats = self.compute_validation_epoch_stats(
            epoch_episode_rewards,
            epoch_episode_discounted_rewards,
            epoch_episode_nr_frames,
            epoch_max_qs,
            epoch_time,
        )
        return epoch_stats

    def compute_validation_epoch_stats(
        self,
        episode_rewards,
        episode_discounted_rewards,
        episode_nr_frames,
        ep_max_qs,
        epoch_time,
    ) -> Dict:
        """Computes the statistics of the current validation epoch.

        Args:
            episode_rewards (List): list contraining the final reward of each episode in the current epoch.
            episode_discounted_rewards (List): list contraining the final discounted reward of each episode in the current epoch.
            episode_nr_frames (List): list contraining the final number of frames of each episode in the current epoch.
            ep_max_qs (List): list contraining maximum Q values from the current epoch.
            epoch_time (float): How much time the epoch took to compute in seconds.

        Returns:
            Dict: Dictionary with the relevant statistics
        """
        stats = {}

        stats["frame_stamp"] = self.t

        stats["episode_rewards"] = self.get_vector_stats(episode_rewards)
        stats["episode_discounted_rewards"] = self.get_vector_stats(
            episode_discounted_rewards
        )
        stats["episode_frames"] = self.get_vector_stats(episode_nr_frames)
        stats["episode_max_qs"] = self.get_vector_stats(ep_max_qs)
        stats["epoch_time"] = epoch_time

        return stats

    def validate_episode(self):
        """Do a single validation episode.

        Returns:
            Tuple[int, int, List, Dict]: Tuple parameters relevant to the validation episode.
                                    The first element is the cumulative reward of the episode.
                                    The second element is the number of frames that were part of the episode.
                                    The third element is a list of the maximum Q values seen.
                                    The fourth element is a dictionary containing the number of times each reward was seen.
        """
        current_episode_reward = 0.0
        current_episode_discounted_reward = 0.0
        discount_factor = 1.0
        ep_frames = 0
        max_qs = []

        # Remake the env because we cycle through setups
        self.validation_env = self._make_validation_env()
        s, info = self.validation_env.reset()
        s = torch.tensor(s, device=device).float()

        is_terminated = False
        truncated = False
        while (
            (not is_terminated)
            and (not truncated)
            and (ep_frames < self.validation_step_cnt)
        ):
            action, betas, w, max_q = self.select_action(
                torch.tensor(s, dtype=torch.float32),
                epsilon=self.validation_epsilon,
                action_noise=False,
            )
            action = np.squeeze(action)
            if not np.isfinite(action).all():
                self.logger.error(
                    f"[EARLY STOP] Non-finite action at t={self.t}: {action}"
                )
                raise EarlyStop("Non-finite action encountered.")
            s_prime, reward, is_terminated, truncated, info = self.validation_env.step(
                action
            )
            s_prime = torch.tensor(s_prime, device=device).float()

            max_qs.append(max_q)
            current_episode_reward += reward
            current_episode_discounted_reward += discount_factor * reward
            discount_factor *= self.gamma
            ep_frames += 1
            s = s_prime

        return (
            current_episode_reward,
            current_episode_discounted_reward,
            ep_frames,
            max_qs,
        )

    def display_validation_epoch_info(self, stats):
        self.logger.info(
            "VALIDATION STATS"
            + " | Max reward: "
            + str(stats["episode_rewards"]["max"])
            + " | Avg reward: "
            + str(stats["episode_rewards"]["mean"])
            + " | Avg frames (episode): "
            + str(stats["episode_frames"]["mean"])
            + " | Avg max Q: "
            + str(stats["episode_max_qs"]["mean"])
            + " | Validation epoch time: "
            + str(stats["epoch_time"])
        )


def main():
    pass


if __name__ == "__main__":
    seed_everything(0)
    main()
    # play_game_visual("breakout")
