import numpy as np

from rl_envs_forge.envs.network_graph.network_graph import NetworkGraph


class EnvironmentFactory:
    def __init__(self):
        """
        Initializes the environment factory with a base configuration.
        """
        self.supported_dynamics = [
            "laplacian",
            "degroot",
            "coca",
            "friedkinjohnsen",
            "hegselmannkrause",
            "nonlinearinfluence",
            "repulsion",
        ]

        self.default_experiment_dynamics = [
            "laplacian",
            "coca",
            "friedkinjohnsen",
            "hegselmannkrause",
            "nonlinearinfluence",
            # "repulsion",   # keep disabled because it does not converge too easily 
        ]

        self.base_config = {
            "num_agents": 15,
            "graph_model": "barabasi_albert",
            "ba_m": 2,
            "ba_prune_max_frac": 0.5,
            "ba_qsc_tol": 1e-8,
            "ba_max_tries": 500,
            "max_u": 0.2,
            "budget": 1000.0,
            "desired_opinion": 1,
            "t_campaign": 2,
            "t_s": 0.5,
            "max_steps": 3000,
            "opinion_end_tolerance": 0.05,
            "control_beta": 0.4,
            "normalize_reward": True,
            "terminal_reward": 0.5,
            "seed": 2,
            "terminate_when_converged": True,
            "dynamics_model": "laplacian",
            # generic defaults
            "fj_lambda": 0.99,
            "fj_prejudice": None,
            "hk_epsilon": 0.20,
            "hk_include_self": True,
            "nonlinear_beta": 2.0,
            "repulsion_epsilon": 0.20,
            "repulsion_strength": 0.5,
        }

        # Per-dynamics overrides discovered in development.
        self.model_specs = {
            "laplacian": {},
            "degroot": {},
            "coca": {},
            "friedkinjohnsen": {
                "fj_lambda": 0.99,
                "fj_prejudice": None,
            },
            "hegselmannkrause": {
                "hk_epsilon": 0.20,
                "hk_include_self": True,
            },
            "nonlinearinfluence": {
                "nonlinear_beta": 2.0,
            },
            "repulsion": {
                "repulsion_epsilon": 0.20,
                "repulsion_strength": 0.5,
            },
        }

        self.use_centrality_resistance = False
        self.validation_versions = [0]

    def _build_config(self, seed: int = None, dynamics_model: str = None):
        config = self.base_config.copy()

        if seed is not None:
            config["seed"] = seed

        dyn = dynamics_model if dynamics_model is not None else config["dynamics_model"]
        if dyn not in self.supported_dynamics:
            raise ValueError(f"Unknown dynamics_model: {dyn}")

        config["dynamics_model"] = dyn
        config.update(self.model_specs.get(dyn, {}))

        num_agents = config["num_agents"]
        config["initial_opinions"] = np.linspace(0.01, 0.99, num_agents)

        return config

    def get_randomized_env(self, seed: int = None, dynamics_model: str = None):
        config = self._build_config(seed=seed, dynamics_model=dynamics_model)

        env = NetworkGraph(**config)
        if self.use_centrality_resistance:
            self.apply_centrality_based_control_resistance(env)
        return env

    def get_validation_env(self, version: int = 0, dynamics_model: str = None):
        """Returns a validation environment with controlled variation by version."""
        config = self.base_config.copy()

        dyn = dynamics_model if dynamics_model is not None else config["dynamics_model"]
        if dyn not in self.supported_dynamics:
            raise ValueError(f"Unknown dynamics_model: {dyn}")

        config["dynamics_model"] = dyn
        config.update(self.model_specs.get(dyn, {}))

        num_agents = config["num_agents"]

        if version == 0:
            config["initial_opinions"] = np.linspace(0.01, 0.99, num_agents)
        else:
            raise ValueError(f"Unknown validation version: {version}")

        env = NetworkGraph(**config)
        if self.use_centrality_resistance:
            self.apply_centrality_based_control_resistance(env)
        return env

    def apply_centrality_based_control_resistance(
        self,
        env: NetworkGraph,
        low: float = 0.0,
        high: float = 0.9,
    ) -> NetworkGraph:
        """
        Linearly maps node centralities to control_resistance in [low, high].
        Lowest centrality -> low, highest centrality -> high.
        """
        c = np.asarray(env.centralities, dtype=float)
        c_min, c_max = float(c.min()), float(c.max())

        if c_max - c_min < 1e-12:
            scaled = np.full_like(c, 0.5)
        else:
            scaled = (c - c_min) / (c_max - c_min)

        env.control_resistance = low + scaled * (high - low)
        return env