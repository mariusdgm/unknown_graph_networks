import numpy as np

from rl_envs_forge.envs.network_graph.network_graph import NetworkGraph


class EnvironmentFactory:
    def __init__(self):
        """
        Initializes the environment factory with a base configuration.
        """
        self.base_config = {
            "num_agents": 15,
            "graph_model": "barabasi_albert",  # (alias: "albert_barabasi")
            "ba_m": 2,  # paper: AP=2
            "ba_prune_max_frac": 0.5,  # paper: prune randomly up to half the edges
            "ba_qsc_tol": 1e-8,  # eigenvalue tolerance for the "single 0 eig" check
            "ba_max_tries": 500,  # more tries reduces chance of fallback warning
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
            "dynamics_model": "laplacian",  # or "coca"
        }

        self.use_centrality_resistance = False
        self.validation_versions = [0]

    def get_randomized_env(self, seed: int = None):
        """Returns a training environment with randomized opinions and optional seed."""
        config = self.base_config.copy()
        if seed is not None:
            config["seed"] = seed

        num_agents = config["num_agents"]
        initial_opinions = np.linspace(0.01, 0.99, num_agents)
        config["initial_opinions"] = initial_opinions

        env = NetworkGraph(**config)
        if self.use_centrality_resistance:
            self.apply_centrality_based_control_resistance(env)
        return env

    def get_validation_env(self, version: int = 0):
        """Returns a validation environment with controlled variation by version."""
        config = self.base_config.copy()
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
        low: float = 0.0,  # M (assigned to lowest centrality)
        high: float = 0.9,  # N (assigned to highest centrality)
    ) -> NetworkGraph:
        """
        Linearly maps node centralities to control_resistance in [low, high].
        Lowest centrality -> low (M), highest centrality -> high (N).
        """
        c = np.asarray(env.centralities, dtype=float)
        c_min, c_max = float(c.min()), float(c.max())

        if c_max - c_min < 1e-12:
            # All nodes have (almost) identical centrality.
            # Put them in the middle of [low, high].
            scaled = np.full_like(c, 0.5)
        else:
            scaled = (c - c_min) / (c_max - c_min)  # in [0,1]

        env.control_resistance = low + scaled * (high - low)
        return env


# Prefer the EnvironmentFactory as single source of truth
# def build_environment(random_initial_opinions=False):

#     num_agents = 20

#     if random_initial_opinions:
#         initial_opinions = np.random.uniform(low=0.1, high=0.99, size=num_agents)

#     else:
#         initial_opinions = np.linspace(0.3, 0.1, num_agents)

#     env = NetworkGraph(
#         num_agents=num_agents,
#         initial_opinions=initial_opinions,
#         max_u=0.4,
#         budget=1000.0,
#         desired_opinion=1,
#         t_campaign=1,
#         t_s=0.1,
#         connection_prob_range=(0.05, 0.1),
#         bidirectional_prob=0.1,
#         max_steps=50,
#         opinion_end_tolerance=0.05,
#         control_beta=0.4,
#         normalize_reward=True,
#         terminal_reward=0.5,
#         seed=42,
#         terminate_when_converged=False,
#         dynamics_model="coca",
#         # dynamics_model="laplacian",
#     )

#     env.reset()
#     return env

# Small env, premade links
# def build_environment(random_initial_opinions=False):
#     links = [
#         (1, 3),
#         (3, 2),
#         (2, 3),
#         (2, 0),
#         (0, 2),
#         (1, 2),
#         (0, 1),
#         # (3, 4),
#         # (4, 3)
#     ]

#     num_agents = 4
#     connectivity_matrix = create_adjacency_matrix_from_links(num_nodes, links)
#     # connectivity_matrix = normalize_adjacency_matrix(connectivity_matrix)

#     if random_initial_opinions:
#         initial_opinions = np.random.uniform(low=0.0, high=1.0, size=num_nodes)

#     else:
#         initial_opinions = np.linspace(0.3, 0, num_agents)

#     # initial_opinions = np.linspace(0, 1, num_agents)

#     # initial_opinions = (np.mod(np.arange(0, 0.1 * num_agents, 0.1), 0.9)) + 0.1

#     env = NetworkGraph(
#         connectivity_matrix=connectivity_matrix,
#         initial_opinions=initial_opinions,
#         max_u=0.4,
#         budget=1000.0,
#         desired_opinion=1,
#         tau=0.1,
#         max_steps=50,
#         opinion_end_tolerance=0.05,
#         control_beta=0.4,
#         normalize_reward=True,
#         terminal_reward=0.5
#     )

#     env.reset()

#     return env
