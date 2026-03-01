import os, sys


def get_dir_n_levels_up(path, n):
    # Go up n levels from the given path
    for _ in range(n):
        path = os.path.dirname(path)
    return path


proj_root = get_dir_n_levels_up(os.path.abspath("__file__"), 4)
sys.path.append(proj_root)

import torch
import yaml
import datetime
import collections
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict

from opinion_dynamics.opinion_dqn import AgentDQN
from opinion_dynamics.utils.my_logging import setup_logger


def create_path_to_experiment_folder(
    config: Dict,
    experiments_output_folder: str,
    timestamp_folder: str = None,
) -> str:
    """Build the path for the nested experiment structure:
    base_outputs / timestamp / experiment / environment / seed

    Args:
        config (Dict): Configuration of the experiment.
        experiments_output_folder (str): Root path for the folder where the outputs
                                        of paralelized experiments are stored.
        timestamp_folder (str, optional): Path to the previous top level output folder. If None, then a new top level folder
                                        is created with a string matching the current time. Defaults to None.

    Returns:
        str: The path to the folder that stores the output for this singular experiment
    """
    experiment = config["experiment_name"]
    env = config["environment"]
    seed = config["seed"]

    prev_output_not_expected = True

    if timestamp_folder is None:
        timestamp_folder = datetime.datetime.now().strftime(r"%Y_%m_%d-%H_%M_%S")
        prev_output_not_expected = False  # disable creation of prev folder

    exp_folder_path = os.path.join(
        experiments_output_folder,
        timestamp_folder,
        experiment,
        env,
        str(seed),
    )

    if prev_output_not_expected:
        Path(exp_folder_path).mkdir(parents=True, exist_ok=True)

    return exp_folder_path


def process_experiment(root_dir):
    rows = []

    for name in os.listdir(root_dir):
        experiment_path = os.path.join(root_dir, name)
        if os.path.isdir(experiment_path):
            for seed_name in os.listdir(experiment_path):
                seed_path = os.path.join(experiment_path, seed_name)
                if os.path.isdir(seed_path):
                    row_data = process_subexperiment(
                        seed_path, os.path.basename(root_dir)
                    )
                    for data in row_data:
                        data["seed"] = seed_name
                        data["experiment_name"] = name
                        data["sub_experiment_path"] = seed_path
                    rows.extend(row_data)

    # Create a DataFrame from the rows
    df = pd.DataFrame(rows)
    return df


def process_subexperiment(seed_folder_path, experiment_name):
    cfg_data = read_config(os.path.join(seed_folder_path, "cfg.yaml"), experiment_name)
    cfg_data["sub_experiment_path"] = seed_folder_path
    train_stats_file = find_train_stats_file(seed_folder_path)
    if train_stats_file:
        experiment_results = process_training_stats(train_stats_file, cfg_data)
        return experiment_results
    else:
        return []  # Return an empty list if no train stats file is found


def read_config(cfg_path, experiment_name):
    with open(cfg_path, "r") as file:
        config = yaml.safe_load(file)
        full_title = config.get("full_title", "")
        variable_part = remove_experiment_name(full_title, experiment_name)
        return parse_config_variables(variable_part)


def remove_experiment_name(full_title, experiment_name):
    to_remove = experiment_name + "_"
    return (
        full_title[len(to_remove) :].strip()
        if full_title.startswith(to_remove)
        else full_title
    )


def parse_config_variables(variable_str):
    variables = {}
    for part in variable_str.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            key = f"sub_exp_cfg_{key.strip()}"  # Add prefix
            variables[key] = value.strip()
    return variables


def find_train_stats_file(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith("_train_stats"):
            return os.path.join(folder_path, file)
    return None


def process_training_stats(train_stats_file, cfg_data):

    checkpoint = torch.load(train_stats_file, weights_only=False)
    training_stats = checkpoint.get("training_stats", [])
    validation_stats = checkpoint.get("validation_stats", [])
    redo_stats = checkpoint.get("redo_scores", [])

    stats_records = process_stats(training_stats, cfg_data, "training") + process_stats(
        validation_stats, cfg_data, "validation"
    )

    # Combine stats records with redo scores
    combined_records = []
    for record in stats_records:
        combined_record = record.copy()  # Copy the stats record
        combined_records.append(combined_record)

    return combined_records


def process_stats(stats, cfg_data, stats_type):
    records = []
    for epoch_stats in stats:
        record = {"epoch_type": stats_type}
        record.update(flatten(epoch_stats))  # Flatten the epoch_stats if it's nested
        record.update(cfg_data)  # Add configuration data
        records.append(record)
    return records


def flatten(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def create_adjacency_matrix_from_links(num_nodes, links):
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for link in links:
        from_node, to_node = link
        # adjacency_matrix[from_node, to_node] = 1
        adjacency_matrix[to_node, from_node] = 1

    return adjacency_matrix


def instantiate_agent(exp_subdir_path: str, checkpoint: int) -> AgentDQN:
    """
    Instantiate an AgentDQN using the configuration stored in a YAML file
    in the provided experiment subdirectory. The agent is created with the
    given training and validation environments and loads its previous state.

    Args:
        exp_subdir_path (str): Path to the experiment subdirectory containing the config YAML and checkpoint files.
        checkpoint (int): The checkpoint number to load.

    Returns:
        AgentDQN: An instance of AgentDQN initialized using the experiment configuration and saved state.
    """
    # Assume the YAML configuration is stored as 'config.yaml' in the experiment folder.
    config_path = os.path.join(exp_subdir_path, "cfg.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Instantiate the agent.
    # The resume_training_path is set to the experiment folder so that the agent loads saved weights/stats.
    agent = AgentDQN(
        resume_training_path=exp_subdir_path,
        experiment_name=config["experiment"],
        config=config,
        save_checkpoints=False,  # you can set this as needed
        logger=setup_logger("dqn"),
    )
    if checkpoint is not None:
        agent.load_models_at(
            checkpoint=checkpoint, resume_training_path=exp_subdir_path
        )

    return agent
