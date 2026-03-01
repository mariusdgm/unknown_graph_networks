import os, sys
import logging

def get_dir_n_levels_up(path, n):
    # Go up n levels from the given path
    for _ in range(n):
        path = os.path.dirname(path)
    return path


proj_root = get_dir_n_levels_up(os.path.abspath(__file__), 2)
sys.path.append(proj_root)

import traceback
from typing import Dict

from liftoff import parse_opts


from opinion_dqn import AgentDQN
from utils import my_logging
from utils.experiment import create_path_to_experiment_folder
from utils.generic import convert_namespace_to_dict, seed_everything


def run(opts: Dict) -> True:
    """Start a training experiment using input configuration.

    Args:
        opts (NameSpace): Configuration to use in the experiment.

    Returns:
        bool: Returns True on experiment end.
    """

    try:
        config = convert_namespace_to_dict(opts)
        seed = int(os.path.basename(config["out_dir"]))

        seed_everything(seed)

        logs_file = os.path.join(config["out_dir"], "experiment_log.log")

        logger = my_logging.setup_logger(
            name=config["experiment"],
            log_file=logs_file,
            # level=logging.DEBUG,
        )

        logger.info(f"Starting experiment: {config['full_title']}")

        ### Setup output and loading paths ###

        path_previous_experiments_outputs = None
        if "restart_training_timestamp" in config:
            path_previous_experiments_outputs = create_path_to_experiment_folder(
                config,
                config["out_dir"],
                config["restart_training_timestamp"],
            )

        experiment_agent = AgentDQN(
            experiment_output_folder=config["out_dir"],
            experiment_name=config["experiment"],
            resume_training_path=path_previous_experiments_outputs,
            save_checkpoints=True,
            logger=logger,
            config=config
        )
        
        logger.info(
            f'Initialized agent with models: {experiment_agent.policy_model}'
        )

        experiment_agent.train(train_epochs=config["epochs_to_train"])

        logger.info(
            f'Finished training experiment: {config["full_title"]}, run_id: {config["run_id"]}'
        )

        my_logging.cleanup_file_handlers(experiment_logger=logger)

        return True

    except Exception as exc:
        logger.error("An error occurred: %s", exc)
        error_info = traceback.format_exc()
        logger.error("An error occurred: %s", error_info)
        return error_info

### Liftoff implementation
def main():
    opts = parse_opts()
    run(opts)

if __name__ == "__main__":
    main()
