from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from scripts.env.conf import AirHockeyChallengeEnv

import os
import yaml

def get_args():
    parser = ArgumentParser()
    arg_test = parser.add_argument_group("override parameters")

    env_choices = ["hit", "defend", "tournament"]

    arg_test.add_argument(
        "-e",
        "--env",
        nargs="+",
        choices=env_choices,
        help="Environments to be used.",
    )

    arg_test.add_argument(
        "--n_cores", type=int, help="Number of CPU cores used for evaluation."
    )

    arg_test.add_argument(
        "-n",
        "--n_episodes",
        type=int,
        help="Each seed will run for this number of Episodes.",
    )

    arg_test.add_argument(
        "--steps_per_game",
        type=int,
        help="Number of steps per game",
    )

    arg_test.add_argument(
        "--log_dir", type=str, help="The directory in which the logs are written"
    )

    arg_test.add_argument(
        "--example",
        type=str,
        choices=["hit-agent", "defend-agent", "baseline", "ppo_baseline", "atacom"],
        default="",
    )

    default_path = Path(__file__).parent.joinpath("air_hockey_agent/agent_config.yml")
    arg_test.add_argument(
        "-c",
        "--config",
        type=str,
        default=default_path,
        help="Path to the config file.",
    )

    arg_test.add_argument(
        "-r", "--render", action="store_true", help="If set renders the environment"
    )

    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":
    args = get_args()

    # Remove all None entries
    filtered_args = {k: v for k, v in args.items() if v is not None}

    # Load Environment

    # Begin Game
