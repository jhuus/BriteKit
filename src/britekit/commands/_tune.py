# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Optional

import click

from britekit.core.config_loader import get_config
from britekit.core import util


def tune(
    cfg_path: Optional[str] = None,
    param_path: Optional[str] = None,
    output_path: str = "",
    annotations_path: str = "",
    metric: str = "micro_roc",
    recordings_path: str = "",
    train_log_path: str = "",
    num_trials: int = 0,
    num_runs: int = 1,
    extract: bool = False,
    skip_training: bool = False,
    classes_path: Optional[str] = None,
):
    """
    Find and print the best hyperparameter settings based on exhaustive or random search.

    This command performs hyperparameter optimization by training models with different
    parameter combinations and evaluating them using the specified metric. It can perform
    either exhaustive search (testing all combinations) or random search (testing a
    specified number of random combinations). To tune spectrogram settings, the --extract
    CLI flag or API parameter specifies that new spectrograms will be extracted before training.
    To tune inference settings, the --notrain CLI flag (skip_training API parameter) specifies
    that training will be skipped.

    The param_path specifies a YAML file that defines the parameters to be tuned, as described in the README.

    Args:
    - cfg_path (str, optional): Path to YAML file defining configuration overrides.
    - param_path (str, optional): Path to YAML file defining hyperparameters to tune and their search space.
    - output_path (str): Directory where reports will be saved.
    - annotations_path (str): Path to CSV file containing ground truth annotations.
    - metric (str): Metric used to compare runs. Options include various MAP and ROC metrics.
    - recordings_path (str, optional): Directory containing audio recordings. Defaults to annotations directory.
    - train_log_path (str, optional): Training log directory. Defaults to "logs".
    - num_trials (int): Number of random trials to run. If 0, performs exhaustive search.
    - num_runs (int): Number of runs to average for each parameter combination. Default is 1.
    - extract (bool): Extract new spectrograms before training, to tune spectrogram parameters.
    - skip_training (bool): Iterate on inference only, using checkpoints from the last training run.
    - classes_path (str, optional): Path to CSV containing class names for extract option. Default is all classes.
    """
    import yaml
    from britekit.core.tuner import Tuner

    try:
        get_config(cfg_path)  # apply any YAML cfg updates
        if extract and skip_training:
            logging.error(
                "Performing spectrogram extract is incompatible with skipping training."
            )
            return

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if not recordings_path:
            recordings_path = str(Path(annotations_path).parent)

        if not train_log_path:
            train_log_path = "logs"

        if param_path is not None:
            with open(param_path) as input_file:
                param_space = yaml.safe_load(input_file)
        else:
            param_space = None

        start_time = time.time()
        tuner = Tuner(
            recordings_path,
            output_path,
            annotations_path,
            train_log_path,
            metric,
            param_space,
            num_trials,
            num_runs,
            extract,
            skip_training,
            classes_path,
        )
        best_score, best_params = tuner.run()
        if best_params:
            logging.info(f"\nBest score = {best_score:.4f}")
            logging.info(f"Best params = {best_params}")
            logging.info(f"See reports in {output_path}")

        if cfg_path:
            to_path = os.path.join(output_path, Path(cfg_path).name)
            shutil.copy(cfg_path, to_path)

        if param_path:
            to_path = os.path.join(output_path, Path(param_path).name)
            shutil.copy(param_path, to_path)

        elapsed_time = util.format_elapsed_time(start_time, time.time())
        logging.info(f"Elapsed time = {elapsed_time}")

    except Exception as e:
        logging.error(e)


@click.command(
    name="tune",
    short_help="Tune hyperparameters using exhaustive or random search.",
    help=util.cli_help_from_doc(tune.__doc__),
)
@click.option(
    "-c",
    "--cfg",
    "cfg_path",
    type=click.Path(exists=True),
    help="Path to YAML file defining config overrides.",
)
@click.option(
    "-p",
    "--param",
    "param_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to YAML file defining hyperparameters to tune.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Path to output directory.",
)
@click.option(
    "-a",
    "--annotations",
    "annotations_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to CSV file containing annotations or ground truth).",
)
@click.option(
    "-m",
    "--metric",
    "metric",
    type=click.Choice(
        [
            "macro_pr",
            "micro_pr",
            "macro_roc",
            "micro_roc",
        ]
    ),
    default="micro_roc",
    help="Metric used to compare runs. Macro-averaging uses annotated classes only, but micro-averaging uses all classes.",
)
@click.option(
    "-r",
    "--recordings",
    "recordings_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Recordings directory. Default is directory containing annotations file.",
)
@click.option(
    "--log",
    "train_log_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Training log directory.",
)
@click.option(
    "--trials",
    "num_trials",
    type=int,
    default=0,
    help="If specified, run this many random trials. Otherwise do an exhaustive search.",
)
@click.option(
    "--runs",
    "num_runs",
    type=int,
    default=1,
    help="Use the average score of this many runs in each case. Default = 1.",
)
@click.option(
    "--extract",
    "extract",
    is_flag=True,
    help="Extract new spectrograms before training, to tune spectrogram parameters.",
)
@click.option(
    "--notrain",
    "skip_training",
    is_flag=True,
    help="Iterate on inference only, using checkpoints from the last training run.",
)
@click.option(
    "--classes",
    "classes_path",
    required=False,
    help="Path to CSV containing class names for extract option. Default is all classes.",
)
def _tune_cmd(
    cfg_path: str,
    param_path: Optional[str],
    output_path: str,
    annotations_path: str,
    metric: str,
    recordings_path: str,
    train_log_path: str,
    num_trials: int,
    num_runs: int,
    extract: bool,
    skip_training: bool,
    classes_path: Optional[str],
):
    util.set_logging()
    tune(
        cfg_path,
        param_path,
        output_path,
        annotations_path,
        metric,
        recordings_path,
        train_log_path,
        num_trials,
        num_runs,
        extract,
        skip_training,
        classes_path,
    )
