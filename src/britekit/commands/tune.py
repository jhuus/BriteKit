from pathlib import Path
import time
from typing import Optional
import yaml

import click
import pandas as pd

from britekit.core.config_loader import get_config
from britekit.core.util import format_elapsed_time, cli_help_from_doc
from britekit.tuning.tuner import Tuner


def tune_impl(
    cfg_path: str,
    param_path: Optional[str],
    annotations_path: str,
    metric: str,
    recordings_path: str,
    train_log_path: str,
    num_trials: int,
    num_runs: int,
):
    """
    Find and print the best hyperparameter settings based on exhaustive or random search.

    This command performs hyperparameter optimization by training models with different
    parameter combinations and evaluating them using the specified metric. It can perform
    either exhaustive search (testing all combinations) or random search (testing a
    specified number of random combinations).

    The param_path specifies a YAML file that contains a sequence of parameters such as:

    - name: prob_simple_merge
      type: float
      bounds:
      - 0.36
      - 0.42
      step: 0.02

    The name defines a hyperparameter to tune, with given type, bounds and step size.

    Args:
        cfg_path (str): Path to YAML file defining configuration overrides.
        param_path (str, optional): Path to YAML file defining hyperparameters to tune and their search space.
        annotations_path (str): Path to CSV file containing ground truth annotations.
        metric (str): Metric used to compare runs. Options include various MAP and ROC metrics.
        recordings_path (str, optional): Directory containing audio recordings. Defaults to annotations directory.
        train_log_path (str, optional): Training log directory. Defaults to "logs/fold-0".
        num_trials (int): Number of random trials to run. If 0, performs exhaustive search.
        num_runs (int): Number of runs to average for each parameter combination. Default is 1.
    """

    _, fn_cfg = get_config(cfg_path)
    fn_cfg.echo = click.echo

    try:
        if not recordings_path:
            recordings_path = str(Path(annotations_path).parent)

        if not train_log_path:
            train_log_path = str(Path("logs") / "fold-0")

        if param_path is not None:
            with open(param_path) as input_file:
                param_space = yaml.safe_load(input_file)
        else:
            param_space = None

        start_time = time.time()
        tuner = Tuner(
            recordings_path,
            annotations_path,
            train_log_path,
            metric,
            param_space,
            num_trials,
            num_runs,
        )
        best_score, best_params = tuner.run()
        if best_params:
            click.echo(f"\nBest score = {best_score:.4f}")
            click.echo(f"Best params = {best_params}")

        elapsed_time = format_elapsed_time(start_time, time.time())
        click.echo(f"Elapsed time = {elapsed_time}")

    except Exception as e:
        click.echo(e)


@click.command(
    name="tune",
    short_help="Tune hyperparameters using exhaustive or random search.",
    help=cli_help_from_doc(tune_impl.__doc__),
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
def tune_cmd(
    cfg_path: str,
    param_path: Optional[str],
    annotations_path: str,
    metric: str,
    recordings_path: str,
    train_log_path: str,
    num_trials: int,
    num_runs: int,
):
    tune_impl(
        cfg_path,
        param_path,
        annotations_path,
        metric,
        recordings_path,
        train_log_path,
        num_trials,
        num_runs,
    )
