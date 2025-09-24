# File name starts with _ to keep it out of typeahead for API users
import click
import time
from typing import Optional

from britekit.core.config_loader import get_config
from britekit.core.exceptions import TrainingError
from britekit.core.util import format_elapsed_time, cli_help_from_doc
from britekit.core.trainer import Trainer


def train(
    cfg_path: Optional[str]=None,
):
    """
    Train a bioacoustic recognition model using the specified configuration.

    This command initiates the complete training pipeline for a bioacoustic model.
    It loads training data from the database, configures the model architecture,
    and runs the training process with the specified hyperparameters. The training
    includes validation, checkpointing, and progress monitoring.

    Training progress is displayed in real-time, and model checkpoints are saved
    automatically. The final trained model can be used for inference and evaluation.

    Args:
        cfg_path (str, optional): Path to YAML file defining configuration overrides.
                                 If not specified, uses default configuration.
    """
    cfg, fn_cfg = get_config(cfg_path)
    fn_cfg.echo = click.echo
    try:
        start_time = time.time()
        Trainer().run()
        elapsed_time = format_elapsed_time(start_time, time.time())
        click.echo(f"Elapsed time = {elapsed_time}")
    except TrainingError as e:
        click.echo(e)


@click.command(
    name="train", short_help="Run training.", help=cli_help_from_doc(train.__doc__)
)
@click.option(
    "-c",
    "--cfg",
    "cfg_path",
    type=click.Path(exists=True),
    required=False,
    help="Path to YAML file defining config overrides.",
)
def _train_cmd(
    cfg_path: str,
):
    train(cfg_path)


def find_lr(cfg_path: str, num_batches: int):
    """
    Find an optimal learning rate for model training using the learning rate finder.

    This command runs a learning rate finder that tests a range of learning rates
    on a small number of training batches to determine the optimal learning rate.
    It generates a plot showing loss vs. learning rate and suggests the best rate
    based on the steepest negative gradient in the loss curve.

    The suggested learning rate helps ensure stable and efficient training by
    avoiding rates that are too high (causing instability) or too low (slow convergence).

    Args:
        cfg_path (str, optional): Path to YAML file defining configuration overrides.
                                 If not specified, uses default configuration.
        num_batches (int): Number of training batches to analyze for learning rate finding.
                          Default is 100. Higher values provide more accurate results but take longer.
    """
    cfg, fn_cfg = get_config(cfg_path)
    fn_cfg.echo = click.echo
    try:
        suggested_lr, fig = Trainer().find_lr(num_batches)
        fig.savefig("learning_rates.jpeg")
        click.echo(f"Suggested learning rate = {suggested_lr:.6f}")
        click.echo("See plot in learning_rates.jpeg")
    except TrainingError as e:
        click.echo(e)


@click.command(
    name="find-lr",
    short_help="Suggest a learning rate.",
    help=cli_help_from_doc(find_lr.__doc__),
)
@click.option(
    "-c",
    "--cfg",
    "cfg_path",
    type=click.Path(exists=True),
    required=False,
    help="Path to YAML file defining config overrides.",
)
@click.option(
    "-n", "--num-batches", type=int, default=100, help="Number of batches to analyze"
)
def _find_lr_cmd(cfg_path: str, num_batches: int):
    find_lr(cfg_path, num_batches)
