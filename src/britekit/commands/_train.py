#!/usr/bin/env python3

# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import logging
import time
from typing import Optional

import click

from britekit.core.config_loader import get_config
from britekit.core.exceptions import TrainingError
from britekit.core import util


def train(
    cfg_path: Optional[str] = None,
    seed: Optional[int] = None,
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
    - cfg_path (str, optional): Path to YAML file defining configuration overrides.
        If not specified, uses default configuration.
    - seed (int, optional): Integer seed.
    """
    from britekit.core.trainer import Trainer

    cfg = get_config(cfg_path)  # apply any YAML cfg updates
    if seed is not None:
        cfg.train.seed = seed
    try:
        start_time = time.time()
        Trainer().run()
        elapsed_time = util.format_elapsed_time(start_time, time.time())
        logging.info(f"Elapsed time = {elapsed_time}")
    except TrainingError as e:
        logging.error(e)


@click.command(
    name="train", short_help="Run training.", help=util.cli_help_from_doc(train.__doc__)
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
    "--seed",
    "seed",
    type=int,
    required=False,
    help="Integer seed.",
)
def _train_cmd(
    cfg_path: str,
    seed: int,
):
    util.set_logging()

    import platform
    import torch

    if platform.system() == "Windows" and not torch.cuda.is_available():
        logging.warning(
            "CUDA is not available. On Windows, reinstall a CUDA-enabled PyTorch build like this:\n"
            "  pip uninstall -y torch torchvision torchaudio\n"
            "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuxxx"
            "For example, use cu126 for CUDA 12.6."
        )

    train(cfg_path, seed)


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
    - cfg_path (str, optional): Path to YAML file defining configuration overrides.
        If not specified, uses default configuration.
    - num_batches (int): Number of training batches to analyze for learning rate finding.
        Default is 100. Higher values provide more accurate results but take longer.
    """
    from britekit.core.trainer import Trainer

    get_config(cfg_path)  # apply any YAML cfg updates
    try:
        suggested_lr, fig = Trainer().find_lr(num_batches)
        fig.savefig("learning_rates.jpeg")
        logging.info(f"Suggested learning rate = {suggested_lr:.6f}")
        logging.info("See plot in learning_rates.jpeg")
    except TrainingError as e:
        logging.error(e)


@click.command(
    name="find-lr",
    short_help="Suggest a learning rate.",
    help=util.cli_help_from_doc(find_lr.__doc__),
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
    util.set_logging()
    find_lr(cfg_path, num_batches)
