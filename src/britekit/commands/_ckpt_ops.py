# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import glob
import logging
import os
from typing import Optional

import click

from britekit.core.config_loader import get_config
from britekit.core import util


def ckpt_avg(input_path: str = "", output_path: Optional[str] = None):
    """
    Average the weights of multiple model checkpoints to create an ensemble checkpoint.

    This command loads multiple checkpoint files from a directory and creates a new checkpoint
    with averaged weights.

    Args:
    - input_path (str): Directory containing checkpoint files (*.ckpt) to average.
    - output_path (str, optional): Path for the output averaged checkpoint.
        Defaults to "average.ckpt" in the input directory.
    """
    import torch

    ckpt_paths = glob.glob(os.path.join(input_path, "*.ckpt"))
    if not output_path:
        output_path = os.path.join(input_path, "average.ckpt")

    checkpoints = [torch.load(str(path), map_location="cpu") for path in ckpt_paths]
    averaged_ckpt = checkpoints[0].copy()

    avg_state_dict = {}
    for key in checkpoints[0]["state_dict"]:
        tensors = [ckpt["state_dict"][key] for ckpt in checkpoints]

        if torch.is_floating_point(tensors[0]) or tensors[0].is_complex():
            # Safe to average directly
            stacked = torch.stack(tensors, dim=0)
            avg_state_dict[key] = stacked.mean(dim=0)
        else:
            # Integer tensors – use majority vote or just take the first (common for counters)
            avg_state_dict[key] = tensors[0]  # or use mode if preferred

    averaged_ckpt["state_dict"] = avg_state_dict
    torch.save(averaged_ckpt, str(output_path))
    logging.info(f"Saved checkpoint with average weights in {output_path}")


@click.command(
    name="ckpt-avg",
    short_help="Average the weights of several checkpoints.",
    help=util.cli_help_from_doc(ckpt_avg.__doc__),
)
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory containing checkpoints to average",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=False,
    type=click.Path(file_okay=True, dir_okay=False),
    help="Optional path to output checkpoint. Default is average.ckpt in the input directory",
)
def _ckpt_avg_cmd(input_path: str, output_path: str):
    util.set_logging()
    ckpt_avg(input_path, output_path)


def ckpt_freeze(input_path: str = ""):
    """
    Freeze the backbone weights of a checkpoint to reduce file size and improve inference speed.

    This command loads a PyTorch checkpoint and freezes the backbone weights, which removes
    training-specific information like gradients and optimizer states. This significantly
    reduces the checkpoint file size and can improve inference performance.

    The original checkpoint is preserved with a ".original" extension, and the frozen
    version replaces the original file. Frozen checkpoints are optimized for deployment
    and inference rather than continued training.

    Args:
    - input_path (str): Path to the checkpoint file to freeze.
    """
    import pytorch_lightning as pl
    from britekit.models.model_loader import load_from_checkpoint

    renamed_path = input_path + ".original"
    os.rename(input_path, renamed_path)
    model = load_from_checkpoint(renamed_path)
    model.freeze()

    trainer = pl.Trainer()
    trainer.strategy.connect(model)
    trainer.save_checkpoint(input_path)


@click.command(
    name="ckpt-freeze",
    short_help="Freeze the backbone weights of a checkpoint.",
    help=util.cli_help_from_doc(ckpt_freeze.__doc__),
)
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False),
    help="Path to checkpoint to freeze",
)
def _ckpt_freeze_cmd(input_path: str):
    util.set_logging()
    ckpt_freeze(input_path)


def ckpt_onnx(
    cfg_path: Optional[str] = None,
    input_path: str = "",
):
    """
    Convert a PyTorch checkpoint to ONNX format for deployment with OpenVINO.

    This command converts a trained PyTorch model checkpoint to ONNX (Open Neural Network
    Exchange) format, which enables deployment using Intel's OpenVINO toolkit. ONNX format
    allows for optimized inference on CPU.

    The conversion process creates a new ONNX file with the same base name as the input
    checkpoint.

    Args:
    - cfg_path (str, optional): Path to YAML file defining configuration overrides.
    - input_path (str): Path to the PyTorch checkpoint file to convert.
    """
    import torch
    from britekit.models.model_loader import load_from_checkpoint

    cfg = get_config(cfg_path)
    base, _ = os.path.splitext(input_path)
    output_path = base + ".onnx"
    model = load_from_checkpoint(input_path)
    input_sample = torch.randn(
        (cfg.infer.openvino_block_size, 1, cfg.audio.spec_height, cfg.audio.spec_width)
    )
    model.to_onnx(output_path, input_sample, export_params=True)


@click.command(
    name="ckpt-onnx",
    short_help="Convert a checkpoint to onnx format for use with openvino.",
    help=util.cli_help_from_doc(ckpt_onnx.__doc__),
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
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False),
    help="Path to checkpoint to convert to ONNX format",
)
def _ckpt_onnx_cmd(
    cfg_path: str,
    input_path: str,
):
    util.set_logging()
    ckpt_onnx(cfg_path, input_path)
