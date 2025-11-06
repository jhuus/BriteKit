# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import logging
import os
from pathlib import Path
from typing import Optional

import click

from britekit.core.exceptions import InputError
from britekit.core import util


def calibrate(
    cfg_path: Optional[str] = None,
    annotations_path: str = "",
    label_dir: str = "",
    output_path: str = "",
    recordings_path: Optional[str] = None,
    cutoff: float = 0.4,
    coef: Optional[float] = None,
    inter: Optional[float] = None,
):
    """
    Calibrate model predictions using per-segment test results.

    This command generates calibration plots and analysis to assess how well
    model prediction scores align with actual probabilities. It compares
    predicted scores against ground truth annotations to determine if the
    model is overconfident or underconfident in its predictions.

    The calibration process helps improve model reliability by adjusting
    prediction scores to better reflect true probabilities.

    Args:
    - cfg_path (str, optional): Path to YAML file defining configuration overrides.
    - annotations_path (str): Path to CSV file containing ground truth annotations.
    - label_dir (str): Directory containing model prediction labels (Audacity format).
    - output_path (str): Directory where calibration reports will be saved.
    - recordings_path (str, optional): Directory containing audio recordings. Defaults to annotations directory.
    - cutoff (float): Ignore predictions below this threshold during calibration. Default is 0.4.
    - coef (float, optional): Use this coefficient for the calibration plot.
    - inter (float, optional): Use this intercept for the calibration plot.
    """
    from britekit.testing.per_segment_tester import PerSegmentTester

    if (coef is None and inter is not None) or (coef is not None and inter is None):
        logging.error("If --coef or --inter is specified, both must be specified.")
        quit()

    try:
        if not recordings_path:
            recordings_path = str(Path(annotations_path).parent)

        labels_path = os.path.join(recordings_path, label_dir)
        if not os.path.exists(labels_path):
            if os.path.exists(label_dir):
                # not just name of subdirectory of recordings
                labels_path = label_dir
            else:
                logging.error(f"Label directory {label_dir} not found.")
                quit()

        PerSegmentTester(
            annotations_path,
            recordings_path,
            labels_path,
            output_path,
            0,
            True,
            cutoff,
            coef,
            inter,
        ).run()

    except InputError as e:
        logging.error(e)


@click.command(
    name="calibrate",
    short_help="Calibrate an ensemble based on per-segment test results.",
    help=util.cli_help_from_doc(calibrate.__doc__),
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
    "-a",
    "--annotations",
    "annotations_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to CSV file containing annotations or ground truth).",
)
@click.option(
    "-l",
    "--labels",
    "label_dir",
    type=str,
    required=True,
    help="Directory containing Audacity labels. If a subdirectory of recordings directory, only the subdirectory name is needed.",
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
    "-r",
    "--recordings",
    "recordings_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=False,
    help="Recordings directory. Default is directory containing annotations file.",
)
@click.option(
    "--cutoff",
    required=False,
    type=float,
    default=0.4,
    help="When calibrating, ignore predictions below this (default = .4)",
)
@click.option(
    "--coef",
    required=False,
    type=float,
    help="Use this coefficient in the calibration plot.",
)
@click.option(
    "--inter",
    required=False,
    type=float,
    help="Use this intercept in the calibration plot.",
)
def _calibrate_cmd(
    cfg_path: str,
    annotations_path: str,
    label_dir: str,
    output_path: str,
    recordings_path: Optional[str],
    cutoff: float,
    coef: Optional[float] = None,
    inter: Optional[float] = None,
):
    util.set_logging()
    calibrate(
        cfg_path,
        annotations_path,
        label_dir,
        output_path,
        recordings_path,
        cutoff,
        coef,
        inter,
    )
