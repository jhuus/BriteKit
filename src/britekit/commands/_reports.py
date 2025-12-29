#!/usr/bin/env python3

# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import glob
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional

import click

from britekit.core.config_loader import get_config
from britekit.core.exceptions import InputError
from britekit.core import util


def rpt_ann(
    annotations_path: str = "",
    output_path: str = "",
):
    """
    Summarize per-segment annotations from a test dataset.

    This command reads annotation data from a CSV file and generates summary reports
    showing the total duration of each class across all recordings and per-recording
    breakdowns.

    Args:
    - annotations_path (str): Path to CSV file containing per-segment annotations.
    - output_path (str): Directory where summary reports will be saved.
    """
    import pandas as pd

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df = pd.read_csv(annotations_path, dtype={"recording": str, "class": str})

    # report counts for all recordings combined
    total = {}
    for i, row in df.iterrows():
        _class = row["class"]
        if type(_class) is not str:
            # class is omitted for "empty" recordings
            continue

        seconds = row["end_time"] - row["start_time"]
        if _class not in total:
            total[_class] = 0

        total[_class] += seconds

    class_col = []
    seconds_col = []
    for _class in sorted(total.keys()):
        class_col.append(_class)
        seconds_col.append(total[_class])

    output_df = pd.DataFrame()
    output_df["class"] = class_col
    output_df["seconds"] = seconds_col
    summary_path = os.path.join(output_path, "test_summary.csv")
    output_df.to_csv(summary_path, index=False, float_format="%.1f")

    # report counts per recording
    per_recording: dict[str, dict] = {}
    for i, row in df.iterrows():
        recording = row["recording"]
        _class = row["class"]
        seconds = row["end_time"] - row["start_time"]
        if recording not in per_recording:
            per_recording[recording] = {}

        if _class not in per_recording[recording]:
            per_recording[recording][_class] = 0

        per_recording[recording][_class] += seconds

    recording_col = []
    class_col = []
    seconds_col = []
    for recording in sorted(per_recording.keys()):
        for _class in sorted(per_recording[recording].keys()):
            recording_col.append(recording)
            class_col.append(_class)
            seconds_col.append(per_recording[recording][_class])

    output_df = pd.DataFrame()
    output_df["recording"] = recording_col
    output_df["class"] = class_col
    output_df["seconds"] = seconds_col
    details_path = os.path.join(output_path, "test_details.csv")
    output_df.to_csv(details_path, index=False, float_format="%.1f")

    logging.info(f"See summary and details reports in {output_path}")


@click.command(
    name="rpt-ann",
    short_help="Summarize annotations in a per-segment test.",
    help=util.cli_help_from_doc(rpt_ann.__doc__),
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
    "-o",
    "--output",
    "output_path",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Path to output directory.",
)
def _rpt_ann_cmd(
    annotations_path: str,
    output_path: str,
):
    util.set_logging()
    rpt_ann(annotations_path, output_path)


def rpt_db(
    cfg_path: Optional[str] = None, db_path: Optional[str] = None, output_path: str = ""
):
    """
    Generate a comprehensive summary report of the training database.

    This command analyzes the training database and generates detailed reports
    about class distributions, spectrogram groups, and data organization.
    The reports help understand the composition and quality of training data
    and can be used for data management and quality control.

    Args:
    - cfg_path (str, optional): Path to YAML file defining configuration overrides.
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    - output_path (str): Directory where database reports will be saved.
    """
    from britekit.training_db.training_db import TrainingDatabase
    from britekit.training_db.training_data_provider import TrainingDataProvider

    cfg = get_config(cfg_path)
    if db_path is not None:
        cfg.train.train_db = db_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.info(f"Generating report for database {cfg.train.train_db}")
    with TrainingDatabase(cfg.train.train_db) as db:
        provider = TrainingDataProvider(db)
        summary_df, details_df = provider.class_info()
        summary_df.to_csv(os.path.join(output_path, "class_summary.csv"), index=False)
        details_df.to_csv(os.path.join(output_path, "class_details.csv"), index=False)

        spec_group_df = provider.spec_group_info()
        spec_group_df.to_csv(os.path.join(output_path, "spec_groups.csv"), index=False)


@click.command(
    name="rpt-db",
    short_help="Generate a database summary report.",
    help=util.cli_help_from_doc(rpt_db.__doc__),
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
    "-d", "--db", "db_path", required=False, help="Path to the training database."
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Path to output directory.",
)
def _rpt_db_cmd(cfg_path, db_path, output_path):
    util.set_logging()
    rpt_db(cfg_path, db_path, output_path)


def rpt_epochs(
    cfg_path: Optional[str] = "",
    input_path: str = "",
    annotations_path: str = "",
    output_path: str = "",
):
    """
    Given a checkpoint directory and a test, run every checkpoint against the test
    and measure the macro-averaged ROC and AP scores, and then plot them.
    This is useful to determine the number of training epochs needed.

    Args:
    - cfg_path (str, optional): Path to YAML file defining configuration overrides.
    - input_path (str): Checkpoint directory generated by training.
    - annotations_path (str): Path to CSV file containing ground truth annotations.
    - output_path (str): Directory where the graph image will be saved.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import pandas as pd

    from britekit.core.analyzer import Analyzer
    from britekit.testing.per_segment_tester import PerSegmentTester

    cfg = get_config(cfg_path)
    ckpt_paths = glob.glob(str(Path(input_path) / "*.ckpt"))
    if len(ckpt_paths) == 0:
        logging.error(f"No checkpoint files found in {input_path}")
        quit()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    epoch_to_ckpt = {}
    for ckpt_path in ckpt_paths:
        stem = Path(ckpt_path).stem
        index = stem.rfind("-e")
        if index == -1:
            logging.error(f"Invalid checkpoint name: {Path(ckpt_path).name}")
            quit()

        ckpt_num = int(stem[index + 2 :])
        epoch_to_ckpt[ckpt_num] = ckpt_path

    epoch_nums = sorted(epoch_to_ckpt)

    # assume recordings are in the same directory as annotations for now
    recording_dir = str(Path(annotations_path).parent)

    # process each ckpt and save the scores
    max_pr_score, max_pr_epoch = 0, 0
    max_roc_score, max_roc_epoch = 0, 0
    pr_scores = []
    roc_scores = []
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg.misc.ckpt_folder = temp_dir

        for epoch_num in epoch_nums:
            util.set_logging()  # restore console output
            logging.info(f"Processing epoch {epoch_num}")
            # suppress output from Analyzer and PerSegmentTester
            util.set_logging(level=logging.ERROR)

            # copy checkpoint to temp dir
            from_path = epoch_to_ckpt[epoch_num]
            to_path = str(Path(temp_dir) / Path(from_path).name)
            shutil.copyfile(from_path, to_path)

            # run inference
            label_dir = "temp"
            inference_output_dir = str(Path(temp_dir) / label_dir)
            cfg.infer.min_score = 0
            Analyzer().run(recording_dir, inference_output_dir)

            # get test metrics
            tester = PerSegmentTester(
                annotations_path,
                recording_dir,
                inference_output_dir,
                temp_dir,
                threshold=0.8,
            )

            tester.initialize()

            pr_stats = tester.get_pr_auc_stats()
            pr_score = pr_stats["micro_pr_auc_trained"]
            pr_scores.append(pr_score)
            if pr_score > max_pr_score:
                max_pr_score = pr_score
                max_pr_epoch = epoch_num

            roc_stats = tester.get_roc_auc_stats()
            roc_score = roc_stats["micro_roc_auc_trained"]
            roc_scores.append(roc_score)
            if roc_score > max_roc_score:
                max_roc_score = roc_score
                max_roc_epoch = epoch_num

            os.remove(to_path)

    # Save CSV
    df = pd.DataFrame()
    df["epoch"] = epoch_nums
    df["PR-AUC"] = pr_scores
    df["ROC-AUC"] = roc_scores
    csv_path = str(Path(output_path) / "training_scores.csv")
    df.to_csv(csv_path, index=False, float_format="%.3f")

    # Plot PR-AUC Score with a solid line
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_nums, pr_scores, linestyle="-", marker="o", label="PR-AUC Score")

    # Plot ROC-AUC Score with a dashed line
    plt.plot(epoch_nums, roc_scores, linestyle="--", marker="s", label="ROC-AUC Score")

    # Labels and title
    plt.xlabel("Epoch #")
    plt.ylabel("Score")
    plt.title("Training Progress: PR-AUC and ROC-AUC")

    # Force integer ticks on the x-axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add grid and legend
    plt.grid(True, linestyle=":")
    plt.legend()

    # Save the figure
    plot_path = str(Path(output_path) / "training_scores.jpeg")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    logging.info(
        f"Maximum micro-averaged PR-AUC score = {max_pr_score:.3f} at epoch {max_pr_epoch}"
    )
    logging.info(
        f"Maximum micro-averaged ROC-AUC score = {max_roc_score:.3f} at epoch {max_roc_epoch}"
    )
    logging.info(f"See plot at {plot_path}")


@click.command(
    name="rpt-epochs",
    short_help="Plot the test score for every training epoch.",
    help=util.cli_help_from_doc(rpt_epochs.__doc__),
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
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Path to checkpoint directory generated by training.",
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
    "-o",
    "--output",
    "output_path",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Path to output directory.",
)
def _rpt_epochs_cmd(
    cfg_path: Optional[str],
    input_path: str,
    annotations_path: str,
    output_path: str,
):
    util.set_logging()
    rpt_epochs(
        cfg_path,
        input_path,
        annotations_path,
        output_path,
    )


def rpt_labels(
    label_dir: str = "",
    output_path: str = "",
    min_score: Optional[float] = None,
):
    """
    Summarize the output of an inference run.

    This command processes inference results (from CSV files or Audacity labels)
    and generates summary reports showing the total duration of detections
    per class and per recording. It filters results by confidence threshold
    and removes overlapping detections to provide clean statistics.

    The reports help understand model performance and detection patterns
    across different recordings and classes.

    Args:
    - label_dir (str): Directory containing inference output (CSV or Audacity labels).
    - output_path (str): Directory where summary reports will be saved.
    - min_score (float, optional): Ignore detections below this confidence threshold.
    """
    import pandas as pd

    cfg = get_config()
    if min_score is None:
        min_score = cfg.infer.min_score

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # get labels in a dataframe, sort it, then filter on threshold
    df = util.inference_output_to_dataframe(label_dir)
    df = df.sort_values(by=["recording", "name", "start_time"])
    df = df[df["score"] >= min_score]

    if df.empty:
        logging.error(f"No labels found with scores >= {min_score}")
        quit()

    # remove any overlap between adjacent labels
    is_first = True
    for i, row in df.iterrows():
        if is_first:
            prev = row
            is_first = False
            continue

        if prev["recording"] == row["recording"] and prev["name"] == row["name"]:
            if row["start_time"] < prev["end_time"]:
                df.loc[i, "start_time"] = prev["end_time"]

        prev = row

    # add the number of labelled seconds per class and per recording/class
    seconds_per_class: dict[str, float] = {}
    seconds_per_rec_class: dict[str, dict[str, float]] = {}
    recordings = df["recording"].unique()

    for recording in recordings:
        seconds_per_rec_class[recording] = {}
        rec_df = df[df["recording"] == recording]

        names = rec_df["name"].unique()
        for name in names:
            if name not in seconds_per_rec_class:
                seconds_per_rec_class[recording][name] = 0

            if name not in seconds_per_class:
                seconds_per_class[name] = 0

            name_df = rec_df[rec_df["name"] == name]
            for i, row in name_df.iterrows():
                elapsed = row["end_time"] - row["start_time"]
                seconds_per_rec_class[recording][name] += elapsed
                seconds_per_class[name] += elapsed

    # save the summary CSV
    class_names = sorted(seconds_per_class.keys())
    rows = []
    for class_name in class_names:
        rows.append([class_name, seconds_per_class[class_name]])

    df = pd.DataFrame(rows, columns=["class", "seconds"])
    df.to_csv(os.path.join(output_path, "summary.csv"), index=False)

    # save the details CSV
    rows = []
    for recording in sorted(seconds_per_rec_class.keys()):
        row = [recording]
        for class_name in class_names:
            if class_name in seconds_per_rec_class[recording]:
                row.append(seconds_per_rec_class[recording][class_name])
            else:
                row.append(0)

        rows.append(row)

    df = pd.DataFrame(rows, columns=["recording"] + class_names)
    df.to_csv(os.path.join(output_path, "details.csv"), index=False)
    logging.info(f"See output reports in {output_path}.")


@click.command(
    name="rpt-labels",
    short_help="Summarize the output of an inference run.",
    help=util.cli_help_from_doc(rpt_labels.__doc__),
)
@click.option(
    "-l",
    "--labels",
    "label_dir",
    type=str,
    required=True,
    help="Directory containing inference output (CSV file or Audacity label files).",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Directory to store output reports.",
)
@click.option(
    "-m",
    "--min_score",
    "min_score",
    type=float,
    required=False,
    help="Ignore scores below this threshold.",
)
def _rpt_labels_cmd(
    label_dir: str,
    output_path: str,
    min_score: Optional[float],
):
    util.set_logging()
    rpt_labels(label_dir, output_path, min_score)


def rpt_test(
    cfg_path: Optional[str] = None,
    granularity: str = "segment",
    annotations_path: str = "",
    label_dir: str = "",
    output_path: str = "",
    recordings_path: Optional[str] = None,
    min_score: Optional[float] = None,
    block_size: int = 60,
    precision: float = 0.95,
):
    """
    Generate comprehensive test metrics and reports comparing model predictions to ground truth.

    This command evaluates model performance by comparing inference results against
    ground truth annotations. It supports three granularity levels:
    - "recording": Evaluate at the recording level (presence/absence)
    - "block": Evaluate at the block level (presence/absence per block)
    - "segment": Evaluate at the segment level (detailed temporal alignment)

    The command generates detailed performance metrics including precision, recall,
    F1 scores, and various visualization plots to help understand model behavior.

    Args:
    - cfg_path (str, optional): Path to YAML file defining configuration overrides.
    - granularity (str): Evaluation granularity ("recording", "block", or "segment"). Default is "segment".
    - annotations_path (str): Path to CSV file containing ground truth annotations.
    - label_dir (str): Directory containing model prediction labels (Audacity format).
    - output_path (str): Directory where test reports will be saved.
    - recordings_path (str, optional): Directory containing audio recordings. Defaults to annotations directory.
    - min_score (float, optional): Provide detailed reports for this confidence threshold.
    - block_size (int, optional): block_size in seconds (default=60).
    - precision (float): For recording granularity, report true positive seconds at this precision. Default is 0.95.
    """
    from britekit.testing.per_block_tester import PerBlockTester
    from britekit.testing.per_recording_tester import PerRecordingTester
    from britekit.testing.per_segment_tester import PerSegmentTester

    cfg = get_config(cfg_path)
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

        if min_score is None:
            min_score = cfg.infer.min_score

        if granularity.startswith("rec"):
            PerRecordingTester(
                annotations_path,
                recordings_path,
                labels_path,
                output_path,
                min_score,
                precision,
            ).run()
        elif granularity.startswith("bl"):
            PerBlockTester(
                annotations_path,
                recordings_path,
                labels_path,
                output_path,
                min_score,
                block_size,
            ).run()
        elif granularity.startswith("seg"):
            PerSegmentTester(
                annotations_path,
                recordings_path,
                labels_path,
                output_path,
                min_score,
            ).run()
        else:
            logging.error(
                'Invalid granularity (expected "recording", "block" or "segment").'
            )

    except InputError as e:
        logging.error(e)


@click.command(
    name="rpt-test",
    short_help="Generate metrics and reports from test results.",
    help=util.cli_help_from_doc(rpt_test.__doc__),
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
    "-g",
    "--granularity",
    "granularity",
    type=str,
    default="segment",
    help='Test annotation and reporting granularity ("recording", "block" or "segment"). Default = "segment".',
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
    "-m",
    "--min_score",
    "min_score",
    type=float,
    required=False,
    help="Provide detailed reports for this threshold.",
)
@click.option(
    "-b",
    "--block",
    "block_size",
    type=int,
    required=False,
    default=60,
    help="Block size in seconds, when granularity=block (default=60).",
)
@click.option(
    "--precision",
    required=False,
    type=float,
    default=0.95,
    help="For granularity=recording, report TP seconds at this precision (default=.95).",
)
def _rpt_test_cmd(
    cfg_path: str,
    granularity: str,
    annotations_path: str,
    label_dir: str,
    output_path: str,
    recordings_path: Optional[str],
    min_score: Optional[float],
    block_size: int,
    precision: float,
):
    util.set_logging()
    rpt_test(
        cfg_path,
        granularity,
        annotations_path,
        label_dir,
        output_path,
        recordings_path,
        min_score,
        block_size,
        precision,
    )
