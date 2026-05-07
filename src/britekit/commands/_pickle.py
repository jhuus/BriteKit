#!/usr/bin/env python3

# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import logging
import os
from pathlib import Path
from typing import Optional

import click

from britekit.core.config_loader import get_config
from britekit.core import util


def pickle_occurrence(
    cfg_path: Optional[str] = None,
    db_path: Optional[str] = None,
    output_path: Optional[str] = None,
    root_dir: str = "",
) -> None:
    """
    Convert an occurrence database to a pickle file for use in inference.

    Args:
    - cfg_path (str, optional): Path to YAML file defining config overrides.
    - db_path (str, optional): Path to the occurrence database. Defaults to "data/occurrence.db".
    - output_path (str, optional): Output pickle file path. Defaults to "data/occurrence.pkl".
    - root_dir (str, optional): Root directory containing data directory. Defaults to working directory.
    """
    from britekit.core.pickler import OccurrencePickler

    get_config(cfg_path)
    if db_path is None:
        db_path = str(Path("data") / "occurrence.db")

    if output_path is None:
        output_path = str(Path(root_dir) / "data" / "occurrence.pkl")

    pickler = OccurrencePickler(db_path, output_path)
    pickler.pickle()


@click.command(
    name="pickle-occurrence",
    short_help="Convert an occurrence database to a pickle file for use in inference.",
    help=util.cli_help_from_doc(pickle_occurrence.__doc__),
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
    "-d",
    "--db",
    "db_path",
    required=False,
    help="Path to the occurrence database. Defaults to 'data/occurrence.db'",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=False,
    help='Output file path. Default is "data/occurrence.pkl".',
)
def _pickle_occurrence_cmd(
    cfg_path: Optional[str],
    db_path: Optional[str],
    output_path: Optional[str],
) -> None:
    util.set_logging()
    pickle_occurrence(
        cfg_path,
        db_path,
        output_path,
    )


def pickle_train(
    cfg_path: Optional[str] = None,
    classes_path: Optional[str] = None,
    db_path: Optional[str] = None,
    output_path: Optional[str] = None,
    root_dir: str = "",
    max_per_class: Optional[int] = None,
    spec_group: Optional[str] = None,
) -> None:
    """
    Convert database spectrograms to a pickle file for use in training.

    This command extracts spectrograms from the training database and saves them in a pickle file
    that can be efficiently loaded during model training. It can process all classes in the database
    or specific classes specified by a CSV file.

    Args:
    - cfg_path (str, optional): Path to YAML file defining config overrides.
    - classes_path (str, optional): Path to CSV file containing class names to include.
        If omitted, includes all classes in the database.
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    - output_path (str, optional): Output pickle file path. Defaults to "data/training.pkl".
    - root_dir (str, optional): Root directory containing data directory. Defaults to working directory.
    - max_per_class (int, optional): Maximum number of spectrograms to include per class.
    - spec_group (str, optional): Spectrogram group name to extract from. Defaults to 'default'.
    """
    from britekit.core.pickler import TrainingPickler

    cfg = get_config(cfg_path)
    if db_path is None:
        db_path = cfg.train.train_db

    if classes_path is not None:
        if not os.path.exists(classes_path):
            logging.error(f"Error: file {classes_path} not found.")
            return

    if output_path is None:
        output_path = str(Path(root_dir) / "data" / "training.pkl")

    pickler = TrainingPickler(
        db_path, output_path, classes_path, max_per_class, spec_group
    )
    pickler.pickle()


@click.command(
    name="pickle-train",
    short_help="Convert a training database to a pickle file for use in training.",
    help=util.cli_help_from_doc(pickle_train.__doc__),
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
    "--classes",
    "classes_path",
    required=False,
    help="Path to CSV containing class names to pickle (optional). Default is all classes.",
)
@click.option(
    "-d", "--db", "db_path", required=False, help="Path to the training database."
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=False,
    help='Output file path. Default is "data/training.pkl".',
)
@click.option(
    "--root",
    "root_dir",
    default=".",
    type=click.Path(file_okay=False),
    help="Root directory containing data directory.",
)
@click.option(
    "-m",
    "--max",
    "max_per_class",
    required=False,
    type=int,
    help="Maximum spectrograms per class.",
)
@click.option(
    "--sgroup",
    "spec_group",
    required=False,
    default="default",
    help="Spectrogram group name. Defaults to 'default'.",
)
def _pickle_train_cmd(
    cfg_path: Optional[str],
    classes_path: Optional[str],
    db_path: Optional[str],
    output_path: Optional[str],
    root_dir: str,
    max_per_class: Optional[int],
    spec_group: Optional[str],
) -> None:
    util.set_logging()
    pickle_train(
        cfg_path,
        classes_path,
        db_path,
        output_path,
        root_dir,
        max_per_class,
        spec_group,
    )


def _compute_frame_labels(score, left_scores, right_scores, alpha, num_frames, pad=0):
    import numpy as np

    threshold = alpha * score

    # key_from_left: the leftmost frame the call needs, found by masking from the left.
    # When masking frames 1..k drops the score, frame k is the leftmost frame the call
    # requires (frames 1..k-1 were removable without effect). Fallback: num_frames (call
    # extends to the last frame).
    key_from_left = num_frames
    for k, s in enumerate(left_scores, start=1):
        if s < threshold:
            key_from_left = k
            break

    # key_from_right: the rightmost frame the call needs, found by masking from the right.
    # When masking the last j frames drops the score, frame (num_frames - j + 1) is the
    # rightmost frame the call requires (frames j+1..num_frames were removable without
    # effect). Fallback: 1 (call extends to frame 1).
    key_from_right = 1
    for j, s in enumerate(right_scores, start=1):
        if s < threshold:
            key_from_right = num_frames - j + 1
            break

    # Active interval spans both key frames. Using min/max handles the case where the
    # two directions identify the same frame (point call) or adjacent frames, as well as
    # the all-frames fallback when neither boundary is found.
    lo = min(key_from_left, key_from_right) - 1  # convert to 0-indexed
    hi = max(key_from_left, key_from_right)  # exclusive upper bound

    # Extend by pad frames on each side, clamped to valid range.
    lo = max(0, lo - pad)
    hi = min(num_frames, hi + pad)

    labels = np.zeros(num_frames, dtype=np.float32)
    labels[lo:hi] = 1.0
    return labels


def pickle_frame(
    input_dir: str,
    output_path: str,
    cfg_path: Optional[str] = None,
    alpha: float = 0.05,
    csv_dir: Optional[str] = None,
    names_path: Optional[str] = None,
    pad: int = 0,
) -> None:
    """
    Create a frame-label pickle from analyze-db occlusion sensitivity CSVs.

    For each segment, uses occlusion-sensitivity scores produced by the analyze-db
    --occlude flag to determine which spectrogram frames contain the target class.
    The number of frames is derived from cfg.audio.spec_duration * cfg.train.sed_fps.

    Args:
    - input_dir (str): Directory containing per-class CSV files from analyze-db --occlude.
    - output_path (str): Output pickle file path.
    - cfg_path (str, optional): Path to YAML file defining config overrides.
    - alpha (float, optional): Score drop threshold as a fraction of the original score
        (default 0.05). Lower values are more conservative (wider active regions).
    - csv_dir (str, optional): Directory for per-class output CSVs with frame labels.
    - names_path (str, optional): Path to a text file listing CSV filenames (one per line)
        to process. Default is all CSV files in the input directory.
    - pad (int, optional): Number of extra frame labels to add on each side of the active
        region (default 0). Padding is clamped to the segment boundary, so a region that
        already starts at frame 0 will only be extended on the right, and vice versa.
    """
    import pickle

    import numpy as np
    import pandas as pd

    cfg = get_config(cfg_path)
    num_frames = round(cfg.audio.spec_duration * cfg.train.sed_fps)

    if names_path is not None:
        with open(names_path) as f:
            csv_files = [line.strip() for line in f if line.strip()]
    else:
        csv_files = sorted(
            f
            for f in os.listdir(input_dir)
            if f.endswith(".csv") and f != "class_summary.csv"
        )

    if not csv_files:
        logging.error(f"No per-class CSV files found in {input_dir}")
        return

    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    frame_labels = {}
    total_segments = 0
    total_full_clip = 0

    for csv_file in csv_files:
        csv_path = os.path.join(input_dir, csv_file)
        df = pd.read_csv(csv_path)

        required = (
            {"Segment ID", "Score"}
            | {f"score_L{n}" for n in range(1, num_frames)}
            | {f"score_R{n}" for n in range(1, num_frames)}
        )
        missing = required - set(df.columns)
        if missing:
            logging.warning(f"Skipping {csv_file}: missing columns {missing}")
            continue

        left_cols = [f"score_L{n}" for n in range(1, num_frames)]
        right_cols = [f"score_R{n}" for n in range(1, num_frames)]

        class_full_clip = 0
        out_segment_ids = []
        out_frame_labels = []

        for _, row in df.iterrows():
            segment_id = int(row["Segment ID"])
            score = float(row["Score"])

            if score <= 0:
                # No reliable baseline; fall back to all-ones
                labels = np.ones(num_frames, dtype=np.float32)
            else:
                left_scores = [float(row[c]) for c in left_cols]
                right_scores = [float(row[c]) for c in right_cols]
                labels = _compute_frame_labels(
                    score, left_scores, right_scores, alpha, num_frames, pad
                )

            if labels.sum() == num_frames:
                class_full_clip += 1

            out_segment_ids.append(segment_id)
            out_frame_labels.append(labels)

            # Keep first occurrence; a segment may appear in multiple class CSVs
            # but the training pickle also assigns each segment to one class only.
            if segment_id not in frame_labels:
                frame_labels[segment_id] = labels

        if csv_dir:
            out_df = pd.DataFrame({"Segment ID": out_segment_ids})
            for n in range(num_frames):
                out_df[f"frame_{n + 1}"] = [fl[n] for fl in out_frame_labels]
            out_csv_path = os.path.join(csv_dir, csv_file)
            out_df.to_csv(out_csv_path, index=False, float_format="%.0f")

        num_rows = len(df)
        total_segments += num_rows
        total_full_clip += class_full_clip
        logging.info(
            f"{csv_file}: {num_rows} segments, "
            f"{class_full_clip} ({100 * class_full_clip / max(num_rows, 1):.0f}%) full-clip labels"
        )

    logging.info(
        f"Total: {total_segments} segments, "
        f"{total_full_clip} ({100 * total_full_clip / max(total_segments, 1):.0f}%) full-clip labels"
    )
    logging.info(f"Writing {output_path}")
    with open(output_path, "wb") as out_f:
        pickle.dump(frame_labels, out_f)


@click.command(
    name="pickle-frame",
    short_help="Create a frame-label pickle from analyze-db occlusion sensitivity CSVs.",
    help=util.cli_help_from_doc(pickle_frame.__doc__),
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
    "input_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing per-class CSV files from analyze-db --occlude.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    help="Output pickle file path.",
)
@click.option(
    "--alpha",
    "alpha",
    type=float,
    default=0.05,
    help="Score drop threshold as a fraction of the original score (default 0.05). "
    "Lower values are more conservative (wider active regions).",
)
@click.option(
    "--csv",
    "csv_dir",
    type=click.Path(file_okay=False, dir_okay=True),
    required=False,
    help="Optional directory for per-class output CSVs with segment IDs and frame labels.",
)
@click.option(
    "--names",
    "names_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=False,
    help="Optional path to a text file listing CSV filenames (one per line) to process. "
    "Default is all CSV files in the input directory.",
)
@click.option(
    "--pad",
    "pad",
    type=int,
    default=0,
    help="Number of extra frame labels to add on each side of the active region (default 0). "
    "Clamped to segment boundaries.",
)
def _pickle_frame_cmd(
    cfg_path: Optional[str],
    input_dir: str,
    output_path: str,
    alpha: float,
    csv_dir: Optional[str],
    names_path: Optional[str],
    pad: int,
) -> None:
    util.set_logging()
    pickle_frame(input_dir, output_path, cfg_path, alpha, csv_dir, names_path, pad)
