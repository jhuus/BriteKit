#!/usr/bin/env python3

# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import logging
from pathlib import Path
import tempfile
from typing import Optional

import click

from britekit.core.config_loader import get_config
from britekit.core import util


def _build_prediction_mapping(tester, metadata_df):
    """Map long-format prediction rows to the tester's dense prediction matrix."""
    import numpy as np

    row_indexes = {
        row_id: index for index, row_id in enumerate(tester.y_pred_trained_df[""])
    }
    step = tester.segment_len - tester.overlap
    coordinate_sources = {}
    rows = metadata_df.itertuples(index=False, name=None)
    for source_index, (recording, class_code, start_time, _) in enumerate(rows):
        segment = int(start_time // step)
        row_index = row_indexes.get(f"{recording}-{segment}")
        if row_index is None:
            continue
        column_index = tester.trained_class_indexes[class_code]
        flat_index = row_index * len(tester.trained_classes) + column_index
        # Match init_y_pred(use_max_score=False): the last duplicate wins.
        coordinate_sources[flat_index] = source_index

    if not coordinate_sources:
        raise ValueError("No prediction rows matched the test segments")

    flat_indexes = np.fromiter(coordinate_sources.keys(), dtype=np.int64)
    source_indexes = np.fromiter(coordinate_sources.values(), dtype=np.int64)
    return source_indexes, flat_indexes


def _eval_scores(scores, tester, metric, source_indexes, flat_indexes):
    """Evaluate one in-memory score vector using one requested scalar metric."""
    import numpy as np

    y_pred = np.zeros_like(tester.y_pred_trained, dtype=np.float32)
    # convert_to_numpy() commonly produces a Fortran-contiguous array after
    # dropping the dataframe's identifier column. ravel() uses C order and
    # returns a copy for that layout, so assigning through it can silently leave
    # y_pred full of zeros. flat always assigns into the original array.
    y_pred.flat[flat_indexes] = scores[source_indexes]
    return float(tester.get_auc_metric(metric, y_pred))


def _eval_ensemble(ensemble, score_dict, tester, metric, source_indexes, flat_indexes):
    """Average an ensemble's checkpoint scores and evaluate the selected metric."""
    import numpy as np

    scores = np.zeros_like(score_dict[ensemble[0]], dtype=np.float64)
    for ckpt_path in ensemble:
        scores += score_dict[ckpt_path]
    scores /= len(ensemble)
    return _eval_scores(scores, tester, metric, source_indexes, flat_indexes)


def ensemble(
    cfg_path: Optional[str] = None,
    ckpt_dir: str = "",
    ensemble_size: int = 3,
    num_tries: int = 100,
    metric: str = "micro_roc",
    annotations_path: str = "",
    recordings_path: Optional[str] = None,
    save_dir: Optional[str] = None,
    greedy: bool = False,
) -> None:
    """
    Find the best ensemble of a given size from a group of checkpoints.

    Given a directory containing checkpoints, and an ensemble size (default=3), select random
    ensembles of the given size and test each one to identify the best ensemble.

    Args:
    - cfg_path (str, optional): Path to YAML file defining config overrides.
    - ckpt_dir (str): Required path to directory containing checkpoints.
    - ensemble_size (int): Number of checkpoints in ensemble (default=3).
    - num_tries (int): Maximum number of ensembles to try (default=100).
    - metric (str): Metric to use to compare ensembles (default=micro_roc).
    - annotations_path (str): Required path to CSV file containing ground truth annotations.
    - recordings_path (str, optional): Directory containing audio recordings. Defaults to annotations directory.
    - save_dir (str, optional): Directory to copy ensemble into.
    - greedy (bool): If true, use a greedy algorithm.
    """
    import glob
    import itertools
    import math
    import os
    import random
    import shutil

    import numpy as np
    import pandas as pd

    from britekit.core.analyzer import Analyzer
    from britekit.testing.per_segment_tester import PerSegmentTester

    if metric not in ["macro_pr", "micro_pr", "macro_roc", "micro_roc"]:
        logging.error(f"Error: invalid metric ({metric})")
        return
    if ensemble_size < 1:
        logging.error("Error: ensemble size must be at least 1")
        return
    if num_tries < 1:
        logging.error("Error: number of tries must be at least 1")
        return

    cfg = get_config(cfg_path)
    ckpt_paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    num_ckpts = len(ckpt_paths)
    if num_ckpts == 0:
        logging.error(f"Error: no checkpoints found in {ckpt_dir}")
        return
    elif num_ckpts < ensemble_size:
        logging.error(
            f"Error: number of checkpoints ({num_ckpts}) is less than requested ensemble size ({ensemble_size})"
        )
        return

    if not recordings_path:
        recordings_path = str(Path(annotations_path).parent)

    original_ckpt_folder = cfg.misc.ckpt_folder
    original_min_score = cfg.infer.min_score
    best_score = float("-inf")
    best_ensemble = None
    try:
        with tempfile.TemporaryDirectory() as ensemble_dir:
            cfg.misc.ckpt_folder = ensemble_dir
            cfg.infer.min_score = 0

            # Run inference once per checkpoint. Keep shared row metadata once and
            # retain only compact score arrays for ensemble evaluation.
            inference_output_dir = str(Path(ensemble_dir) / "inference")
            scores_csv_path = str(Path(inference_output_dir) / "scores.csv")
            score_dict = {}
            metadata_df = None
            metadata_columns = ["recording", "name", "start_time", "end_time"]
            for ckpt_path in ckpt_paths:
                ckpt_name = Path(ckpt_path).name
                logging.info(f"Running inference with {ckpt_name}")
                dest_path = str(Path(ensemble_dir) / ckpt_name)
                shutil.copyfile(ckpt_path, dest_path)

                util.set_logging(level=logging.ERROR)
                try:
                    Analyzer().run(recordings_path, inference_output_dir, rtype="csv")
                finally:
                    util.set_logging()

                df = pd.read_csv(scores_csv_path)
                checkpoint_metadata = df.loc[:, metadata_columns].reset_index(drop=True)
                if metadata_df is None:
                    metadata_df = checkpoint_metadata
                elif not metadata_df.equals(checkpoint_metadata):
                    raise ValueError(
                        f"Prediction rows for {ckpt_name} do not match the other checkpoints"
                    )
                score_dict[ckpt_path] = df["score"].to_numpy(
                    dtype=np.float64, copy=True
                )
                os.remove(dest_path)

            assert metadata_df is not None

            # Initialize static ground truth, segment metadata, and class mappings
            # once. Candidate ensembles only replace the prediction scores.
            util.set_logging(level=logging.ERROR)
            try:
                tester = PerSegmentTester(
                    annotations_path,
                    recordings_path,
                    inference_output_dir,
                    str(Path(ensemble_dir) / "tester_output"),
                    threshold=0.8,
                    save_matrices=False,
                )
                tester.initialize()
            finally:
                util.set_logging()

            source_indexes, flat_indexes = _build_prediction_mapping(
                tester, metadata_df
            )

            count = 1
            total_combinations = math.comb(len(ckpt_paths), ensemble_size)
            if greedy:
                # Find the best single checkpoint, then add the checkpoint that
                # improves the ensemble most. Reuse the selected checkpoints' sum.
                logging.info("Using greedy algorithm")
                current_ensemble: list = []
                remaining_ckpts = set(ckpt_paths)
                current_score_sum = np.zeros_like(
                    score_dict[ckpt_paths[0]], dtype=np.float64
                )

                for i in range(ensemble_size):
                    best_addition = None
                    best_addition_score = float("-inf")

                    for candidate in sorted(remaining_ckpts):
                        candidate_scores = (
                            current_score_sum + score_dict[candidate]
                        ) / (i + 1)
                        score = _eval_scores(
                            candidate_scores,
                            tester,
                            metric,
                            source_indexes,
                            flat_indexes,
                        )
                        logging.info(
                            f"Step {i + 1}/{ensemble_size}, testing {Path(candidate).name}: score = {score:.4f}"
                        )
                        if score > best_addition_score:
                            best_addition_score = score
                            best_addition = candidate

                    assert best_addition is not None
                    current_ensemble.append(best_addition)
                    remaining_ckpts.remove(best_addition)
                    current_score_sum += score_dict[best_addition]
                    logging.info(
                        f"Added {Path(best_addition).name}, ensemble score = {best_addition_score:.4f}"
                    )

                best_ensemble = tuple(current_ensemble)
                best_score = best_addition_score
            elif total_combinations <= num_tries:
                logging.info("Doing exhaustive search")
                for candidate_ensemble in itertools.combinations(
                    ckpt_paths, ensemble_size
                ):
                    score = _eval_ensemble(
                        candidate_ensemble,
                        score_dict,
                        tester,
                        metric,
                        source_indexes,
                        flat_indexes,
                    )
                    logging.info(
                        f"For ensemble {count} of {total_combinations}, score = {score:.4f}"
                    )
                    if score > best_score:
                        best_score = score
                        best_ensemble = candidate_ensemble

                    count += 1
            else:
                logging.info("Doing random sampling")
                seen: set = set()
                while len(seen) < num_tries:
                    candidate_ensemble = tuple(
                        sorted(random.sample(ckpt_paths, ensemble_size))
                    )
                    if candidate_ensemble not in seen:
                        seen.add(candidate_ensemble)
                        score = _eval_ensemble(
                            candidate_ensemble,
                            score_dict,
                            tester,
                            metric,
                            source_indexes,
                            flat_indexes,
                        )
                        logging.info(
                            f"For ensemble {count} of {num_tries}, score = {score:.4f}"
                        )
                        if score > best_score:
                            best_score = score
                            best_ensemble = candidate_ensemble

                    count += 1
    finally:
        cfg.misc.ckpt_folder = original_ckpt_folder
        cfg.infer.min_score = original_min_score
        util.set_logging()

    logging.info(f"Best score = {best_score:.4f}")

    assert best_ensemble is not None
    best_names = [Path(ckpt_path).name for ckpt_path in best_ensemble]
    logging.info(f"Best ensemble = {best_names}")

    if save_dir is not None:
        # Copy the selected ensemble
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for name in best_names:
            from_path = os.path.join(ckpt_dir, name)
            dest_path = os.path.join(save_dir, name)
            shutil.copyfile(from_path, dest_path)


@click.command(
    name="ensemble",
    short_help="Find the best ensemble of a given size from a group of checkpoints.",
    help=util.cli_help_from_doc(ensemble.__doc__),
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
    "--ckpt",
    "ckpt_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing checkpoints.",
)
@click.option(
    "-e",
    "--ensemble_size",
    "ensemble_size",
    type=int,
    default=3,
    help="Number of checkpoints in ensemble (default=3).",
)
@click.option(
    "-n",
    "--num_tries",
    "num_tries",
    type=int,
    default=100,
    help="Maximum number of ensembles to try (default=100).",
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
    help="Metric used to compare ensembles (default=micro_roc). Macro-averaging uses annotated classes only, but micro-averaging uses all classes.",
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
    "-r",
    "--recordings",
    "recordings_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=False,
    help="Recordings directory. Default is directory containing annotations file.",
)
@click.option(
    "--save",
    "save_dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    required=False,
    help="Directory to copy ensemble into.",
)
@click.option(
    "--greedy",
    "greedy",
    is_flag=True,
    help="If specified, use a greedy algorithm, which runs faster.",
)
def _ensemble_cmd(
    cfg_path: Optional[str],
    ckpt_dir: str,
    ensemble_size: int,
    num_tries: int,
    metric: str,
    annotations_path: str,
    recordings_path: Optional[str],
    save_dir: Optional[str],
    greedy: bool,
) -> None:
    util.set_logging()
    ensemble(
        cfg_path,
        ckpt_dir,
        ensemble_size,
        num_tries,
        metric,
        annotations_path,
        recordings_path,
        save_dir,
        greedy,
    )
