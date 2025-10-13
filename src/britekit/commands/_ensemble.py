# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import logging
import os
from pathlib import Path
import tempfile
from typing import Optional

import click

from britekit.core.config_loader import get_config
from britekit.core import util

def _eval_ensemble(ensemble, temp_dir, annotations_path, recording_dir):
    import shutil

    from britekit.core.analyzer import Analyzer
    from britekit.testing.per_segment_tester import PerSegmentTester

    # delete any checkpoints in the temp dir
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        os.remove(file_path)

    # copy checkpoints to the temp dir
    for file_path in ensemble:
        file_name = Path(file_path).name
        dest_path = os.path.join(temp_dir, file_name)
        shutil.copyfile(file_path, dest_path)

    # run inference on the given test
    util.set_logging(level=logging.ERROR) # suppress logging during inference and analysis
    label_dir = "ensemble_evaluation_labels"
    inference_output_dir = str(Path(recording_dir) / label_dir)
    Analyzer().run(recording_dir, inference_output_dir)

    min_score = 0.8 # irrelevant really
    with tempfile.TemporaryDirectory() as output_dir:
        tester = PerSegmentTester(
            annotations_path,
            recording_dir,
            inference_output_dir,
            output_dir,
            min_score,
        )
        tester.initialize()

        pr_stats = tester.get_pr_auc_stats()
        roc_stats = tester.get_roc_auc_stats()

        scores = {
            "macro_pr": pr_stats["macro_pr_auc"],
            "micro_pr": pr_stats["micro_pr_auc_trained"],
            "macro_roc": roc_stats["macro_roc_auc"],
            "micro_roc": roc_stats["micro_roc_auc_trained"]
        }

        shutil.rmtree(inference_output_dir)
        util.set_logging() # restore logging

    return scores

def ensemble(
    cfg_path: Optional[str]=None,
    ckpt_path: str="",
    ensemble_size: int=3,
    num_tries: int=100,
    metric: str = "micro_roc",
    annotations_path: str = "",
    recordings_path: Optional[str] = None,
) -> None:
    """
    Find the best ensemble of a given size from a group of checkpoints.

    Given a directory containing checkpoints, and an ensemble size (default=3), select random
    ensembles of the given size and test each one to identify the best ensemble.

    Args:
        cfg_path (str, optional): Path to YAML file defining configuration overrides.
        ckpt_path (str): Path to directory containing checkpoints.
        ensemble_size (int): Number of checkpoints in ensemble (default=3).
        num_tries (int): Maximum number of ensembles to try (default=100).
        metric (str): Metric to use to compare ensembles (default=micro_roc).
        annotations_path (str): Path to CSV file containing ground truth annotations.
        recordings_path (str, optional): Directory containing audio recordings. Defaults to annotations directory.
    """
    import glob
    import itertools
    import math
    import random

    if metric not in ["macro_pr", "micro_pr", "macro_roc", "micro_roc"]:
        logging.error(f"Error: invalid metric ({metric})")
        return

    cfg = get_config(cfg_path)
    ckpt_paths = sorted(glob.glob(os.path.join(ckpt_path, "*.ckpt")))
    num_ckpts = len(ckpt_paths)
    if num_ckpts == 0:
        logging.error(f"Error: no checkpoints found in {ckpt_path}")
        return
    elif num_ckpts < ensemble_size:
        logging.error(f"Error: number of checkpoints ({num_ckpts}) is less than requested ensemble size ({ensemble_size})")
        return

    if not recordings_path:
        recordings_path = str(Path(annotations_path).parent)

    with tempfile.TemporaryDirectory() as temp_dir:
        cfg.misc.ckpt_folder = temp_dir
        cfg.infer.min_score = 0

        best_score = 0
        best_ensemble = None
        count = 1
        total_combinations = math.comb(len(ckpt_paths), ensemble_size)
        if total_combinations <= num_tries:
            # Exhaustive search
            logging.info("Doing exhaustive search")
            for ensemble in itertools.combinations(ckpt_paths, ensemble_size):
                scores = _eval_ensemble(ensemble, temp_dir, annotations_path, recordings_path)
                logging.info(f"For ensemble {count} of {total_combinations}, score = {scores[metric]:.4f}")
                if scores[metric] > best_score:
                    best_score = scores[metric]
                    best_ensemble = ensemble

                count += 1
        else:
            # Random sampling without replacement
            logging.info("Doing random sampling")
            seen: set = set()
            while len(seen) < num_tries:
                ensemble = tuple(sorted(random.sample(ckpt_paths, ensemble_size)))
                if ensemble not in seen:
                    seen.add(ensemble)
                    scores = _eval_ensemble(ensemble, temp_dir, annotations_path, recordings_path)
                    logging.info(f"For ensemble {count} of {num_tries}, score = {scores[metric]:.4f}")
                    if scores[metric] > best_score:
                        best_score = scores[metric]
                        best_ensemble = ensemble

                count += 1

    logging.info(f"Best score = {best_score:.4f}")

    assert best_ensemble is not None
    best_names = [Path(ckpt_path).name for ckpt_path in best_ensemble]
    logging.info(f"Best ensemble = {best_names}")

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
    "--ckpt_path",
    "ckpt_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing checkpoints."
)
@click.option(
    "-e",
    "--ensemble_size",
    "ensemble_size",
    type=int,
    default=3,
    help="Number of checkpoints in ensemble (default=3)."
)
@click.option(
    "-n",
    "--num_tries",
    "num_tries",
    type=int,
    default=100,
    help="Maximum number of ensembles to try (default=100)."
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
def _ensemble_cmd(
    cfg_path: Optional[str],
    ckpt_path: str,
    ensemble_size: int,
    num_tries: int,
    metric: str,
    annotations_path: str,
    recordings_path: Optional[str],
) -> None:
    util.set_logging()
    ensemble(cfg_path, ckpt_path, ensemble_size, num_tries, metric, annotations_path, recordings_path)
