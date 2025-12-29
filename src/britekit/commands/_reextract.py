#!/usr/bin/env python3

# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import logging
from pathlib import Path
import time
from typing import Optional

import click

from britekit.core.config_loader import get_config
from britekit.core import util


def reextract(
    cfg_path: Optional[str] = None,
    db_path: Optional[str] = None,
    class_name: Optional[str] = None,
    classes_path: Optional[str] = None,
    check: bool = False,
    spec_group: str = "default",
):
    """
    Re-generate spectrograms from audio recordings and update the training database.

    This command extracts spectrograms from audio recordings and imports them into the training database.
    It can process all classes in the database or specific classes specified by name or CSV file.
    If the specified spectrogram group already exists, it will be deleted and recreated.

    In check mode, it only verifies that all required audio files are accessible without
    updating the database.

    Args:
    - cfg_path (str, optional): Path to YAML file defining configuration overrides.
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.training_db.
    - class_name (str, optional): Name of a specific class to reextract. If omitted, processes all classes.
    - classes_path (str, optional): Path to CSV file listing classes to reextract. Alternative to class_name.
    - check (bool): If True, only check that all recording paths are accessible without updating database.
    - spec_group (str): Spectrogram group name for storing the extracted spectrograms. Defaults to 'default'.
    """
    from britekit.core.reextractor import Reextractor
    from britekit.training_db.training_db import TrainingDatabase

    cfg = get_config(cfg_path)

    if class_name and classes_path:
        logging.error("Only one of --name and --classes may be specified.")
        return

    if db_path is None:
        db_path = str(Path(cfg.train.train_db).resolve())

    start_time = time.time()
    Reextractor(db_path, class_name, classes_path, check, spec_group).run()

    with TrainingDatabase(cfg.train.train_db) as db:
        db.optimize()

    elapsed_time = util.format_elapsed_time(start_time, time.time())
    logging.info(f"Elapsed time = {elapsed_time}")


@click.command(
    name="reextract",
    short_help="Re-generate the spectrograms in a database, and add them to the database.",
    help=util.cli_help_from_doc(reextract.__doc__),
)
@click.option(
    "-c",
    "--cfg",
    "cfg_path",
    type=click.Path(exists=True),
    help="Path to YAML file defining config overrides.",
)
@click.option(
    "-d",
    "--db",
    "db_path",
    type=click.Path(file_okay=True, dir_okay=False),
    help="Path to the database. Defaults to value of cfg.train.training_db.",
)
@click.option(
    "--name",
    "class_name",
    type=str,
    help="Optional class name. If this and --classes are omitted, do all classes.",
)
@click.option(
    "--classes",
    "classes_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to CSV listing classes to reextract. Alternative to --name. If this and --name are omitted, do all classes.",
)
@click.option(
    "--check",
    "check",
    is_flag=True,
    help="If specified, just check if all specified recordings are accessible and do not update the database.",
)
@click.option(
    "--sgroup",
    "spec_group",
    required=False,
    default="default",
    help="Spectrogram group name. Defaults to 'default'.",
)
def _reextract_cmd(
    cfg_path: Optional[str] = None,
    db_path: Optional[str] = None,
    class_name: Optional[str] = None,
    classes_path: Optional[str] = None,
    check: bool = False,
    spec_group: str = "default",
):
    util.set_logging()
    reextract(cfg_path, db_path, class_name, classes_path, check, spec_group)
