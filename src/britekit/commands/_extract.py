# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import logging
from typing import Optional

import click

from britekit.core.config_loader import get_config
from britekit.core import util


def extract_all(
    cfg_path: Optional[str] = None,
    db_path: Optional[str] = None,
    cat_name: Optional[str] = None,
    class_code: Optional[str] = None,
    class_name: str = "",
    dir_path: str = "",
    overlap: Optional[float] = None,
    src_name: Optional[str] = None,
    spec_group: Optional[str] = None,
) -> None:
    """
    Extract all spectrograms from audio recordings and insert them into the training database.

    This command processes all audio files in a directory and extracts spectrograms using
    sliding windows with optional overlap. The spectrograms are then inserted into the
    training database for use in model training. If the specified class doesn't exist,
    it will be automatically created.

    Args:
    - cfg_path (str, optional): Path to YAML file defining configuration overrides.
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    - cat_name (str, optional): Category name for new class creation (e.g., "bird"). Defaults to "default".
    - class_code (str, optional): Class code for new class creation (e.g., "COYE").
    - class_name (str): Name of the class for the recordings (e.g., "Common Yellowthroat").
    - dir_path (str): Path to directory containing audio recordings to process.
    - overlap (float, optional): Spectrogram overlap in seconds. Defaults to config value.
    - src_name (str, optional): Source name for the recordings (e.g., "Xeno-Canto"). Defaults to "default".
    - spec_group (str, optional): Spectrogram group name for organizing extractions. Defaults to "default".
    """
    from britekit.training_db.extractor import Extractor
    from britekit.training_db.training_db import TrainingDatabase

    cfg = get_config(cfg_path)
    if db_path is not None:
        cfg.train.train_db = db_path

    with TrainingDatabase(cfg.train.train_db) as db:
        extractor = Extractor(
            db, class_name, class_code, cat_name, src_name, overlap, spec_group
        )
        count = extractor.extract_all(dir_path)
        db.optimize()
        logging.info(f"Inserted {count} spectrograms")


@click.command(
    name="extract-all",
    short_help="Insert all spectrograms from recordings into database.",
    help=util.cli_help_from_doc(extract_all.__doc__),
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
    "--cat",
    "cat_name",
    required=False,
    help="Category name, e.g. 'bird' for when new class is added. Defaults to 'default'.",
)
@click.option(
    "--code",
    "class_code",
    required=False,
    help="Class code for when new class is added.",
)
@click.option("--name", "class_name", required=True, help="Class name.")
@click.option(
    "--dir",
    "dir_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Path to directory containing recordings.",
)
@click.option(
    "--overlap",
    "overlap",
    type=float,
    required=False,
    help="Spectrogram overlap in seconds. Defaults to value in the config file.",
)
@click.option(
    "--src",
    "src_name",
    required=False,
    help="Source name for inserted recordings. Defaults to 'default'.",
)
@click.option(
    "--sgroup",
    "spec_group",
    required=False,
    help="Spectrogram group name. Defaults to 'default'.",
)
def _extract_all_cmd(
    cfg_path: Optional[str],
    db_path: Optional[str],
    cat_name: Optional[str],
    class_code: Optional[str],
    class_name: str,
    dir_path: str,
    overlap: Optional[float],
    src_name: Optional[str],
    spec_group: Optional[str],
) -> None:
    util.set_logging()
    extract_all(
        cfg_path,
        db_path,
        cat_name,
        class_code,
        class_name,
        dir_path,
        overlap,
        src_name,
        spec_group,
    )


def extract_by_csv(
    cfg_path: Optional[str] = None,
    db_path: Optional[str] = None,
    cat_name: Optional[str] = None,
    class_code: Optional[str] = None,
    class_name: str = "",
    rec_dir: str = "",
    csv_path: str = "",
    dest_dir: Optional[str] = None,
    src_name: Optional[str] = None,
    spec_group: Optional[str] = None,
) -> None:
    """
    Extract spectrograms that correspond to rows in a CSV file.

    This command parses a CSV file to identify the corresponding audio
    segments and extracts those spectrograms from the original recordings.
    This is useful when you have pre-selected spectrograms (e.g., from manual review
    or search results) and want to extract only those specific segments. The CSV file
    needs two columns: recording and start_time, where recording is the stem of the
    recording file name (e.g. XC12345) and start_time is the offset in seconds from the
    start of the recording.

    Args:
    - cfg_path (str, optional): Path to YAML file defining configuration overrides.
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    - cat_name (str, optional): Category name for new class creation (e.g., "bird"). Defaults to "default".
    - class_code (str, optional): Class code for new class creation (e.g., "COYE").
    - class_name (str): Name of the class for the recordings (e.g., "Common Yellowthroat").
    - rec_dir (str): Path to directory containing the original audio recordings.
    - csv_path (str): Path to CSV file containing two columns (recording and offset) to identify segments to extract.
    - dest_dir (str, optional): If specified, copy used recordings to this directory.
    - src_name (str, optional): Source name for the recordings (e.g., "Xeno-Canto"). Defaults to "default".
    - spec_group (str, optional): Spectrogram group name for organizing extractions. Defaults to "default".
    """
    from britekit.training_db.extractor import Extractor
    from britekit.training_db.training_db import TrainingDatabase

    cfg = get_config(cfg_path)
    if db_path is not None:
        cfg.train.train_db = db_path

    with TrainingDatabase(cfg.train.train_db) as db:
        extractor = Extractor(
            db, class_name, class_code, cat_name, src_name, spec_group=spec_group
        )
        count = extractor.extract_by_csv(rec_dir, csv_path, dest_dir)
        db.optimize()
        logging.info(f"Inserted {count} spectrograms")


@click.command(
    name="extract-by-csv",
    short_help="Insert spectrograms that correspond to rows in a CSV file.",
    help=util.cli_help_from_doc(extract_by_csv.__doc__),
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
    "--cat",
    "cat_name",
    required=False,
    help="Category name, e.g. 'bird' for when new class is added. Defaults to 'default'.",
)
@click.option(
    "--code",
    "class_code",
    required=False,
    help="Class code for when new class is added.",
)
@click.option("--name", "class_name", required=True, help="Class name.")
@click.option(
    "--rec-dir",
    "rec_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Path to directory containing recordings.",
)
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to CSV file containing two columns (recording and offset) to identify segments to extract.",
)
@click.option(
    "--dest-dir",
    "dest_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=False,
    help="Copy used recordings to this directory if specified.",
)
@click.option(
    "--src",
    "src_name",
    required=False,
    help="Source name for inserted recordings. Defaults to 'default'.",
)
@click.option(
    "--sgroup",
    "spec_group",
    required=False,
    help="Spectrogram group name. Defaults to 'default'.",
)
def _extract_by_csv_cmd(
    cfg_path: Optional[str],
    db_path: Optional[str],
    cat_name: Optional[str],
    class_code: Optional[str],
    class_name: str,
    rec_dir: str,
    csv_path: str,
    dest_dir: Optional[str],
    src_name: Optional[str],
    spec_group: Optional[str],
) -> None:
    util.set_logging()
    extract_by_csv(
        cfg_path,
        db_path,
        cat_name,
        class_code,
        class_name,
        rec_dir,
        csv_path,
        dest_dir,
        src_name,
        spec_group,
    )


def extract_by_image(
    cfg_path: Optional[str] = None,
    db_path: Optional[str] = None,
    cat_name: Optional[str] = None,
    class_code: Optional[str] = None,
    class_name: str = "",
    rec_dir: str = "",
    spec_dir: str = "",
    dest_dir: Optional[str] = None,
    src_name: Optional[str] = None,
    spec_group: Optional[str] = None,
) -> None:
    """
    Extract spectrograms that correspond to existing spectrogram images.

    This command parses spectrogram image filenames to identify the corresponding audio
    segments and extracts those specific spectrograms from the original recordings.
    This is useful when you have pre-selected spectrograms (e.g., from manual review
    or search results) and want to extract only those specific segments.

    The images contain metadata in their filenames (recording name and time offset)
    that allows the command to locate and extract the corresponding audio segments.

    Args:
    - cfg_path (str, optional): Path to YAML file defining configuration overrides.
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    - cat_name (str, optional): Category name for new class creation (e.g., "bird"). Defaults to "default".
    - class_code (str, optional): Class code for new class creation (e.g., "COYE").
    - class_name (str): Name of the class for the recordings (e.g., "Common Yellowthroat").
    - rec_dir (str): Path to directory containing the original audio recordings.
    - spec_dir (str): Path to directory containing spectrogram image files.
    - dest_dir (str, optional): If specified, copy used recordings to this directory.
    - src_name (str, optional): Source name for the recordings (e.g., "Xeno-Canto"). Defaults to "default".
    - spec_group (str, optional): Spectrogram group name for organizing extractions. Defaults to "default".
    """
    from britekit.training_db.extractor import Extractor
    from britekit.training_db.training_db import TrainingDatabase

    cfg = get_config(cfg_path)
    if db_path is not None:
        cfg.train.train_db = db_path

    with TrainingDatabase(cfg.train.train_db) as db:
        extractor = Extractor(
            db, class_name, class_code, cat_name, src_name, spec_group=spec_group
        )
        count = extractor.extract_by_image(rec_dir, spec_dir, dest_dir)
        db.optimize()
        logging.info(f"Inserted {count} spectrograms")


@click.command(
    name="extract-by-image",
    short_help="Insert spectrograms that correspond to images.",
    help=util.cli_help_from_doc(extract_by_image.__doc__),
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
    "--cat",
    "cat_name",
    required=False,
    help="Category name, e.g. 'bird' for when new class is added. Defaults to 'default'.",
)
@click.option(
    "--code",
    "class_code",
    required=False,
    help="Class code for when new class is added.",
)
@click.option("--name", "class_name", required=True, help="Class name.")
@click.option(
    "--rec-dir",
    "rec_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Path to directory containing recordings.",
)
@click.option(
    "--spec-dir",
    "spec_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Path to directory containing spectrogram images.",
)
@click.option(
    "--dest-dir",
    "dest_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=False,
    help="Copy used recordings to this directory if specified.",
)
@click.option(
    "--src",
    "src_name",
    required=False,
    help="Source name for inserted recordings. Defaults to 'default'.",
)
@click.option(
    "--sgroup",
    "spec_group",
    required=False,
    help="Spectrogram group name. Defaults to 'default'.",
)
def _extract_by_image_cmd(
    cfg_path: Optional[str],
    db_path: Optional[str],
    cat_name: Optional[str],
    class_code: Optional[str],
    class_name: str,
    rec_dir: str,
    spec_dir: str,
    dest_dir: Optional[str],
    src_name: Optional[str],
    spec_group: Optional[str],
) -> None:
    util.set_logging()
    extract_by_image(
        cfg_path,
        db_path,
        cat_name,
        class_code,
        class_name,
        rec_dir,
        spec_dir,
        dest_dir,
        src_name,
        spec_group,
    )
