# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import logging
import os
import re
from typing import Optional, Dict, List

import click

from britekit.core.config_loader import get_config
from britekit.core import util


def del_cat(db_path: Optional[str] = None, name: Optional[str] = None) -> None:
    """
    Delete a category and all its associated data from the training database.

    This command performs a cascading delete that removes the category, all its classes,
    all recordings belonging to those classes, and all spectrograms from those recordings.
    This is a destructive operation that cannot be undone.

    Args:
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    - name (str): Name of the category to delete (e.g., "Birds", "Mammals").
    """
    from britekit.training_db.training_db import TrainingDatabase

    cfg = get_config()
    if db_path is None:
        db_path = cfg.train.train_db

    if name is None:
        logging.error("Error: category name is missing but required.")
        quit()

    with TrainingDatabase(db_path) as db:
        results = db.get_category({"Name": name})
        if not results:
            logging.error(f"No category found with name {name}")
        else:
            cat_id = results[0].id
            class_results = db.get_class({"CategoryID": cat_id})
            for c in class_results:
                logging.info(f'Deleting class "{c.name}"')
                rec_results = db.get_recording_by_class(c.name)
                for r in rec_results:
                    db.delete_recording({"ID": r.id})

            db.delete_category({"Name": name})
            logging.info(f'Successfully deleted category "{name}"')


@click.command(
    name="del-cat",
    short_help="Delete a category (class group) and its classes from a database.",
    help=util.cli_help_from_doc(del_cat.__doc__),
)
@click.option(
    "-d", "--db", "db_path", required=False, help="Path to the training database."
)
@click.option("--name", "name", required=True, help="Category name.")
def _del_cat_cmd(db_path: Optional[str], name: str) -> None:
    util.set_logging()
    del_cat(db_path, name)


def del_class(db_path: Optional[str] = None, name: Optional[str] = None) -> None:
    """
    Delete a class and all its associated data from the training database.

    This command removes the class, all recordings belonging to that class, and all
    spectrograms from those recordings. This is a destructive operation that cannot
    be undone and will affect any training data associated with this class.

    Args:
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    - name (str): Name of the class to delete (e.g., "Common Yellowthroat").
    """
    from britekit.training_db.training_db import TrainingDatabase

    cfg = get_config()
    if db_path is None:
        db_path = cfg.train.train_db

    if name is None:
        logging.error("Error: class name is missing but required.")
        quit()

    with TrainingDatabase(db_path) as db:
        results = db.get_class({"Name": name})
        if not results:
            logging.error(f"No class found with name {name}")
        else:
            # cascading deletes don't fully handle this case,
            # so have to delete recordings first
            results = db.get_recording_by_class(name)
            for r in results:
                db.delete_recording({"ID": r.id})

            db.delete_class({"Name": name})
            logging.info(f'Successfully deleted class "{name}"')


@click.command(
    name="del-class",
    short_help="Delete a class and associated records from a database.",
    help=util.cli_help_from_doc(del_class.__doc__),
)
@click.option(
    "-d", "--db", "db_path", required=False, help="Path to the training database."
)
@click.option("--name", "class_name", required=True, help="Class name.")
def _del_class_cmd(db_path: Optional[str], class_name: str) -> None:
    util.set_logging()
    del_class(db_path, class_name)


def del_rec(db_path: Optional[str] = None, file_name: Optional[str] = None) -> None:
    """
    Delete a recording and all its spectrograms from the training database.

    This command removes a specific audio recording and all spectrograms that were
    extracted from it.

    Args:
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    - file_name (str): Name of the recording file to delete (e.g., "XC123456.mp3").
    """
    from britekit.training_db.training_db import TrainingDatabase

    cfg = get_config()
    if db_path is None:
        db_path = cfg.train.train_db

    if file_name is None:
        logging.error("Error: file name is missing but required.")
        quit()

    with TrainingDatabase(db_path) as db:
        results = db.get_recording({"FileName": file_name})
        if not results:
            logging.error(f"No recording found with file name {file_name}")
        else:
            db.delete_recording({"FileName": file_name})
            logging.info(f'Successfully deleted recording "{file_name}"')


@click.command(
    name="del-rec",
    short_help="Delete a recording and associated records from a database.",
    help=util.cli_help_from_doc(del_rec.__doc__),
)
@click.option(
    "-d", "--db", "db_path", required=False, help="Path to the training database."
)
@click.option("--name", "file_name", required=True, help="Recording file name.")
def _del_rec_cmd(db_path: Optional[str], file_name: str) -> None:
    util.set_logging()
    del_rec(db_path, file_name)


def del_sgroup(db_path: Optional[str] = None, name: Optional[str] = None) -> None:
    """
    Delete a spectrogram group and all its spectrogram values from the training database.

    Spectrogram groups organize spectrograms by processing parameters or extraction method.
    This command removes the entire group and all spectrograms within it.

    Args:
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    - name (str): Name of the spectrogram group to delete (e.g., "default", "augmented").
    """
    from britekit.training_db.training_db import TrainingDatabase

    cfg = get_config()
    if db_path is None:
        db_path = cfg.train.train_db

    if name is None:
        logging.error("Error: name is missing but required.")
        quit()

    with TrainingDatabase(db_path) as db:
        results = db.get_specgroup({"Name": name})
        if not results:
            logging.error(f"No spectrogram group found with name {name}")
        else:
            db.delete_specgroup({"ID": results[0].id})
            logging.info(f'Successfully deleted spectrogram group "{name}"')


@click.command(
    name="del-sgroup",
    short_help="Delete a spectrogram group from the database.",
    help=util.cli_help_from_doc(del_sgroup.__doc__),
)
@click.option(
    "-d", "--db", "db_path", required=False, help="Path to the training database."
)
@click.option("--name", "name", required=True, help="Spec group name.")
def _del_sgroup_cmd(db_path: Optional[str], name: str) -> None:
    util.set_logging()
    del_sgroup(db_path, name)


def del_stype(db_path: Optional[str] = None, name: Optional[str] = None) -> None:
    """
    Delete a sound type from the training database.

    This command removes a sound type definition but preserves the spectrograms that were
    labeled with this sound type. The spectrograms will have their soundtype_id field set
    to null, effectively removing the sound type classification while keeping the audio data.

    Args:
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    - name (str): Name of the sound type to delete (e.g., "Song", "Call", "Alarm").
    """
    from britekit.training_db.training_db import TrainingDatabase

    cfg = get_config()
    if db_path is None:
        db_path = cfg.train.train_db

    if name is None:
        logging.error("Error: name is missing but required.")
        quit()

    with TrainingDatabase(db_path) as db:
        results = db.get_soundtype({"Name": name})
        if not results:
            logging.error(f"No sound type found with name {name}")
        else:
            db.delete_soundtype({"Name": name})
            logging.info(f'Successfully deleted sound type "{name}"')


@click.command(
    name="del-stype",
    short_help="Delete a sound type from a database.",
    help=util.cli_help_from_doc(del_stype.__doc__),
)
@click.option(
    "-d", "--db", "db_path", required=False, help="Path to the training database."
)
@click.option("--name", "name", required=True, help="Sound type name.")
def _del_stype_cmd(db_path: Optional[str], name: str) -> None:
    util.set_logging()
    del_stype(db_path, name)


def del_src(db_path: Optional[str] = None, name: Optional[str] = None) -> None:
    """
    Delete a recording source and all its associated data from the training database.

    This command performs a cascading delete that removes the source, all recordings
    from that source, and all spectrograms from those recordings. This is useful for
    removing entire datasets from a specific source (e.g., removing all Xeno-Canto data).

    Args:
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    - name (str): Name of the source to delete (e.g., "Xeno-Canto", "Macaulay Library").
    """
    from britekit.training_db.training_db import TrainingDatabase

    cfg = get_config()
    if db_path is None:
        db_path = cfg.train.train_db

    if name is None:
        logging.error("Error: name is missing but required.")
        quit()

    with TrainingDatabase(db_path) as db:
        results = db.get_source({"Name": name})
        if not results:
            logging.error(f"No source found with name {name}")
        else:
            db.delete_source({"Name": name})
            logging.info(f'Successfully deleted source "{name}"')


@click.command(
    name="del-src",
    short_help="Delete a recording source and associated records from a database.",
    help=util.cli_help_from_doc(del_src.__doc__),
)
@click.option(
    "-d", "--db", "db_path", required=False, help="Path to the training database."
)
@click.option("--name", "name", required=True, help="Source name.")
def _del_src_cmd(db_path: Optional[str], name: str) -> None:
    util.set_logging()
    del_src(db_path, name)


def del_seg(
    db_path: Optional[str] = None,
    class_name: Optional[str] = None,
    csv_path: Optional[str] = None,
    dir_path: Optional[str] = None,
) -> None:
    """
    Delete segments that correspond to images in a given directory.

    This command parses image filenames to identify and delete corresponding segments
    from the database. Images are typically generated by the plot-db or search commands,
    and their filenames contain the recording name and time offset.

    This is useful for removal of segments based on visual inspection of plots,
    allowing you to remove low-quality or incorrectly labeled segments.
    Exactly one of the csv_path and dir_path arguments must be specified.

    Args:
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    - class_name (str): Name of the class whose segments should be considered for deletion.
    - csv_path (str): Path to CSV file containing two columns (recording and offset) to identify segments to extract.
    - dir_path (str): Path to directory containing spectrogram image files.
    """
    from britekit.training_db.training_db import TrainingDatabase

    assert csv_path or dir_path, "Either csv_path or dir_path must be specified."
    assert not (
        csv_path and dir_path
    ), "Only one of csv_path and dir_path may be specified."

    cfg = get_config()
    if db_path is None:
        db_path = cfg.train.train_db

    if class_name is None:
        logging.error("Error: class name is missing but required.")
        return

    with TrainingDatabase(db_path) as db:
        count = db.get_class_count({"Name": class_name})
        if count == 0:
            logging.error(f"Error: class {class_name} not found")
            return
        elif count > 1:
            logging.error(f"Error: found multiple classes called {class_name}")
            return

        recording_dict: Dict[str, int] = {}
        results = db.get_recording_by_class(class_name)
        logging.info(f"Found {len(results)} recordings")
        for r in results:
            tokens = r.filename.split(".")
            recording_dict[tokens[0]] = r.id

        # collect the info
        offsets_per_file: dict[str, list] = {}
        if csv_path:
            import pandas as pd

            df = pd.read_csv(csv_path)
            for i, row in df.iterrows():
                recording = row["recording"]
                if recording not in offsets_per_file:
                    offsets_per_file[recording] = []

                offsets_per_file[recording].append(row["offset"])
        else:
            assert dir_path is not None
            file_names: List[str] = os.listdir(dir_path)
            for file_name in file_names:
                if os.path.isfile(os.path.join(dir_path, file_name)):
                    base, ext = os.path.splitext(file_name)
                    if ext == ".jpeg":
                        if "~" in base:
                            result = re.split("\\S+~(\\S+)-(\\S+)~.*", base)
                        else:
                            result = re.split("(.+)-(.+)", base)

                        if len(result) != 4:
                            logging.error(
                                f"Error: unknown file name format: {base} (ignored)"
                            )
                            continue
                        else:
                            recording = result[1]
                            offset = float(result[2])
                            if recording not in offsets_per_file:
                                offsets_per_file[recording] = []

                            offsets_per_file[recording].append(offset)

        # delete the segments
        deleted = 0
        for recording in sorted(offsets_per_file):
            if recording not in recording_dict.keys():
                logging.error(f"Error: recording not found: {recording}")
                continue

            recording_id = recording_dict[recording]
            for offset in offsets_per_file[recording]:
                result = db.get_segment({"RecordingID": recording_id, "Offset": offset})
                if result is None:
                    logging.error(f"Error: segment not found: {recording}-{offset}")
                else:
                    # should only be one, but loop just in case
                    for r in result:
                        logging.info(f"Deleting segment ID {r.id}")
                        db.delete_segment({"ID": r.id})
                        deleted += 1

        logging.info(f"Deleted {deleted} segments")


@click.command(
    name="del-seg",
    short_help="Delete segments that match given images.",
    help=util.cli_help_from_doc(del_seg.__doc__),
)
@click.option(
    "-d", "--db", "db_path", required=False, help="Path to the training database."
)
@click.option("--name", "class_name", required=True, help="Class name.")
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=False,
    help="Path to CSV file containing two columns (recording and offset) to identify segments to delete. Exactly one of --csv and --dir must be specified.",
)
@click.option(
    "--dir",
    "dir_path",
    required=False,
    help="Path to directory containing images. Exactly one of --csv and --dir must be specified.",
)
def _del_seg_cmd(
    db_path: Optional[str],
    class_name: str,
    csv_path: Optional[str],
    dir_path: Optional[str],
) -> None:
    util.set_logging()
    if not csv_path and not dir_path:
        logging.error("Either --csv or --dir must be specified.")
        return

    if csv_path and dir_path:
        logging.error("Only one of --csv or --dir may be specified.")
        return

    del_seg(db_path, class_name, csv_path, dir_path)
