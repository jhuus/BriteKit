# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import logging
from typing import Optional

import click

from britekit.core.config_loader import get_config
from britekit.core import util


def add_cat(db_path: Optional[str] = None, name: str = "") -> None:
    """
    Add a category (class group) record to the training database.

    Categories are used to group related classes together in the database.
    For example, you might have categories like "Birds", "Mammals", or "Insects"
    that contain multiple related species classes.

    Args:
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    - name (str): Name of the category to add (e.g., "Birds", "Mammals").
    """
    from britekit.training_db.training_db import TrainingDatabase

    cfg = get_config()
    if db_path is None:
        db_path = cfg.train.train_db

    with TrainingDatabase(db_path) as db:
        results = db.get_category({"Name": name})
        if results:
            logging.error(f"Error: category {name} already exists.")
            quit()

        db.insert_category(name)
        logging.info(f"Successfully added category {name}")


@click.command(
    name="add-cat",
    short_help="Add a category (class group) record to a database.",
    help=util.cli_help_from_doc(add_cat.__doc__),
)
@click.option("-d", "--db", "db_path", required=False, help="Path to the database.")
@click.option("--name", "name", required=True, help="Category name")
def _add_cat_cmd(db_path: Optional[str], name: str) -> None:
    util.set_logging()
    add_cat(db_path, name)


def add_stype(db_path: Optional[str] = None, name: str = "") -> None:
    """
    Add a sound type record to the training database.

    Sound types describe the nature of the audio content, such as "Song", "Call",
    "Alarm", "Drumming", etc. This helps categorize different types of vocalizations
    or sounds produced by the same species.

    Args:
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    - name (str): Name of the sound type to add (e.g., "Song", "Call", "Alarm").
    """
    from britekit.training_db.training_db import TrainingDatabase

    cfg = get_config()
    if db_path is None:
        db_path = cfg.train.train_db

    with TrainingDatabase(db_path) as db:
        results = db.get_soundtype({"Name": name})
        if results:
            logging.error(f"Error: soundtype {name} already exists.")
            quit()

        db.insert_soundtype(name)
        logging.info(f"Successfully added soundtype {name}")


@click.command(
    name="add-stype",
    short_help="Add a soundtype record to a database.",
    help=util.cli_help_from_doc(add_stype.__doc__),
)
@click.option("-d", "--db", "db_path", required=False, help="Path to the database.")
@click.option("--name", "name", required=True, help="Soundtype name")
def _add_stype_cmd(db_path: Optional[str], name: str) -> None:
    util.set_logging()
    add_stype(db_path, name)


def add_src(db_path: Optional[str] = None, name: str = "") -> None:
    """
    Add a source record to the training database.

    Sources track where audio recordings originated from, such as "Xeno-Canto",
    "Macaulay Library", "iNaturalist", or custom field recordings. This helps
    maintain provenance and can be useful for data quality analysis.

    Args:
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    - name (str): Name of the source to add (e.g., "Xeno-Canto", "Macaulay Library").
    """
    from britekit.training_db.training_db import TrainingDatabase

    cfg = get_config()
    if db_path is None:
        db_path = cfg.train.train_db

    with TrainingDatabase(db_path) as db:
        results = db.get_source({"Name": name})
        if results:
            logging.error(f"Error: source {name} already exists.")
            quit()

        db.insert_source(name)
        logging.info(f"Successfully added source {name}")


@click.command(
    name="add-src",
    short_help="Add a source (e.g. 'Xeno-Canto') record to a database.",
    help=util.cli_help_from_doc(add_src.__doc__),
)
@click.option("-d", "--db", "db_path", required=False, help="Path to the database.")
@click.option("--name", "name", required=True, help="Source name")
def _add_src_cmd(db_path: Optional[str], name: str) -> None:
    util.set_logging()
    add_src(db_path, name)


def add_class(
    db_path: Optional[str] = None,
    category: str = "default",
    name: Optional[str] = None,
    code: Optional[str] = None,
    alt_name: Optional[str] = None,
    alt_code: Optional[str] = None,
) -> None:
    """
    Add a class record to the training database.

    Classes represent the target species or sound categories for training and inference.
    Each class must belong to a category and can have both primary and alternate names/codes.
    This is typically used to add new species or sound types to the training database.

    Args:
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    - category (str): Name of the category this class belongs to. Defaults to "default".
    - name (str): Primary name of the class (e.g., "Common Yellowthroat").
    - code (str): Primary code for the class (e.g., "COYE").
    - alt_name (str, optional): Alternate name for the class (e.g., scientific name).
    - alt_code (str, optional): Alternate code for the class (e.g., scientific code).
    """
    from britekit.training_db.training_db import TrainingDatabase

    cfg = get_config()
    if db_path is None:
        db_path = cfg.train.train_db

    if name is None:
        logging.error("Error: class name is missing but required.")
        quit()

    with TrainingDatabase(db_path) as db:
        results = db.get_category({"Name": category})
        if not results:
            logging.error(f"Error: category {category} not found.")
            quit()

        category_id = results[0].id
        results = db.get_class({"Name": name})
        if results:
            logging.error(f"Error: class {name} already exists.")
            quit()

        db.insert_class(category_id, name, alt_name, code, alt_code)
        logging.info(f"Successfully added class {name}")


@click.command(
    name="add-class",
    short_help="Add a class record to a database.",
    help=util.cli_help_from_doc(add_class.__doc__),
)
@click.option("-d", "--db", "db_path", required=False, help="Path to the database.")
@click.option(
    "--cat", "category", required=False, default="default", help="Category name"
)
@click.option("--name", "name", required=True, help="Class name")
@click.option("--code", "code", required=True, help="Class code")
@click.option(
    "--alt_name", "alt_name", required=False, default="", help="Class alternate name"
)
@click.option(
    "--alt_code", "alt_code", required=False, default="", help="Class alternate code"
)
def _add_class_cmd(
    db_path: Optional[str],
    category: str,
    name: str,
    code: str,
    alt_name: str,
    alt_code: str,
) -> None:
    util.set_logging()
    add_class(db_path, category, name, code, alt_name, alt_code)
