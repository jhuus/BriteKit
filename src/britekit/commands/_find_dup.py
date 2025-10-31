# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import logging
from typing import List, Optional
import zlib

import click

from britekit.core.config_loader import get_config
from britekit.core import util
from britekit.training_db.training_db import TrainingDatabase
from britekit.training_db.training_data_provider import TrainingDataProvider


def find_dup(
    cfg_path: Optional[str]=None,
    db_path: Optional[str]=None,
    class_name: str="",
    delete: bool=False,
    spec_group: str="default",
) -> None:
    """
    Find and optionally delete duplicate recordings in the training database.

    This command scans the database for recordings of the same class that appear to be duplicates.
    It uses a two-stage detection approach:
    1. Compare recording durations (within 0.1 seconds tolerance)
    2. Compare spectrogram embeddings of the first few spectrograms (within 0.02 cosine distance)

    Duplicates are identified by comparing the first 3 spectrogram embeddings from each recording
    using cosine distance.

    Args:
    - cfg_path (str, optional): Path to YAML file defining configuration overrides.
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    - class_name (str): Name of the class to scan for duplicates (e.g., "Common Yellowthroat").
    - delete (bool): If True, remove duplicate recordings from the database. If False, only report them.
    - spec_group (str): Spectrogram group name to use for embedding comparison. Defaults to "default".
    """

    class Recording:
        def __init__(self, id: int, filename: str, seconds: float):
            self.id: int = id
            self.filename: str = filename
            self.seconds: float = seconds
            self.embeddings: List = []

    def get_spectrogram_embeddings(
        db: TrainingDatabase, recording: Recording, specgroup_id: int
    ) -> None:
        import numpy as np

        results = db.get_specvalue(
            {"RecordingID": recording.id, "SpecGroupID": specgroup_id}
        )
        if len(results) == 0:
            logging.error(f"Error: no matching spectrograms for {recording.filename}.")
            quit()

        for i in range(min(3, len(results))):  # just need the first few for comparison
            r = results[i]
            if r.embedding is None:
                logging.error(f"Error: embeddings missing for {recording.filename}.")
                quit()

            recording.embeddings.append(
                np.frombuffer(zlib.decompress(r.embedding), dtype=np.float32)
            )

    def get_recordings(
        db: TrainingDatabase, class_name: str, specgroup_id: int
    ) -> List[Recording]:
        recordings: List[Recording] = []
        results = db.get_recording_by_class(class_name)
        for r in results:
            recording = Recording(r.id, r.filename, r.seconds)
            recordings.append(recording)
            get_spectrogram_embeddings(db, recording, specgroup_id)

        return recordings

    # return true iff the two recordings appear to be duplicates
    def match_recordings(recording1: Recording, recording2: Recording) -> bool:
        import scipy

        SECONDS_FUDGE = 0.1  # treat durations as equal if within this many seconds
        DISTANCE_FUDGE = 0.02  # treat spectrograms as equal if within this distance

        if (recording1.seconds > recording2.seconds - SECONDS_FUDGE) and (
            recording1.seconds < recording2.seconds + SECONDS_FUDGE
        ):
            if len(recording1.embeddings) == 0 or len(recording2.embeddings) == 0:
                return False

            if len(recording1.embeddings) == len(recording2.embeddings):
                for i in range(len(recording1.embeddings)):
                    distance = scipy.spatial.distance.cosine(
                        recording1.embeddings[i], recording2.embeddings[i]
                    )
                    if distance > DISTANCE_FUDGE:
                        return False
                return True
            else:
                return False
        else:
            return False

    cfg = get_config(cfg_path)
    if db_path is None:
        db_path = cfg.train.train_db

    # get recordings from the database
    logging.info("Opening database")
    db = TrainingDatabase(db_path)
    provider = TrainingDataProvider(db)
    specgroup_id = provider.specgroup_id(spec_group)

    recordings = get_recordings(db, class_name, specgroup_id)
    logging.info(f"Fetched {len(recordings)} recordings")

    # sort recordings by length, then process in a loop
    recordings = sorted(recordings, key=lambda recording: recording.seconds)
    i = 0
    while i < len(recordings) - 1:
        if match_recordings(recordings[i], recordings[i + 1]):
            logging.info(
                f"{recordings[i].filename} and {recordings[i + 1].filename} are possible duplicates"
            )
            if delete:
                logging.info(f"Removing {recordings[i].filename} from database")
                db.delete_recording({"ID": recordings[i].id})

            i += 2
        else:
            i += 1


@click.command(
    name="find-dup",
    short_help="Find and optionally delete duplicate recordings in a database.",
    help=util.cli_help_from_doc(find_dup.__doc__),
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
    type=click.Path(file_okay=True, dir_okay=False),
    help="Path to the database. Defaults to value of cfg.train.training_db.",
)
@click.option("--name", "class_name", type=str, required=True, help="Class name")
@click.option(
    "--del",
    "delete",
    is_flag=True,
    help="If specified, remove duplicate recordings from the database.",
)
@click.option(
    "--sgroup",
    "spec_group",
    required=False,
    default="default",
    help="Spectrogram group name. Defaults to 'default'.",
)
def _find_dup_cmd(
    cfg_path: Optional[str],
    db_path: Optional[str],
    class_name: str,
    delete: bool,
    spec_group: str,
) -> None:
    util.set_logging()
    find_dup(cfg_path, db_path, class_name, delete, spec_group)
