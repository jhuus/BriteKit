# File name starts with _ to keep it out of typeahead for API users.
# Defer some imports to improve --help performance.
import logging
import os
from typing import List, Optional
import zlib

import click

from britekit.core.config_loader import get_config
from britekit.core import plot, util
from britekit.training_db.training_db import TrainingDatabase
from britekit.training_db.training_data_provider import TrainingDataProvider

# =============================================================================
# Helper Classes and Methods
# =============================================================================


class Recording:
    def __init__(self, id: int, filename: str, seconds: float):
        self.id: int = id
        self.filename: str = filename
        self.seconds: float = seconds
        self.embeddings: List = []
        self.segment_ids: List = []
        self.num_segments: int = 0


def _get_spectrogram_embeddings(
    db: TrainingDatabase,
    recording: Recording,
    specgroup_id: int,
    max_embeddings: Optional[int] = None,
) -> None:
    import numpy as np

    results = db.get_specvalue(
        {"RecordingID": recording.id, "SpecGroupID": specgroup_id}
    )
    if len(results) == 0:
        logging.error(f"Error: no matching spectrograms for {recording.filename}.")
        quit()

    recording.num_segments = len(results)
    num_to_get = len(results)
    if max_embeddings:
        # just need the first few for dedup-rec
        num_to_get = min(max_embeddings, num_to_get)

    for i in range(num_to_get):
        r = results[i]
        if r.embedding is None:
            logging.error(f"Error: embeddings missing for {recording.filename}.")
            quit()

        recording.embeddings.append(
            np.frombuffer(zlib.decompress(r.embedding), dtype=np.float32)
        )
        recording.segment_ids.append(r.segment_id)


def _get_recordings(
    db: TrainingDatabase,
    class_name: str,
    specgroup_id: int,
    max_embeddings: Optional[int] = None,
) -> List[Recording]:
    recordings: List[Recording] = []
    results = db.get_recording_by_class(class_name)
    for r in results:
        recording = Recording(r.id, r.filename, r.seconds)
        recordings.append(recording)
        _get_spectrogram_embeddings(db, recording, specgroup_id, max_embeddings)

    return recordings


# =============================================================================
# Commands
# =============================================================================


def dedup_rec(
    cfg_path: Optional[str] = None,
    db_path: Optional[str] = None,
    class_name: str = "",
    delete: bool = False,
    spec_group: str = "default",
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

    # return true iff the two recordings appear to be duplicates
    def _match_recordings(recording1: Recording, recording2: Recording) -> bool:
        import scipy

        SECONDS_FUDGE = 0.1  # treat durations as equal if within this many seconds
        DISTANCE_FUDGE = 0.001  # treat spectrograms as equal if within this distance

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

    max_embeddings = 3  # just compare the first few embeddings
    recordings = _get_recordings(db, class_name, specgroup_id, max_embeddings)
    logging.info(f"Fetched {len(recordings)} recordings")

    # sort recordings by length, then process in a loop
    recordings = sorted(recordings, key=lambda recording: recording.seconds)
    i = 0
    num_found = 0
    while i < len(recordings) - 1:
        if _match_recordings(recordings[i], recordings[i + 1]):
            num_found += 1
            logging.info(
                f"{recordings[i].filename} ({recordings[i].num_segments} segments) and {recordings[i + 1].filename} "
                f"({recordings[i].num_segments} segments) are possible duplicates"
            )
            if delete:
                logging.info(f"Removing {recordings[i].filename} from database")
                db.delete_recording({"ID": recordings[i].id})

            i += 2
        else:
            i += 1

    logging.info(f"Found {num_found} duplicate pairs")


@click.command(
    name="dedup-rec",
    short_help="Find and optionally delete duplicate recordings in a database.",
    help=util.cli_help_from_doc(dedup_rec.__doc__),
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
def _dedup_rec_cmd(
    cfg_path: Optional[str],
    db_path: Optional[str],
    class_name: str,
    delete: bool,
    spec_group: str,
) -> None:
    util.set_logging()
    dedup_rec(cfg_path, db_path, class_name, delete, spec_group)


def dedup_seg(
    cfg_path: Optional[str] = None,
    db_path: Optional[str] = None,
    output_path: str = "",
    class_name: str = "",
    delete: bool = False,
    spec_group: str = "default",
    threshold: float = 0.99,
    no_plot: bool = False,
) -> None:
    """
    Find and optionally delete duplicate segments in the training database.

    This command scans the database for segments of the same class that are very similar.

    Args:
    - cfg_path (str, optional): Path to YAML file defining configuration overrides.
    - db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    - output_path (str): Path to the output directory for reports and plots.
    - class_name (str): Name of the class to scan for duplicates (e.g., "Common Yellowthroat").
    - delete (bool): If True, remove duplicate segments from the database. If False, only report them.
    - spec_group (str): Spectrogram group name to use for embedding comparison. Defaults to "default".
    - threshold (float): Treat as duplicates if cosine similarity >= threshold (default = 0.99).
    - no_plot (bool): If specified, do not plot spectrograms.
    """

    def build_clusters(pairs, n):
        """Used to remove redundant entries from the duplicates list"""
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for a, b in pairs:
            union(a, b)

        clusters = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(i)

        return list(clusters.values())

    def prune_pairs(clusters):
        """Used to remove redundant entries from the duplicates list"""
        pruned = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            base = cluster[0]
            for other in cluster[1:]:
                pruned.append((base, other))
        return pruned

    import faiss
    import numpy as np

    cfg = get_config(cfg_path)
    if db_path is None:
        db_path = cfg.train.train_db

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get recordings from the database
    logging.info("Opening database")
    db = TrainingDatabase(db_path)
    provider = TrainingDataProvider(db)
    specgroup_id = provider.specgroup_id(spec_group)

    recordings = _get_recordings(db, class_name, specgroup_id)
    logging.info(f"Fetched {len(recordings)} recordings")

    # Create an array of embeddings and a matching array with (recording, segment_id) pairs
    embedding_list = []
    metadata = []  # (recording, segment_index) per corresponding embedding
    for recording in recordings:
        for i, embedding in enumerate(recording.embeddings):
            # normalize for use with FAISS
            embedding_list.append(embedding / np.linalg.norm(embedding))
            metadata.append((recording, recording.segment_ids[i]))

    # Build FAISS index for cosine similarity
    embeddings = np.array(embedding_list)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Search for top-10 near-duplicates of each
    num_matches = 10
    similarities, indexes = index.search(embeddings, k=num_matches)

    assert similarities.shape == (len(embeddings), num_matches)
    assert indexes.shape == (len(embeddings), num_matches)

    duplicates = set()

    for i in range(len(embeddings)):
        # Skip j = 0 because it's always the (i,i) self-match with similarity ~1.0
        for j in range(1, num_matches):
            if similarities[i][j] >= threshold:
                min_index = min(i, indexes[i][j].item())
                max_index = max(i, indexes[i][j].item())
                duplicates.add((min_index, max_index))

    # Prune duplicate entries
    sorted_duplicates = sorted(list(duplicates), key=lambda x: (x[0], x[1]))
    clusters = build_clusters(sorted_duplicates, len(embeddings))
    pruned_duplicates = prune_pairs(clusters)

    logging.info(f"Found {len(pruned_duplicates)} pairs of duplicates")

    # Optionally plot and/or delete the duplicates
    for i, (index1, index2) in enumerate(pruned_duplicates):
        recording1, segment_id1 = metadata[index1]
        if not no_plot:
            offset1 = db.get_segment({"ID": segment_id1})[0].offset
            spec1 = db.get_specvalue(
                {"SegmentID": segment_id1, "SpecGroupID": specgroup_id}
            )[0].value
            spec1 = util.expand_spectrogram(spec1)
            spec_path1 = os.path.join(
                output_path, f"{i + 1}-{recording1.filename}-{offset1:.2f}.jpeg"
            )
            if not os.path.exists(spec_path1):
                plot.plot_spec(spec1, spec_path1)

        recording2, segment_id2 = metadata[index2]
        if not no_plot:
            offset2 = db.get_segment({"ID": segment_id2})[0].offset
            spec2 = db.get_specvalue(
                {"SegmentID": segment_id2, "SpecGroupID": specgroup_id}
            )[0].value
            spec2 = util.expand_spectrogram(spec2)
            spec_path2 = os.path.join(
                output_path, f"{i + 1}-{recording2.filename}-{offset2:.2f}.jpeg"
            )
            if not os.path.exists(spec_path2):
                plot.plot_spec(spec2, spec_path2)

        if delete:
            # delete the second segment of each pair
            db.delete_segment({"ID": segment_id2})


@click.command(
    name="dedup-seg",
    short_help="Find and optionally delete duplicate segments in a database.",
    help=util.cli_help_from_doc(dedup_seg.__doc__),
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
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Path to output directory.",
)
@click.option("--name", "class_name", type=str, required=True, help="Class name")
@click.option(
    "--del",
    "delete",
    is_flag=True,
    help="If specified, remove duplicate segments from the database.",
)
@click.option(
    "--sgroup",
    "spec_group",
    required=False,
    default="default",
    help="Spectrogram group name. Defaults to 'default'.",
)
@click.option(
    "--threshold",
    "threshold",
    type=float,
    default=0.99,
    help="Treat as duplicates if cosine similarity >= threshold. Default = 0.99.",
)
@click.option("--name", "class_name", type=str, required=True, help="Class name")
@click.option(
    "--noplot",
    "no_plot",
    is_flag=True,
    help="If specified, do not plot spectrograms.",
)
def _dedup_seg_cmd(
    cfg_path: Optional[str],
    db_path: Optional[str],
    output_path: str,
    class_name: str,
    delete: bool,
    spec_group: str,
    threshold: float,
    no_plot: bool,
) -> None:
    util.set_logging()
    dedup_seg(
        cfg_path,
        db_path,
        output_path,
        class_name,
        delete,
        spec_group,
        threshold,
        no_plot,
    )
