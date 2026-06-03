#!/usr/bin/env python3

import logging

import click

from britekit.core import util


def copy_class(
    src_db_path: str,
    dest_db_path: str,
    class_name: str,
) -> None:
    """
    Copy a class and all its associated records from one training database to another.

    Copies the class record along with its category, all associated recordings (and their
    sources), segments, spectrograms, and SegmentClass links. Records that already exist
    in the destination database (matched by name or filename+source) are reused rather than
    duplicated. The class must not already exist in the destination database.

    Args:
    - src_db_path (str): Path to the source training database.
    - dest_db_path (str): Path to the destination training database.
    - class_name (str): Name of the class to copy.
    """
    from britekit.training_db.training_db import TrainingDatabase

    with TrainingDatabase(src_db_path) as src, TrainingDatabase(dest_db_path) as dest:
        # Validate class exists in source and not in destination
        src_classes = src.get_class({"Name": class_name})
        if not src_classes:
            raise ValueError(f"Class '{class_name}' not found in source database")
        src_class = src_classes[0]

        if dest.get_class({"Name": class_name}):
            raise ValueError(
                f"Class '{class_name}' already exists in destination database"
            )

        # Ensure category exists in destination
        src_cats = src.get_category({"ID": src_class.category_id})
        cat_name = src_cats[0].name if src_cats else "default"
        dest_cats = dest.get_category({"Name": cat_name})
        dest_cat_id = dest_cats[0].id if dest_cats else dest.insert_category(cat_name)

        # Insert class in destination
        dest_class_id = dest.insert_class(
            dest_cat_id,
            src_class.name,
            src_class.alt_name,
            src_class.code,
            src_class.alt_code,
        )

        # Specgroup ID map: src specgroup ID -> dest specgroup ID (populated lazily)
        specgroup_id_map: dict[int, int] = {}

        def dest_specgroup_id(src_id: int) -> int:
            if src_id not in specgroup_id_map:
                src_groups = src.get_specgroup({"ID": src_id})
                name = src_groups[0].name if src_groups else "default"
                dest_groups = dest.get_specgroup({"Name": name})
                dest_id = (
                    dest_groups[0].id if dest_groups else dest.insert_specgroup(name)
                )
                specgroup_id_map[src_id] = dest_id
            return specgroup_id_map[src_id]

        # Copy recordings: build src recording ID -> dest recording ID map
        src_recordings = src.get_recording_by_class(class_name)
        recording_id_map: dict[int, int] = {}

        for rec in src_recordings:
            src_sources = src.get_source({"ID": rec.source_id})
            src_source_name = src_sources[0].name if src_sources else "default"
            dest_source_results = dest.get_source({"Name": src_source_name})
            dest_source_id = (
                dest_source_results[0].id
                if dest_source_results
                else dest.insert_source(src_source_name)
            )

            dest_recs = dest.get_recording(
                {"FileName": rec.filename, "SourceID": dest_source_id}
            )
            dest_rec_id = (
                dest_recs[0].id
                if dest_recs
                else dest.insert_recording(
                    dest_source_id, rec.filename, rec.path, rec.seconds
                )
            )
            recording_id_map[rec.id] = dest_rec_id

        # Copy segments, spectrograms, and SegmentClass links
        src_segments = src.get_segment_by_class(class_name)
        num_segments = 0
        num_specvalues = 0

        for seg in src_segments:
            if seg.recording_id not in recording_id_map:
                logging.warning(
                    f"Skipping segment {seg.id}: recording {seg.recording_id} not found"
                )
                continue

            dest_rec_id = recording_id_map[seg.recording_id]
            dest_segs = dest.get_segment(
                {"RecordingID": dest_rec_id, "Offset": seg.offset}
            )

            if dest_segs:
                dest_seg_id = dest_segs[0].id
            else:
                dest_seg_id = dest.insert_segment(dest_rec_id, seg.offset)
                num_segments += 1

                for sv in src.get_specvalue({"SegmentID": seg.id}):
                    dest.insert_specvalue(
                        sv.value, dest_specgroup_id(sv.specgroup_id), dest_seg_id
                    )
                    num_specvalues += 1

            if (
                dest.get_segment_class_count(
                    {"SegmentID": dest_seg_id, "ClassID": dest_class_id}
                )
                == 0
            ):
                dest.insert_segment_class(dest_seg_id, dest_class_id)

        dest.optimize()
        logging.info(
            f"Copied '{class_name}': {len(src_recordings)} recordings, "
            f"{num_segments} segments, {num_specvalues} spectrograms"
        )


@click.command(
    name="copy-class",
    short_help="Copy a class and its records from one database to another.",
    help=util.cli_help_from_doc(copy_class.__doc__),
)
@click.option(
    "--src-db",
    "src_db_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to the source training database.",
)
@click.option(
    "--dest-db",
    "dest_db_path",
    type=click.Path(file_okay=True, dir_okay=False),
    required=True,
    help="Path to the destination training database.",
)
@click.option("--name", "class_name", required=True, help="Name of the class to copy.")
def _copy_class_cmd(
    src_db_path: str,
    dest_db_path: str,
    class_name: str,
) -> None:
    util.set_logging()
    copy_class(src_db_path, dest_db_path, class_name)
