import click
import os
from pathlib import Path
import time
from typing import Optional
import zlib

import torch
import torchaudio.transforms as T
import numpy as np
import pandas as pd

from britekit.core.audio import Audio
from britekit.core.config_loader import get_config
from britekit.core import util
from britekit.core.util import cli_help_from_doc
from britekit.training_db.training_db import TrainingDatabase
from britekit.training_db.training_data_provider import TrainingDataProvider


def reextract_impl(
    cfg_path: Optional[str] = None,
    db_path: Optional[str] = None,
    class_name: Optional[str] = None,
    class_csv_path: Optional[str] = None,
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
        cfg_path (str, optional): Path to YAML file defining configuration overrides.
        db_path (str, optional): Path to the training database. Defaults to cfg.train.training_db.
        class_name (str, optional): Name of a specific class to reextract. If omitted, processes all classes.
        class_csv_path (str, optional): Path to CSV file listing classes to reextract. Alternative to class_name.
        check (bool): If True, only check that all recording paths are accessible without updating database.
        spec_group (str): Spectrogram group name for storing the extracted spectrograms. Defaults to 'default'.
    """

    # Resample audio data
    def resample(waveform, original_sampling_rate, desired_sampling_rate):
        waveform = torch.from_numpy(waveform)
        resampler = T.Resample(
            original_sampling_rate, desired_sampling_rate, dtype=waveform.dtype
        )
        return resampler(waveform).numpy()

    cfg, _ = get_config(cfg_path)
    if db_path is None:
        db_path = str(Path(cfg.train.train_db).resolve())

    start_time = time.time()
    with TrainingDatabase(db_path) as db:
        if class_name and class_csv_path:
            click.echo("Only one of --name and --csv may be specified.")
            quit()

        if class_name is None and class_csv_path is None:
            recordings = db.get_recording()
        elif class_csv_path:
            df = pd.read_csv(class_csv_path)
            if "Name" not in df:
                click.echo(f'Error: column "Name" not found in {class_csv_path}.')
                quit()

            class_names = df["Name"].to_list()
            recordings = []
            for name in class_names:
                recordings.extend(db.get_recording_by_class(name))
        else:
            assert class_name is not None
            recordings = db.get_recording_by_class(class_name)

        if len(recordings) == 0:
            click.echo("No matching recordings found.")
            quit()

        click.echo(f"Found {len(recordings)} matching recordings.")

        # Check if we have audio for all recordings.
        # We call get_segments more than once if --check is not specified.
        # If we kept them all we might run out of memory.
        have_all_audio = True
        for recording in recordings:
            if recording.path:
                if not os.path.exists(recording.path):
                    click.echo(f"Error: path {recording.path} not found.")
                    have_all_audio = False
            else:
                segments = db.get_segment(
                    {"RecordingID": recording.id}, include_audio=True
                )
                for segment in segments:
                    if segment.audio is None:
                        click.echo(
                            f"Error: no audio found for recording with ID={recording.id} and filename = {recording.filename}."
                        )
                        have_all_audio = False
                        break

        if have_all_audio:
            click.echo("Found all required recordings.")

        if check or not have_all_audio:
            quit()

        # Delete existing spec_values of the same group and class
        specgroup_results = db.get_specgroup({"Name": spec_group})
        if len(specgroup_results) > 0:
            if class_name is None:
                # Deleting the spec_group efficiently deletes all corresponding spec_values
                db.delete_specgroup({"Name": spec_group})
            else:
                # We're doing just one class, so have to delete spec_values individually
                for recording in recordings:
                    segments = db.get_segment(
                        {"RecordingID": recording.id}, include_audio=False
                    )
                    for segment in segments:
                        db.delete_specvalue(
                            {
                                "SegmentID": segment.id,
                                "SpecGroupID": specgroup_results[0].id,
                            }
                        )

        # Get the ID of the spec_group, inserting a new record if we just deleted it
        specgroup_id = TrainingDataProvider(db).specgroup_id(spec_group)

        # Do the extract
        audio_obj = Audio()
        for recording in recordings:
            click.echo(f"Processing {recording.filename}")
            segments = db.get_segment({"RecordingID": recording.id}, include_audio=True)
            if recording.path:
                audio_obj.load(recording.path)
                offsets = [segment.offset for segment in segments]
                spectrograms, _ = audio_obj.get_spectrograms(offsets)

                for i, spec in enumerate(spectrograms):
                    compressed_spec = util.compress_spectrogram(spec)
                    segment = segments[i]
                    db.insert_specvalue(compressed_spec, specgroup_id, segment.id)
            else:
                # Assume all segments for this recording have audio
                for segment in segments:
                    # Unzip then make a copy so it is writeable
                    audio_blob = np.frombuffer(
                        zlib.decompress(segment.audio), np.float32
                    ).copy()
                    if segment.sampling_rate != cfg.audio.sampling_rate:
                        audio_blob = resample(
                            audio_blob, segment.sampling_rate, cfg.audio.sampling_rate
                        )

                    # If embedded audio is longer than needed, adjust offset so only relevant central audio is used
                    seconds = len(audio_blob) / cfg.audio.sampling_rate
                    fudge_factor = (
                        0.001  # consider lengths the same if difference less than this
                    )
                    offset = 0.0
                    if seconds > cfg.audio.spec_duration - fudge_factor:
                        offset = round(
                            (seconds - cfg.audio.spec_duration) / 2, 1
                        )  # round to nearest tenth of a second

                    audio_obj.signal = audio_blob
                    spectrograms, _ = audio_obj.get_spectrograms([offset])
                    compressed_spec = util.compress_spectrogram(spectrograms[0])
                    db.insert_specvalue(compressed_spec, specgroup_id, segment.id)

        elapsed_time = util.format_elapsed_time(start_time, time.time())
        click.echo(f"Elapsed time = {elapsed_time}")


@click.command(
    name="reextract",
    short_help="Re-generate the spectrograms in a database, and add them to the database.",
    help=cli_help_from_doc(reextract_impl.__doc__),
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
    help="Optional class name. If this and --csv are omitted, do all classes.",
)
@click.option(
    "--classes",
    "class_csv_path",
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
def reextract_cmd(
    cfg_path: Optional[str] = None,
    db_path: Optional[str] = None,
    class_name: Optional[str] = None,
    class_csv_path: Optional[str] = None,
    check: bool = False,
    spec_group: str = "default",
):
    reextract_impl(cfg_path, db_path, class_name, class_csv_path, check, spec_group)
