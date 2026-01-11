#!/usr/bin/env python3

# Defer some imports to improve initialization performance.
import logging
import os
from typing import Optional

from britekit.core import util
from britekit.training_db.training_db import TrainingDatabase
from britekit.training_db.training_data_provider import TrainingDataProvider


class Reextractor:
    """
    Re-generate spectrograms from audio recordings and update the training database.

    This class extracts spectrograms from audio recordings and imports them into the training database.
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

    def __init__(
        self,
        db_path: Optional[str] = None,
        class_name: Optional[str] = None,
        classes_path: Optional[str] = None,
        offset: float = 0.0,
        check: bool = False,
        spec_group: str = "default",
    ):

        self.db_path = db_path
        self.class_name = class_name
        self.classes_path = classes_path
        self.offset = offset
        self.check = check
        self.spec_group = spec_group

    # Resample audio data
    def _resample(self, waveform, original_sampling_rate, desired_sampling_rate):
        import torch
        import torchaudio.transforms as T

        waveform = torch.from_numpy(waveform)
        resampler = T.Resample(
            original_sampling_rate, desired_sampling_rate, dtype=waveform.dtype
        )
        return resampler(waveform).numpy()

    # Perform the re-extract
    def run(self, quiet=False):
        import pandas as pd

        with TrainingDatabase(self.db_path) as db:
            if self.class_name is None and self.classes_path is None:
                recordings = db.get_recording()
            elif self.classes_path:
                df = pd.read_csv(self.classes_path)
                if "Name" not in df:
                    logging.error(
                        f'Error: column "Name" not found in {self.classes_path}.'
                    )
                    quit()

                class_names = df["Name"].to_list()
                recordings = []
                for name in class_names:
                    recordings.extend(db.get_recording_by_class(name))
            else:
                assert self.class_name is not None
                recordings = db.get_recording_by_class(self.class_name)

            if len(recordings) == 0:
                logging.error("No matching recordings found.")
                quit()

            if not quiet:
                logging.info(f"Found {len(recordings)} matching recordings.")

            # Check if we have audio for all recordings.
            # We call get_segments more than once if check is not specified.
            # If we kept them all we might run out of memory.
            have_all_audio = True
            for recording in recordings:
                if recording.path:
                    if not os.path.exists(recording.path):
                        logging.error(f"Error: path {recording.path} not found.")
                        have_all_audio = False
                else:
                    segments = db.get_segment(
                        {"RecordingID": recording.id}, include_audio=True
                    )
                    for segment in segments:
                        if segment.audio is None:
                            logging.error(
                                f"Error: no audio found for recording with ID={recording.id} and filename = {recording.filename}."
                            )
                            have_all_audio = False
                            break

            if have_all_audio and not quiet:
                logging.info("Found all required recordings.")

            if self.check or not have_all_audio:
                return

            # Delete existing spec_values of the same group and class
            specgroup_results = db.get_specgroup({"Name": self.spec_group})
            if len(specgroup_results) > 0:
                if self.class_name is None:
                    # Deleting the spec_group efficiently deletes all corresponding spec_values
                    db.delete_specgroup({"Name": self.spec_group})
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
            specgroup_id = TrainingDataProvider(db).specgroup_id(self.spec_group)

            # Do the extract
            from britekit.core.audio import Audio

            audio_obj = Audio()
            for recording in recordings:
                if not quiet:
                    logging.info(f"Processing {recording.filename}")
                segments = db.get_segment(
                    {"RecordingID": recording.id}, include_audio=True
                )
                if recording.path:
                    audio_obj.load(recording.path)
                    offsets = []
                    for segment in segments:
                        offsets.append(max(0, segment.offset + self.offset))

                    spectrograms, _ = audio_obj.get_spectrograms(offsets)
                    if spectrograms is not None:
                        for i, spec in enumerate(spectrograms):
                            compressed_spec = util.compress_spectrogram(spec)
                            segment = segments[i]
                            db.insert_specvalue(
                                compressed_spec, specgroup_id, segment.id
                            )
                else:
                    logging.warning(
                        f"No recording path specified for recording {recording.ID}"
                    )
