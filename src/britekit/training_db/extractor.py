import logging
import os
import re
import shutil
from pathlib import Path
from typing import Optional

from britekit.core import audio, util
from britekit.core.config_loader import get_config
from britekit.training_db.training_db import TrainingDatabase
from britekit.training_db.training_data_provider import TrainingDataProvider


class Extractor:
    """
    Class for extracting spectrograms from recordings and inserting them into the database.

    Attributes:
        db (TrainingDatabase): Training database
        class_name (str): Name of class
        class_code (str, optional): Class code, only used when new class is inserted into the database
        cat_name (str, optional): Category name used when class is inserted. Defaults to 'default'.
        src_name (str, optional): Source name used when recording is inserted. Defaults to 'default'.
        overlap (float, optional): Spectrogram overlap in seconds
        spec_group (str, optional): Spectrogram group name
    """

    def __init__(
        self,
        db: TrainingDatabase,
        class_name: str,
        class_code: Optional[str] = None,
        cat_name: Optional[str] = None,
        src_name: Optional[str] = None,
        overlap: Optional[float] = None,
        spec_group: Optional[str] = None,
    ):
        self.db = db
        self.class_name = class_name
        self.class_code = class_code
        self.src_name = src_name
        self.cat_name = cat_name if cat_name is not None else "default"
        self.spec_group = spec_group if spec_group is not None else "default"

        self.provider = TrainingDataProvider(self.db)
        self.cfg = get_config()
        self.audio = audio.Audio()

        if overlap is None:
            overlap = self.cfg.infer.overlap
        self.increment = max(0.5, self.cfg.audio.spec_duration - overlap)

        self.category_id = self.provider.category_id(self.cat_name)
        self.class_id = self.provider.class_id(
            self.class_name, self.class_code, self.category_id
        )

        self._get_db_recordings()
        self._get_db_segments()

    def _get_db_recordings(self):
        """Get existing recordings."""
        self.filenames = set()
        results = self.db.get_recording_by_class(self.class_name)
        for r in results:
            if r.filename not in self.filenames:
                self.filenames.add(r.filename)

    def _get_db_segments(self):
        """Get existing segments, so we don't insert duplicates."""
        self.segments = {}
        results = self.db.get_segment_by_class(self.class_name)
        for r in results:
            if r.recording_id not in self.segments:
                self.segments[r.recording_id] = set()

            self.segments[r.recording_id].add(round(r.offset, 0))

    def _process_image_dir(self, spec_dir):
        """
        Get list of specs from directory of images.

        Returns:
            A dict with key per file name and list of offsets per key
        """

        offsets_per_file = {}
        for image_path in sorted(Path().glob(f"{spec_dir}/*.jpeg")):
            name = Path(image_path).stem
            if "~" in name:
                result = re.split("\\S+~(.+)~.*", name)
                result = re.split("(.+)-(.+)", result[1])
            else:
                result = re.split("(.+)-(.+)", name)
                if len(result) != 4:
                    result = re.split("(\\S+)_(\\S+)", name)

            if len(result) != 4:
                logging.error(f"Error: unknown file name format: {image_path}")
                continue
            else:
                file_name = result[1]
                offset = float(result[2])

            if file_name not in offsets_per_file:
                offsets_per_file[file_name] = []

            offsets_per_file[file_name].append(offset)

        return offsets_per_file

    def _insert_by_dict(self, recording_dir, destination_dir, offsets_per_file):
        """
        Given a recording directory and a dict from recording stems to offsets,
        insert the corresponding spectrograms.
        """
        num_inserted = 0
        recording_paths = util.get_audio_files(recording_dir)
        for recording_dir in recording_paths:
            filename = Path(recording_dir).stem
            if filename not in offsets_per_file:
                continue

            if destination_dir is not None:
                dest_path = os.path.join(destination_dir, Path(recording_dir).name)
                if not os.path.exists(dest_path):
                    shutil.copy(recording_dir, dest_path)

                recording_dir = dest_path

            logging.info(f"Processing {recording_dir}")
            try:
                self.audio.load(recording_dir)
            except Exception as e:
                logging.error(f"Caught exception: {e}")
                continue

            num_inserted += self.insert_spectrograms(
                recording_dir, offsets_per_file[filename]
            )

        return num_inserted

    def insert_spectrograms(self, recording_path, offsets):
        """
        Insert a spectrogram at each of the given offsets of the specified file.

        Args:
        - recording_path (str): Path to audio recording.
        - offsets (list[float]): List of offsets, where each represents number of seconds to start of spectrogram.

        Returns:
            Number of spectrograms inserted.
        """

        seconds = self.audio.seconds()
        filename = Path(recording_path).name
        source_id = self.provider.source_id(filename, self.src_name)
        specgroup_id = self.provider.specgroup_id(self.spec_group)
        recording_id = self.provider.recording_id(
            self.class_name, filename, recording_path, source_id, seconds
        )

        num_inserted = 0
        specs, _ = self.audio.get_spectrograms(
            offsets, spec_duration=self.cfg.audio.spec_duration
        )
        if specs is None:
            return 0

        for i in range(len(specs)):
            # check for duplicate before inserting
            if recording_id not in self.segments:
                self.segments[recording_id] = set()

            check_offset = round(offsets[i], 0)
            if check_offset in self.segments[recording_id]:
                continue  # skip if another one is within a second

            num_inserted += 1
            self.segments[recording_id].add(round(check_offset))
            compressed = util.compress_spectrogram(specs[i])
            segment_id = self.db.insert_segment(recording_id, offsets[i])
            self.db.insert_segment_class(segment_id, self.class_id)
            self.db.insert_specvalue(compressed, specgroup_id, segment_id)

        return num_inserted

    def extract_all(self, dir_path: str):
        """
        Extract spectrograms for all recordings in the given directory.

        Args:
        - dir_path (str): Directory containing recordings.

        Returns:
            Number of spectrograms inserted.
        """
        num_inserted = 0
        recording_paths = util.get_audio_files(dir_path)
        for recording_path in recording_paths:
            filename = Path(recording_path).stem
            if filename in self.filenames:
                logging.info(f"Skipping {filename} (already in database)")
                continue

            logging.info(f"Processing {recording_path}")
            try:
                self.audio.load(recording_path)
                seconds = self.audio.seconds()
            except Exception as e:
                logging.error(f"Caught exception: {e}")
                continue

            end_offset = max(self.increment, seconds - self.increment)
            offsets = util.get_range(0, end_offset, self.increment)
            if len(offsets) == 0:
                logging.info(f"Skipping {filename} (too short)")
                continue

            num_inserted += self.insert_spectrograms(recording_path, offsets)

        return num_inserted

    def extract_by_csv(
        self, rec_dir: str, csv_path: str, dest_dir: Optional[str] = None
    ):
        """
        Extract spectrograms that match names of spectrogram images in a given directory.
        Typically the spectrograms were generated using the 'search' or 'plot-db' commands.

        Args:
        - rec_dir (str): Directory containing recordings.
        - csv_path (str): Path to CSV file containing two columns (recording and offset) to identify segments to extract.
        - dest_dir (str, optional): Optionally copy used recordings to this directory.

        Returns:
            Number of spectrograms inserted.
        """
        import pandas as pd

        df = pd.read_csv(csv_path)
        offsets_per_file: dict[str, list] = {}
        for i, row in df.iterrows():
            recording = row["recording"]
            if recording not in offsets_per_file:
                offsets_per_file[recording] = []

            offsets_per_file[recording].append(row["offset"])

        return self._insert_by_dict(rec_dir, dest_dir, offsets_per_file)

    def extract_by_image(
        self, rec_dir: str, spec_dir: str, dest_dir: Optional[str] = None
    ):
        """
        Extract spectrograms that match names of spectrogram images in a given directory.
        Typically the spectrograms were generated using the 'search' or 'plot-db' commands.

        Args:
        - rec_dir (str): Directory containing recordings.
        - spec_dir (str): Directory containing spectrogram images.
        - dest_dir (str, optional): Optionally copy used recordings to this directory.

        Returns:
            Number of spectrograms inserted.
        """
        offsets_per_file = self._process_image_dir(spec_dir)
        return self._insert_by_dict(rec_dir, dest_dir, offsets_per_file)
