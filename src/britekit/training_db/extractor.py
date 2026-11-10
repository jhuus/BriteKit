#!/usr/bin/env python3

import glob
import logging
import os
import random
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from threading import local
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
        pattern = str(Path(spec_dir) / "*.jpeg")
        for image_path in sorted(glob.glob(pattern)):
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

    def _matching_recordings(self, directory, destination_dir, offsets_per_file):
        for source in util.get_audio_files(directory):
            path = Path(source)
            offsets = offsets_per_file.get(path.stem, offsets_per_file.get(path.name))
            if offsets is None:
                continue
            target = source
            if destination_dir is not None:
                target = os.path.join(destination_dir, path.name)
                if not os.path.exists(target):
                    shutil.copy(source, target)
            yield source, target, offsets

    def _insert_by_dict(
        self, recording_dir, destination_dir, offsets_per_file, workers=1
    ):
        """Prepare matching recordings and write them in input order."""
        jobs = self._matching_recordings(
            recording_dir, destination_dir, offsets_per_file
        )
        if workers == 1:
            total = 0
            for source, target, offsets in jobs:
                logging.info(f"Processing {target}")
                try:
                    self.audio.load(source)
                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    continue
                total += self.insert_spectrograms(target, offsets)
            return total

        worker_state = local()

        def prepare(job):
            source, target, offsets = job
            if not hasattr(worker_state, "audio"):
                worker_state.audio = audio.Audio()
            worker_audio = worker_state.audio
            logging.info(f"Processing {target}")
            try:
                worker_audio.load(source)
            except Exception as e:
                logging.error(f"Caught exception: {e}")
                return None
            return (
                target,
                worker_audio.seconds(),
                self._prepare_spectrograms(worker_audio, offsets),
            )

        total = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Bound the queued work and keep SQLite writes on the calling thread.
            while batch := list(islice(jobs, workers * 2)):
                for result in pool.map(prepare, batch):
                    if result is not None:
                        total += self._write_spectrograms(*result)
        return total

    def _prepare_spectrograms(self, worker_audio, offsets):
        specs, _ = worker_audio.get_spectrograms(
            offsets, spec_duration=self.cfg.audio.spec_duration, convert_to_db=False
        )
        if specs is None:
            return []
        return [
            (offset, util.compress_spectrogram(spec, bits=self.cfg.audio.spec_bits))
            for offset, spec in zip(offsets, specs)
        ]

    def insert_spectrograms(self, recording_path, offsets):
        """Extract and insert spectrograms at the specified recording offsets."""
        return self._write_spectrograms(
            recording_path,
            self.audio.seconds(),
            self._prepare_spectrograms(self.audio, offsets),
        )

    def _write_spectrograms(self, recording_path, seconds, prepared):
        filename = Path(recording_path).name
        source_id = self.provider.source_id(filename, self.src_name)
        specgroup_id = self.provider.specgroup_id(self.spec_group)
        recording_id = self.provider.recording_id(
            self.class_name, filename, recording_path, source_id, seconds
        )
        existing = self.segments.setdefault(recording_id, set())
        pending_offsets = set()
        segments = []
        for offset, value in prepared:
            check_offset = round(offset, 0)
            if check_offset in existing or check_offset in pending_offsets:
                continue
            pending_offsets.add(check_offset)
            segments.append((offset, value))

        if segments:
            self.db.insert_extracted_segments(
                recording_id, self.class_id, specgroup_id, segments
            )
            existing.update(pending_offsets)
        return len(segments)

    def extract_all(
        self,
        dir_path: str,
        max_spec: Optional[int] = None,
        max_rec: Optional[int] = None,
        include_existing: bool = False,
        randomize: bool = False,
    ):
        """
        Extract spectrograms for all recordings in the given directory.

        Args:
        - dir_path (str): Directory containing recordings.
        - max_spec (int, optional): Maximum number of spectrograms to extract per recording.
          If the recording would yield more, they are sampled evenly, or randomly when
          randomize is enabled.
        - max_rec (int, optional): Maximum number of recordings to process.
        - include_existing (bool): Also extract new offsets from recordings already in the database.
        - randomize (bool): Randomize recording order and spectrogram offsets.

        Returns:
            Number of spectrograms inserted.
        """
        num_inserted = 0
        num_processed = 0
        recording_paths = util.get_audio_files(dir_path)
        if not include_existing:
            recording_paths = [
                path
                for path in recording_paths
                if Path(path).name not in self.filenames
            ]
        if randomize:
            random.shuffle(recording_paths)

        for recording_path in recording_paths:
            filename = Path(recording_path).name
            if not include_existing and filename in self.filenames:
                logging.info(f"Skipping {filename} (already in database)")
                continue

            if max_rec is not None and num_processed >= max_rec:
                break

            logging.info(f"Processing {recording_path}")
            try:
                self.audio.load(recording_path)
                seconds = self.audio.seconds()
            except Exception as e:
                logging.error(f"Caught exception: {e}")
                continue

            num_processed += 1
            end_offset = max(self.increment, seconds - self.increment)
            offsets = util.get_range(0, end_offset, self.increment)

            if max_spec is not None and len(offsets) > max_spec:
                if randomize:
                    offsets = sorted(random.sample(offsets, max_spec))
                else:
                    step = (len(offsets) - 1) / max(max_spec - 1, 1)
                    offsets = [offsets[round(i * step)] for i in range(max_spec)]

            num_inserted += self.insert_spectrograms(recording_path, offsets)

        return num_inserted

    def extract_by_csv(
        self,
        rec_dir: str,
        csv_path: str,
        dest_dir: Optional[str] = None,
        *,
        workers: int = 1,
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

        if workers < 1:
            raise ValueError("workers must be at least 1")
        df = pd.read_csv(csv_path)
        offsets_per_file: dict[str, list] = {}
        for i, row in df.iterrows():
            recording = row["recording"]
            if recording not in offsets_per_file:
                offsets_per_file[recording] = []

            offsets_per_file[recording].append(row["offset"])

        return self._insert_by_dict(rec_dir, dest_dir, offsets_per_file, workers)

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
