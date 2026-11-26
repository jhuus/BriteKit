"""Export database spectrograms directly to an indexed training store."""

import logging
from pathlib import Path
import random
from typing import Optional

import numpy as np

from britekit.core.convert_pickles import _destination, _manifest, _save
from britekit.core.exceptions import InputError
from britekit.training_db.training_data_provider import TrainingDataProvider
from britekit.training_db.training_db import TrainingDatabase


class TrainingExporter:
    def __init__(
        self,
        db_path: str,
        output_path: str,
        classes_path: Optional[str] = None,
        max_per_class: Optional[int] = None,
        spec_group: Optional[str] = None,
    ):
        if not Path(db_path).is_file():
            raise InputError(f'Database "{db_path}" not found')
        if max_per_class is not None and max_per_class <= 0:
            raise InputError("max_per_class must be positive")
        self.db_path, self.output_path = db_path, output_path
        self.classes_path, self.max_per_class = classes_path, max_per_class
        self.spec_group = spec_group or "default"

    def export(self, quiet=False):
        """Retain lightweight selection metadata, and stream one blob at a time.

        Selection order matches the legacy export: classes in requested order,
        recordings by filename, then offset. A segment appears only once, with all
        selected labels. Limits are assigned to the earliest selected class and
        sampled reproducibly with seed 101, without changing global RNG state.
        """
        with (
            _destination(Path(self.output_path)) as directory,
            TrainingDatabase(self.db_path) as db,
        ):
            # Keep labels and spectrogram selection in one consistent snapshot.
            db.conn.execute("BEGIN")
            classes = db.get_class()
            by_name = {row.name: row for row in classes}
            if self.classes_path:
                import pandas as pd

                names = pd.read_csv(self.classes_path)["Name"].tolist()
            else:
                names = [row.name for row in classes]
            if not names or len(names) != len(set(names)):
                raise InputError(
                    "Class list must be nonempty and contain no duplicates"
                )
            for name in names:
                if name not in by_name:
                    raise InputError(f'Class "{name}" not found in database.')
            name_order = {name: i for i, name in enumerate(names)}
            labels = TrainingDataProvider(db).segment_class_dict()
            specs, used = [], set()
            for name in names:
                rows = db.get_spectrogram_by_class(
                    name, include_value=False, spec_group=self.spec_group
                )
                for row in rows:
                    if row.segment_id not in used:
                        used.add(row.segment_id)
                        specs.append(row)
                if not quiet:
                    logging.info(
                        'Selected %s unique segments after "%s"', len(specs), name
                    )
            del used
            if self.max_per_class is not None:
                random.Random(101).shuffle(specs)
                counts = [0] * len(names)
                selected = []
                for row in specs:
                    primary = min(
                        name_order[name]
                        for name in labels[row.segment_id]
                        if name in name_order
                    )
                    if counts[primary] < self.max_per_class:
                        selected.append(row)
                        counts[primary] += 1
                specs = selected
            selected_labels = [
                sorted(
                    name_order[name]
                    for name in labels[row.segment_id]
                    if name in name_order
                )
                for row in specs
            ]
            del labels
            coverage = np.zeros(len(names), dtype=np.int64)
            for row_labels in selected_labels:
                coverage[row_labels] += 1
            empty = [name for name, count in zip(names, coverage) if count == 0]
            if empty:
                raise InputError(
                    f"No spectrograms for selected classes: {', '.join(empty)}"
                )
            count = len(specs)
            _save(
                directory,
                "segment_ids",
                np.array([row.segment_id for row in specs], dtype=np.int64),
            )
            _save(
                directory,
                "recording_ids",
                np.array([row.recording_id for row in specs], dtype=np.int64),
            )
            label_offsets = np.zeros(count + 1, dtype=np.uint64)
            for i, row_labels in enumerate(selected_labels):
                label_offsets[i + 1] = int(label_offsets[i]) + len(row_labels)
            _save(directory, "label_offsets", label_offsets)
            label_values = np.lib.format.open_memmap(
                directory / "label_values.npy",
                mode="w+",
                dtype=np.int64,
                shape=(int(label_offsets[-1]),),
            )
            for i, row_labels in enumerate(selected_labels):
                label_values[int(label_offsets[i]) : int(label_offsets[i + 1])] = (
                    row_labels
                )
            label_values.flush()
            del label_values, selected_labels
            # A temporary index keeps blob selection inside SQLite. CROSS JOIN
            # fixes the loop order so ORDER BY does not sort/materialize blobs.
            db.conn.execute(
                "CREATE TEMP TABLE export_selection (ordinal INTEGER PRIMARY KEY, value_id INTEGER NOT NULL)"
            )
            db.conn.executemany(
                "INSERT INTO export_selection VALUES (?, ?)",
                ((i, row.specvalue_id) for i, row in enumerate(specs)),
            )
            del specs
            offsets = np.zeros(count + 1, dtype=np.uint64)
            cursor = db.conn.execute(
                "SELECT sv.Value FROM export_selection AS selected CROSS JOIN SpecValue AS sv ON sv.ID = selected.value_id ORDER BY selected.ordinal"
            )
            written = 0
            with (directory / "specs.bin").open("wb") as output:
                for written, (blob,) in enumerate(cursor, start=1):
                    if not isinstance(blob, bytes) or not blob:
                        raise InputError(
                            f"Invalid compressed spectrogram at row {written - 1}"
                        )
                    output.write(blob)
                    offsets[written] = output.tell()
                    if not quiet and written % 10000 == 0:
                        logging.info("Exported %s/%s spectrograms", written, count)
            if written != count:
                raise InputError("Database spectrogram count changed during export")
            _save(directory, "offsets", offsets)
            metadata = dict(
                class_names=names,
                class_codes=[by_name[name].code for name in names],
                alt_names=[by_name[name].alt_name for name in names],
                alt_codes=[by_name[name].alt_code for name in names],
                spec_group=self.spec_group,
            )
            _manifest(directory, "training", Path(self.db_path), count, metadata)
        if not quiet:
            logging.info("Saved %s spectrograms to %s", count, self.output_path)
