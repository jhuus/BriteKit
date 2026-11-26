"""One-time, lossless conversion of trusted training pickles to indexed stores.

Legacy pickle containers must be loaded once; conversion never expands spectrograms
or creates a second full copy of teacher/frame arrays. Output arrays are plain NPY
files suitable for np.load(..., mmap_mode="r", allow_pickle=False).
"""

import argparse
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any, Iterator

import numpy as np


@contextmanager
def _destination(destination: Path) -> Iterator[Path]:
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        yield temporary
        # Never replace an existing store, including one created during conversion.
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _save(directory: Path, name: str, values: Any) -> None:
    """Write and compare in bounded chunks, including dtype and shape."""
    values = np.asarray(values)
    if values.dtype.hasobject:
        raise ValueError(f"{name} cannot contain Python objects")
    path = directory / f"{name}.npy"
    np.save(path, values, allow_pickle=False)
    actual = np.load(path, mmap_mode="r", allow_pickle=False)
    if actual.shape != values.shape or actual.dtype != values.dtype:
        raise ValueError(f"Verification failed for {name}")
    for start in range(0, len(values), 1024):
        if not np.array_equal(
            actual[start : start + 1024], values[start : start + 1024]
        ):
            raise ValueError(f"Verification failed for {name} at row {start}")


def _ids(values: Any, name: str, count: int, unique: bool = False) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (count,) or (count and array.dtype.kind not in "iu"):
        raise ValueError(f"{name} must contain {count} integer IDs")
    if unique and len(np.unique(array)) != count:
        raise ValueError(f"{name} contains duplicate IDs")
    # Preserve integer width/sign; empty lists have no source integer dtype.
    return array if count else np.empty(0, dtype=np.int64)


def _probabilities(
    values: Any, name: str, prefix: tuple[int, ...], ndim: int
) -> np.ndarray:
    array = np.asarray(values)
    if (
        array.ndim != ndim
        or array.shape[: len(prefix)] != prefix
        or array.dtype.kind != "f"
    ):
        raise ValueError(f"{name} must be a floating array with shape prefix {prefix}")
    for start in range(0, len(array), 1024):
        chunk = array[start : start + 1024]
        if not np.isfinite(chunk).all() or np.any((chunk < 0) | (chunk > 1)):
            raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
    return array


def _manifest(
    directory: Path, kind: str, source: Path, count: int, metadata: dict
) -> None:
    manifest = dict(
        format="britekit-indexed",
        format_version=1,
        kind=kind,
        source=str(source.resolve()),
        count=count,
        metadata=metadata,
    )
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _training(data: dict, directory: Path, source: Path) -> None:
    required = (
        "class_names",
        "class_codes",
        "alt_names",
        "alt_codes",
        "spec_values",
        "spec_class_indexes",
        "spec_segment_ids",
        "spec_recording_ids",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Training pickle missing keys: {missing}")
    count = len(data["spec_values"])
    classes = len(data["class_names"])
    if any(len(data[key]) != classes for key in required[1:4]):
        raise ValueError("Class metadata lengths differ")
    labels = data["spec_class_indexes"]
    if len(labels) != count:
        raise ValueError("Label count differs from spectrogram count")
    label_offsets = np.empty(count + 1, dtype=np.uint64)
    label_offsets[0] = 0
    for i, row in enumerate(labels):
        if any(
            not isinstance(index, (int, np.integer)) or not 0 <= index < classes
            for index in row
        ):
            raise ValueError(f"Invalid class indexes at sample {i}")
        label_offsets[i + 1] = int(label_offsets[i]) + len(row)
    label_values = np.lib.format.open_memmap(
        directory / "label_values.npy",
        mode="w+",
        dtype=np.int64,
        shape=(int(label_offsets[-1]),),
    )
    for i, row in enumerate(labels):
        label_values[int(label_offsets[i]) : int(label_offsets[i + 1])] = row
    label_values.flush()
    del label_values
    actual_labels = np.load(
        directory / "label_values.npy", mmap_mode="r", allow_pickle=False
    )
    for i, row in enumerate(labels):
        if not np.array_equal(
            actual_labels[int(label_offsets[i]) : int(label_offsets[i + 1])],
            np.asarray(row, dtype=np.int64),
        ):
            raise ValueError(f"Label verification failed at sample {i}")
    del actual_labels
    _save(directory, "label_offsets", label_offsets)
    for key, unique in (("spec_segment_ids", True), ("spec_recording_ids", False)):
        _save(directory, key.removeprefix("spec_"), _ids(data[key], key, count, unique))
    offsets = np.empty(count + 1, dtype=np.uint64)
    offsets[0] = 0
    with (directory / "specs.bin").open("wb") as file:
        for i, blob in enumerate(data["spec_values"]):
            if not isinstance(blob, bytes) or not blob:
                raise ValueError(f"Spectrogram {i} must be nonempty compressed bytes")
            file.write(blob)
            offsets[i + 1] = file.tell()
    _save(directory, "offsets", offsets)
    with (directory / "specs.bin").open("rb") as file:
        for i, blob in enumerate(data["spec_values"]):
            if file.read(int(offsets[i + 1] - offsets[i])) != blob:
                raise ValueError(f"Spectrogram verification failed at sample {i}")
    metadata = {key: value for key, value in data.items() if key not in required[4:]}
    _manifest(directory, "training", source, count, metadata)


def _teacher(data: dict, directory: Path, source: Path) -> None:
    required = ("format_version", "class_codes", "segment_ids", "probabilities")
    if any(key not in data for key in required) or data["format_version"] != 2:
        raise ValueError("Expected a version 2 teacher-target pickle")
    count = len(data["segment_ids"])
    codes = data["class_codes"]
    if len(set(codes)) != len(codes):
        raise ValueError("Duplicate teacher class codes")
    _save(
        directory, "segment_ids", _ids(data["segment_ids"], "segment_ids", count, True)
    )
    prefix = (count, len(codes))
    _save(
        directory,
        "probabilities",
        _probabilities(data["probabilities"], "probabilities", prefix, 2),
    )
    if "frame_probabilities" in data:
        _save(
            directory,
            "frame_probabilities",
            _probabilities(
                data["frame_probabilities"], "frame_probabilities", prefix, 3
            ),
        )
    metadata = {
        key: value
        for key, value in data.items()
        if key not in ("segment_ids", "probabilities", "frame_probabilities")
    }
    _manifest(directory, "teacher-targets", source, count, metadata)


def _frames(data: dict, directory: Path, source: Path) -> None:
    count = len(data)
    _save(directory, "segment_ids", _ids(list(data), "segment_ids", count, True))
    offsets = np.empty(count + 1, dtype=np.uint64)
    offsets[0] = 0
    dtype = None
    for i, value in enumerate(data.values()):
        value = np.asarray(value)
        if (
            value.ndim != 1
            or value.dtype.kind not in "fbiu"
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Frame labels at row {i} must be a finite numeric vector")
        if np.any((value < 0) | (value > 1)):
            raise ValueError(f"Frame labels at row {i} must be in [0, 1]")
        if dtype is not None and dtype != value.dtype:
            raise ValueError(
                "Frame-label vectors must share a dtype for lossless storage"
            )
        dtype = value.dtype
        offsets[i + 1] = int(offsets[i]) + len(value)
    values = np.lib.format.open_memmap(
        directory / "values.npy",
        mode="w+",
        dtype=dtype or np.dtype("float32"),
        shape=(int(offsets[-1]),),
    )
    for i, value in enumerate(data.values()):
        values[int(offsets[i]) : int(offsets[i + 1])] = value
    values.flush()
    del values
    actual = np.load(directory / "values.npy", mmap_mode="r", allow_pickle=False)
    for i, value in enumerate(data.values()):
        if not np.array_equal(actual[int(offsets[i]) : int(offsets[i + 1])], value):
            raise ValueError(f"Frame-label verification failed at row {i}")
    del actual
    _save(directory, "offsets", offsets)
    _manifest(directory, "training-frames", source, count, {})


def convert_pickle(kind: str, source: str, destination: str) -> None:
    """Convert a trusted pickle; preserve source and refuse existing destinations.

    Conversion needs RAM for the legacy pickle once. The destination is published
    only after all arrays/blobs are verified. Spectrograms remain compressed.
    """
    converters = {
        "training": _training,
        "teacher-targets": _teacher,
        "training-frames": _frames,
    }
    if kind not in converters:
        raise ValueError(f"Unknown pickle kind: {kind}")
    source_path, destination_path = Path(source), Path(destination)
    with _destination(destination_path) as directory:
        logging.info(
            "Loading %s (legacy pickle requires RAM for the full input)", source_path
        )
        with source_path.open("rb") as file:
            data = pickle.load(file)
        if not isinstance(data, dict):
            raise ValueError("Expected a dictionary pickle")
        converters[kind](data, directory, source_path)
    logging.info("Verified and saved %s", destination_path)


def main(kind: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Convert a trusted {kind} pickle to an indexed store. Requires RAM to unpickle the source once."
    )
    parser.add_argument("source", help="Existing trusted pickle file")
    parser.add_argument("destination", help="New output directory (must not exist)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    convert_pickle(kind, args.source, args.destination)
