"""Read-only indexed training stores, with mappings reopened in worker processes."""

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np


class DiskArray:
    """Small pickleable handle; never serialize an open memory map."""

    def __init__(self, path):
        self.path = str(Path(path).resolve())
        array = np.load(self.path, mmap_mode="r", allow_pickle=False)
        self.shape, self.dtype = array.shape, array.dtype
        if array.dtype.hasobject:
            raise ValueError(f"Object array is not supported: {path}")
        self._array = None

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        if self._array is None:
            self._array = np.load(self.path, mmap_mode="r", allow_pickle=False)
        return self._array[index]

    def __getstate__(self):
        return dict(self.__dict__, _array=None)


class IndexView(Sequence):
    """Subset without reading any samples from its backing sequence."""

    def __init__(self, source, indices):
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return IndexView(self.source, self.indices[index])
        return self.source[int(self.indices[index])]


class RaggedRows(Sequence):
    def __init__(self, offsets, values, as_list=False):
        self.offsets, self.values, self.as_list = offsets, values, as_list

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        if isinstance(index, slice):
            return IndexView(self, range(len(self))[index])
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        value = self.values[start:end]
        return value.tolist() if self.as_list else value


class SpectrogramRows(RaggedRows):
    def __init__(self, offsets, path):
        self.offsets = offsets
        self.path = str(Path(path).resolve())
        self._data = None

    def __getitem__(self, index):
        if isinstance(index, slice):
            return IndexView(self, range(len(self))[index])
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._data is None:
            self._data = np.memmap(self.path, mode="r", dtype=np.uint8)
        return self._data[
            int(self.offsets[index]) : int(self.offsets[index + 1])
        ].tobytes()

    def __getstate__(self):
        return dict(self.__dict__, _data=None)


class FrameLabels(Mapping):
    def __init__(self, ids, rows):
        self.ids, self.rows = ids, rows
        values = ids[:]
        self.order = np.argsort(values)
        self.sorted_ids = values[self.order]
        if len(values) > 1 and np.any(self.sorted_ids[1:] == self.sorted_ids[:-1]):
            raise ValueError("Duplicate frame-label segment IDs")

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        return (int(value) for value in self.ids[:])

    def __getitem__(self, key):
        position = int(np.searchsorted(self.sorted_ids, key))
        if position == len(self) or self.sorted_ids[position] != key:
            raise KeyError(key)
        return self.rows[int(self.order[position])]


class AlignedTargets:
    """Resolve one target row at a time; missing rows produce zeros on demand."""

    def __init__(self, source, order, dtype=None):
        self.source, self.order = source, order
        self.shape = (len(order), *source.shape[1:])
        self.dtype = np.dtype(dtype or source.dtype)

    def __getitem__(self, index):
        row = int(self.order[index])
        if row < 0:
            return np.zeros(self.shape[1:], dtype=self.dtype)
        value = np.array(self.source[row], dtype=self.dtype, copy=True)
        if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError(f"Invalid teacher probabilities at source row {row}")
        return value


def _array(path, name, shape=None, kind=None):
    array = DiskArray(path / f"{name}.npy")
    if shape is not None and array.shape != shape:
        raise ValueError(f"Invalid shape for {name}: {array.shape}, expected {shape}")
    if kind is not None and array.dtype.kind not in kind:
        raise ValueError(f"Invalid dtype for {name}: {array.dtype}")
    return array


def _offsets(path, name, count, size):
    offsets = _array(path, name, (count + 1,), "u")
    if int(offsets[0]) != 0 or int(offsets[-1]) != size:
        raise ValueError(f"Invalid boundaries in {name}")
    for start in range(0, count, 65536):
        chunk = offsets[start : min(count + 1, start + 65537)]
        if np.any(chunk[1:] < chunk[:-1]):
            raise ValueError(f"Nonmonotonic boundaries in {name}")
    return offsets


def read_store(path: str, kind: str) -> dict[str, Any]:
    directory = Path(path)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("format"),
        manifest.get("format_version"),
        manifest.get("kind"),
    ) != ("britekit-indexed", 1, kind):
        raise ValueError(f"Unsupported {kind} store: {path}")
    count = manifest["count"]
    if not isinstance(count, int) or count < 0:
        raise ValueError("Invalid store count")
    data = dict(manifest["metadata"])
    ids = _array(directory, "segment_ids", (count,), "iu")
    if kind == "training":
        for name in ("class_names", "class_codes", "alt_names", "alt_codes"):
            if name not in data or len(data[name]) != len(data["class_names"]):
                raise ValueError(f"Invalid class metadata: {name}")
        offsets = _offsets(
            directory, "offsets", count, (directory / "specs.bin").stat().st_size
        )
        labels = _array(directory, "label_values", kind="iu")
        if len(labels.shape) != 1:
            raise ValueError("Label values must be one dimensional")
        label_offsets = _offsets(directory, "label_offsets", count, len(labels))
        data.update(
            spec_values=SpectrogramRows(offsets, directory / "specs.bin"),
            spec_class_indexes=RaggedRows(label_offsets, labels, as_list=True),
            spec_segment_ids=ids,
            spec_recording_ids=_array(directory, "recording_ids", (count,), "iu"),
        )
    elif kind == "training-frames":
        values = _array(directory, "values", kind="fbiu")
        if len(values.shape) != 1:
            raise ValueError("Frame values must be one dimensional")
        offsets = _offsets(directory, "offsets", count, len(values))
        data["labels"] = FrameLabels(ids, RaggedRows(offsets, values))
    elif kind == "teacher-targets":
        classes = len(data["class_codes"])
        data["segment_ids"] = ids
        data["probabilities"] = _array(
            directory, "probabilities", (count, classes), "f"
        )
        frames = _array(directory, "frame_probabilities", kind="f")
        if len(frames.shape) != 3 or frames.shape[:2] != (count, classes):
            raise ValueError("Invalid teacher frame probability shape")
        data["frame_probabilities"] = frames
    return data


def align_targets(data, class_codes, segment_ids):
    if data.get("format_version") != 2 or list(data["class_codes"]) != list(
        class_codes
    ):
        raise ValueError(
            "Teacher-target version or class codes and ordering do not match training data"
        )
    ids = data["segment_ids"][:]
    sort_order = np.argsort(ids)
    sorted_ids = ids[sort_order]
    if len(ids) > 1 and np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise ValueError("Teacher-target store contains duplicate segment IDs")
    wanted = np.asarray(
        segment_ids[:] if isinstance(segment_ids, DiskArray) else segment_ids
    )
    positions = np.searchsorted(sorted_ids, wanted)
    available = positions < len(ids)
    if len(ids):
        available &= sorted_ids[np.minimum(positions, len(ids) - 1)] == wanted
    order = np.full(len(wanted), -1, dtype=np.int64)
    order[available] = sort_order[positions[available]]
    return (
        AlignedTargets(data["probabilities"], order, np.float32),
        AlignedTargets(data["frame_probabilities"], order),
        available,
    )
