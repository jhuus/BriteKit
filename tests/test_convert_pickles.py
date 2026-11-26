"""Conversion round trips, validation and safe publication."""

import json
import pickle

import numpy as np
import pytest

from britekit.core.convert_pickles import convert_pickle


def convert(tmp_path, kind, data):
    source = tmp_path / "source.pkl"
    source.write_bytes(pickle.dumps(data))
    output = tmp_path / "store"
    convert_pickle(kind, str(source), str(output))
    assert pickle.loads(source.read_bytes()).keys() == data.keys()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["kind"] == kind
    assert manifest["format_version"] == 1
    return output, manifest


def load(output, name):
    array = np.load(output / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    assert isinstance(array, np.memmap)
    return array


def test_training_blob_and_multilabel_round_trip(tmp_path):
    blobs = [b"\x00compressed-8", b"BKSPcompressed-16"]
    data = dict(
        class_names=["Bird", "Noise"],
        class_codes=["B", "N"],
        alt_names=[None, "Noise"],
        alt_codes=[None, "N"],
        spec_values=blobs,
        spec_class_indexes=[[1, 0], []],
        spec_segment_ids=[52, 11],
        spec_recording_ids=[3, 3],
    )
    output, manifest = convert(tmp_path, "training", data)
    assert manifest["metadata"]["class_codes"] == ["B", "N"]
    assert manifest["metadata"]["alt_names"] == [None, "Noise"]
    offsets = load(output, "offsets")
    with (output / "specs.bin").open("rb") as file:
        for i, blob in enumerate(blobs):
            file.seek(int(offsets[i]))
            assert file.read(int(offsets[i + 1] - offsets[i])) == blob
    np.testing.assert_array_equal(load(output, "label_offsets"), [0, 2, 2])
    np.testing.assert_array_equal(load(output, "label_values"), [1, 0])
    np.testing.assert_array_equal(load(output, "segment_ids"), [52, 11])
    np.testing.assert_array_equal(load(output, "recording_ids"), [3, 3])


@pytest.mark.parametrize("frames", [False, True])
def test_teacher_preserves_precision_order_and_provenance(tmp_path, frames):
    data = dict(
        format_version=2,
        class_codes=["Z", "A"],
        segment_ids=[81, 12],
        probabilities=np.array([[0.1, 0.8], [0, 1]], dtype=np.float32),
        source={"sha256": "source-hash"},
        teacher={"hard_labels": False},
    )
    if frames:
        data["frame_probabilities"] = (
            np.arange(12, dtype=np.float16).reshape(2, 2, 3) / 16
        )
    output, manifest = convert(tmp_path, "teacher-targets", data)
    for name in ("probabilities", "frame_probabilities"):
        if name in data:
            actual = load(output, name)
            assert actual.dtype == data[name].dtype
            np.testing.assert_array_equal(actual, data[name])
    assert manifest["metadata"]["source"] == data["source"]
    assert manifest["metadata"]["class_codes"] == ["Z", "A"]
    np.testing.assert_array_equal(load(output, "segment_ids"), [81, 12])


def test_frame_labels_preserve_ragged_vectors(tmp_path):
    data = {
        81: np.array([0, 1, 0.5], dtype=np.float32),
        12: np.array([], dtype=np.float32),
        99: np.array([1], dtype=np.float32),
    }
    output, _ = convert(tmp_path, "training-frames", data)
    ids, offsets, values = (
        load(output, name) for name in ("segment_ids", "offsets", "values")
    )
    assert values.dtype == np.float32
    for i, segment_id in enumerate(ids):
        np.testing.assert_array_equal(
            values[int(offsets[i]) : int(offsets[i + 1])], data[segment_id]
        )


@pytest.mark.parametrize("data", [{}, {1: np.array([], dtype=np.float32)}])
def test_empty_frame_labels(tmp_path, data):
    output, manifest = convert(tmp_path, "training-frames", data)
    assert manifest["count"] == len(data)
    assert load(output, "values").size == 0


@pytest.mark.parametrize(
    "kind,data",
    [
        ("training", {}),
        (
            "teacher-targets",
            dict(
                format_version=2,
                class_codes=["A"],
                segment_ids=[1],
                probabilities=np.array([[np.nan]]),
            ),
        ),
        (
            "teacher-targets",
            dict(
                format_version=2,
                class_codes=["A"],
                segment_ids=[1, 1],
                probabilities=np.zeros((2, 1)),
            ),
        ),
        ("training-frames", {1: np.array([2.0])}),
        (
            "training-frames",
            {1: np.zeros(2, dtype=np.float32), 2: np.zeros(2, dtype=np.float16)},
        ),
    ],
)
def test_invalid_input_does_not_publish_or_leave_temporary_files(tmp_path, kind, data):
    source = tmp_path / "source.pkl"
    source.write_bytes(pickle.dumps(data))
    with pytest.raises(ValueError):
        convert_pickle(kind, str(source), str(tmp_path / "store"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["source.pkl"]


def test_refuse_existing_destination_before_loading_source(tmp_path):
    output = tmp_path / "store"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        convert_pickle("training", str(tmp_path / "missing.pkl"), str(output))
    assert marker.read_text() == "keep"


def test_truncated_pickle_cleanup(tmp_path):
    source = tmp_path / "source.pkl"
    source.write_bytes(b"\x80\x05")
    with pytest.raises(EOFError):
        convert_pickle("training-frames", str(source), str(tmp_path / "store"))
    assert not (tmp_path / "store").exists()
    assert not list(tmp_path.glob(".store-*"))
