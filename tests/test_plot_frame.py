import json
import pickle
from copy import deepcopy

import numpy as np
import pytest
from click.testing import CliRunner

from britekit.cli import cli
from britekit.commands import plot_frame
from britekit.commands import _plot_frame
from britekit.core import config_loader, util
from britekit.core.base_config import BaseConfig


@pytest.fixture
def example(monkeypatch, tmp_path):
    monkeypatch.setattr(config_loader, "_base_config", BaseConfig())
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """audio:
  spec_height: 8
  spec_width: 16
  spec_duration: 2
  convert_to_db: true
  top_db: 40
  freq_scale: linear
train:
  sed_fps: 2
"""
    )
    spec = util.compress_spectrogram(
        np.geomspace(0.001, 1, 128).astype(np.float32).reshape(8, 16)
    )
    data = dict(
        class_codes=["a", "b"],
        class_names=["Alpha", "Beta"],
        spec_values=[spec] * 8,
        spec_segment_ids=list(range(10, 18)),
        spec_recording_ids=[100, 100, 101, 101, 200, 200, 201, 201],
        spec_class_indexes=[[0]] * 4 + [[1]] * 4,
    )
    labels = {i: np.array([0.1, 0.8, 0.1, 0.7], np.float32) for i in range(10, 18)}
    train_path, frame_path = tmp_path / "train.pkl", tmp_path / "frames.pkl"
    train_path.write_bytes(pickle.dumps(data))
    frame_path.write_bytes(pickle.dumps(labels))
    return data, labels, train_path, frame_path, cfg


def test_sampling_is_reproducible_recording_diverse_and_score_independent(example):
    data, labels, *_ = example
    sample = _plot_frame._select_sample(data, labels, 2, 42)
    assert len(sample["spec_segment_ids"]) == 4
    assert len(set(sample["spec_recording_ids"])) == 4
    assert sample["spec_class_indexes"] == [[0], [0], [1], [1]]
    changed = {key: np.ones(4, np.float32) for key in labels}
    reversed_data = {
        key: list(reversed(value)) if key.startswith("spec_") else value
        for key, value in data.items()
    }
    assert _plot_frame._select_sample(reversed_data, changed, 2, 42) == sample
    # A class with fewer examples contributes all available segments, without duplicates.
    all_samples = _plot_frame._select_sample(data, labels, 10, 42)
    assert len(set(all_samples["spec_segment_ids"])) == 8


def test_review_bundle_and_comparison_reuse_exact_sample_and_display(
    monkeypatch, tmp_path, example
):
    from PIL import Image

    data, labels, train_path, frame_path, cfg_path = example
    first = tmp_path / "round1"
    result = CliRunner().invoke(
        cli,
        [
            "plot-frame",
            str(frame_path),
            "--train-pickle",
            str(train_path),
            "--cfg",
            str(cfg_path),
            "--output",
            str(first),
            "--per-class",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads((first / "selection.json").read_text())
    saved_labels = pickle.loads((first / "labels.pkl").read_bytes())
    saved_sample = pickle.loads((first / "sample.pkl").read_bytes())
    assert len(manifest["samples"]) == len(saved_labels) == 4
    assert manifest["num_frames"] == 4
    assert manifest["audio_config"]["top_db"] == 40
    assert manifest["baseline"] is None
    for record in manifest["samples"]:
        with Image.open(first / record["image"]) as image:
            assert image.size == (1200, 648)
        np.testing.assert_array_equal(
            saved_labels[record["segment_id"]], labels[record["segment_id"]]
        )
    assert "Frame-label inspection" in (first / "index.html").read_text()
    original_files = {p.name: p.read_bytes() for p in first.iterdir() if p.is_file()}

    # Comparison uses only the saved sample/config, even without the original training file.
    train_path.unlink()
    cfg_path.unlink()
    later_labels = {key: value + 0.05 for key, value in labels.items()}
    later_path = tmp_path / "later.pkl"
    later_path.write_bytes(pickle.dumps(later_labels))
    calls = []
    original_plot = _plot_frame._plot_frame_spec

    def capture(spec, curve, previous, record, cfg, output):
        assert cfg.audio.top_db == 40
        np.testing.assert_array_equal(previous, saved_labels[record["segment_id"]])
        np.testing.assert_array_equal(curve, later_labels[record["segment_id"]])
        calls.append(record["segment_id"])
        original_plot(spec, curve, previous, record, cfg, output)

    monkeypatch.setattr(_plot_frame, "_plot_frame_spec", capture)
    second = tmp_path / "round2"
    plot_frame(
        str(later_path), str(second), baseline_path=str(first), seed=999, per_class=99
    )
    updated = json.loads((second / "selection.json").read_text())
    assert calls == saved_sample["spec_segment_ids"]
    assert updated["seed"] == 42 and updated["per_class"] == 2
    assert pickle.loads((second / "sample.pkl").read_bytes()) == saved_sample
    assert all(
        r["mean_absolute_change"] == pytest.approx(0.05) for r in updated["samples"]
    )
    assert "Frame-label comparison" in (second / "index.html").read_text()
    assert original_files == {
        p.name: p.read_bytes() for p in first.iterdir() if p.is_file()
    }


@pytest.mark.parametrize(
    "problem, message",
    [
        ("missing", "missing selected segment"),
        ("length", "frame shape"),
        ("nan", "invalid frame probabilities"),
    ],
)
def test_comparison_rejects_invalid_labels_without_resampling(
    monkeypatch, tmp_path, example, problem, message
):
    _, labels, train_path, frame_path, cfg_path = example
    monkeypatch.setattr(_plot_frame, "_plot_frame_spec", lambda *args: None)
    first = tmp_path / "round1"
    plot_frame(
        str(frame_path),
        str(first),
        train_pickle_path=str(train_path),
        cfg_path=str(cfg_path),
        per_class=1,
    )
    manifest = json.loads((first / "selection.json").read_text())
    selected_id = manifest["samples"][0]["segment_id"]
    updated = deepcopy(labels)
    if problem == "missing":
        del updated[selected_id]
    elif problem == "length":
        updated[selected_id] = np.ones(5)
    else:
        updated[selected_id][0] = np.nan
    frame_path.write_bytes(pickle.dumps(updated))
    destination = tmp_path / "round2"
    with pytest.raises(ValueError, match=message):
        plot_frame(str(frame_path), str(destination), baseline_path=str(first))
    assert not destination.exists()


def test_existing_review_is_not_overwritten(tmp_path, example):
    _, _, train_path, frame_path, _ = example
    with pytest.raises(FileExistsError):
        plot_frame(str(frame_path), str(tmp_path), train_pickle_path=str(train_path))
