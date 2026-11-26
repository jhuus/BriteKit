import pickle
import random

from click.testing import CliRunner
import pytest

from britekit.cli import cli
from britekit.commands import export_training, pickle_train
from britekit.core.exceptions import InputError
from britekit.core.indexed_store import read_store
from britekit.core.pickler import TrainingPickler
from britekit.core.training_exporter import TrainingExporter
from britekit.training_db.training_db import TrainingDatabase


@pytest.fixture
def source(tmp_path):
    path = str(tmp_path / "source.db")
    with TrainingDatabase(path) as db:
        category = db.insert_category("Birds")
        a = db.insert_class(category, "A", code="a", alt_name="Alt A")
        b = db.insert_class(category, "B", code="b")
        source_id = db.insert_source("test")
        recording = db.insert_recording(source_id, "one.wav", "/one.wav", 10)
        group = db.get_specgroup({"Name": "default"})[0].id
        other = db.insert_specgroup("other")
        for i in range(6):
            segment = db.insert_segment(recording, i)
            db.insert_segment_class(segment, a if i < 3 else b)
            if i == 0:
                db.insert_segment_class(segment, b)
            db.insert_specvalue(b"compressed-" + bytes([i]), group, segment)
            db.insert_specvalue(b"other-" + bytes([i]), other, segment)
    return path


@pytest.mark.parametrize("limit", [None, 2])
def test_export_matches_legacy_selection_without_fetching_blob_lists(
    source, tmp_path, monkeypatch, limit
):
    legacy_path = tmp_path / "legacy.pkl"
    legacy = TrainingPickler(source, str(legacy_path), max_per_class=limit)
    try:
        legacy.pickle(quiet=True)
    finally:
        legacy.db.close()
    expected = pickle.loads(legacy_path.read_bytes())
    original = TrainingDatabase.get_spectrogram_by_class

    def metadata_only(self, *args, **kwargs):
        assert kwargs.get("include_value") is False
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TrainingDatabase, "get_spectrogram_by_class", metadata_only)
    state = random.getstate()
    output = tmp_path / "store"
    TrainingExporter(source, str(output), max_per_class=limit).export(quiet=True)
    assert random.getstate() == state
    actual = read_store(str(output), "training")
    for key in ("class_names", "class_codes", "alt_names", "alt_codes"):
        assert actual[key] == expected[key]
    for key in ("spec_values", "spec_segment_ids", "spec_recording_ids"):
        assert list(actual[key]) == expected[key]
    assert [set(row) for row in actual["spec_class_indexes"]] == [
        set(row) for row in expected["spec_class_indexes"]
    ]


def test_class_subset_order_and_group(source, tmp_path):
    classes = tmp_path / "classes.csv"
    classes.write_text("Name\nB\nA\n")
    output = tmp_path / "store"
    export_training(
        db_path=source,
        output_path=str(output),
        classes_path=str(classes),
        spec_group="other",
    )
    data = read_store(str(output), "training")
    assert data["class_names"] == ["B", "A"]
    assert len(data["spec_values"]) == 6
    assert all(blob.startswith(b"other-") for blob in data["spec_values"])
    assert data["spec_class_indexes"][0] == [0, 1]
    assert data["spec_class_indexes"][1] == [0]
    assert data["spec_class_indexes"][-1] == [1]


def test_cli_default_and_deprecated_alias(source, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli, ["export-training", "--db", source, "--root", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "data/training/manifest.json").is_file()
    output = tmp_path / "alias"
    result = runner.invoke(cli, ["pickle-train", "--db", source, "-o", str(output)])
    assert result.exit_code == 0, result.output
    assert "deprecated" in result.output.lower()
    assert read_store(str(output), "training")["class_codes"] == ["a", "b"]
    with pytest.warns(FutureWarning, match="export_training"):
        pickle_train(db_path=source, output_path=str(tmp_path / "api-alias"))


@pytest.mark.parametrize(
    "problem", ["empty-group", "unknown-class", "duplicate-class", "zero-limit"]
)
def test_invalid_selection_is_not_published(source, tmp_path, problem):
    output = tmp_path / "store"
    kwargs = {}
    if problem == "empty-group":
        kwargs["spec_group"] = "missing"
    elif problem == "zero-limit":
        kwargs["max_per_class"] = 0
    else:
        classes = tmp_path / "classes.csv"
        classes.write_text(
            "Name\n" + ("Missing\n" if problem == "unknown-class" else "A\nA\n")
        )
        kwargs["classes_path"] = str(classes)
    with pytest.raises(InputError):
        TrainingExporter(source, str(output), **kwargs).export(quiet=True)
    assert not output.exists()
    assert not list(tmp_path.glob(".store-*"))


def test_existing_output_preserved(source, tmp_path):
    output = tmp_path / "store"
    output.mkdir()
    (output / "keep").write_text("original")
    with pytest.raises(FileExistsError):
        TrainingExporter(source, str(output)).export()
    assert (output / "keep").read_text() == "original"


def test_bad_blob_cleanup(source, tmp_path):
    with TrainingDatabase(source) as db:
        db.conn.execute("UPDATE SpecValue SET Value = X''")
        db.conn.commit()
    output = tmp_path / "store"
    with pytest.raises(InputError, match="Invalid compressed"):
        TrainingExporter(source, str(output)).export(quiet=True)
    assert not output.exists()
    assert not list(tmp_path.glob(".store-*"))


def test_export_can_feed_training_loader(source, tmp_path):
    import numpy as np

    from britekit.core import config_loader
    from britekit.core.base_config import BaseConfig
    from britekit.core.data_module import DataModule
    from britekit.core.util import compress_spectrogram

    blob = compress_spectrogram(np.full((2, 3), 0.5, dtype=np.float32), bits=16)
    with TrainingDatabase(source) as db:
        db.conn.execute("UPDATE SpecValue SET Value = ?", (blob,))
        db.conn.commit()
    output = tmp_path / "store"
    TrainingExporter(source, str(output)).export(quiet=True)
    previous = config_loader.get_config()
    cfg = BaseConfig()
    cfg.audio.spec_height = 2
    cfg.audio.spec_width = 3
    cfg.train.train_pickle = str(output)
    cfg.train.augment = False
    cfg.train.num_workers = 0
    cfg.train.batch_size = 2
    try:
        config_loader.set_base_config(cfg)
        module = DataModule()
        module.prepare_fold(0)
        batch = next(iter(module.train_dataloader()))
        assert tuple(batch["input"].shape) == (2, 1, 2, 3)
    finally:
        config_loader.set_base_config(previous)
