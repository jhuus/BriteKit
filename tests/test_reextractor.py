import numpy as np
import pytest

from britekit.core.base_config import BaseConfig
from britekit.core.config_loader import set_base_config
from britekit.core.reextractor import Reextractor
from britekit.training_db.training_db import TrainingDatabase


@pytest.mark.parametrize("class_names", ["Bird A\nBird B\n", "Bird A\nBird A\n"])
def test_class_list_extracts_each_recording_once(tmp_path, monkeypatch, class_names):
    config = BaseConfig()
    set_base_config(config)
    loads = []

    class FakeAudio:
        def load(self, path):
            loads.append(path)

        def get_spectrograms(self, offsets, **kwargs):
            return [np.ones((2, 4), dtype=np.float32) for _ in offsets], None

    monkeypatch.setattr("britekit.core.audio.Audio", FakeAudio)
    db_path = str(tmp_path / "training.db")
    recording_path = tmp_path / "recording.wav"
    recording_path.touch()
    classes_path = tmp_path / "classes.csv"
    classes_path.write_text("Name\n" + class_names)
    try:
        with TrainingDatabase(db_path) as db:
            category_id = db.insert_category("Birds")
            class_ids = [
                db.insert_class(category_id, name) for name in ("Bird A", "Bird B")
            ]
            source_id = db.insert_source("Test")
            recording_id = db.insert_recording(
                source_id, recording_path.name, str(recording_path)
            )
            segment_ids = [db.insert_segment(recording_id, offset) for offset in (0, 5)]
            for segment_id in segment_ids:
                for class_id in class_ids:
                    db.insert_segment_class(segment_id, class_id)

        reextractor = Reextractor(
            db_path=db_path, classes_path=str(classes_path), spec_group="__temp__"
        )
        # A second trial must also replace the group successfully.
        for _ in range(2):
            reextractor.run(quiet=True)
            with TrainingDatabase(db_path) as db:
                values = db.get_specvalue()
                assert sorted(value.segment_id for value in values) == segment_ids
                assert db.get_segment_class_count() == 4
        assert loads == [str(recording_path)] * 2
    finally:
        set_base_config(BaseConfig())
