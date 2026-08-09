#!/usr/bin/env python3

import os
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
from britekit import Extractor, TrainingDatabase

db_path = str(Path("tests") / "db" / "_test.db")


@pytest.fixture(scope="module")
def db():
    """Setup logic for all tests."""
    if os.path.exists(db_path):
        os.remove(db_path)

    db = TrainingDatabase(db_path=db_path)
    yield db
    db.close()


@pytest.fixture(scope="session", autouse=True)
def finalize_at_end():
    """Cleanup after all tests are done."""
    yield


def test_extract_all(db: TrainingDatabase):
    class_name = "Test Class"
    class_code = "ABCD"
    dir_path = str(Path("tests") / "recordings")
    extractor = Extractor(db, class_name, class_code, overlap=0)
    extractor.extract_all(dir_path)

    results = db.get_source()
    assert len(results) == 1
    assert results[0].name == "default"

    results = db.get_category()
    assert len(results) == 1
    assert results[0].name == "default"

    assert db.get_recording_count() == 2
    assert db.get_segment_count() == 4
    assert db.get_segment_class_count() == 4
    assert db.get_specvalue_count() == 4


def test_extract_all_can_include_existing_recordings():
    extractor = Extractor.__new__(Extractor)
    extractor.filenames = {"existing.wav"}
    extractor.increment = 1
    extractor.audio = Mock()
    extractor.audio.seconds.return_value = 10
    extractor.insert_spectrograms = Mock(return_value=2)

    with patch(
        "britekit.training_db.extractor.util.get_audio_files",
        return_value=["/recordings/existing.wav"],
    ):
        assert extractor.extract_all("/recordings", include_existing=False) == 0
        assert extractor.extract_all("/recordings", include_existing=True) == 2

    extractor.insert_spectrograms.assert_called_once()


def test_extract_all_randomizes_recordings_and_offsets():
    extractor = Extractor.__new__(Extractor)
    extractor.filenames = set()
    extractor.increment = 1
    extractor.audio = Mock()
    extractor.audio.seconds.return_value = 10
    extractor.insert_spectrograms = Mock(return_value=2)
    recording_paths = [f"/recordings/{name}.wav" for name in ("a", "b", "c")]

    with (
        patch(
            "britekit.training_db.extractor.util.get_audio_files",
            return_value=recording_paths,
        ),
        patch("britekit.training_db.extractor.random.shuffle") as shuffle,
        patch(
            "britekit.training_db.extractor.random.sample",
            side_effect=lambda offsets, count: offsets[-count:],
        ) as sample,
    ):
        count = extractor.extract_all(
            "/recordings", max_spec=2, max_rec=2, randomize=True
        )

    assert count == 4
    shuffle.assert_called_once()
    assert sample.call_count == 2
    assert extractor.insert_spectrograms.call_count == 2
    for call in extractor.insert_spectrograms.call_args_list:
        assert call.args[1] == [8, 9]
