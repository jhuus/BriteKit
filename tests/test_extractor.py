import os
import pytest
from britekit import Extractor, TrainingDatabase

db_path = os.path.join("data", "_test.db")


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
    dir_path = os.path.join("tests", "recordings")
    extractor = Extractor(db, class_name, class_code, overlap=0)
    extractor.extract_all(dir_path)

    results = db.get_source()
    assert len(results) == 1
    assert results[0].name == "default"

    results = db.get_category()
    assert len(results) == 1
    assert results[0].name == "default"

    assert db.get_recording_count() == 2
    assert db.get_segment_count() == 3
    assert db.get_segment_class_count() == 3
    assert db.get_specvalue_count() == 3
