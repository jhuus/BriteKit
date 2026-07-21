import pickle
import sqlite3

import numpy as np
import pytest

from britekit.core.exceptions import DatabaseError
from britekit.core.pickler import OccurrencePickler
from britekit.occurrence_db.occurrence_database_v2 import OccurrenceDatabaseV2
from britekit.occurrence_db.occurrence_pickle import OccurrencePickleProvider
from britekit.occurrence_db.occurrence_pickle_v2 import (
    FORMAT_NAME,
    FORMAT_VERSION,
    compile_occurrence_pickle_v2,
)


def _create_database(path):
    with OccurrenceDatabaseV2(path, create=True) as database:
        database.set_region_pack("test", "Test Region", "2026.1")
        database.insert_area(
            code="XY",
            name="Example Country",
            level=0,
            area_type="country",
            selectable=False,
        )
        # Insert in the opposite order from display order to exercise offsets.
        beta_id = database.insert_area(
            code="XY-B",
            name="Beta County",
            level=1,
            area_type="county",
            selectable=True,
            parent_code="XY",
            min_longitude=-80,
            max_longitude=-70,
            min_latitude=40,
            max_latitude=50,
        )
        alpha_id = database.insert_area(
            code="XY-A",
            name="Alpha County",
            level=1,
            area_type="county",
            selectable=True,
            parent_code="XY",
            min_longitude=-100,
            max_longitude=-90,
            min_latitude=40,
            max_latitude=50,
        )
        first_class = database.insert_class("First bird")
        second_class = database.insert_class("Second bird")

        alpha_first = np.linspace(0, 0.47, 48, dtype=np.float32)
        beta_first = np.linspace(0.47, 0, 48, dtype=np.float32)
        alpha_second = np.full(48, 0.25, dtype=np.float32)
        database.insert_occurrences(alpha_id, first_class, alpha_first)
        database.insert_occurrences(beta_id, first_class, beta_first)
        database.insert_occurrences(alpha_id, second_class, alpha_second)
    return alpha_first.astype(np.float16), beta_first.astype(np.float16)


def test_compact_pickle_preserves_provider_behavior(tmp_path):
    database_path = tmp_path / "occurrence.db"
    pickle_path = tmp_path / "occurrence.pkl"
    alpha_values, beta_values = _create_database(database_path)

    report = compile_occurrence_pickle_v2(database_path, pickle_path)
    provider = OccurrencePickleProvider(pickle_path)

    assert report.area_count == 2
    assert report.class_count == 2
    assert report.occurrence_count == 3
    assert provider.format_version == FORMAT_VERSION
    assert [area.code for area in provider.find_counties("XY")] == ["XY-B", "XY-A"]
    assert provider.find_county(45, -95).code == "XY-A"

    found, class_found, value = provider.occurrence_value(
        "First bird", region_code="XY-A", week_num=10, smoothed=False
    )
    assert found and class_found
    assert value == alpha_values[10]

    _, _, smoothed = provider.occurrence_value(
        "First bird", region_code="XY-A", week_num=0
    )
    assert smoothed == max(alpha_values[-1], alpha_values[0], alpha_values[1])

    _, _, maximum = provider.occurrence_value("First bird", region_code="XY-A")
    assert maximum == alpha_values.max()

    _, _, regional = provider.occurrence_value(
        "First bird", region_code="XY", week_num=12, smoothed=False
    )
    assert regional == pytest.approx(
        (float(alpha_values[12]) + float(beta_values[12])) / 2
    )

    found, class_found, value = provider.occurrence_value(
        "Second bird", region_code="XY-B", week_num=1
    )
    assert found and not class_found and value is None


def test_compact_pickle_has_versioned_array_payload(tmp_path):
    database_path = tmp_path / "occurrence.db"
    pickle_path = tmp_path / "occurrence.pkl"
    _create_database(database_path)
    compile_occurrence_pickle_v2(database_path, pickle_path)

    with pickle_path.open("rb") as stream:
        payload = pickle.load(stream)

    assert payload["format"] == FORMAT_NAME
    assert payload["version"] == FORMAT_VERSION
    assert payload["values"].shape == (3, 48)
    assert payload["values"].dtype == np.float16
    assert "smoothed" not in payload
    assert "max" not in payload


def test_compiler_rejects_non_48_week_rows(tmp_path):
    database_path = tmp_path / "occurrence.db"
    pickle_path = tmp_path / "occurrence.pkl"
    _create_database(database_path)
    with OccurrenceDatabaseV2(database_path) as database:
        area_id = database.get_area("XY-A").id
        class_id = database.insert_class("Bad calendar bird")
        database.insert_occurrences(area_id, class_id, np.ones(52))

    with pytest.raises(DatabaseError, match="expected exactly 48"):
        compile_occurrence_pickle_v2(database_path, pickle_path)

    assert not pickle_path.exists()


def test_provider_rejects_unknown_compiled_version(tmp_path):
    pickle_path = tmp_path / "occurrence.pkl"
    with pickle_path.open("wb") as stream:
        pickle.dump({"format": FORMAT_NAME, "version": 999}, stream)

    with pytest.raises(ValueError, match="Unsupported occurrence pickle version 999"):
        OccurrencePickleProvider(pickle_path)


def test_pickler_detects_explicit_v1_schema_version(tmp_path):
    database_path = tmp_path / "occurrence.db"
    with sqlite3.connect(database_path) as database:
        database.execute("CREATE TABLE SchemaVersion (Version INTEGER NOT NULL)")
        database.execute("INSERT INTO SchemaVersion VALUES (1)")

    assert OccurrencePickler._schema_version(str(database_path)) == 1
