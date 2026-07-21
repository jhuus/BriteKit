import sqlite3

import pytest

from britekit.occurrence_db.location_catalog import compile_location_catalog
from britekit.occurrence_db.occurrence_database_v2 import OccurrenceDatabaseV2


def _create_occurrence_database(path):
    with OccurrenceDatabaseV2(path, create=True) as database:
        database.set_region_pack("test", "Test Region", "2026.1", "tax-1")
        database.insert_area(
            code="XY",
            name="Example Country",
            level=0,
            area_type="country",
            selectable=False,
            display_order=1,
        )
        database.insert_area(
            code="XY-ONE",
            name="First Province",
            level=1,
            area_type="province",
            selectable=False,
            parent_code="XY",
        )
        leaf_id = database.insert_area(
            code="XY-ONE-A",
            name="Example County",
            level=2,
            area_type="county",
            selectable=True,
            parent_code="XY-ONE",
            min_longitude=-80,
            max_longitude=-70,
            min_latitude=40,
            max_latitude=50,
        )
        database.insert_level("XY", 1, "Province", "Provinces")
        database.insert_level("XY", 2, "County", "Counties")
        class_id = database.insert_class("Example bird")
        database.insert_occurrences(leaf_id, class_id, __import__("numpy").ones(48))


def test_compile_location_catalog_copies_only_picker_metadata(tmp_path):
    source = tmp_path / "occurrence.db"
    output = tmp_path / "locations.db"
    _create_occurrence_database(source)

    report = compile_location_catalog(source, output)

    assert report.region_code == "test"
    assert report.area_count == 3
    assert report.selectable_area_count == 1
    assert report.level_count == 2

    with sqlite3.connect(output) as connection:
        connection.row_factory = sqlite3.Row
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert tables == {
            "CatalogMetadata",
            "AdministrativeArea",
            "AdministrativeLevel",
        }
        metadata = connection.execute("SELECT * FROM CatalogMetadata").fetchone()
        assert metadata["RegionCode"] == "test"
        assert metadata["DataVersion"] == "2026.1"
        assert metadata["TaxonomyVersion"] == "tax-1"
        assert (
            connection.execute(
                "SELECT Code FROM AdministrativeArea WHERE Selectable = 1"
            ).fetchone()[0]
            == "XY-ONE-A"
        )
        bounds = connection.execute(
            """
            SELECT MinLongitude, MaxLongitude, MinLatitude, MaxLatitude
            FROM AdministrativeArea WHERE Code = 'XY-ONE-A'
            """
        ).fetchone()
        assert tuple(bounds) == (-80, -70, 40, 50)
        assert (
            connection.execute(
                "SELECT PluralName FROM AdministrativeLevel WHERE Level = 1"
            ).fetchone()[0]
            == "Provinces"
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_compile_location_catalog_does_not_overwrite_output(tmp_path):
    source = tmp_path / "occurrence.db"
    output = tmp_path / "locations.db"
    _create_occurrence_database(source)
    output.write_text("keep me")

    with pytest.raises(FileExistsError):
        compile_location_catalog(source, output)

    assert output.read_text() == "keep me"
