import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from britekit import OccurrenceDatabase, OccurrenceDatabaseV2
from britekit.core.exceptions import InputError
from britekit.occurrence_db.occurrence_migration import migrate_occurrence_v1_to_v2


def create_v1_database(path: Path) -> dict[str, np.ndarray]:
    database = OccurrenceDatabase(str(path))
    ottawa_id = database.insert_county(
        "Ottawa", "CA-ON-OT", -76.35, -75.24, 44.96, 45.54
    )
    edmonton_id = database.insert_county(
        "Edmonton", "CA-AB-EL", -114.65, -112.91, 51.88, 52.89
    )
    owl_id = database.insert_class("Barred Owl")
    goldfinch_id = database.insert_class("American Goldfinch")
    values = {
        "ottawa_owl": np.linspace(0, 1, 48, dtype=np.float32),
        "ottawa_goldfinch": np.linspace(1, 0, 48, dtype=np.float32),
        "edmonton_owl": np.full(48, 0.25, dtype=np.float32),
    }
    database.insert_occurrences(ottawa_id, owl_id, values["ottawa_owl"])
    database.insert_occurrences(ottawa_id, goldfinch_id, values["ottawa_goldfinch"])
    database.insert_occurrences(edmonton_id, owl_id, values["edmonton_owl"])
    database.close()
    return values


def write_metadata(path: Path) -> Path:
    metadata = {
        "region_pack": {
            "code": "canada",
            "name": "Canada test pack",
            "data_version": "test-1",
            "taxonomy_version": "test-taxonomy",
        },
        "areas": [
            {
                "code": "CA",
                "name": "Canada",
                "level": 0,
                "area_type": "country",
            },
            {
                "code": "CA-AB",
                "parent_code": "CA",
                "name": "Alberta",
                "level": 1,
                "area_type": "province",
            },
            {
                "code": "CA-ON",
                "parent_code": "CA",
                "name": "Ontario",
                "level": 1,
                "area_type": "province",
            },
        ],
        "levels": [
            {
                "country_code": "CA",
                "level": 1,
                "singular_name": "Province/Territory",
                "plural_name": "Provinces/Territories",
            },
            {
                "country_code": "CA",
                "level": 2,
                "singular_name": "County",
                "plural_name": "Counties",
            },
        ],
    }
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_migrate_v1_database_without_changing_values(tmp_path: Path):
    source_path = tmp_path / "occurrence-v1.db"
    output_path = tmp_path / "occurrence-v2.db"
    values = create_v1_database(source_path)
    metadata_path = write_metadata(tmp_path / "areas.json")

    report = migrate_occurrence_v1_to_v2(source_path, output_path, metadata_path)

    assert report.source_area_count == 2
    assert report.parent_area_count == 3
    assert report.total_area_count == 5
    assert report.class_count == 2
    assert report.occurrence_count == 3

    source = sqlite3.connect(source_path)
    assert source.execute("SELECT Version FROM SchemaVersion").fetchone()[0] == 1
    assert source.execute("SELECT COUNT(*) FROM County").fetchone()[0] == 2
    source.close()

    with OccurrenceDatabaseV2(output_path) as migrated:
        assert migrated.validate() == []
        assert migrated.get_area("CA").name == "Canada"
        assert migrated.get_area("CA-ON").parent_code == "CA"
        ottawa = migrated.get_area("CA-ON-OT")
        assert ottawa.name == "Ottawa"
        assert ottawa.parent_code == "CA-ON"
        assert ottawa.area_type == "subnational2"
        assert ottawa.selectable
        assert ottawa.min_longitude == -76.35
        assert [area.code for area in migrated.get_children("CA")] == [
            "CA-AB",
            "CA-ON",
        ]
        assert {area.code for area in migrated.get_descendants("CA")} == {
            "CA-AB",
            "CA-AB-EL",
            "CA-ON",
            "CA-ON-OT",
        }
        np.testing.assert_array_equal(
            migrated.get_occurrences(ottawa.id, "Barred Owl"),
            values["ottawa_owl"].astype(np.float16).astype(np.float32),
        )
        np.testing.assert_array_equal(
            migrated.get_occurrences(ottawa.id, "American Goldfinch"),
            values["ottawa_goldfinch"].astype(np.float16).astype(np.float32),
        )
        assert migrated.get_occurrences(ottawa.id, "Missing class") == []
        level = migrated.conn.execute(
            """
            SELECT SingularName, PluralName
            FROM AdministrativeLevel
            WHERE CountryAreaID = ? AND Level = 1
            """,
            (migrated.get_area("CA").id,),
        ).fetchone()
        assert tuple(level) == ("Province/Territory", "Provinces/Territories")


def test_v2_allows_selectable_country_without_placeholder_levels(tmp_path: Path):
    path = tmp_path / "guernsey.db"
    with OccurrenceDatabaseV2(path, create=True) as database:
        database.set_region_pack("europe", "Europe", "test-1")
        area_id = database.insert_area(
            code="GG",
            name="Guernsey",
            level=0,
            area_type="country",
            selectable=True,
        )
        class_id = database.insert_class("Common Blackbird")
        database.insert_occurrences(area_id, class_id, np.ones(48))

        assert database.get_area("GG").selectable
        assert database.get_children("GG") == []
        assert database.get_occurrences(area_id, "Common Blackbird")[0] == 1
        assert database.validate() == []


def test_migration_failure_does_not_publish_partial_database(tmp_path: Path):
    source_path = tmp_path / "occurrence-v1.db"
    output_path = tmp_path / "occurrence-v2.db"
    create_v1_database(source_path)
    metadata_path = write_metadata(tmp_path / "areas.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["areas"] = [metadata["areas"][0]]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(InputError, match="missing parent definitions"):
        migrate_occurrence_v1_to_v2(source_path, output_path, metadata_path)

    assert not output_path.exists()
