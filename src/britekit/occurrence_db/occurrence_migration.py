"""Non-destructive migration from occurrence schema v1 to schema v2."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

from britekit.core.exceptions import DatabaseError, InputError
from britekit.occurrence_db.occurrence_database_v2 import OccurrenceDatabaseV2


@dataclass(frozen=True)
class OccurrenceMigrationReport:
    source_area_count: int
    parent_area_count: int
    total_area_count: int
    class_count: int
    occurrence_count: int


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise InputError(f"Could not read area metadata: {error}") from error
    except json.JSONDecodeError as error:
        raise InputError(f"Invalid area metadata JSON: {error}") from error

    if not isinstance(metadata, dict):
        raise InputError("Area metadata must be a JSON object")
    pack = metadata.get("region_pack")
    areas = metadata.get("areas")
    levels = metadata.get("levels", [])
    if not isinstance(pack, dict):
        raise InputError("Area metadata requires a region_pack object")
    for key in ("code", "name", "data_version"):
        if not isinstance(pack.get(key), str) or not pack[key].strip():
            raise InputError(f"region_pack.{key} is required")
    if not isinstance(areas, list) or not areas:
        raise InputError("Area metadata requires a non-empty areas list")
    if not isinstance(levels, list):
        raise InputError("Area metadata levels must be a list")
    return metadata


def _validate_v1_source(connection: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    required = {"SchemaVersion", "County", "Class", "Occurrence"}
    missing = required - tables
    if missing:
        raise InputError(
            "Input is not a v1 occurrence database; missing tables: "
            + ", ".join(sorted(missing))
        )
    row = connection.execute("SELECT Version FROM SchemaVersion").fetchone()
    if row is None or row[0] != 1:
        actual = None if row is None else row[0]
        raise InputError(f"Expected occurrence schema version 1, found {actual}")


def _validate_area_metadata(
    areas: list[dict[str, Any]], county_codes: list[str]
) -> None:
    codes: set[str] = set()
    for index, area in enumerate(areas):
        if not isinstance(area, dict):
            raise InputError(f"areas[{index}] must be an object")
        for key in ("code", "name", "area_type"):
            if not isinstance(area.get(key), str) or not area[key].strip():
                raise InputError(f"areas[{index}].{key} is required")
        if not isinstance(area.get("level"), int) or area["level"] < 0:
            raise InputError(f"areas[{index}].level must be a non-negative integer")
        if area["code"] in codes:
            raise InputError(f"Duplicate administrative area code: {area['code']}")
        codes.add(area["code"])

    available_codes = codes | set(county_codes)
    missing_parents: set[str] = set()
    for county_code in county_codes:
        if "-" in county_code:
            parent_code = county_code.rsplit("-", 1)[0]
            if parent_code not in available_codes:
                missing_parents.add(parent_code)
    for area in areas:
        metadata_parent = area.get("parent_code")
        if metadata_parent is not None and metadata_parent not in codes:
            missing_parents.add(str(metadata_parent))
    if missing_parents:
        raise InputError(
            "Area metadata is missing parent definitions: "
            + ", ".join(sorted(missing_parents))
        )


def migrate_occurrence_v1_to_v2(
    input_path: str | Path,
    output_path: str | Path,
    metadata_path: str | Path,
) -> OccurrenceMigrationReport:
    """Create a schema-v2 database from an existing schema-v1 database.

    The input file is opened read-only and never modified. Parent area names,
    types, relationships, and GUI level labels come from the metadata JSON.
    """
    source_path = Path(input_path).expanduser().resolve()
    destination_path = Path(output_path).expanduser().resolve()
    area_metadata_path = Path(metadata_path).expanduser().resolve()
    if not source_path.is_file():
        raise InputError(f"Input occurrence database not found: {source_path}")
    if destination_path.exists():
        raise FileExistsError(f"Output occurrence database exists: {destination_path}")

    metadata = _read_metadata(area_metadata_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{source_path.as_posix()}?mode=ro", uri=True)
    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=destination_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    temporary_path.unlink()

    target: OccurrenceDatabaseV2 | None = None
    try:
        _validate_v1_source(source)
        counties = source.execute(
            "SELECT ID, Name, Code, MinX, MaxX, MinY, MaxY FROM County ORDER BY ID"
        ).fetchall()
        county_codes = [row[2] for row in counties]
        areas = metadata["areas"]
        _validate_area_metadata(areas, county_codes)

        target = OccurrenceDatabaseV2(temporary_path, create=True)
        connection = target.conn
        connection.execute("BEGIN")
        pack = metadata["region_pack"]
        connection.execute(
            """
            INSERT INTO RegionPack(
                ID, Code, Name, DataVersion, TaxonomyVersion
            ) VALUES (1, ?, ?, ?, ?)
            """,
            (
                pack["code"],
                pack["name"],
                pack["data_version"],
                pack.get("taxonomy_version"),
            ),
        )

        area_ids: dict[str, int] = {}
        ordered_areas = sorted(areas, key=lambda area: (area["level"], area["code"]))
        for area in ordered_areas:
            parent_code = area.get("parent_code")
            parent_id = area_ids.get(parent_code) if parent_code else None
            if parent_code is not None and parent_id is None:
                raise InputError(
                    f"Parent area must precede child {area['code']}: {parent_code}"
                )
            cursor = connection.execute(
                """
                INSERT INTO AdministrativeArea(
                    ParentID, Code, Name, Level, AreaType, Selectable,
                    MinLongitude, MaxLongitude, MinLatitude, MaxLatitude,
                    DisplayOrder
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    parent_id,
                    area["code"],
                    area["name"],
                    area["level"],
                    area["area_type"],
                    int(bool(area.get("selectable", False))),
                    area.get("min_longitude"),
                    area.get("max_longitude"),
                    area.get("min_latitude"),
                    area.get("max_latitude"),
                    int(area.get("display_order", 0)),
                ),
            )
            if cursor.lastrowid is None:
                raise DatabaseError("SQLite did not return an administrative area ID")
            area_ids[area["code"]] = cursor.lastrowid

        old_county_to_area: dict[int, int] = {}
        for county_id, name, code, min_x, max_x, min_y, max_y in counties:
            if code in area_ids:
                area_id = area_ids[code]
                connection.execute(
                    """
                    UPDATE AdministrativeArea
                    SET Selectable = 1,
                        MinLongitude = ?, MaxLongitude = ?,
                        MinLatitude = ?, MaxLatitude = ?
                    WHERE ID = ?
                    """,
                    (min_x, max_x, min_y, max_y, area_id),
                )
            else:
                parent_code = code.rsplit("-", 1)[0] if "-" in code else None
                parent_id = area_ids.get(parent_code) if parent_code else None
                cursor = connection.execute(
                    """
                    INSERT INTO AdministrativeArea(
                        ParentID, Code, Name, Level, AreaType, Selectable,
                        MinLongitude, MaxLongitude, MinLatitude, MaxLatitude
                    ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                    """,
                    (
                        parent_id,
                        code,
                        name,
                        code.count("-"),
                        f"subnational{code.count('-')}",
                        min_x,
                        max_x,
                        min_y,
                        max_y,
                    ),
                )
                if cursor.lastrowid is None:
                    raise DatabaseError("SQLite did not return a migrated area ID")
                area_id = cursor.lastrowid
                area_ids[code] = area_id
            old_county_to_area[county_id] = area_id

        for level in metadata.get("levels", []):
            try:
                country_id = area_ids[level["country_code"]]
                connection.execute(
                    """
                    INSERT INTO AdministrativeLevel(
                        CountryAreaID, Level, SingularName, PluralName
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        country_id,
                        level["level"],
                        level["singular_name"],
                        level["plural_name"],
                    ),
                )
            except (KeyError, TypeError) as error:
                raise InputError(
                    f"Invalid administrative level metadata: {level}"
                ) from error

        classes = source.execute("SELECT ID, Name FROM Class ORDER BY ID").fetchall()
        connection.executemany("INSERT INTO Class(ID, Name) VALUES (?, ?)", classes)
        occurrence_rows = source.execute(
            "SELECT CountyID, ClassID, Value FROM Occurrence"
        ).fetchall()
        connection.executemany(
            "INSERT INTO Occurrence(AreaID, ClassID, Value) VALUES (?, ?, ?)",
            (
                (old_county_to_area[county_id], class_id, value)
                for county_id, class_id, value in occurrence_rows
            ),
        )
        connection.commit()

        if (
            len(classes)
            != connection.execute("SELECT COUNT(*) FROM Class").fetchone()[0]
        ):
            raise DatabaseError("Class count changed during occurrence migration")
        if (
            len(occurrence_rows)
            != connection.execute("SELECT COUNT(*) FROM Occurrence").fetchone()[0]
        ):
            raise DatabaseError("Occurrence row count changed during migration")
        problems = target.validate()
        if problems:
            raise DatabaseError(
                "Migrated occurrence database is invalid: " + "; ".join(problems)
            )

        total_area_count = connection.execute(
            "SELECT COUNT(*) FROM AdministrativeArea"
        ).fetchone()[0]
        report = OccurrenceMigrationReport(
            source_area_count=len(counties),
            parent_area_count=total_area_count - len(counties),
            total_area_count=total_area_count,
            class_count=len(classes),
            occurrence_count=len(occurrence_rows),
        )
        target.close()
        target = None
        os.replace(temporary_path, destination_path)
        return report
    except Exception:
        if target is not None:
            target.close()
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        source.close()
