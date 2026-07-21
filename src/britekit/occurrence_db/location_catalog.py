"""Build and read lightweight administrative-area catalogs."""

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import tempfile

from britekit.core.exceptions import DatabaseError
from britekit.occurrence_db.occurrence_database_v2 import OccurrenceDatabaseV2

CATALOG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LocationCatalogReport:
    """Summary of a compiled location catalog."""

    region_code: str
    area_count: int
    selectable_area_count: int
    level_count: int


def compile_location_catalog(
    input_path: str | Path, output_path: str | Path
) -> LocationCatalogReport:
    """Compile picker metadata from an occurrence-v2 database.

    The destination is created atomically and must not already exist. Species,
    occurrence values, and boundary geometry are deliberately omitted.
    """
    source_path = Path(input_path)
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"Location catalog already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with OccurrenceDatabaseV2(source_path) as source:
        problems = source.validate()
        if problems:
            raise DatabaseError(
                "Invalid source occurrence database: " + "; ".join(problems)
            )

        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

        try:
            report = _write_catalog(source.conn, temporary_path)
            temporary_path.replace(destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    return report


def _write_catalog(
    source: sqlite3.Connection, destination: Path
) -> LocationCatalogReport:
    connection = sqlite3.connect(destination)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            BEGIN;

            CREATE TABLE CatalogMetadata (
                ID INTEGER PRIMARY KEY CHECK (ID = 1),
                SchemaVersion INTEGER NOT NULL CHECK (SchemaVersion = 1),
                RegionCode TEXT NOT NULL,
                RegionName TEXT NOT NULL,
                DataVersion TEXT NOT NULL,
                TaxonomyVersion TEXT,
                SourceCreatedAt TEXT NOT NULL,
                GeneratedAt TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            );

            CREATE TABLE AdministrativeArea (
                ID INTEGER PRIMARY KEY,
                ParentID INTEGER REFERENCES AdministrativeArea(ID),
                Code TEXT NOT NULL UNIQUE,
                Name TEXT NOT NULL,
                Level INTEGER NOT NULL CHECK (Level >= 0),
                AreaType TEXT NOT NULL,
                Selectable INTEGER NOT NULL CHECK (Selectable IN (0, 1)),
                MinLongitude REAL,
                MaxLongitude REAL,
                MinLatitude REAL,
                MaxLatitude REAL,
                DisplayOrder INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX AdministrativeArea_ParentID_idx
                ON AdministrativeArea(ParentID);

            CREATE TABLE AdministrativeLevel (
                CountryAreaID INTEGER NOT NULL
                    REFERENCES AdministrativeArea(ID) ON DELETE CASCADE,
                Level INTEGER NOT NULL CHECK (Level > 0),
                SingularName TEXT NOT NULL,
                PluralName TEXT NOT NULL,
                PRIMARY KEY (CountryAreaID, Level)
            );

            COMMIT;
            """
        )

        pack = source.execute("SELECT * FROM RegionPack WHERE ID = 1").fetchone()
        if pack is None:
            raise DatabaseError("Source occurrence database has no region pack")
        connection.execute(
            """
            INSERT INTO CatalogMetadata(
                ID, SchemaVersion, RegionCode, RegionName, DataVersion,
                TaxonomyVersion, SourceCreatedAt
            ) VALUES (1, ?, ?, ?, ?, ?, ?)
            """,
            (
                CATALOG_SCHEMA_VERSION,
                pack["Code"],
                pack["Name"],
                pack["DataVersion"],
                pack["TaxonomyVersion"],
                pack["CreatedAt"],
            ),
        )

        areas = source.execute(
            """
            SELECT ID, ParentID, Code, Name, Level, AreaType, Selectable,
                   MinLongitude, MaxLongitude, MinLatitude, MaxLatitude,
                   DisplayOrder
            FROM AdministrativeArea
            ORDER BY Level, DisplayOrder, Name
            """
        ).fetchall()
        connection.executemany(
            """
            INSERT INTO AdministrativeArea(
                ID, ParentID, Code, Name, Level, AreaType, Selectable,
                MinLongitude, MaxLongitude, MinLatitude, MaxLatitude,
                DisplayOrder
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [tuple(row) for row in areas],
        )

        levels = source.execute(
            """
            SELECT CountryAreaID, Level, SingularName, PluralName
            FROM AdministrativeLevel
            ORDER BY CountryAreaID, Level
            """
        ).fetchall()
        connection.executemany(
            """
            INSERT INTO AdministrativeLevel(
                CountryAreaID, Level, SingularName, PluralName
            ) VALUES (?, ?, ?, ?)
            """,
            [tuple(row) for row in levels],
        )
        connection.commit()

        problems = list(connection.execute("PRAGMA foreign_key_check"))
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if problems or integrity != "ok":
            raise DatabaseError(
                f"Invalid compiled location catalog: foreign keys={problems}, "
                f"integrity={integrity}"
            )

        selectable_count = sum(row["Selectable"] for row in areas)
        return LocationCatalogReport(
            region_code=pack["Code"],
            area_count=len(areas),
            selectable_area_count=selectable_count,
            level_count=len(levels),
        )
    finally:
        connection.close()
