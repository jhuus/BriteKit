"""Normalized occurrence database with a variable-depth area hierarchy."""

from pathlib import Path
import sqlite3
from types import SimpleNamespace
from typing import Optional
import zlib

from britekit.core.exceptions import DatabaseError

SCHEMA_VERSION = 2


class OccurrenceDatabaseV2:
    """SQLite interface for occurrence schema version 2.

    This class intentionally lives beside the v1 ``OccurrenceDatabase`` while
    the compiled occurrence format and inference provider are transitioned.
    """

    def __init__(self, db_path: str | Path, *, create: bool = False):
        self.path = Path(db_path)
        if create and self.path.exists():
            raise FileExistsError(f"Occurrence database already exists: {self.path}")
        if not create and not self.path.is_file():
            raise FileNotFoundError(f"Occurrence database not found: {self.path}")

        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        try:
            if create:
                self._create_schema()
            self._validate_schema_version()
        except Exception:
            self.conn.close()
            raise

    def __enter__(self) -> "OccurrenceDatabaseV2":
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore[no-untyped-def]
        self.close()

    def close(self) -> None:
        self.conn.close()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            BEGIN;

            CREATE TABLE SchemaVersion (
                Version INTEGER NOT NULL CHECK (Version = 2)
            );
            INSERT INTO SchemaVersion(Version) VALUES (2);

            CREATE TABLE RegionPack (
                ID INTEGER PRIMARY KEY CHECK (ID = 1),
                Code TEXT NOT NULL CHECK (length(trim(Code)) > 0),
                Name TEXT NOT NULL CHECK (length(trim(Name)) > 0),
                DataVersion TEXT NOT NULL CHECK (length(trim(DataVersion)) > 0),
                TaxonomyVersion TEXT,
                CreatedAt TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            );

            CREATE TABLE AdministrativeArea (
                ID INTEGER PRIMARY KEY,
                ParentID INTEGER REFERENCES AdministrativeArea(ID),
                Code TEXT NOT NULL UNIQUE CHECK (length(trim(Code)) > 0),
                Name TEXT NOT NULL CHECK (length(trim(Name)) > 0),
                Level INTEGER NOT NULL CHECK (Level >= 0),
                AreaType TEXT NOT NULL CHECK (length(trim(AreaType)) > 0),
                Selectable INTEGER NOT NULL DEFAULT 0
                    CHECK (Selectable IN (0, 1)),
                MinLongitude REAL,
                MaxLongitude REAL,
                MinLatitude REAL,
                MaxLatitude REAL,
                DisplayOrder INTEGER NOT NULL DEFAULT 0,
                CHECK (
                    (
                        MinLongitude IS NULL
                        AND MaxLongitude IS NULL
                        AND MinLatitude IS NULL
                        AND MaxLatitude IS NULL
                    )
                    OR
                    (
                        MinLongitude IS NOT NULL
                        AND MaxLongitude IS NOT NULL
                        AND MinLatitude IS NOT NULL
                        AND MaxLatitude IS NOT NULL
                        AND MinLongitude <= MaxLongitude
                        AND MinLatitude <= MaxLatitude
                    )
                )
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

            CREATE TABLE Class (
                ID INTEGER PRIMARY KEY,
                Name TEXT NOT NULL UNIQUE CHECK (length(trim(Name)) > 0)
            );

            CREATE TABLE Occurrence (
                AreaID INTEGER NOT NULL
                    REFERENCES AdministrativeArea(ID) ON DELETE CASCADE,
                ClassID INTEGER NOT NULL
                    REFERENCES Class(ID) ON DELETE CASCADE,
                Value BLOB NOT NULL,
                PRIMARY KEY (AreaID, ClassID)
            );

            CREATE INDEX Occurrence_ClassID_idx ON Occurrence(ClassID);

            CREATE TABLE AdministrativeBoundary (
                AreaID INTEGER PRIMARY KEY
                    REFERENCES AdministrativeArea(ID) ON DELETE CASCADE,
                GeometryFormat TEXT NOT NULL,
                Geometry BLOB NOT NULL
            );

            COMMIT;
            """
        )

    def _validate_schema_version(self) -> None:
        try:
            row = self.conn.execute("SELECT Version FROM SchemaVersion").fetchone()
        except sqlite3.Error as error:
            raise DatabaseError(
                "Not an occurrence database with schema version metadata"
            ) from error
        if row is None or row["Version"] != SCHEMA_VERSION:
            actual = None if row is None else row["Version"]
            raise DatabaseError(
                f"Occurrence schema version {actual} is not supported by "
                f"OccurrenceDatabaseV2 (expected {SCHEMA_VERSION})"
            )

    def set_region_pack(
        self,
        code: str,
        name: str,
        data_version: str,
        taxonomy_version: Optional[str] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO RegionPack(
                ID, Code, Name, DataVersion, TaxonomyVersion
            ) VALUES (1, ?, ?, ?, ?)
            """,
            (code, name, data_version, taxonomy_version),
        )
        self.conn.commit()

    def insert_area(
        self,
        *,
        code: str,
        name: str,
        level: int,
        area_type: str,
        selectable: bool,
        parent_code: Optional[str] = None,
        min_longitude: Optional[float] = None,
        max_longitude: Optional[float] = None,
        min_latitude: Optional[float] = None,
        max_latitude: Optional[float] = None,
        display_order: int = 0,
    ) -> int:
        parent_id = None
        if parent_code is not None:
            parent = self.conn.execute(
                "SELECT ID FROM AdministrativeArea WHERE Code = ?", (parent_code,)
            ).fetchone()
            if parent is None:
                raise ValueError(f"Parent area does not exist: {parent_code}")
            parent_id = parent["ID"]

        try:
            cursor = self.conn.execute(
                """
                INSERT INTO AdministrativeArea(
                    ParentID, Code, Name, Level, AreaType, Selectable,
                    MinLongitude, MaxLongitude, MinLatitude, MaxLatitude,
                    DisplayOrder
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    parent_id,
                    code,
                    name,
                    level,
                    area_type,
                    int(selectable),
                    min_longitude,
                    max_longitude,
                    min_latitude,
                    max_latitude,
                    display_order,
                ),
            )
            self.conn.commit()
        except sqlite3.Error as error:
            raise DatabaseError(
                f"Could not insert administrative area: {error}"
            ) from error
        if cursor.lastrowid is None:
            raise DatabaseError("SQLite did not return the administrative area ID")
        return cursor.lastrowid

    def insert_level(
        self,
        country_code: str,
        level: int,
        singular_name: str,
        plural_name: str,
    ) -> None:
        country = self.get_area(country_code)
        self.conn.execute(
            """
            INSERT INTO AdministrativeLevel(
                CountryAreaID, Level, SingularName, PluralName
            ) VALUES (?, ?, ?, ?)
            """,
            (country.id, level, singular_name, plural_name),
        )
        self.conn.commit()

    def insert_class(self, name: str, *, class_id: Optional[int] = None) -> int:
        cursor = self.conn.execute(
            "INSERT INTO Class(ID, Name) VALUES (?, ?)", (class_id, name)
        )
        self.conn.commit()
        if cursor.lastrowid is None:
            raise DatabaseError("SQLite did not return the class ID")
        return cursor.lastrowid

    def insert_occurrences(self, area_id: int, class_id: int, value) -> None:  # type: ignore[no-untyped-def]
        reduced = value.astype("float16")
        compressed = zlib.compress(reduced.tobytes())
        self.insert_compressed_occurrences(area_id, class_id, compressed)

    def insert_compressed_occurrences(
        self, area_id: int, class_id: int, value: bytes
    ) -> None:
        self.conn.execute(
            "INSERT INTO Occurrence(AreaID, ClassID, Value) VALUES (?, ?, ?)",
            (area_id, class_id, value),
        )
        self.conn.commit()

    def get_area(self, code: str):  # type: ignore[no-untyped-def]
        row = self.conn.execute(
            """
            SELECT area.*, parent.Code AS ParentCode
            FROM AdministrativeArea area
            LEFT JOIN AdministrativeArea parent ON parent.ID = area.ParentID
            WHERE area.Code = ?
            """,
            (code,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Administrative area not found: {code}")
        return self._area_from_row(row)

    def get_all_areas(self):  # type: ignore[no-untyped-def]
        rows = self.conn.execute(
            """
            SELECT area.*, parent.Code AS ParentCode
            FROM AdministrativeArea area
            LEFT JOIN AdministrativeArea parent ON parent.ID = area.ParentID
            ORDER BY area.Level, area.DisplayOrder, area.Name
            """
        ).fetchall()
        return [self._area_from_row(row) for row in rows]

    def get_children(self, parent_code: str):  # type: ignore[no-untyped-def]
        parent = self.get_area(parent_code)
        rows = self.conn.execute(
            """
            SELECT child.*, parent.Code AS ParentCode
            FROM AdministrativeArea child
            JOIN AdministrativeArea parent ON parent.ID = child.ParentID
            WHERE child.ParentID = ?
            ORDER BY child.DisplayOrder, child.Name
            """,
            (parent.id,),
        ).fetchall()
        return [self._area_from_row(row) for row in rows]

    def get_descendants(self, code: str):  # type: ignore[no-untyped-def]
        area = self.get_area(code)
        rows = self.conn.execute(
            """
            WITH RECURSIVE descendants(ID) AS (
                SELECT ID FROM AdministrativeArea WHERE ParentID = ?
                UNION ALL
                SELECT child.ID
                FROM AdministrativeArea child
                JOIN descendants parent ON child.ParentID = parent.ID
            )
            SELECT area.*, parent.Code AS ParentCode
            FROM AdministrativeArea area
            LEFT JOIN AdministrativeArea parent ON parent.ID = area.ParentID
            WHERE area.ID IN descendants
            ORDER BY area.Level, area.DisplayOrder, area.Name
            """,
            (area.id,),
        ).fetchall()
        return [self._area_from_row(row) for row in rows]

    def get_occurrences(self, area_id: int, class_name: str):  # type: ignore[no-untyped-def]
        import numpy as np

        row = self.conn.execute(
            """
            SELECT occurrence.Value
            FROM Occurrence occurrence
            JOIN Class class ON class.ID = occurrence.ClassID
            WHERE occurrence.AreaID = ? AND class.Name = ?
            """,
            (area_id, class_name),
        ).fetchone()
        if row is None:
            return []
        values = np.frombuffer(zlib.decompress(row["Value"]), dtype=np.float16)
        return values.astype(np.float32)

    def validate(self) -> list[str]:
        """Return integrity problems; an empty list means the database is valid."""
        problems = [
            f"foreign key error: {tuple(row)}"
            for row in self.conn.execute("PRAGMA foreign_key_check")
        ]
        pack_count = self.conn.execute("SELECT COUNT(*) FROM RegionPack").fetchone()[0]
        if pack_count != 1:
            problems.append(f"expected one RegionPack row, found {pack_count}")
        roots = self.conn.execute(
            "SELECT COUNT(*) FROM AdministrativeArea WHERE ParentID IS NULL"
        ).fetchone()[0]
        if roots == 0:
            problems.append("no root administrative areas")
        bad_levels = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM AdministrativeArea child
            JOIN AdministrativeArea parent ON parent.ID = child.ParentID
            WHERE child.Level <= parent.Level
            """
        ).fetchone()[0]
        if bad_levels:
            problems.append(f"{bad_levels} child areas do not have a greater level")
        return problems

    @staticmethod
    def _area_from_row(row: sqlite3.Row):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            id=row["ID"],
            parent_id=row["ParentID"],
            parent_code=row["ParentCode"],
            code=row["Code"],
            name=row["Name"],
            level=row["Level"],
            area_type=row["AreaType"],
            selectable=bool(row["Selectable"]),
            min_longitude=row["MinLongitude"],
            max_longitude=row["MaxLongitude"],
            min_latitude=row["MinLatitude"],
            max_latitude=row["MaxLatitude"],
            display_order=row["DisplayOrder"],
        )
