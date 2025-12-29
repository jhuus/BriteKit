#!/usr/bin/env python3

# Defer some imports to improve initialization performance.
import os
import sqlite3
from types import SimpleNamespace
import zlib

from britekit.core.exceptions import DatabaseError


class OccurrenceDatabase:
    """
    SQLite database interface for class occurrence data.

    Attributes:
        db_path: Path to the database file.

    """

    def __init__(self, db_path: str = os.path.join("data", "occurrence.db")):
        self.conn = None
        try:
            self.conn = sqlite3.connect(db_path)
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.cursor = self.conn.cursor()
            self._init_database()
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in OccurrenceDatabase::init: {e}")

    def __enter__(self) -> "OccurrenceDatabase":
        """Allow use in 'with' blocks."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Ensure the connection closes when done."""
        self.close()

    def _migrate_schema(self, current_version: int):
        """Add schema migration code here when schema version 2 is created"""
        return

    def _init_database(self):
        try:
            # Schema version handling
            self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='SchemaVersion'"
            )
            if not self.cursor.fetchone():
                # create the SchemaVersion table
                self.cursor.execute(
                    "CREATE TABLE SchemaVersion (Version INTEGER NOT NULL)"
                )
                self.cursor.execute("INSERT INTO SchemaVersion (Version) VALUES (1)")

                self.cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='Species'"
                )
                if self.cursor.fetchone():
                    # convert HawkEars 1.0 DB to new format
                    self.cursor.execute("ALTER TABLE Species RENAME TO Class")

                    # rename old Occurrence table and copy to new one to create foreign keys
                    self.cursor.execute(
                        "ALTER TABLE Occurrence RENAME TO Occurrence_old"
                    )
                    self._create_occurrence_table()
                    sql = """
                        INSERT INTO Occurrence (CountyID, ClassID, Value)
                        SELECT o.CountyID, o.SpeciesID, o.Value
                        FROM Occurrence_old o
                    """
                    self.cursor.execute(sql)
                    self.cursor.execute("DROP TABLE Occurrence_old")
                    self.cursor.execute("DROP INDEX IF EXISTS idx_county_id")
                    self.cursor.execute("DROP INDEX IF EXISTS idx_species_id")
            else:
                self.cursor.execute("SELECT Version FROM SchemaVersion")
                row = self.cursor.fetchone()
                if row:
                    current_version = row[0]
                    self._migrate_schema(current_version)
                else:
                    raise RuntimeError("SchemaVersion table is empty or corrupt")

            self._create_county_table()
            self._create_class_table()
            self._create_occurrence_table()
            self._create_indexes()
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in OccurrenceDatabase::_init_database: {e}")

    def _create_county_table(self):
        """Create County table if it doesn't exist."""
        query = """
            CREATE TABLE IF NOT EXISTS County (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Name TEXT NOT NULL,
                Code TEXT NOT NULL,
                MinX REAL NOT NULL,
                MaxX REAL NOT NULL,
                MinY REAL NOT NULL,
                MaxY REAL NOT NULL
                )
        """
        self.cursor.execute(query)

    def _create_class_table(self):
        """Create Class table if it doesn't exist."""
        query = """
            CREATE TABLE IF NOT EXISTS Class (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Name TEXT NOT NULL
                )
        """
        self.cursor.execute(query)

    def _create_occurrence_table(self):
        """Create Occurrence table if it doesn't exist."""
        query = """
            CREATE TABLE IF NOT EXISTS Occurrence (
                CountyID INTEGER NOT NULL,
                ClassID INTEGER NOT NULL,
                Value BLOB NOT NULL,
                FOREIGN KEY (CountyID) REFERENCES County(ID) ON DELETE CASCADE,
                FOREIGN KEY (ClassID) REFERENCES Class(ID) ON DELETE CASCADE
                )
        """
        self.cursor.execute(query)

    def _create_indexes(self):
        """Create indexes for efficiency."""
        query = "CREATE UNIQUE INDEX IF NOT EXISTS idx_class_name ON Class (Name)"
        self.cursor.execute(query)

        query = "CREATE INDEX IF NOT EXISTS idx_county_id ON Occurrence (CountyID)"
        self.cursor.execute(query)

        query = "CREATE INDEX IF NOT EXISTS idx_class_id ON Occurrence (ClassID)"
        self.cursor.execute(query)

        self.conn.commit()

    def close(self):
        """Close the database."""
        try:
            self.conn.close()
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in OccurrenceDatabase::close: {e}")

    def get_all_counties(self):
        """Return a list of all counties."""
        try:
            query = """
                SELECT ID, Name, Code, MinX, MaxX, MinY, MaxY FROM County ORDER BY ID
            """
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            results = []
            for row in rows:
                id, name, code, min_x, max_x, min_y, max_y = row
                county = SimpleNamespace(
                    id=id,
                    name=name,
                    code=code,
                    min_x=min_x,
                    max_x=max_x,
                    min_y=min_y,
                    max_y=max_y,
                )
                results.append(county)

            return results
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in OccurrenceDatabase::get_all_counties: {e}")

    def get_all_classes(self):
        """Return a list of all classes."""
        try:
            query = """
                SELECT ID, Name FROM Class ORDER BY ID
            """
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            results = []
            for row in rows:
                id, name = row
                _class = SimpleNamespace(id=id, name=name)
                results.append(_class)

            return results
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in OccurrenceDatabase::get_all_classes: {e}")

    def get_all_occurrences(self):
        """
        Return a list with the CountyID and ClassID for every defined occurrence.
        """
        try:
            query = """
                SELECT CountyID, ClassID FROM Occurrence
            """
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            results = []
            for row in rows:
                county_id, class_id = row
                result = SimpleNamespace(county_id=county_id, class_id=class_id)
                results.append(result)

            return results
        except sqlite3.Error as e:
            raise DatabaseError(
                f"Error in OccurrenceDatabase::get_all_occurrences: {e}"
            )

    def get_occurrences(self, county_id, class_name):
        """
        Return the occurrence values for a given county ID and class name.
        """
        import numpy as np

        try:
            query = """
                SELECT o.ClassID, o.Value FROM Occurrence o JOIN Class s on o.ClassID = s.ID WHERE CountyID = ? AND s.Name = ?
            """
            self.cursor.execute(query, [county_id, class_name])
            result = self.cursor.fetchone()

            if result is None:
                return []

            class_id, compressed = result
            bytes = zlib.decompress(compressed)
            values = np.frombuffer(bytes, dtype=np.float16)
            values = values.astype(np.float32)
            return values
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in OccurrenceDatabase::get_occurrences: {e}")

    def insert_county(self, name, code, min_x, max_x, min_y, max_y):
        """Insert a county record and return the ID."""
        try:
            query = """
                INSERT INTO County (Name, Code, MinX, MaxX, MinY, MaxY) Values (?, ?, ?, ?, ?, ?)
            """
            self.cursor.execute(query, (name, code, min_x, max_x, min_y, max_y))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in OccurrenceDatabase::insert_county: {e}")

    def insert_occurrences(self, county_id, class_id, value):
        """Insert an occurrence record for a given county and class."""
        import numpy as np

        try:
            # value is a numpy array of 48 floats (occurrence per week, four weeks/month);
            # convert it to a float16 array and zip that to keep the database small
            reduced = value.astype(np.float16)
            bytes = reduced.tobytes()
            compressed = zlib.compress(bytes)

            query = """
                INSERT INTO Occurrence (CountyID, ClassID, Value) Values (?, ?, ?)
            """
            self.cursor.execute(query, (county_id, class_id, compressed))
            self.conn.commit()
            return True

        except sqlite3.Error as e:
            raise DatabaseError(f"Error in OccurrenceDatabase::insert_occurrences: {e}")

    def insert_class(self, name):
        """Insert a class record and return the ID."""
        try:
            query = """
                INSERT INTO Class (Name) Values (?)
            """
            self.cursor.execute(query, (name,))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in OccurrenceDatabase::insert_class: {e}")

    def delete_county(self, id):
        """Delete a county record specified by ID."""
        try:
            query = """
                DELETE FROM County WHERE ID = ?
            """
            self.cursor.execute(query, (id,))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in OccurrenceDatabase::delete_county: {e}")

    def delete_class(self, id):
        """Delete a class record specified by ID."""
        try:
            query = """
                DELETE FROM Class WHERE ID = ?
            """
            self.cursor.execute(query, (id,))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in OccurrenceDatabase::delete_class: {e}")
