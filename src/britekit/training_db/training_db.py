from datetime import datetime
import logging
import os
import sqlite3
from types import SimpleNamespace
from typing import Optional

from britekit.core.exceptions import DatabaseError


class TrainingDatabase:
    """
    Handle the creation, querying, and updating of a simple SQLite database storing
    training data, including a class table, recording table and spectrogram tables.

    Attributes:
        db_path: Path to the database file.
    """

    def __init__(self, db_path: str = os.path.join("data", "training.db")):
        self.today = datetime.today().strftime("%Y-%m-%d")
        try:
            self.conn = sqlite3.connect(db_path)
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.cursor = self.conn.cursor()
            self._init_database()

            # allows callers to specify filter names such as CategoryName in the Class table,
            # and trigger corresponding joins
            self.class_relationships = {
                "CategoryName": {
                    "table": "Category",
                    "local_key": "CategoryID",
                    "remote_key": "ID",
                    "remote_field": "Name",
                }
            }

            self.recording_relationships = {
                "SourceName": {
                    "table": "Source",
                    "local_key": "SourceID",
                    "remote_key": "ID",
                    "remote_field": "Name",
                },
            }

            self.segment_relationships = {
                "FileName": {
                    "table": "Recording",
                    "local_key": "RecordingID",
                    "remote_key": "ID",
                    "remote_field": "FileName",
                },
            }

            self.segment_class_relationships = {
                "ClassName": {
                    "table": "Class",
                    "local_key": "ClassID",
                    "remote_key": "ID",
                    "remote_field": "Name",
                },
                "SoundType": {
                    "table": "SoundType",
                    "local_key": "SoundTypeID",
                    "remote_key": "ID",
                    "remote_field": "Name",
                },
            }

            self.specvalue_relationships = {
                "RecordingID": {
                    "table": "Segment",
                    "local_key": "SegmentID",
                    "remote_key": "ID",
                    "remote_field": "RecordingID",
                },
            }

        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::init: {e}")

    def __enter__(self) -> "TrainingDatabase":
        """Allow use in 'with' blocks."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Ensure the connection closes when done."""
        self.close()

    def _migrate_schema(self, current_version: int):
        """Add schema migration code here when schema version 2 is created"""
        return

    def _create_category_table(self):
        """Record per category, e.g. birds or mammals"""
        sql = """
            CREATE TABLE IF NOT EXISTS Category (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Name TEXT NOT NULL UNIQUE,
                InsertedDate TEXT)
        """
        self.cursor.execute(sql)

    def _create_class_table(self):
        """Record per class, e.g. a species"""
        sql = """
            CREATE TABLE IF NOT EXISTS Class (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                CategoryID INTEGER,
                Name TEXT NOT NULL UNIQUE,
                Code TEXT,
                AltName TEXT,
                AltCode TEXT,
                InsertedDate TEXT,
                FOREIGN KEY (CategoryID) REFERENCES Category(ID) ON DELETE CASCADE)
        """
        self.cursor.execute(sql)

    def _create_recording_table(self):
        """Record per recording"""
        sql = """
            CREATE TABLE IF NOT EXISTS Recording (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                SourceID INTEGER,
                FileName TEXT NOT NULL,
                Path TEXT,
                Seconds INTEGER,
                InsertedDate TEXT,
                FOREIGN KEY (SourceID) REFERENCES Source(ID) ON DELETE CASCADE)
        """
        self.cursor.execute(sql)

    def _create_segment_table(self):
        """
        Record per segment extracted from recordings, including a link to the recording
        and offset within it. The Audio field is rarely used, but is a way of storing
        the audio in the database if needed. SamplingRate refers to the Audio field.
        """
        sql = """
            CREATE TABLE IF NOT EXISTS Segment (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                RecordingID INTEGER,
                Offset REAL NOT NULL,
                Audio BLOB,
                SamplingRate INTEGER,
                InsertedDate TEXT,
                FOREIGN KEY (RecordingID) REFERENCES Recording(ID) ON DELETE CASCADE,
                UNIQUE (RecordingID, Offset))
        """
        self.cursor.execute(sql)

    def _create_segment_class_table(self):
        """A segment may have multiple classes, and this table joins them."""
        sql = """
            CREATE TABLE IF NOT EXISTS SegmentClass (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                SoundTypeID INTEGER,
                SegmentID INTEGER,
                ClassID INTEGER,
                FOREIGN KEY (SoundTypeID) REFERENCES SoundType(ID) ON DELETE SET NULL,
                FOREIGN KEY (SegmentID) REFERENCES Segment(ID) ON DELETE CASCADE,
                FOREIGN KEY (ClassID) REFERENCES Class(ID) ON DELETE CASCADE,
                UNIQUE (SegmentID, ClassID))
        """
        self.cursor.execute(sql)

    def _create_soundtype_table(self):
        """Record per sound type, e.g. chip or tink"""
        sql = """
            CREATE TABLE IF NOT EXISTS SoundType (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Name TEXT NOT NULL UNIQUE,
                Description TEXT,
                InsertedDate TEXT)
        """
        self.cursor.execute(sql)

    def _create_source_table(self):
        """Record per data source, e.g. Xeno-Canto"""
        sql = """
            CREATE TABLE IF NOT EXISTS Source (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Name TEXT NOT NULL UNIQUE,
                InsertedDate TEXT)
        """
        self.cursor.execute(sql)

    def _create_spec_group_table(self):
        """Represent different spectrogram parameter groups, e.g. frequency scaling."""
        sql = """
            CREATE TABLE IF NOT EXISTS SpecGroup (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Name TEXT NOT NULL UNIQUE,
                InsertedDate TEXT)
        """
        self.cursor.execute(sql)

    def _create_spec_value_table(self):
        """The actual spectrograms."""
        sql = """
            CREATE TABLE IF NOT EXISTS SpecValue (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Value BLOB NOT NULL,
                Embedding BLOB,
                SpecGroupID INTEGER,
                SegmentID INTEGER,
                FOREIGN KEY (SpecGroupID) REFERENCES SpecGroup(ID) ON DELETE CASCADE,
                FOREIGN KEY (SegmentID) REFERENCES Segment(ID) ON DELETE CASCADE,
                UNIQUE (SpecGroupID, SegmentID))
        """
        self.cursor.execute(sql)

    def _init_database(self):
        try:
            # Schema version handling
            self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='SchemaVersion'"
            )
            if not self.cursor.fetchone():
                # Create the SchemaVersion table
                self.cursor.execute(
                    "CREATE TABLE SchemaVersion (Version INTEGER NOT NULL)"
                )
                self.cursor.execute("INSERT INTO SchemaVersion (Version) VALUES (1)")

                self.cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='Subcategory'"
                )
                if self.cursor.fetchone():
                    # Convert database from HawkEars 1.x to schema version 1 of BriteKit
                    logging.info("")
                    logging.info("Upgrading training database schema from HawkEars 1.0")

                    # settings to improve migration performance
                    self.conn.commit()
                    self.cursor.execute("PRAGMA journal_mode = OFF")
                    self.cursor.execute("PRAGMA synchronous = OFF")
                    self.cursor.execute("PRAGMA locking_mode = EXCLUSIVE")
                    self.cursor.execute("PRAGMA temp_store = MEMORY")
                    self.cursor.execute("BEGIN TRANSACTION")

                    # copy Subcategory to new Class table so foreign keys are defined with cascading delete etc,
                    # but retain ID so ClassID in Recording table is still valid
                    logging.info("Migrate Class table")
                    self._create_class_table()
                    sql = """
                        INSERT INTO Class (ID, CategoryID, Name, Code, AltName)
                        SELECT s.ID, s.CategoryID, s.Name, s.Code, s.Synonym
                        FROM Subcategory s
                    """
                    self.cursor.execute(sql)
                    self.cursor.execute("DROP TABLE Subcategory")
                    self.cursor.execute("DROP INDEX IF EXISTS idx_subcategory")

                    # rename SoundType table, copy to new one, then drop the old one
                    logging.info("Migrate SoundType table")
                    self.cursor.execute("ALTER TABLE SoundType RENAME TO SoundType_old")
                    self._create_soundtype_table()
                    sql = """
                        INSERT INTO SoundType (ID, Name)
                        SELECT st.ID, st.Name
                        FROM SoundType_old st
                    """
                    self.cursor.execute(sql)
                    self.cursor.execute("DROP TABLE SoundType_old")

                    # rename Recording table, copy to new one, then drop the old one
                    # (create a temporary ClassID column, which we drop below)
                    logging.info("Migrate Recording table")
                    self.cursor.execute("ALTER TABLE Recording RENAME TO Recording_old")
                    self._create_recording_table()
                    self.cursor.execute(
                        "ALTER TABLE Recording ADD Column ClassID INTEGER"
                    )
                    sql = """
                        INSERT INTO Recording (ID, SourceID, FileName, Path, Seconds, ClassID)
                        SELECT r.ID, r.SourceID, r.FileName, r.Path, r.Seconds, r.SubcategoryID
                        FROM Recording_old r
                    """
                    self.cursor.execute(sql)
                    self.cursor.execute("DROP TABLE Recording_old")
                    self.cursor.execute("DROP INDEX IF EXISTS idx_recording")

                    # rename Spectrogram table, then copy to new tables
                    self.cursor.execute(
                        "ALTER TABLE Spectrogram RENAME TO Spectrogram_old"
                    )
                    self._create_segment_table()
                    self._create_spec_group_table()
                    self.insert_specgroup("default")
                    self._create_spec_value_table()
                    self.conn.commit()  # do commits to free up disk space

                    logging.info("Migrate Segment table")
                    sql = """
                        INSERT INTO Segment (ID, RecordingID, Offset, Audio, SamplingRate, InsertedDate)
                        SELECT s.ID, s.RecordingID, s.Offset, s.Audio, s.SamplingRate, s.Inserted
                        FROM Spectrogram_old s
                    """
                    self.cursor.execute(sql)
                    self.conn.commit()  # do commits to free up disk space

                    logging.info("Migrate SpecValue table")
                    sql = """
                        INSERT INTO SpecValue (SegmentID, Value, Embedding, SpecGroupID)
                        SELECT s.ID, s.Value, s.Embedding, 1
                        FROM Spectrogram_old s
                    """
                    self.cursor.execute(sql)
                    self.conn.commit()  # do commits to free up disk space

                    # create and populate SegmentClass table
                    logging.info("Migrate SegmentClass table")
                    self._create_segment_class_table()
                    sql = """
                        INSERT INTO SegmentClass (SegmentID, ClassID, SoundTypeID)
                        SELECT s.ID, r.ClassID, s.SoundTypeID
                        FROM Spectrogram_old s
                        JOIN Recording r ON s.RecordingID = r.ID
                        WHERE r.ClassID IS NOT NULL
                    """
                    self.cursor.execute(sql)
                    self.conn.commit()  # do commits to free up disk space

                    logging.info("Finalize migration")

                    self.cursor.execute("DROP TABLE Spectrogram_old")
                    self.cursor.execute("DROP INDEX IF EXISTS idx_spectrogram")
                    self.cursor.execute("DROP INDEX IF EXISTS idx_spectrogram_ignore")
                    self.cursor.execute("DROP INDEX IF EXISTS idx_spectrogram_recid")
                    self.cursor.execute("DROP INDEX IF EXISTS idx_spectrogram_sndid")

                    # ClassID was needed only until we populated SegmentClass
                    self.cursor.execute("ALTER TABLE Recording DROP Column ClassID")

                    # Add missing InsertedDate columns
                    self.cursor.execute(
                        "ALTER TABLE Category ADD Column InsertedDate TEXT"
                    )
                    self.cursor.execute(
                        "ALTER TABLE Source ADD Column InsertedDate TEXT"
                    )

                self.conn.commit()
            else:
                self.cursor.execute("SELECT Version FROM SchemaVersion")
                row = self.cursor.fetchone()
                if row:
                    current_version = row[0]
                    self._migrate_schema(current_version)
                else:
                    raise RuntimeError("SchemaVersion table is empty or corrupt")

            self._create_source_table()
            self._create_category_table()
            self._create_class_table()
            self._create_recording_table()
            self._create_segment_table()
            self._create_segment_class_table()
            self._create_soundtype_table()
            self._create_spec_group_table()
            self._create_spec_value_table()

            # Insert default category and source records if none found
            self.cursor.execute("SELECT ID FROM Category WHERE Name = 'default'")
            row = self.cursor.fetchone()
            if not row:
                self.insert_category("default")

            self.cursor.execute("SELECT ID FROM Source WHERE Name = 'default'")
            row = self.cursor.fetchone()
            if not row:
                self.insert_source("default")

            self.cursor.execute("SELECT ID FROM SpecGroup WHERE Name = 'default'")
            row = self.cursor.fetchone()
            if not row:
                self.insert_specgroup("default")

            # Create indexes for efficiency
            sql = (
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_category_name ON Category (Name)"
            )
            self.cursor.execute(sql)

            sql = "CREATE UNIQUE INDEX IF NOT EXISTS idx_class_name ON Class (Name)"
            self.cursor.execute(sql)
            sql = "CREATE UNIQUE INDEX IF NOT EXISTS idx_class_code ON Class (Code)"

            sql = (
                "CREATE INDEX IF NOT EXISTS idx_recording_name ON Recording (FileName)"
            )
            self.cursor.execute(sql)

            sql = "CREATE INDEX IF NOT EXISTS idx_recording_path ON Recording (Path)"
            self.cursor.execute(sql)

            sql = (
                "CREATE INDEX IF NOT EXISTS idx_segment_recid ON Segment (RecordingID)"
            )
            self.cursor.execute(sql)

            sql = "CREATE INDEX IF NOT EXISTS idx_segment_class_sndid ON SegmentClass (SoundTypeID)"
            self.cursor.execute(sql)

            sql = "CREATE INDEX IF NOT EXISTS idx_segment_class_segid ON SegmentClass (SegmentID)"
            self.cursor.execute(sql)

            sql = "CREATE INDEX IF NOT EXISTS idx_segment_class_classid ON SegmentClass (ClassID)"
            self.cursor.execute(sql)

            sql = "CREATE UNIQUE INDEX IF NOT EXISTS idx_specgroup_name ON SpecGroup (Name)"
            self.cursor.execute(sql)

            sql = "CREATE INDEX IF NOT EXISTS idx_specvalue_groupid ON SpecValue (SpecGroupID)"
            self.cursor.execute(sql)

            sql = "CREATE INDEX IF NOT EXISTS idx_specvalue_segmentid ON SpecValue (SegmentID)"
            self.cursor.execute(sql)

            sql = "CREATE UNIQUE INDEX IF NOT EXISTS idx_soundtype_name ON SoundType (Name)"
            self.cursor.execute(sql)

            sql = "CREATE UNIQUE INDEX IF NOT EXISTS idx_source_name ON Source (Name)"
            self.cursor.execute(sql)

            self.conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::_init_database: {e}")

    def close(self):
        """
        Close the database.
        """
        try:
            self.conn.close()
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::close: {e}")

    @staticmethod
    def _parse_filters(
        table_name: str,
        filters: Optional[dict] = None,
        relationships: Optional[dict] = None,
    ):
        join_clauses = []
        where_clauses = []
        params = []
        if filters:
            for column, value in filters.items():
                if relationships is not None and column in relationships:
                    rel = relationships[column]
                    join_clauses.append(
                        f"JOIN {rel['table']} ON {rel['local_key']} = {rel['table']}.{rel['remote_key']}"
                    )
                    where_clauses.append(f"{rel['table']}.{rel['remote_field']} = ?")
                    params.append(value)
                else:
                    where_clauses.append(f"{table_name}.{column} = ?")
                    params.append(value)

        return join_clauses, where_clauses, params

    def _get_count(
        self,
        table_name: str,
        filters: Optional[dict] = None,
        relationships: Optional[dict] = None,
    ):
        sql = f"SELECT COUNT(*) FROM {table_name}"
        join_clauses, where_clauses, params = self._parse_filters(
            table_name, filters, relationships
        )
        if join_clauses:
            sql += " " + " ".join(join_clauses)

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        self.cursor.execute(sql, params)
        result = self.cursor.fetchone()
        if result is not None:
            return result[0]

        return None

    # delete records and return the number deleted
    def _delete_records(
        self,
        table_name: str,
        filters: Optional[dict] = None,
        relationships: Optional[dict] = None,
    ):
        # Step 1: Build SELECT to find matching IDs
        sql_select = f"SELECT {table_name}.id FROM {table_name}"
        join_clauses = []
        where_clauses = []
        params = []

        if filters is not None:
            for column, value in filters.items():
                if relationships is not None and column in relationships:
                    rel = relationships[column]
                    join_clauses.append(
                        f"JOIN {rel['table']} ON {table_name}.{rel['local_key']} = {rel['table']}.{rel['remote_key']}"
                    )
                    where_clauses.append(f"{rel['table']}.{rel['remote_field']} = ?")
                    params.append(value)
                else:
                    where_clauses.append(f"{table_name}.{column} = ?")
                    params.append(value)

        if join_clauses:
            sql_select += " " + " ".join(join_clauses)

        if where_clauses:
            sql_select += " WHERE " + " AND ".join(where_clauses)

        # Execute the SELECT
        cur = self.cursor.execute(sql_select, params)
        ids = [row[0] for row in cur.fetchall()]

        if not ids:
            return 0

        # Step 2: DELETE based on IDs
        placeholders = ",".join("?" for _ in ids)
        sql_delete = f"DELETE FROM {table_name} WHERE id IN ({placeholders})"
        self.cursor.execute(sql_delete, ids)
        self.conn.commit()

        return len(ids)

    def optimize(self):
        """
        Optimize database performance (important after extract or re-extract)
        """
        self.cursor.execute("ANALYZE")
        self.cursor.execute("PRAGMA optimize")
        self.conn.commit()

    # ------------------------------- #
    # Source
    # ------------------------------- #

    def insert_source(self, name: str):
        """
        Insert a Source record.

        Args:
        - name (str): Name of the source (e.g. "Xeno-Canto").

        Returns:
            row_id (int): ID of the inserted record.
        """
        try:
            sql = "INSERT INTO Source (Name, InsertedDate) Values (?, ?)"
            self.cursor.execute(sql, (name, self.today))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::insert_source: {e}")

    def delete_source(self, filters: Optional[dict] = None):
        """
        Delete one or more Source records.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters. Valid
        column names for the Source table are:
            - ID (int): record ID
            - Name (str): source name

        Returns:
            Number of records deleted.
        """
        try:
            return self._delete_records("Source", filters)
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::delete_source: {e}")

    def get_source(self, filters: Optional[dict] = None):
        """
        Query the Source table.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters. Valid
        column names for the Source table are:
            - ID (int): record ID
            - Name (str): source name

        Returns:
            A list of entries, each as a SimpleNamespace object with the following attributes:
            - id (int): Unique ID of the entry.
            - name (str): Name of the source.
        """
        try:
            sql = "SELECT ID, Name FROM Source"
            _, where_clauses, params = self._parse_filters("Source", filters)
            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

            sql += " ORDER BY ID"

            self.cursor.execute(sql, params)
            rows = self.cursor.fetchall()
            if rows is None:
                return []

            results = []
            for row in rows:
                id, name = row
                result = SimpleNamespace(id=id, name=name)
                results.append(result)

            return results

        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::get_source: {e}")

    def get_source_count(self, filters: Optional[dict] = None):
        """
        Get the number of records in the Source table.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters. Valid
        column names for the Source table are:
            - ID (int): record ID
            - Name (str): source name

        Returns:
            Number of records that match the criteria.
        """
        try:
            return self._get_count("Source", filters)
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::get_source_count: {e}")

    # ------------------------------- #
    # Category
    # ------------------------------- #

    def insert_category(self, name: str):
        """
        Insert a Category record.

        Args:
        - name (str): Name of the category (e.g. "bird").

        Returns:
            row_id (int): ID of the inserted record.
        """
        try:
            sql = "INSERT INTO Category (Name, InsertedDate) Values (?, ?)"
            self.cursor.execute(sql, (name, self.today))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::insert_category: {e}")

    def delete_category(self, filters: Optional[dict] = None):
        """
        Delete one or more Category records.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters. Valid
        column names for the Category table are:
            - ID (int): record ID
            - Name (str): source name

        Returns:
            Number of records deleted.
        """
        try:
            return self._delete_records("Category", filters)
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::delete_category: {e}")

    def get_category(self, filters: Optional[dict] = None):
        """
        Query the Category table.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters. Valid
        column names for the Category table are:
            - ID (int): record ID
            - Name (str): category name

        Returns:
            A list of entries, each as a SimpleNamespace object with the following attributes:
            - id (int): Unique ID of the entry.
            - name (str): Name of the category.
        """
        try:
            sql = "SELECT ID, Name FROM Category"
            _, where_clauses, params = self._parse_filters("Category", filters)
            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

            sql += " ORDER BY ID"

            self.cursor.execute(sql, params)
            rows = self.cursor.fetchall()
            if rows is None:
                return []

            results = []
            for row in rows:
                id, name = row
                result = SimpleNamespace(id=id, name=name)
                results.append(result)

            return results

        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::get_category: {e}")

    def get_category_count(self, filters: Optional[dict] = None):
        """
        Get the number of records in the Category table.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters. Valid
        column names for the Category table are:
            - ID (int): record ID
            - Name (str): category name

        Returns:
            Number of records that match the criteria.
        """
        try:
            return self._get_count("Category", filters)
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::get_category_count: {e}")

    # ------------------------------- #
    # Class
    # ------------------------------- #

    def insert_class(
        self,
        category_id: int,
        name: str,
        alt_name: Optional[str] = None,
        code: Optional[str] = None,
        alt_code: Optional[str] = None,
    ):
        """
        Insert a Class record.

        Args:
        - category_id (int, required): Record ID of the category (e.g. ID of "bird" in the Category table).
        - name (str, required): Name of the class (e.g. "White-winged Crossbill").
        - alt_name (str, optional): Alternate name of the class (e.g. "Two-barred Crossbill").
        - code (str, optional): Code for the class (e.g. "WWCR").
        - alt_code (str, optional): Alternate code

        Returns:
            row_id (int): ID of the inserted record.
        """
        try:
            sql = """
                INSERT INTO Class (CategoryID, Name, AltName, Code, AltCode, InsertedDate) Values (?, ?, ?, ?, ?, ?)
            """
            self.cursor.execute(
                sql, (category_id, name, alt_name, code, alt_code, self.today)
            )
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::insert_class: {e}")

    def delete_class(self, filters: Optional[dict] = None):
        """
        Delete one ore more Class records.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters.

        Returns:
            Number of records deleted.
        """
        try:
            return self._delete_records("Class", filters, self.class_relationships)
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::delete_class: {e}")

    def get_class(self, filters: Optional[dict] = None):
        """
        Query the Class table.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters.

        Returns:
            A list of entries, each as a SimpleNamespace object with the following attributes:
            - id (int): Unique ID of the entry.
            - category_id (int): ID of the corresponding category.
            - name (str): Class name
            - alt_name (str): Class alt_name
            - code (str): Class code
        """
        try:
            sql = "SELECT Class.ID, Class.CategoryID, Class.Name, Class.AltName, Class.Code, Class.AltCode FROM Class"

            join_clauses, where_clauses, params = self._parse_filters(
                "Class", filters, self.class_relationships
            )
            if join_clauses:
                sql += " " + " ".join(join_clauses)

            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

            sql += " ORDER BY Class.Name"

            self.cursor.execute(sql, params)
            rows = self.cursor.fetchall()
            if rows is None:
                return []

            results = []
            for row in rows:
                id, categoryID, name, alt_name, code, alt_code = row
                result = SimpleNamespace(
                    id=id,
                    category_id=categoryID,
                    name=name,
                    code=code,
                    alt_name=alt_name,
                    alt_code=alt_code,
                )
                results.append(result)

            return results

        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::get_class: {e}")

    def get_class_count(self, filters: Optional[dict] = None):
        """
        Get the number of records in the Class table.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters.

        Returns:
            Number of records that match the criteria.
        """
        try:
            return self._get_count("Class", filters)
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::get_class_count: {e}")

    # ------------------------------- #
    # SoundType
    # ------------------------------- #

    def insert_soundtype(self, name: str):
        """
        Insert a SoundType record.

        Args:
        - name (str, required): Name of the sound type.

        Returns:
            row_id (int): ID of the inserted record.
        """
        try:
            sql = "INSERT INTO SoundType (Name, InsertedDate) Values (?, ?)"
            self.cursor.execute(sql, (name, self.today))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::insert_soundtype: {e}")

    def delete_soundtype(self, filters: Optional[dict] = None):
        """
        Delete one or more SoundType records.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters.

        Returns:
            Number of records deleted.
        """
        try:
            return self._delete_records("SoundType", filters)
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::delete_soundtype: {e}")

    def get_soundtype(self, filters: Optional[dict] = None):
        """
        Query the SoundType table.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters.

        Returns:
            A list of entries, each as a SimpleNamespace object with the following attributes:
            - id (int): Unique ID of the entry.
            - source_id (int): ID of the corresponding source.
            - class_id (int): ID of the corresponding class.
            - filename (str): File name
            - path (str): Path
            - seconds (float): Duration in seconds
        """
        try:
            sql = "SELECT ID, Name FROM SoundType"
            _, where_clauses, params = self._parse_filters("SoundType", filters)
            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

            sql += " ORDER BY ID"

            self.cursor.execute(sql, params)
            rows = self.cursor.fetchall()
            if rows is None:
                return []

            results = []
            for row in rows:
                id, name = row
                result = SimpleNamespace(id=id, name=name)
                results.append(result)

            return results

        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::get_soundtype: {e}")

    def get_soundtype_count(self, filters: Optional[dict] = None):
        """
        Get the number of records in the SoundType table.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters.

        Returns:
            Number of records that match the criteria.
        """
        try:
            return self._get_count("SoundType", filters)
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::get_soundtype_count: {e}")

    # ------------------------------- #
    # Recording
    # ------------------------------- #

    def insert_recording(
        self,
        source_id: int,
        filename: str,
        path: str,
        seconds: float = 0,
    ):
        """
        Insert a Recording record.

        Args:
        - source_id (int, required): Record ID of the source (e.g. ID of "Xeno-Canto" in the Source table).
        - filename (str, required): Name of the recording (e.g. "XC12345.mp3").
        - path (str, required): Full path to the recording.
        - seconds (float, optional): Duration of the recording in seconds.

        Returns:
            row_id (int): ID of the inserted record.
        """
        try:
            sql = """
                INSERT INTO Recording (SourceID, FileName, Path, Seconds, InsertedDate)
                Values (?, ?, ?, ?, ?)
            """
            self.cursor.execute(sql, (source_id, filename, path, seconds, self.today))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::insert_recording: {e}")

    def delete_recording(self, filters: Optional[dict] = None):
        """
        Delete one or more Recording records.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters.

        Returns:
            Number of records deleted.
        """
        try:
            return self._delete_records(
                "Recording", filters, self.recording_relationships
            )
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::delete_recording: {e}")

    def get_recording(self, filters: Optional[dict] = None):
        """
        Query the Recording table.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters.

        Returns:
            A list of entries, each as a SimpleNamespace object with the following attributes:
            - id (int): Unique ID of the entry.
            - source_id (int): ID of the corresponding source.
            - class_id (int): ID of the corresponding class.
            - filename (str): File name
            - path (str): Path
            - seconds (float): Duration in seconds
        """
        try:
            sql = """
                SELECT Recording.ID, Recording.SourceID, Recording.FileName, Recording.Path, Recording.Seconds
                FROM Recording
            """

            join_clauses, where_clauses, params = self._parse_filters(
                "Recording", filters, self.recording_relationships
            )
            if join_clauses:
                sql += " " + " ".join(join_clauses)

            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

            sql += " ORDER BY Recording.ID"

            self.cursor.execute(sql, params)
            rows = self.cursor.fetchall()
            if rows is None:
                return []

            results = []
            for row in rows:
                id, sourceID, filename, path, seconds = row
                result = SimpleNamespace(
                    id=id,
                    source_id=sourceID,
                    filename=filename,
                    path=path,
                    seconds=seconds,
                )
                results.append(result)

            return results

        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::get_recording: {e}")

    def get_recording_by_class(self, class_name: str):
        """
        Return all recordings that have segments with the given class.

        Args:
        - class_name (str): name of the class.

        Returns:
            A list of entries, each as a SimpleNamespace object with the following attributes:
            - id (int): Unique ID of the entry.
            - source_id (int): ID of the corresponding source.
            - class_id (int): ID of the corresponding class.
            - filename (str): File name
            - path (str): Path
            - seconds (float): Duration in seconds
        """
        try:
            sql = """
                SELECT DISTINCT r.ID, r.SourceID, r.FileName, r.Path, r.Seconds
                FROM Recording r
                JOIN Segment s ON r.ID = s.RecordingID
                JOIN SegmentClass sc ON s.ID = sc.SegmentID
                JOIN Class c ON sc.ClassID = c.ID
                WHERE c.Name = ?
            """

            self.cursor.execute(sql, [class_name])
            rows = self.cursor.fetchall()
            if rows is None:
                return []

            results = []
            for row in rows:
                id, sourceID, filename, path, seconds = row
                result = SimpleNamespace(
                    id=id,
                    source_id=sourceID,
                    filename=filename,
                    path=path,
                    seconds=seconds,
                )
                results.append(result)

            return results

        except sqlite3.Error as e:
            raise DatabaseError(
                f"Error in TrainingDatabase::get_recording_by_class: {e}"
            )

    def update_recording(self, id: int, field: str, value):
        """
        Update a record in the Recording table.

        Args:
        - id (int): ID that identifies the record to update
        - field (str): Name of column to update.
        - value: New value.
        """
        try:
            sql = f"""
                UPDATE Recording SET {field} = ? WHERE ID = ?
            """
            self.cursor.execute(sql, (value, id))
            self.conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::update_recording: {e}")

    def get_recording_count(self, filters: Optional[dict] = None):
        """
        Get the number of records in the Recording table.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters.

        Returns:
            Number of records that match the criteria.
        """
        try:
            return self._get_count("Recording", filters, self.recording_relationships)
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::get_recording_count: {e}")

    # ------------------------------- #
    # Segment
    # ------------------------------- #

    def insert_segment(
        self,
        recording_id: int,
        offset: float,
    ):
        """
        Insert a Segment record.

        Args:
        - recording_id (int, required): Record ID of the recording.
        - offset (float, required): offset in seconds from start of the recording.
        - audio (blob, optional): corresponding raw audio.

        Returns:
            row_id (int): ID of the inserted record.
        """
        try:
            sql = "INSERT INTO Segment (RecordingID, Offset, InsertedDate) Values (?, ?, ?)"
            self.cursor.execute(sql, (recording_id, offset, self.today))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::insert_segment: {e}")

    def delete_segment(self, filters: Optional[dict] = None):
        """
        Delete one or more Segment records.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters.
        """
        try:
            self._delete_records("Segment", filters, self.segment_relationships)
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::delete_segment: {e}")

    def get_segment(
        self,
        filters: Optional[dict] = None,
        include_audio: bool = False,
    ):
        """
        Query the Segment table.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters.
        - include_audio (bool, optional): if True, include audio in the returned objects. Default = False.

        Returns:
            A list of entries, each as a SimpleNamespace object with the following attributes:
            - id (int): Unique ID of the entry.
            - audio (blob): raw audio, or None.
            - sampling_rate (int): if there is audio, this is its sampling_rate
            - offset (float): number of seconds from the start of the recording to the start of the segment.
            - recording_id (int): ID of the corresponding Recording record.

        """
        try:
            fields = "Segment.ID, RecordingID, Offset, SamplingRate"

            # audio is a large blob and slow to fetch, so it is optional
            if include_audio:
                fields += ", Audio"

            sql = f"SELECT {fields} FROM Segment"

            join_clauses, where_clauses, params = self._parse_filters(
                "Segment", filters, self.segment_relationships
            )
            if join_clauses:
                sql += " " + " ".join(join_clauses)

            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

            sql += " ORDER BY Segment.ID"

            self.cursor.execute(sql, params)
            rows = self.cursor.fetchall()
            if rows is None:
                return []

            results = []
            for row in rows:
                (id, recordingID, offset, sampling_rate) = row[:4]
                if include_audio:
                    audio = row[4]
                else:
                    audio = None

                result = SimpleNamespace(
                    id=id,
                    recording_id=recordingID,
                    offset=offset,
                    audio=audio,
                    sampling_rate=sampling_rate,
                )
                results.append(result)

            return results

        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::get_segment: {e}")

    def get_segment_by_class(
        self,
        class_name: str,
        include_audio: bool = False,
    ):
        """
        Get segment info for the given class.

        Args:
        - class_name (str): class name.
        - include_audio (bool, optional): if True, include audio in the returned objects. Default = False.

        Returns:
            A list of entries, each as a SimpleNamespace object with the following attributes:
            - id (int): ID of the Segment record.
            - audio (blob): raw audio, or None.
            - recording_id (int): ID of the corresponding Recording record.
            - offset (float): Number of seconds from the start of the recording to the start of the segment.
        """
        try:
            fields = "s.ID, s.RecordingID, s.Offset"

            # audio is a large blob and slow to fetch, so it is optional
            if include_audio:
                fields += ", s.Audio"

            sql = f"""
                SELECT {fields} FROM Segment s
                JOIN SegmentClass sc ON s.ID = sc.SegmentID
                JOIN Class c ON sc.ClassID = c.ID
                WHERE c.Name = ?
                ORDER BY s.RecordingID, s.Offset
            """

            self.cursor.execute(sql, (class_name,))
            rows = self.cursor.fetchall()

            results = []
            for row in rows:
                id, recordingID, offset = row[:3]
                if include_audio:
                    audio = row[3]
                else:
                    audio = None

                result = SimpleNamespace(
                    id=id,
                    audio=audio,
                    recording_id=recordingID,
                    offset=offset,
                )
                results.append(result)

            return results

        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::get_segment_by_class: {e}")

    def get_segment_count(self, filters: Optional[dict] = None):
        """
        Get the number of records in the Segment table.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters.

        Returns:
            Number of records that match the criteria.
        """
        try:
            return self._get_count("Segment", filters, self.segment_relationships)
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::get_segment_count: {e}")

    def get_all_segment_counts(self):
        """
        Get the class name and segment count for all classes.

        Returns:
            A list of entries, each as a SimpleNamespace object with the following attributes:
            - class_name (str): Class Name.
            - count (int): Number of segments.
        """
        try:
            sql = """
                SELECT c.Name, COUNT(s.ID)
                FROM Segment s
                JOIN Recording r ON s.RecordingID = r.ID
                JOIN SegmentClass sc ON s.ID = sc.SegmentID
                JOIN Class c ON sc.ClassID = c.ID
                GROUP BY c.ID
                ORDER BY c.Name;
            """

            self.cursor.execute(sql)
            rows = self.cursor.fetchall()
            results = []
            for row in rows:
                class_name, count = row
                result = SimpleNamespace(class_name=class_name, count=count)
                results.append(result)

            return results
        except sqlite3.Error as e:
            raise DatabaseError(
                f"Error in TrainingDatabase::get_all_segment_counts: {e}"
            )

    def update_segment(self, id: int, field: str, value):
        """
        Update a record in the Segment table.

        Args:
        - id (int): ID that identifies the record to update
        - field (str): Name of column to update.
        - value: New value.
        """
        try:
            sql = f"""
                UPDATE Segment SET {field} = ? WHERE ID = ?
            """
            self.cursor.execute(sql, (value, id))
            self.conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::update_segment: {e}")

    # ------------------------------- #
    # SegmentClass
    # ------------------------------- #

    def get_segment_class(self):
        """
        Get all records from the SegmentClass table.

        Returns:
            A list of entries, each as a SimpleNamespace object with the following attributes:
            - segment_id (int): Segment ID.
            - class_id (int): Class ID.
            - soundtype_id: SoundType ID.
        """
        try:
            sql = """
                SELECT SegmentID, ClassID, SoundTypeID FROM SegmentClass
                ORDER BY SegmentID, ClassID, SoundTypeID
            """

            self.cursor.execute(sql)
            rows = self.cursor.fetchall()
            if rows is None:
                return []

            results = []
            for row in rows:
                segment_id, class_id, soundtype_id = row
                result = SimpleNamespace(
                    segment_id=segment_id,
                    class_id=class_id,
                    soundtype_id=soundtype_id,
                )
                results.append(result)

            return results

        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::get_segment_class: {e}")

    def get_segment_class_count(self, filters: Optional[dict] = None):
        """
        Get the number of records in the SegmentClass table.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters.

        Returns:
            Number of records that match the criteria.
        """
        try:
            return self._get_count(
                "SegmentClass", filters, self.segment_class_relationships
            )
        except sqlite3.Error as e:
            raise DatabaseError(
                f"Error in TrainingDatabase::get_segment_class_count: {e}"
            )

    def insert_segment_class(self, segment_id: int, class_id: int):
        """
        Insert a SegmentClass record, to identify a segment as containing a sound of the class.
        Spectrograms can contain sounds of multiple classes, represented by multiple SegmentClass
        records.

        Args:
        - segment_id (int, required): Segment ID.
        - class_id (int, required): Class ID.

        Returns:
            row_id (int): ID of the inserted record.
        """
        try:
            sql = "INSERT INTO SegmentClass (SegmentID, ClassID) Values (?, ?)"
            self.cursor.execute(sql, (segment_id, class_id))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::insert_segment_class: {e}")

    def update_segment_class(self, id: int, field: str, value):
        """
        Update a record in the SegmentClass table.

        Args:
        - id (int): ID that identifies the record to update
        - field (str): Name of column to update.
        - value: New value.
        """
        try:
            sql = f"""
                UPDATE SegmentClass SET {field} = ? WHERE ID = ?
            """
            self.cursor.execute(sql, (value, id))
            self.conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::update_segment_class: {e}")

    # ------------------------------- #
    # SpecGroup
    # ------------------------------- #

    def insert_specgroup(self, name: str):
        """
        Insert a SpecGroup record.

        Args:
        - name (str): Name of the group (e.g. "logscale").

        Returns:
            row_id (int): ID of the inserted record.
        """
        try:
            sql = "INSERT INTO SpecGroup (Name, InsertedDate) Values (?, ?)"
            self.cursor.execute(sql, (name, self.today))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::insert_specgroup: {e}")

    def get_specgroup(self, filters: Optional[dict] = None):
        """
        Query the SpecGroup table.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters. Valid
        column names for the SpecGroup table are:
            - ID (int): record ID
            - Name (str): specgroup name

        Returns:
            A list of entries, each as a SimpleNamespace object with the following attributes:
            - id (int): Unique ID of the entry.
            - name (str): Name of the specgroup.
        """
        try:
            sql = "SELECT ID, Name FROM SpecGroup"
            _, where_clauses, params = self._parse_filters("SpecGroup", filters)
            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

            sql += " ORDER BY ID"

            self.cursor.execute(sql, params)
            rows = self.cursor.fetchall()
            if rows is None:
                return []

            results = []
            for row in rows:
                id, name = row
                result = SimpleNamespace(id=id, name=name)
                results.append(result)

            return results

        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::get_specgroup: {e}")

    def delete_specgroup(self, filters: Optional[dict] = None):
        """
        Delete one or more SpecGroup records.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters. Valid
        column names for the Category table are:
            - ID (int): record ID
            - Name (str): specgroup name

        Returns:
            Number of records deleted.
        """
        try:
            return self._delete_records("SpecGroup", filters)
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::delete_specgroup: {e}")

    def get_specgroup_count(self, filters: Optional[dict] = None):
        """
        Get the number of records in the SpecGroup table.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters. Valid
        column names for the Source table are:
            - ID (int): record ID
            - Name (str): spec group name

        Returns:
            Number of records that match the criteria.
        """
        try:
            return self._get_count("SpecGroup", filters)
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::get_specgroup_count: {e}")

    # ------------------------------- #
    # SpecValue
    # ------------------------------- #

    def insert_specvalue(
        self,
        value: bytes,
        spec_group_id: int,
        segment_id: int,
    ):
        """
        Insert a SpecValue record.

        Args:
        - value (blob, required): the actual compressed spectrogram
        - spec_group_id (int, required): ID of spec group record
        - segment_id (int, required): ID of segment record
        - sampling_rate (int): sampling rate used to create it

        Returns:
            row_id (int): ID of the inserted record.
        """
        try:
            sql = (
                "INSERT INTO SpecValue (Value, SpecGroupID, SegmentID) Values (?, ?, ?)"
            )
            self.cursor.execute(sql, (value, spec_group_id, segment_id))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::insert_specvalue: {e}")

    def get_specvalue(self, filters: Optional[dict] = None):
        """
        Query the SpecValue table.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters.

        Returns:
            A list of entries, each as a SimpleNamespace object with the following attributes:
            - id (int): Unique ID of the entry.
            - value (bytes): The compressed spectrogram.
            - embedding (bytes): The embedding.
            - specgroup_id (str): ID of the corresponding specgroup.
            - segment_id (str): ID of the corresponding segment.
        """
        try:
            sql = """
                SELECT SpecValue.ID, SpecValue.Value, SpecValue.Embedding, SpecValue.SpecGroupID, SpecValue.SegmentID
                FROM SpecValue
            """

            join_clauses, where_clauses, params = self._parse_filters(
                "SpecValue", filters, self.specvalue_relationships
            )
            if join_clauses:
                sql += " " + " ".join(join_clauses)

            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

            sql += " ORDER BY SpecValue.ID"

            self.cursor.execute(sql, params)
            rows = self.cursor.fetchall()
            if rows is None:
                return []

            results = []
            for row in rows:
                id, value, embedding, specgroup_id, segment_id = row
                result = SimpleNamespace(
                    id=id,
                    value=value,
                    embedding=embedding,
                    specgroup_id=specgroup_id,
                    segment_id=segment_id,
                )
                results.append(result)

            return results

        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::get_specvalue: {e}")

    def delete_specvalue(self, filters: Optional[dict] = None):
        """
        Delete one or more SpecValue records.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters. Valid
        column names for the SpecValue table are:
            - ID (int): record ID

        Returns:
            Number of records deleted.
        """
        try:
            return self._delete_records("SpecValue", filters)
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::delete_specvalue: {e}")

    def update_specvalue(self, id: int, field: str, value):
        """
        Update a record in the SpecValue table.

        Args:
        - id (int): ID that identifies the record to update
        - field (str): Name of column to update.
        - value: New value.
        """
        try:
            sql = f"""
                UPDATE SpecValue SET {field} = ? WHERE ID = ?
            """
            self.cursor.execute(sql, (value, id))
            self.conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::update_specvalue: {e}")

    def get_specvalue_count(self, filters: Optional[dict] = None):
        """
        Get the number of records in the SpecValue table.

        Args:
        - filters (dict, optional): a dict of column_name/value pairs that define filters.

        Returns:
            Number of records that match the criteria.
        """
        try:
            return self._get_count("SpecValue", filters)
        except sqlite3.Error as e:
            raise DatabaseError(f"Error in TrainingDatabase::get_specvalue_count: {e}")

    # --------------------------------------- #
    # Queries on joined Spectrogram tables
    # --------------------------------------- #

    def get_spectrogram_by_class(
        self,
        class_name: str,
        include_value: bool = True,
        include_embedding: bool = False,
        spec_group: str = "default",
        limit: Optional[int] = None,
    ):
        """
        Fetch joined spectrogram records for the given class.

        Args:
        - class_name (str): class name.
        - include_value (bool, optional): If True, include the compressed spectrogram. Default = True.
        - include_embedding (bool, optional): If True, include embeddings in the returned objects. Default = False.
        - spec_group (str): Name of spectrogram group. Default = "default".
        - limit (int, optional): If specified, only return up to this many records.

        Returns:
            A list of entries, each as a SimpleNamespace object with the following attributes:
            - segment_id (int): ID of the Segment record.
            - specvalue_id (int): ID of the SpecValue record.
            - value (bytes): The spectrogram itself.
            - embedding (bytes): The embedding, if include_embedding=True.
            - recording_id (int): ID of the corresponding recording record.
            - filename (str): Name of the audio file.
            - offset (float): Number of seconds from the start of the recording to the start of the segment.
        """
        try:

            fields = "s.ID, sv.ID, s.RecordingID, r.FileName, s.Offset"
            if include_value:
                fields += ", sv.Value"

            if include_embedding:
                fields += ", sv.Embedding"

            if limit is None:
                limit_clause = ""
            else:
                limit_clause = f"LIMIT {limit}"

            results = self.get_specgroup()
            if len(results) == 1 and results[0].name == spec_group:
                # Skip spec_group join for performance
                sql = f"""
                    SELECT {fields} FROM SpecValue sv
                    JOIN Segment s on sv.SegmentID = s.ID
                    JOIN SegmentClass sc ON s.ID = sc.SegmentID
                    JOIN Class c ON sc.ClassID = c.ID
                    JOIN Recording r ON s.RecordingID = r.ID
                    WHERE c.Name = "{class_name}"
                    ORDER BY r.FileName, s.Offset
                    {limit_clause}
                """
            else:
                sql = f"""
                    SELECT {fields} FROM SpecValue sv
                    JOIN SpecGroup sg on sv.SpecGroupID = sg.ID
                    JOIN Segment s on sv.SegmentID = s.ID
                    JOIN SegmentClass sc ON s.ID = sc.SegmentID
                    JOIN Class c ON sc.ClassID = c.ID
                    JOIN Recording r ON s.RecordingID = r.ID
                    WHERE c.Name = "{class_name}" AND sg.Name = "{spec_group}"
                    ORDER BY r.FileName, s.Offset
                    {limit_clause}
                """

            self.cursor.execute(sql)
            rows = self.cursor.fetchall()

            results = []
            for row in rows:
                segment_id, value_id, recordingID, filename, offset = row[:5]
                if include_value:
                    value = row[5]
                    next_idx = 6
                else:
                    value = None
                    next_idx = 5

                if include_embedding:
                    embedding = row[next_idx]
                else:
                    embedding = None

                result = SimpleNamespace(
                    segment_id=segment_id,
                    specvalue_id=value_id,
                    value=value,
                    embedding=embedding,
                    recording_id=recordingID,
                    filename=filename,
                    offset=offset,
                )
                results.append(result)

            return results

        except sqlite3.Error as e:
            raise DatabaseError(
                f"Error in TrainingDatabase::get_spectrogram_by_class: {e}"
            )
