"""Compact, versioned occurrence-pickle compiler."""

from dataclasses import dataclass
from pathlib import Path
import pickle
import tempfile
import zlib

import numpy as np

from britekit.core.exceptions import DatabaseError
from britekit.occurrence_db.occurrence_database_v2 import OccurrenceDatabaseV2

FORMAT_NAME = "britekit-occurrence"
FORMAT_VERSION = 2
WEEK_COUNT = 48


@dataclass(frozen=True)
class OccurrencePickleV2Report:
    """Summary of a compiled occurrence pickle."""

    region_code: str
    area_count: int
    class_count: int
    occurrence_count: int


def compile_occurrence_pickle_v2(
    input_path: str | Path, output_path: str | Path
) -> OccurrencePickleV2Report:
    """Compile an occurrence-v2 database into the compact pickle format."""
    source_path = Path(input_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with OccurrenceDatabaseV2(source_path) as database:
        problems = database.validate()
        if problems:
            raise DatabaseError(
                "Invalid source occurrence database: " + "; ".join(problems)
            )
        payload, report = _build_payload(database)

    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            pickle.dump(payload, temporary, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.flush()
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    temporary_path.replace(destination)
    return report


def _build_payload(
    database: OccurrenceDatabaseV2,
) -> tuple[dict[str, object], OccurrencePickleV2Report]:
    connection = database.conn
    pack = connection.execute("SELECT * FROM RegionPack WHERE ID = 1").fetchone()
    if pack is None:
        raise DatabaseError("Source occurrence database has no region pack")

    area_rows = connection.execute(
        """
        SELECT ID, Code, Name, MinLongitude, MaxLongitude,
               MinLatitude, MaxLatitude
        FROM AdministrativeArea
        WHERE Selectable = 1
        ORDER BY ID
        """
    ).fetchall()
    class_rows = connection.execute("SELECT ID, Name FROM Class ORDER BY ID").fetchall()
    area_index = {row["ID"]: index for index, row in enumerate(area_rows)}
    class_index = {row["ID"]: index for index, row in enumerate(class_rows)}

    occurrence_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM Occurrence occurrence
        JOIN AdministrativeArea area ON area.ID = occurrence.AreaID
        WHERE area.Selectable = 1
        """
    ).fetchone()[0]
    class_indices = np.empty(occurrence_count, dtype=np.int32)
    values = np.empty((occurrence_count, WEEK_COUNT), dtype=np.float16)
    area_offsets = np.zeros(len(area_rows) + 1, dtype=np.int64)

    counts = {
        row["AreaID"]: row["OccurrenceCount"]
        for row in connection.execute(
            """
            SELECT occurrence.AreaID, COUNT(*) AS OccurrenceCount
            FROM Occurrence occurrence
            JOIN AdministrativeArea area ON area.ID = occurrence.AreaID
            WHERE area.Selectable = 1
            GROUP BY occurrence.AreaID
            """
        )
    }
    for index, area in enumerate(area_rows):
        area_offsets[index + 1] = area_offsets[index] + counts.get(area["ID"], 0)
    next_positions = area_offsets[:-1].copy()

    position = 0
    rows = connection.execute(
        """
        SELECT occurrence.AreaID, occurrence.ClassID, occurrence.Value
        FROM Occurrence occurrence
        JOIN AdministrativeArea area ON area.ID = occurrence.AreaID
        WHERE area.Selectable = 1
        ORDER BY occurrence.AreaID, occurrence.ClassID
        """
    )
    for row in rows:
        current_area_index = area_index[row["AreaID"]]
        position = int(next_positions[current_area_index])
        occurrence_values = np.frombuffer(
            zlib.decompress(row["Value"]), dtype=np.float16
        )
        if occurrence_values.shape != (WEEK_COUNT,):
            raise DatabaseError(
                f"Occurrence row for area {row['AreaID']} and class "
                f"{row['ClassID']} has {len(occurrence_values)} weekly values; "
                f"expected exactly {WEEK_COUNT}"
            )
        class_indices[position] = class_index[row["ClassID"]]
        values[position] = occurrence_values
        next_positions[current_area_index] += 1

    if not np.array_equal(next_positions, area_offsets[1:]):
        raise DatabaseError(
            f"Expected {occurrence_count} occurrence rows, compiled "
            f"{int(next_positions.sum() - area_offsets[:-1].sum())}"
        )

    payload: dict[str, object] = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "region": {
            "code": pack["Code"],
            "name": pack["Name"],
            "data_version": pack["DataVersion"],
            "taxonomy_version": pack["TaxonomyVersion"],
        },
        "areas": [
            (
                row["Code"],
                row["Name"],
                row["MinLongitude"],
                row["MaxLongitude"],
                row["MinLatitude"],
                row["MaxLatitude"],
            )
            for row in area_rows
        ],
        "class_names": tuple(row["Name"] for row in class_rows),
        "area_offsets": area_offsets,
        "class_indices": class_indices,
        "values": values,
    }
    return payload, OccurrencePickleV2Report(
        region_code=pack["Code"],
        area_count=len(area_rows),
        class_count=len(class_rows),
        occurrence_count=occurrence_count,
    )
