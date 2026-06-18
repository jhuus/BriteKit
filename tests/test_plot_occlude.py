#!/usr/bin/env python3

from britekit import TrainingDatabase
from britekit.commands._plot import _get_segment_spectrograms


def test_get_segment_spectrograms_filters_by_recording(tmp_path):
    db_path = tmp_path / "training.db"
    with TrainingDatabase(str(db_path)) as db:
        source_id = db.get_source({"Name": "default"})[0].id
        spec_group_id = db.get_specgroup({"Name": "default"})[0].id

        rec1_id = db.insert_recording(source_id, "XC123.mp3", "/audio/XC123.mp3", 12.0)
        rec2_id = db.insert_recording(source_id, "XC456.mp3", "/audio/XC456.mp3", 12.0)
        seg1_id = db.insert_segment(rec1_id, 1.5)
        seg2_id = db.insert_segment(rec2_id, 3.0)
        db.insert_specvalue(b"spec1", spec_group_id, seg1_id)
        db.insert_specvalue(b"spec2", spec_group_id, seg2_id)

        result = _get_segment_spectrograms(
            db, [seg1_id, seg2_id], "default", recording="XC123.mp3"
        )

    assert result == {seg1_id: ("XC123.mp3", 1.5, b"spec1")}
