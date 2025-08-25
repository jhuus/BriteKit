import os
import pytest
from britekit import TrainingDatabase


@pytest.fixture(scope="module")
def db():
    db_path = os.path.join("data", "_test.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    db = TrainingDatabase(db_path=db_path)
    yield db
    db.close()


def test_source_table(db: TrainingDatabase):
    # there should always be a 'default' record
    results = db.get_source()
    assert len(results) == 1
    assert results[0].name == "default"

    id1 = db.insert_source("Xeno-Canto")
    db.insert_source("iNaturalist")
    count = db.get_source_count()
    assert count == 3

    num_deleted = db.delete_source({"ID": id1})
    assert num_deleted == 1
    count = db.get_source_count()
    assert count == 2

    db.delete_source({"Name": "iNaturalist"})
    count = db.get_source_count()
    assert count == 1

    db.insert_source("Xeno-Canto")
    db.insert_source("iNaturalist")

    results = db.get_source()
    assert len(results) == 3

    results = db.get_source({"Name": "iNaturalist"})
    assert len(results) == 1
    assert results[0].name == "iNaturalist"

    # clean up
    db.delete_source({"Name": "Xeno-Canto"})
    db.delete_source({"Name": "iNaturalist"})
    results = db.get_source()
    assert len(results) == 1


def test_category_table(db: TrainingDatabase):
    # there should always be a 'default' record
    results = db.get_category()
    assert len(results) == 1
    assert results[0].name == "default"

    id1 = db.insert_category("bird")
    db.insert_category("other")
    count = db.get_category_count()
    assert count == 3

    num_deleted = db.delete_category({"ID": id1})
    assert num_deleted == 1
    count = db.get_category_count()
    assert count == 2

    num_deleted = db.delete_category({"Name": "other"})
    assert num_deleted == 1
    count = db.get_category_count()
    assert count == 1

    db.insert_category("bird")
    db.insert_category("other")

    results = db.get_category()
    assert len(results) == 3

    results = db.get_category({"Name": "other"})
    assert len(results) == 1
    assert results[0].name == "other"

    # clean up
    db.delete_category()
    assert db.get_category_count() == 0


def test_class_table(db: TrainingDatabase):
    cat_id = db.insert_category("bird")
    id1 = db.insert_class(cat_id, "Yellow Warbler", "", "YEWA")
    db.insert_class(cat_id, "White-winged Crossbill", "Two-barred Crossbill", "WWCR")
    count = db.get_class_count()
    assert count == 2

    num_deleted = db.delete_class({"ID": id1})
    assert num_deleted == 1
    count = db.get_class_count()
    assert count == 1

    num_deleted = db.delete_class({"Name": "White-winged Crossbill"})
    assert num_deleted == 1
    count = db.get_class_count()
    assert count == 0

    db.insert_class(cat_id, "Yellow Warbler", "", "YEWA")
    db.insert_class(cat_id, "White-winged Crossbill", "Two-barred Crossbill", "WWCR")
    count = db.get_class_count()
    assert count == 2

    results = db.get_class()
    assert len(results) == 2

    results = db.get_class({"Name": "White-winged Crossbill"})
    assert len(results) == 1
    assert results[0].name == "White-winged Crossbill"
    assert results[0].alt_name == "Two-barred Crossbill"
    assert results[0].code == "WWCR"

    results = db.get_class({"CategoryID": cat_id, "Name": "Yellow Warbler"})
    assert len(results) == 1
    assert results[0].name == "Yellow Warbler"
    assert results[0].alt_name == ""
    assert results[0].code == "YEWA"

    results = db.get_class({"CategoryName": "other", "Name": "Yellow Warbler"})
    assert not results

    results = db.get_class({"CategoryName": "bird", "Name": "Yellow Warbler"})
    assert len(results) == 1
    assert results[0].name == "Yellow Warbler"
    assert results[0].alt_name == ""
    assert results[0].code == "YEWA"

    # test cascading delete of category
    db.delete_category({"Name": "bird"})
    assert db.get_class_count() == 0

    # clean up
    db.delete_category()
    assert db.get_category_count() == 0
    db.delete_class()
    assert db.get_class_count() == 0


def test_soundtype_table(db: TrainingDatabase):
    id1 = db.insert_soundtype("woodpecker drumming")
    db.insert_soundtype("chip")
    count = db.get_soundtype_count()
    assert count == 2

    num_deleted = db.delete_soundtype({"ID": id1})
    assert num_deleted == 1
    count = db.get_soundtype_count()
    assert count == 1

    num_deleted = db.delete_soundtype({"Name": "chip"})
    assert num_deleted == 1
    count = db.get_soundtype_count()
    assert count == 0

    id1 = db.insert_soundtype("woodpecker drumming")
    db.insert_soundtype("chip")
    results = db.get_soundtype()
    assert len(results) == 2

    results = db.get_soundtype({"Name": "chip"})
    assert len(results) == 1
    assert results[0].name == "chip"

    # clean up
    db.delete_soundtype()


def test_recording_table(db: TrainingDatabase):
    source1_name = "Xeno-Canto"
    source2_name = "iNaturalist"
    source1_id = db.insert_source(source1_name)
    source2_id = db.insert_source(source2_name)

    rec1_name = "r1.mp3"
    rec1_path = "/data/bird/YEWA/r1.mp3"
    rec1_seconds = 120.0
    rec1_id = db.insert_recording(source1_id, rec1_name, rec1_path, rec1_seconds)

    rec2_name = "r2.mp3"
    rec2_path = "/data/bird/SOSP/r2.mp3"
    rec2_seconds = 179.5
    rec2_id = db.insert_recording(source1_id, rec2_name, rec2_path, rec2_seconds)
    count = db.get_recording_count()
    assert count == 2

    cat_id = db.insert_category("bird")
    class1_name = "Yellow Warbler"
    class2_name = "Song Sparrow"
    class1_id = db.insert_class(cat_id, class1_name)
    class2_id = db.insert_class(cat_id, class2_name)

    seg1_id = db.insert_segment(rec1_id, offset=0)
    seg2_id = db.insert_segment(rec2_id, offset=0)

    db.insert_segment_class(seg1_id, class1_id)
    db.insert_segment_class(seg2_id, class2_id)
    results = db.get_recording_by_class(class1_name)
    assert len(results) == 1
    assert results[0].id == rec1_id

    num_deleted = db.delete_recording({"ID": rec1_id})
    assert num_deleted == 1
    count = db.get_recording_count()
    assert count == 1

    num_deleted = db.delete_recording({"FileName": rec2_name})
    assert num_deleted == 1
    count = db.get_recording_count()
    assert count == 0

    rec1_id = db.insert_recording(source1_id, rec1_name, rec1_path, rec1_seconds)
    rec2_id = db.insert_recording(source1_id, rec2_name, rec2_path, rec2_seconds)
    count = db.get_recording_count()
    assert count == 2

    db.delete_recording({"SourceName": source1_name})
    count = db.get_recording_count()
    assert count == 0

    rec1_id = db.insert_recording(source1_id, rec1_name, rec1_path, rec1_seconds)
    rec2_id = db.insert_recording(source2_id, rec2_name, rec2_path, rec2_seconds)
    results = db.get_recording()
    assert len(results) == 2
    assert results[0].filename == rec1_name

    results = db.get_recording({"SourceName": source2_name})
    assert len(results) == 1
    assert results[0].filename == rec2_name

    new_filename = "abc.mp3"
    db.update_recording(rec2_id, "FileName", new_filename)
    results = db.get_recording({"ID": rec2_id})
    assert len(results) == 1
    assert results[0].filename == new_filename

    # clean up
    db.delete_category()
    assert db.get_category_count() == 0
    db.delete_class()
    assert db.get_class_count() == 0
    db.delete_source()
    assert db.get_source_count() == 0
    db.delete_recording()
    assert db.get_recording_count() == 0


def test_spectrogram_tables(db: TrainingDatabase):
    # spectrograms involve the Segment, SpecGroup and SpecValue tables
    soundtype1_name = "woodpecker drumming"
    soundtype1_id = db.insert_soundtype(soundtype1_name)

    soundtype2_name = "chip"
    soundtype2_id = db.insert_soundtype(soundtype2_name)

    source1_name = "Xeno-Canto"
    source2_name = "iNaturalist"
    source1_id = db.insert_source(source1_name)
    source2_id = db.insert_source(source2_name)

    cat_id = db.insert_category("bird")
    class1_name = "Yellow Warbler"
    class2_name = "Song Sparrow"
    class1_id = db.insert_class(cat_id, class1_name)
    class2_id = db.insert_class(cat_id, class2_name)

    rec1_name = "r1.mp3"
    rec1_path = "/data/bird/YEWA/r1.mp3"
    rec1_seconds = 120.0
    rec1_id = db.insert_recording(source1_id, rec1_name, rec1_path, rec1_seconds)

    rec2_name = "r2.mp3"
    rec2_path = "/data/bird/SOSP/r2.mp3"
    rec2_seconds = 179.5
    rec2_id = db.insert_recording(source2_id, rec2_name, rec2_path, rec2_seconds)

    spec_grp1_name = "default"
    spec_grp1_id = db.get_specgroup({"Name": spec_grp1_name})[0].id
    assert spec_grp1_id == 1

    spec_grp2_name = "logscale"
    spec_grp2_id = db.insert_specgroup(spec_grp2_name)
    assert spec_grp2_id == 2

    segment1_offset = 1.5
    segment1_id = db.insert_segment(rec1_id, segment1_offset)

    segment2_offset = 3.0
    segment2_id = db.insert_segment(rec2_id, segment2_offset)

    spec_value = bytes(0)  # dummy spec
    db.insert_specvalue(spec_value, spec_grp2_id, segment1_id)

    count = db.get_segment_count()
    assert count == 2

    count = db.get_segment_count({"FileName": rec1_name})
    assert count == 1
    count = db.get_segment_count({"FileName": rec2_name})
    assert count == 1

    segment_class1_id = db.insert_segment_class(segment1_id, class1_id)
    segment_class2_id = db.insert_segment_class(segment2_id, class2_id)
    count = db.get_segment_class_count()
    assert count == 2

    db.update_segment_class(segment_class1_id, "SoundTypeID", soundtype1_id)
    db.update_segment_class(segment_class2_id, "SoundTypeID", soundtype2_id)

    results = db.get_segment_class()
    assert len(results) == 2
    assert results[0].soundtype_id == soundtype1_id
    assert results[1].soundtype_id == soundtype2_id

    db.delete_specgroup({"Name": spec_grp2_name})
    count = db.get_specgroup_count()
    assert count == 1

    count = db.get_specvalue_count()
    assert count == 0

    """
    db.delete_spectrogram({"FileName": rec1_name})
    count = db.get_spectrogram_count()
    assert count == 1

    db.delete_spectrogram({"FileName": rec2_name})
    count = db.get_spectrogram_count()
    assert count == 0

    spec_id1 = db.insert_spectrogram(
        rec_id1, spec_value, spec1_offset, sound_type_id=soundtype_id1
    )
    spec_id2 = db.insert_spectrogram(
        rec_id2, spec_value, spec2_offset, sound_type_id=soundtype_id2
    )

    db.insert_spectrogram_class(spec_id1, class_id1)
    db.insert_spectrogram_class(spec_id2, class_id2)

    results = db.get_spectrogram_by_class(class1_name)
    assert len(results) == 1
    assert results[0].recording_id == rec_id1

    results = db.get_spectrogram_by_class(class2_name)
    assert len(results) == 1
    assert results[0].recording_id == rec_id2

    db.delete_spectrogram_by_class(class1_name)
    count = db.get_spectrogram_count()
    assert count == 1

    db.delete_spectrogram_by_class(class2_name)
    count = db.get_spectrogram_count()
    assert count == 0

    spec_id1 = db.insert_spectrogram(
        rec_id1, spec_value, spec1_offset, sound_type_id=soundtype_id1
    )
    spec_id2 = db.insert_spectrogram(
        rec_id2, spec_value, spec2_offset, sound_type_id=soundtype_id2
    )

    results = db.get_spectrogram()
    assert len(results) == 2
    assert results[0].recording_id == rec_id1
    assert results[1].recording_id == rec_id2

    results = db.get_spectrogram({"FileName": rec2_name})
    assert len(results) == 1
    assert results[0].recording_id == rec_id2

    results = db.get_spectrogram({"RecordingID": rec_id1, "Offset": spec1_offset})
    assert len(results) == 1
    assert results[0].recording_id == rec_id1

    results = db.get_spectrogram_embeddings()
    assert len(results) == 2

    db.insert_spectrogram_class(spec_id1, class_id1)
    db.insert_spectrogram_class(spec_id2, class_id2)

    results = db.get_spectrogram_embeddings_by_class(class_name=class2_name)
    assert len(results) == 1

    results = db.get_spectrogram_embeddings_by_class(class_code="XXXX")
    assert len(results) == 0

    results = db.get_all_spectrogram_counts()
    assert len(results) == 2

    # test that deleting a sound type sets SoundTypeID=null in spectrogram table
    db.delete_soundtype()
    results = db.get_spectrogram()
    assert len(results) == 2
    assert results[0].sound_type_id is None
    assert results[1].sound_type_id is None
    """

    # clean up
    db.delete_specgroup()
    db.delete_specvalue()
    db.delete_segment()
    db.delete_recording()
    db.delete_class()
    db.delete_soundtype()
    db.delete_category()
    db.delete_source()
