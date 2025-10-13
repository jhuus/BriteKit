import shutil
from pathlib import Path

from britekit import TrainingDatabase, TrainingDataProvider


def test_db_convert():
    # copy the old database (named ".saved" since ".db" is in gitignore.txt)
    from_path = str(Path("tests") / "db" / "old.db.saved")
    to_path = str(Path("tests") / "db" / "new.db")
    shutil.copyfile(from_path, to_path)

    # open and validate it
    db = TrainingDatabase(db_path=to_path)
    assert db.get_source_count() == 3
    assert db.get_category_count() == 2
    assert db.get_soundtype_count() == 3
    assert db.get_class_count() == 2
    assert db.get_recording_count() == 17
    assert db.get_segment_count() == 656

    # get summary and details dataframes and validate them
    provider = TrainingDataProvider(db)
    summary_df, _ = provider.class_info()
    grfl_recording_count = summary_df.loc[
        summary_df["name"] == "Gray Flycatcher", "recordings"
    ].values[0]
    assert grfl_recording_count == 9

    hesp_recording_count = summary_df.loc[
        summary_df["name"] == "Henslow's Sparrow", "recordings"
    ].values[0]
    assert hesp_recording_count == 8

    grfl_segment_count = summary_df.loc[
        summary_df["name"] == "Gray Flycatcher", "segments"
    ].values[0]
    assert grfl_segment_count == 554

    hesp_segment_count = summary_df.loc[
        summary_df["name"] == "Henslow's Sparrow", "segments"
    ].values[0]
    assert hesp_segment_count == 102

    db.close()
