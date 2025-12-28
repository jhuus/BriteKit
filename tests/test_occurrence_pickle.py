import os
from pathlib import Path
import zipfile

from britekit.occurrence_db.occurrence_pickle import OccurrencePickleProvider


def test_all():
    zip_path = str(Path("tests") / "db" / "occurrence.zip")
    zip_dir = str(Path("tests") / "db")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(zip_dir)

    pickle_path = str(Path("tests") / "db" / "occurrence.pkl")
    provider = OccurrencePickleProvider(pickle_path=pickle_path)

    county = provider.find_county(45.3, -75.7)
    assert county.code == "CA-ON-OT"

    counties = provider.find_counties("US-WA")
    assert len(counties) == 39 and counties[0].name == "Adams"

    location_found, class_found, value = provider.occurrence_value(
        "Western Tanager", region_code="US-WA", week_num=None
    )
    assert location_found and class_found
    assert value > 0.4 and value < 0.6

    location_found, class_found, value = provider.occurrence_value(
        "Western Tanager", region_code="CA-NS", week_num=None
    )
    assert not class_found  # not listed for Nova Scotia
    assert value is None

    location_found, class_found, value = provider.occurrence_value(
        "Northern Parula", region_code="CA-ON-OT", week_num=1
    )
    assert location_found and class_found
    assert value < 0.01

    location_found, class_found, value = provider.occurrence_value(
        "Northern Parula", region_code="CA-ON-OT", week_num=18
    )
    assert location_found and class_found
    assert value > 0.2

    location_found, class_found, value = provider.occurrence_value(
        "Northern Parula", region_code="CA-ON-OT", week_num=None
    )
    assert location_found and class_found
    assert value > 0.2

    location_found, class_found, value = provider.occurrence_value(
        "Northern Parula", region_code="CA-ON-OT", week_num=18, smoothed=False
    )
    assert location_found and class_found
    print(value)
    assert value > 0.2

    os.remove(pickle_path)
