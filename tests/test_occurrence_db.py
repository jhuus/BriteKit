import os
import pytest

import numpy as np

from britekit import OccurrenceDatabase, OccurrenceDataProvider


@pytest.fixture(scope="module")
def db():
    db_path = os.path.join("data", "_test_occur.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    db = OccurrenceDatabase(db_path=db_path)
    yield db
    db.close()


def test_all(db: OccurrenceDatabase):
    # insert counties
    name1, code1, min_x1, max_x1, min_y1, max_y1 = (
        "Ottawa",
        "CA-ON-OT",
        -76.35396,
        -75.24663,
        44.96563,
        45.53698,
    )
    county_id1 = db.insert_county(name1, code1, min_x1, max_x1, min_y1, max_y1)

    name2, code2, min_x2, max_x2, min_y2, max_y2 = (
        "Edmonton",
        "CA-AB-EL",
        -114.65416,
        -112.91187,
        51.88177,
        52.88732,
    )
    county_id2 = db.insert_county(name2, code2, min_x2, max_x2, min_y2, max_y2)

    counties = db.get_all_counties()
    assert len(counties) == 2
    assert counties[0].id == county_id1
    assert counties[1].id == county_id2

    provider = OccurrenceDataProvider(db)
    county = provider.find_county(51.9, -113)
    assert county.name == name2

    # insert classes
    class_name1 = "Barred Owl"
    class_id1 = db.insert_class(class_name1)

    class_name2 = "American Goldfinch"
    class_id2 = db.insert_class(class_name2)

    classes = db.get_all_classes()
    assert len(classes) == 2
    assert classes[0].id == class_id1
    assert classes[1].id == class_id2

    # insert occurrences
    ones = np.ones(48)
    db.insert_occurrences(county_id1, class_id1, ones)
    db.insert_occurrences(county_id1, class_id2, ones * 2)
    db.insert_occurrences(county_id2, class_id1, ones * 3)
    db.insert_occurrences(county_id2, class_id2, ones * 4)

    occurrences = db.get_all_occurrences()
    assert len(occurrences) == 4

    values = db.get_occurrences(county_id1, class_name1)
    assert len(values) == 48
    assert values[0] == 1

    values = db.get_occurrences(county_id1, class_name2)
    assert len(values) == 48
    assert values[0] == 2

    provider.refresh()  # refresh cache since we did inserts and deletes
    values = provider.occurrences(code2, class_name2)
    assert len(values) == 48
    assert values[0] == 4

    # delete county
    db.delete_county(county_id1)
    counties = db.get_all_counties()
    assert len(counties) == 1
    assert counties[0].id == county_id2

    # check that delete cascaded to Occurrences table
    occurrences = db.get_all_occurrences()
    assert len(occurrences) == 2

    # delete class1
    db.delete_class(class_id1)

    classes = db.get_all_classes()
    assert len(classes) == 1
    assert classes[0].id == class_id2

    # check that delete cascaded to Occurrences table
    occurrences = db.get_all_occurrences()
    assert len(occurrences) == 1
