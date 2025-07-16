import pandas as pd
import pytest

from ftw.utils import get_harvest_integer_from_bbox, harvest_to_datetime


@pytest.fixture()
def bbox_iowa_aoi():
    return [-93.68708939, 41.9530844, -93.64078526, 41.98070608]


@pytest.fixture()
def bbox_kenya_aoi():
    return [34.71339499, -2.84654017, 40.24044535, 3.1539109]


def test_get_harvest_integer_from_bbox(bbox_iowa_aoi):
    # Test for a valid bounding box
    result = get_harvest_integer_from_bbox(bbox_iowa_aoi)
    assert isinstance(result, list)
    assert len(result) == 2  # Should return start and end harvest days
    assert result[0] == 88
    assert result[1] == 316


def test_get_harvest_integer_multi_harvest_days(bbox_kenya_aoi):
    # Test for a bounding box with multiple harvest days
    result = get_harvest_integer_from_bbox(bbox_kenya_aoi)
    assert isinstance(result, list)
    assert len(result) == 2  # Should return start and end harvest days
    assert result[0] == 243  # Example expected value, adjust as needed
    assert result[1] == 44  # Example expected value, adjust as needed


def test_harvest_to_datetime():
    # Test for a valid harvest day and year
    result = harvest_to_datetime(100, 2023)
    assert result == pd.Timestamp("2023-04-10")
