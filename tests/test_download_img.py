import pandas as pd
import pytest

from ftw_tools.download.download_img import query_stac


@pytest.fixture
def large_aoi():
    [13.83984671, -6.73397741, 23.62140098, 0.0]


def test_query_stac_large_aoi(large_aoi):
    with pytest.raises(ValueError):
        query_stac(bbox=large_aoi, date=pd.Timestamp("2020-01-01"))
