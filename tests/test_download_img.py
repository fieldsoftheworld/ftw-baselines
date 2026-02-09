import pandas as pd
import pystac
import pytest

from ftw_tools.download.download_img import (
    _compute_bbox_nodata_percentage,
    _parse_stac_item,
    get_item,
    query_stac,
)


def test_compute_bbox_nodata_fully_covered():
    """Bbox fully within item footprint -> ~0% nodata."""
    item_geometry = {
        "type": "Polygon",
        "coordinates": [
            [[10.0, 47.0], [11.0, 47.0], [11.0, 48.0], [10.0, 48.0], [10.0, 47.0]]
        ],
    }
    bbox = [10.2, 47.2, 10.8, 47.8]
    result = _compute_bbox_nodata_percentage(item_geometry, bbox)
    assert result < 1.0


def test_compute_bbox_nodata_partial_coverage():
    """Bbox extends beyond item footprint -> ~50% nodata."""
    item_geometry = {
        "type": "Polygon",
        "coordinates": [
            [[10.0, 47.0], [10.5, 47.0], [10.5, 48.0], [10.0, 48.0], [10.0, 47.0]]
        ],
    }
    bbox = [10.0, 47.0, 11.0, 48.0]
    result = _compute_bbox_nodata_percentage(item_geometry, bbox)
    assert 40.0 < result < 60.0


def test_compute_bbox_nodata_no_overlap():
    """Bbox does not overlap item footprint -> 100% nodata."""
    item_geometry = {
        "type": "Polygon",
        "coordinates": [
            [[10.0, 47.0], [11.0, 47.0], [11.0, 48.0], [10.0, 48.0], [10.0, 47.0]]
        ],
    }
    bbox = [20.0, 47.0, 21.0, 48.0]
    result = _compute_bbox_nodata_percentage(item_geometry, bbox)
    assert result > 99.0


def test_compute_bbox_nodata_none_geometry():
    """None geometry -> 100% nodata."""
    result = _compute_bbox_nodata_percentage(None, [10.0, 47.0, 11.0, 48.0])
    assert result == 100.0


@pytest.mark.integration
def test_get_item_from_s3_url():
    s3_url = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/15/T/VG/2021/12/S2A_15TVG_20211225_0_L2A"
    item = get_item(s3_url, stac_host="earthsearch")
    assert type(item) is pystac.Item


@pytest.mark.integration
def test_get_item_from_id():
    item_id = "S2A_15TVG_20211225_0_L2A"
    item = get_item(item_id, stac_host="earthsearch")
    assert type(item) is pystac.Item
    assert item.id == item_id


@pytest.mark.integration
def test_get_item_from_id_single_digit_month():
    item_id = "S2B_33UUP_20210925_0_L2A"
    item = get_item(item_id, stac_host="earthsearch")
    assert type(item) is pystac.Item
    assert item.id == item_id


@pytest.mark.integration
def test_get_item_from_s3_url_single_digit_month():
    s3_url = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/33/U/UP/2021/9/S2B_33UUP_20210925_0_L2A"
    item = get_item(s3_url, stac_host="earthsearch")
    assert type(item) is pystac.Item
    assert item.id == "S2B_33UUP_20210925_0_L2A"


@pytest.mark.integration
def test_query_stac_future_year():
    with pytest.raises(ValueError, match="Crop calendar harvest date"):
        query_stac(
            bbox=[-93.68708939, 41.9530844, -93.64078526, 41.98070608],
            stac_host="mspc",
            date=pd.Timestamp.now() + pd.Timedelta(days=3),
        )


@pytest.fixture
def large_aoi():
    return [13.83984671, -6.73397741, 15.0, -5]


@pytest.mark.integration
def test_query_stac_large_aoi_mspc(large_aoi):
    with pytest.raises(ValueError):
        query_stac(bbox=large_aoi, stac_host="mspc", date=pd.Timestamp("2020-01-01"))


@pytest.mark.integration
def test_query_stac_large_aoi_earthsearch(large_aoi):
    with pytest.raises(ValueError):
        query_stac(
            bbox=large_aoi, stac_host="earthsearch", date=pd.Timestamp("2020-01-01")
        )


@pytest.mark.integration
def test_query_stac_with_nodata_filter_mspc():
    """Test that nodata_max parameter is accepted and used in queries"""
    # This test verifies the parameter is accepted - actual filtering is done by the STAC API
    bbox = [13.0, 48.0, 13.2, 48.2]
    date = pd.Timestamp("2024-06-01")

    # Query with nodata filter should not raise an error
    result = query_stac(
        bbox=bbox,
        stac_host="mspc",
        date=date,
        cloud_cover_max=20,
        buffer_days=14,
        nodata_max=50,  # Filter out scenes with >50% nodata
    )

    # Should return a valid STAC item URL
    assert result is not None
    assert isinstance(result, str)


@pytest.mark.integration
def test_parse_stac_item_includes_nodata():
    """Test that parsed STAC items include nodata percentage"""
    item_id = "S2B_33UUP_20210925_0_L2A"
    item = get_item(item_id, stac_host="earthsearch")

    parsed = _parse_stac_item(item)

    # Check that all expected fields are present
    assert "id" in parsed
    assert "date" in parsed
    assert "mgrs_tile" in parsed
    assert "cloud_cover" in parsed
    assert "nodata_percentage" in parsed
    assert "item" in parsed

    # nodata_percentage may be None if not available in the item
    assert parsed["nodata_percentage"] is None or isinstance(
        parsed["nodata_percentage"], (int, float)
    )
