import pystac
import pytest

from ftw_tools.download.download_img import get_item


def test_get_item_from_s3_url():
    s3_url = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/15/T/VG/2021/12/S2A_15TVG_20211225_0_L2A"
    item = get_item(s3_url)
    assert type(item) is pystac.Item


def test_get_item_from_id():
    item_id = "S2A_15TVG_20211225_0_L2A"
    item = get_item(item_id)
    assert type(item) is pystac.Item
    assert item.id == item_id


def test_get_item_from_id_single_digit_month():
    item_id = "S2B_33UUP_20210925_0_L2A"
    item = get_item(item_id)
    assert type(item) is pystac.Item
    assert item.id == item_id


def test_get_item_from_s3_url_single_digit_month():
    s3_url = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/33/U/UP/2021/9/S2B_33UUP_20210925_0_L2A"
    item = get_item(s3_url)
    assert type(item) is pystac.Item
    assert item.id == "S2B_33UUP_20210925_0_L2A"
