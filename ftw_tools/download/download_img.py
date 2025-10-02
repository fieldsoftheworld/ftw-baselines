import logging
import os
import time
import urllib.parse
from typing import Tuple
from urllib.parse import urlparse

import dask.diagnostics.progress
import geopandas as gpd
import odc.stac
import pandas as pd
import planetary_computer as pc
import pystac
import pystac_client
import rioxarray  # noqa: F401
import xarray as xr  # noqa: F401
from pystac.extensions.eo import EOExtension as eo
from shapely.geometry import box, shape
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ftw_tools.settings import (
    AWS_SENTINEL_URL,
    BANDS_OF_INTEREST,
    COLLECTION_ID,
    EARTHSEARCH_URL,
    MSPC_BANDS_OF_INTEREST,
    MSPC_URL,
    S2_COLLECTIONS,
)
from ftw_tools.utils import get_harvest_integer_from_bbox, harvest_to_datetime

logger = logging.getLogger()


def _get_item_from_mspc(id: str) -> pystac.Item:
    """Get a STAC item from Microsoft Planetary Computer.

    Args:
        id (str): The ID or URL of the STAC item.

    Returns:
        pystac.Item: The retrieved and signed STAC item.
    """
    if "/" not in id:
        uri = MSPC_URL + "/collections/" + COLLECTION_ID + "/items/" + id
    else:
        uri = id

    item = pystac.Item.from_file(uri)

    if uri.startswith(MSPC_URL):
        item = pc.sign(item)

    return item


def _get_item_from_earthsearch(id: str) -> pystac.Item:
    """Get a STAC item from EarthSearch.

    Args:
        id (str): The ID or URL of the STAC item.

    Returns:
        pystac.Item: The retrieved STAC item.
    """
    if "s3://" in id:
        parsed = urlparse(id)
        path = parsed.path.strip("/")  # remove leading slash
        item_id = os.path.basename(path)

        uri = f"{AWS_SENTINEL_URL}/{path}/{item_id}.json"

    elif "/" not in id:
        # Convert ID into full URL
        parts = id.split("_")
        mgrs_tile = parts[1]
        date_str = parts[2]

        utm_zone = mgrs_tile[:2]
        lat_band = mgrs_tile[2]
        grid_square = mgrs_tile[3:]

        year = date_str[:4]
        month = date_str[4:6]
        # Remove leading zero from month to get valid S3 path
        month = str(int(month))

        uri = (
            f"{AWS_SENTINEL_URL}/"
            f"sentinel-s2-l2a-cogs/{utm_zone}/{lat_band}/{grid_square}/"
            f"{year}/{month}/{id}/{id}.json"
        )
    else:
        uri = id

    item = pystac.Item.from_file(uri)

    return item


@retry(wait=wait_random_exponential(max=3), stop=stop_after_attempt(2))
def get_item(id: str, stac_host: str) -> pystac.Item:
    """Get a STAC item from a given ID or URL.
    Args:
        id (str): The ID or URL of the STAC item.
        stac_host (str): The STAC host to use for item retrieval either 'mspc' or 'earthsearch'.
    Returns:
        pystac.Item: The retrieved STAC item.
    """
    if stac_host == "mspc":
        return _get_item_from_mspc(id)
    elif stac_host == "earthsearch":
        return _get_item_from_earthsearch(id)
    else:
        raise ValueError(
            f"Unsupported STAC host: {stac_host}. Use 'mspc' or 'earthsearch'."
        )


def scene_selection(
    bbox: list[float],
    year: int,
    stac_host: str,
    cloud_cover_max: int = 20,
    buffer_days: int = 14,
    s2_collection: str = "c1",
    verbose: bool = False,
) -> Tuple[str, str]:
    """
    Returns sentinel 2 image S3 URL for start and end date within +/- number of days
    of crop calendar indicated dates. If there are multiple images within the date
    range, lowest cloud cover will be returned.

    Args:
        bbox (list[int]): Bounding box in [minx, miny, maxx, maxy] format.
        year (int): Year for filtering scenes.
        cloud_cover_max (int, optional): Maximum allowed cloud cover percentage. Defaults to 20.
        buffer_days (int, optional): Number of days to buffer the date for querying to help balance
            decreasing cloud cover and selecting a date near the crop calendar indicated date.
            Defaults to 14.
        s2_collection (str, optional): Sentinel-2 collection to use (only applies to EarthSearch). Defaults to "c1".

    Returns:
        tuple: Sentinel2 image ids to be used as input into the 2 image crop model
    """
    # Note: s2_collection parameter is ignored when using MSPC
    # MSPC always uses the default collection regardless of s2_collection value
    # get crop calendar days
    start_day, end_day = get_harvest_integer_from_bbox(bbox=bbox)

    start_dt = harvest_to_datetime(harvest_day=start_day, year=year)
    end_dt = harvest_to_datetime(
        harvest_day=end_day, year=year + 1 if end_day < start_day else year
    )  # to account for southern hemisphere harvest

    if verbose:
        print(
            f"\n=== SCENE SELECTION ===\n"
            f"Crop calendar dates: Start={start_dt.date()} (day {start_day}), End={end_dt.date()} (day {end_day})\n"
            f"STAC Host: {stac_host}\n"
            f"S2 Collection: {s2_collection} ({'EarthSearch only' if stac_host == 'earthsearch' else 'ignored for MSPC'})\n"
            f"Search parameters: cloud_cover_max={cloud_cover_max}%, buffer_days=±{buffer_days}\n"
            f"Bounding box: {bbox}\n",
            # search for +/- number of days the crop calendar indicated start and end days
            f"Searching for EARLY SEASON scene around {start_dt.date()} (crop calendar start)",
        )

    win_a = query_stac(
        bbox=bbox,
        date=start_dt,
        stac_host=stac_host,
        cloud_cover_max=cloud_cover_max,
        buffer_days=buffer_days,
        s2_collection=s2_collection,
        verbose=verbose,
    )
    if verbose:
        print(
            f"\nSearching for LATE SEASON scene around {end_dt.date()} (crop calendar end)"
        )
    win_b = query_stac(
        bbox=bbox,
        date=end_dt,
        stac_host=stac_host,
        cloud_cover_max=cloud_cover_max,
        buffer_days=buffer_days,
        s2_collection=s2_collection,
        verbose=verbose,
    )

    return (win_a, win_b)


def query_stac(
    bbox: list[int],
    date: pd.Timestamp,
    stac_host: str,
    cloud_cover_max: int = 20,
    buffer_days=14,
    s2_collection: str = "c1",
    verbose: bool = False,
) -> str:
    """
    Queries Sentinel-2 imagery hosted on STAC hosts via pystac.
    Returns the sentinel 2 image id for start and end date within +/- number of days
    of crop calendar indicated dates, with the lowest percent cloud cover.

    Args:
        bbox: Bounding box in [minx, miny, maxx, maxy] format.
        date: crop calendar indicated date
        cloud_cover_max: threshold for maximum percent cloud cover.
        buffer_days: Number of days to buffer the date for querying.
        stac_host: The STAC host to use ('mspc' or 'earthsearch').
        s2_collection: Sentinel-2 collection to use (only applies to EarthSearch).

    Returns:
        Sentinel-2 image S3 URL.
    """
    _validate_query_date(date, buffer_days)

    if stac_host == "earthsearch":
        items, date_range = _query_earthsearch(
            bbox=bbox,
            date=date,
            cloud_cover_max=cloud_cover_max,
            buffer_days=buffer_days,
            s2_collection=s2_collection,
            verbose=verbose,
        )
    elif stac_host == "mspc":
        items, date_range = _query_microsoft_pc(
            bbox=bbox,
            date=date,
            cloud_cover_max=cloud_cover_max,
            buffer_days=buffer_days,
            verbose=verbose,
        )
    else:
        raise ValueError(
            f"Unsupported STAC host: {stac_host}. Use 'mspc' or 'earthsearch'."
        )

    logger.info(f"Returned {len(items)} Items")
    if verbose:
        print(f"  Found {len(items)} scenes for date range {date_range}")

    if len(items) == 0:
        raise ValueError(
            f"No sentinel scenes within area for {date_range} with less than {cloud_cover_max} percent cloud cover."
        )

    # Log all found scenes with their details using uniform parsing
    if verbose and len(items) > 0:
        print("  Available scenes:")
        for item in items:
            parsed_item = _parse_stac_item(item)
            print(
                f"\t- {parsed_item['id']}: {parsed_item['date']}, "
                f"MGRS: {parsed_item['mgrs_tile']}, "
                f"cloud cover: {parsed_item['cloud_cover']:.2f}%"
            )

    # Check if AOI is approximately greater than 100 km x 100 km and spans multiple Sentinel 2 MGRS tiles
    if len(items) > 1 and (
        gpd.GeoDataFrame(geometry=[box(*bbox)], crs="EPSG:4326")
        .to_crs("EPSG:6933")
        .area[0]
        > 10000000000
    ):
        s2_tile_ids = []
        for item in items:
            parsed_item = _parse_stac_item(item)
            if parsed_item["mgrs_tile"] != "Unknown":
                s2_tile_ids.append(parsed_item["mgrs_tile"])
        if len(set(s2_tile_ids)) > 1:
            raise ValueError(
                f"Multiple MGRS tiles found: {set(s2_tile_ids)}. Please chose a smaller "
                f"search area, support coming soon for multiple Sentinel 2 scenes."
            )

    # Sort by percent cloud cover
    least_cloudy_item = min(items, key=lambda item: eo.ext(item).cloud_cover)
    parsed_selected = _parse_stac_item(least_cloudy_item)

    logger.info(
        f"Choosing {parsed_selected['id']} from {parsed_selected['date']}"
        f" with {parsed_selected['cloud_cover']}% cloud cover"
    )

    if verbose:
        print(
            f"  SELECTED: {parsed_selected['id']} from {parsed_selected['date']}\n"
            f"    Cloud cover: {parsed_selected['cloud_cover']:.2f}% (lowest among {len(items)} candidates)\n"
            f"    STAC URL: {least_cloudy_item.get_self_href()}"
        )

    return least_cloudy_item.get_self_href()


def _log_stac_query(
    host: str,
    collection_name: str,
    date_range: str,
    buffer_days: int,
    date: pd.Timestamp,
    bbox: list,
    cloud_cover_max: int,
) -> None:
    """Log STAC query information and build API URL for verbose output.

    Args:
        host: STAC host URL
        collection_name: Collection name to query
        date_range: Formatted date range string
        buffer_days: Number of buffer days
        date: Center date for the query
        bbox: Bounding box coordinates
        cloud_cover_max: Maximum cloud cover percentage
    """
    print(
        f"  Connecting to STAC catalog: {host}\n"
        f"  STAC Query:\n"
        f"    Host: {host}\n"
        f"    Collection: {collection_name}\n"
        f"    Date range: {date_range} (±{buffer_days} days from {date.date()})\n"
        f"    Bbox: {bbox}\n"
        f"    Cloud cover: <{cloud_cover_max}%"
    )

    # Build the STAC API URL with query parameters
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    cql_filter = f'"eo:cloud_cover" < {cloud_cover_max}'
    encoded_filter = urllib.parse.quote(cql_filter)

    stac_api_url = (
        f"{host}/search?"
        f"collections={collection_name}&"
        f"bbox={bbox_str}&"
        f"datetime={date_range}&"
        f"filter={encoded_filter}&"
        f"filter-lang=cql2-text"
    )

    print(f"\nYou can test this with STAC API URL:\n  {stac_api_url}")


def _format_date_range(date: pd.Timestamp, buffer_days: int) -> str:
    """Format date range for STAC API queries.

    Args:
        date: Center date for the range
        buffer_days: Number of days to buffer around the center date

    Returns:
        Date range string in RFC3339 format for STAC API
    """
    start = (date - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%dT00:00:00Z")
    end = (date + pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%dT23:59:59Z")
    return f"{start}/{end}"


def _validate_query_date(date: pd.Timestamp, buffer_days: int) -> None:
    """Validate that query dates are not in the future.

    Args:
        date: Center date to validate
        buffer_days: Buffer days to check end date

    Raises:
        ValueError: If any part of the date range extends into the future
    """
    start = (date - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    end = (date + pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")

    today = pd.Timestamp.now().normalize()
    if pd.Timestamp(start) > today or pd.Timestamp(end) > today:
        raise ValueError(
            f"Crop calendar harvest date {date} and buffer days {start}, "
            f"{end} can't be in the future, try using an earlier calendar year"
        )


def _query_earthsearch(
    bbox: list[int],
    date: pd.Timestamp,
    cloud_cover_max: int,
    buffer_days: int,
    s2_collection: str,
    verbose: bool = False,
) -> tuple[pystac.ItemCollection, str]:
    """Query EarthSearch for Sentinel-2 imagery.

    Args:
        bbox: Bounding box in [minx, miny, maxx, maxy] format.
        date: Center date for the query.
        cloud_cover_max: Maximum allowed cloud cover percentage.
        buffer_days: Number of days to buffer around the center date.
        s2_collection: Sentinel-2 collection identifier to use.
        verbose: Whether to print verbose output.

    Returns:
        Tuple of (ItemCollection with matching scenes, formatted date range string).
    """
    date_range = _format_date_range(date, buffer_days)

    host = EARTHSEARCH_URL
    collection_name = S2_COLLECTIONS.get(s2_collection, COLLECTION_ID)

    if verbose:
        _log_stac_query(
            host, collection_name, date_range, buffer_days, date, bbox, cloud_cover_max
        )

    catalog = pystac_client.Client.open(host)
    search = catalog.search(
        collections=[collection_name],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": cloud_cover_max}},
    )

    return search.item_collection(), date_range


def _query_microsoft_pc(
    bbox: list[int],
    date: pd.Timestamp,
    cloud_cover_max: int,
    buffer_days: int,
    verbose: bool = False,
) -> tuple[pystac.ItemCollection, str]:
    """Query Microsoft Planetary Computer for Sentinel-2 imagery.

    Args:
        bbox: Bounding box in [minx, miny, maxx, maxy] format.
        date: Center date for the query.
        cloud_cover_max: Maximum allowed cloud cover percentage.
        buffer_days: Number of days to buffer around the center date.
        verbose: Whether to print verbose output.

    Returns:
        Tuple of (ItemCollection with matching scenes, formatted date range string).
    """
    date_range = _format_date_range(date, buffer_days)

    host = MSPC_URL
    collection_name = COLLECTION_ID  # MSPC always uses the default collection

    if verbose:
        _log_stac_query(
            host, collection_name, date_range, buffer_days, date, bbox, cloud_cover_max
        )

    catalog = pystac_client.Client.open(host)
    search = catalog.search(
        collections=[collection_name],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": cloud_cover_max}},
    )

    return search.item_collection(), date_range


def _parse_stac_item(item: pystac.Item) -> dict:
    """Parse a STAC item into a uniform representation.

    Args:
        item: STAC item to parse.

    Returns:
        Dictionary containing parsed item information with keys:
        - id: Item identifier
        - date: Item date
        - mgrs_tile: MGRS tile code
        - cloud_cover: Cloud cover percentage
        - item: Original STAC item object
    """
    cloud_cover = eo.ext(item).cloud_cover
    date_str = item.datetime.date() if item.datetime else "Unknown date"
    mgrs_tile = item.properties.get("grid:code") or item.properties.get(
        "s2:mgrs_tile", "Unknown"
    )

    return {
        "id": item.id,
        "date": date_str,
        "mgrs_tile": mgrs_tile,
        "cloud_cover": cloud_cover,
        "item": item,
    }


def create_input(
    win_a,
    win_b,
    out,
    overwrite,
    stac_host,
    bbox=None,
    s2_collection="c1",
    verbose=False,
):
    """Main function for creating input for inference.

    Args:
        win_a: Window A identifier
        win_b: Window B identifier
        out: Output path
        overwrite: Whether to overwrite existing files
        stac_host: STAC host to use ('mspc' or 'earthsearch')
        bbox: Optional bounding box
        s2_collection: Sentinel-2 collection to use (only applies to EarthSearch)
        verbose: Whether to print verbose output
    """
    out = os.path.abspath(out)
    if os.path.exists(out) and not overwrite:
        print("Output file already exists, use -f to overwrite them. Exiting.")
        return

    os.makedirs(os.path.dirname(out), exist_ok=True)

    # Load the items
    identifiers = [win_a]
    if win_b is not None:
        identifiers.append(win_b)
    items = []
    timestamp = None
    version = 0

    if verbose:
        print(f"\n=== DOWNLOADING IMAGERY ===\nOutput file: {out}\nProcessing scenes:")

    for idx, i in enumerate(identifiers):
        item = get_item(i, stac_host=stac_host)
        items.append(item)

        if verbose:
            season = "Early season" if idx == 0 else "Late season"
            parsed_item = _parse_stac_item(item)
            print(
                f"  {season}: {parsed_item['id']}\n"
                f"    Date: {parsed_item['date']}\n"
                f"    MGRS Tile: {parsed_item['mgrs_tile']}\n"
                f"    Download URL: {i}"
            )

        datetime = item.datetime or item.start_datetime or item.end_datetime
        if datetime and (not timestamp or datetime > timestamp):
            timestamp = datetime

        proc_version = item.properties.get(
            "processing:version", item.properties.get("s2:processing_baseline", 0)
        )
        try:
            proc_version = float(proc_version)
        except TypeError:
            proc_version = 0
        if proc_version > 0:
            if version > 0 and version != proc_version:
                print(
                    f"Warning: Processing version of imagery differs: {version} vs {proc_version}. Continuing anyway."
                )
            version = proc_version

    shapes = [shape(item.geometry) for item in items]
    # Ensure the images intersect
    if len(shapes) > 1:
        if not shapes[0].intersects(shapes[1]):
            print("The provided images do not intersect. Exiting.")
            return

    if stac_host == "mspc":
        bands = MSPC_BANDS_OF_INTEREST
    else:
        bands = BANDS_OF_INTEREST  # for EarthSearch

    if verbose:
        bbox_info = f"\nClipping to bbox: {bbox}" if bbox else ""
        print(
            f"\n=== DATA PROCESSING ===\n"
            f"Bands: {bands}\n"
            f"Processing version: {version}{bbox_info}\n"
            f"Loading and stacking imagery..."
        )

    tic = time.time()
    data = odc.stac.load(
        items,
        bands=bands,
        dtype="uint16",
        resampling="bilinear",
        bbox=bbox,
        chunks={"x": "auto", "y": "auto"},
    )

    data = (
        data.to_array(dim="band")
        .stack(bands=("time", "band"))
        .drop_vars("band")
        .transpose("bands", "y", "x")
    )

    if version < 3 or version >= 4:
        print(
            f"Processing version {version} unknown or untested (< 3.0 or >= 4.0). Inference quality might decrease."
        )
    if version >= 4:
        print(
            f"Rescaling data to processing version 3.0 from processing version {version}."
        )
        data = (data.astype("int32") - 1000).clip(min=0).astype("uint16")

    if verbose:
        print(
            f"Data shape: {data.shape}\n"
            f"Data bounds: {data.rio.bounds()}\n"
            f"Data CRS: {data.rio.crs}"
        )

    print("Writing output")
    with dask.diagnostics.progress.ProgressBar():
        data.rio.to_raster(
            out,
            driver="GTiff",
            compress="deflate",
            dtype="uint16",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            tags={"TIFFTAG_DATETIME": timestamp.strftime("%Y:%m:%d %H:%M:%S")},
        )

    print(f"Finished merging and writing output in {time.time() - tic:0.2f} seconds")

    if verbose:
        print(
            f"\n=== SUMMARY ===\n"
            f"Successfully created: {out}\n"
            f"Processing time: {time.time() - tic:0.2f} seconds\n"
            f"Scenes used:\n"
            f"  Early: {identifiers[0]}\n"
            f"  Late:  {identifiers[1]}\n"
            f"Ready for inference!"
        )
