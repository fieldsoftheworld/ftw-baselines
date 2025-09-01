import logging
import os
import time
from typing import Tuple
from urllib.parse import urlparse

import dask.diagnostics.progress
import geopandas as gpd
import odc.stac
import pandas as pd
import planetary_computer as pc
import pystac
import pystac_client
import rioxarray  # seems unused but is needed
import xarray as xr
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
        print(f"\n=== SCENE SELECTION ===")
        print(f"Crop calendar dates: Start={start_dt.date()} (day {start_day}), End={end_dt.date()} (day {end_day})")
        print(f"STAC Host: {stac_host}")
        print(f"S2 Collection: {s2_collection} ({'EarthSearch only' if stac_host == 'earthsearch' else 'ignored for MSPC'})")
        print(f"Search parameters: cloud_cover_max={cloud_cover_max}%, buffer_days=±{buffer_days}")
        print(f"Bounding box: {bbox}")

    # search for +/- number of days the crop calendar indicated start and end days
    if verbose:
        print(f"\nSearching for EARLY SEASON scene around {start_dt.date()} (crop calendar start)")
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
        print(f"\nSearching for LATE SEASON scene around {end_dt.date()} (crop calendar end)")
    win_b = query_stac(
        bbox=bbox,
        date=end_dt,
        stac_host=stac_host,
        cloud_cover_max=cloud_cover_max,
        buffer_days=buffer_days,
        s2_collection=s2_collection,
        verbose=verbose,
    )

    if verbose:
        print(f"\n=== FINAL SCENE SELECTION ===")
        print(f"Early season: {win_a}")
        print(f"Late season:  {win_b}")

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
    Queries Sentinel-2 imagery hosted on planetary computer via pystac.
    sentinel 2 image id for start and end date within +/- number of days
    of crop calendar indicated dates, with the lowest percent cloud cover is returned.

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
    # Note: s2_collection parameter is ignored when using MSPC
    # MSPC always uses the default collection regardless of s2_collection value
    # Format dates in RFC3339 format for STAC API compliance
    start = (date - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%dT00:00:00Z")
    end = (date + pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%dT23:59:59Z")

    # Format as string
    date_range = f"{start}/{end}"

    host = MSPC_URL if stac_host == "mspc" else EARTHSEARCH_URL
    if verbose:
        print(f"  Connecting to STAC catalog: {host}")
    catalog = pystac_client.Client.open(host)

    # Use s2_collection only for EarthSearch, use default for MSPC
    if stac_host == "earthsearch":
        collection_name = S2_COLLECTIONS.get(s2_collection, COLLECTION_ID)
    else:
        collection_name = COLLECTION_ID  # MSPC always uses the default collection
    
    if verbose:
        print(f"  STAC Query:")
        print(f"    Host: {host}")
        print(f"    Collection: {collection_name}")
        print(f"    Date range: {date_range} (±{buffer_days} days from {date.date()})")
        print(f"    Bbox: {bbox}")
        print(f"    Cloud cover: <{cloud_cover_max}%")

        # Build the STAC API URL with query parameters
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        # URL encode the CQL filter for cloud cover
        import urllib.parse
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
        
        print(f"  \n  You can test this with STAC API URL:")
        print(f"    {stac_api_url}")

    
    search = catalog.search(
        collections=[collection_name],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": cloud_cover_max}},
    )

    items = search.item_collection()
    logger.info(f"Returned {len(items)} Items")
    if verbose:
        print(f"  Found {len(items)} scenes for date range {date_range}")

    if len(items) == 0:
        raise ValueError(
            f"No sentinel scenes within this area for {date_range} with less than {cloud_cover_max} percent cloud cover."
        )
    
    # Log all found scenes with their details
    if verbose and len(items) > 0:
        print(f"  Available scenes:")
        for item in items:
            cloud_cover = eo.ext(item).cloud_cover
            date_str = item.datetime.date() if item.datetime else "Unknown date"
            mgrs_tile = item.properties.get("grid:code") or item.properties.get("s2:mgrs_tile", "Unknown")
            print(f"    - {item.id}: {date_str}, MGRS: {mgrs_tile}, cloud cover: {cloud_cover:.2f}%")
    
    # check if aoi is approximately greater than 100 km x 100 km and spans multiple Sentinel 2 MGRS tiles
    if len(items) > 1 and (
        gpd.GeoDataFrame(geometry=[box(*bbox)], crs="EPSG:4326")
        .to_crs("EPSG:6933")
        .area[0]
        > 10000000000
    ):
        s2_tile_ids = []
        for item in items:
            code = item.properties.get("grid:code")
            if not code:
                code = item.properties.get("s2:mgrs_tile")
            if code:
                s2_tile_ids.append(code)
        if len(set(s2_tile_ids)) > 1:
            raise ValueError(
                f"Multiple MGRS tiles found: {set(s2_tile_ids)}. Please chose a smaller "
                f"search area, support coming soon for multiple Sentinel 2 scenes."
            )
    # sort by percent cloud cover
    least_cloudy_item = min(items, key=lambda item: eo.ext(item).cloud_cover)

    logger.info(
        f"Choosing {least_cloudy_item.id} from {least_cloudy_item.datetime.date()}"
        f" with {eo.ext(least_cloudy_item).cloud_cover}% cloud cover"
    )
    
    if verbose:
        print(f"  SELECTED: {least_cloudy_item.id} from {least_cloudy_item.datetime.date()}")
        print(f"    Cloud cover: {eo.ext(least_cloudy_item).cloud_cover:.2f}% (lowest among {len(items)} candidates)")
        print(f"    STAC URL: {least_cloudy_item.get_self_href()}")
    
    return least_cloudy_item.get_self_href()


def create_input(win_a, win_b, out, overwrite, stac_host, bbox=None, s2_collection="c1", verbose=False):
    """Main function for creating input for inference.
    
    Args:
        win_a: Window A identifier
        win_b: Window B identifier  
        out: Output path
        overwrite: Whether to overwrite existing files
        stac_host: STAC host to use ('mspc' or 'earthsearch')
        bbox: Optional bounding box
        s2_collection: Sentinel-2 collection to use (only applies to EarthSearch)
    """
    # Note: s2_collection parameter is ignored when using MSPC
    # MSPC always uses the default collection regardless of s2_collection value
    out = os.path.abspath(out)
    if os.path.exists(out) and not overwrite:
        print("Output file already exists, use -f to overwrite them. Exiting.")
        return

    # Ensure that the base directory exists
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # Load the items
    identifiers = [win_a, win_b]
    items = []
    timestamp = None
    version = 0
    
    if verbose:
        print(f"\n=== DOWNLOADING IMAGERY ===")
        print(f"Output file: {out}")
        print(f"Processing scenes:")
    
    for idx, i in enumerate(identifiers):
        item = get_item(i, stac_host=stac_host)
        items.append(item)
        
        if verbose:
            season = "Early season" if idx == 0 else "Late season"
            item_date = item.datetime or item.start_datetime or item.end_datetime
            date_str = item_date.date() if item_date else "Unknown date"
            mgrs_tile = item.properties.get("grid:code") or item.properties.get("s2:mgrs_tile", "Unknown")
            print(f"  {season}: {item.id}")
            print(f"    Date: {date_str}")
            print(f"    MGRS Tile: {mgrs_tile}")
            print(f"    Download URL: {i}")

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
    if not shapes[0].intersects(shapes[1]):
        print("The provided images do not intersect. Exiting.")
        return

    if stac_host == "mspc":
        bands = MSPC_BANDS_OF_INTEREST
    else:
        bands = BANDS_OF_INTEREST  # for EarthSearch
    
    if verbose:
        print(f"\n=== DATA PROCESSING ===")
        print(f"Bands: {bands}")
        print(f"Processing version: {version}")
        if bbox:
            print(f"Clipping to bbox: {bbox}")
        print(f"Loading and stacking imagery...")
    
    tic = time.time()
    data = odc.stac.load(
        [items[0], items[1]],
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
        print(f"Data shape: {data.shape}")
        print(f"Data bounds: {data.rio.bounds()}")
        print(f"Data CRS: {data.rio.crs}")
    
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
        print(f"\n=== SUMMARY ===")
        print(f"Successfully created: {out}")
        print(f"Processing time: {time.time() - tic:0.2f} seconds")
        print(f"Scenes used:")
        print(f"  Early: {identifiers[0]}")
        print(f"  Late:  {identifiers[1]}")
        print(f"Ready for inference!")
