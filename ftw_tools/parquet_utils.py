"""Utility functions for writing GeoParquet files with fiboa metadata.

This module provides lightweight replacements for fiboa-cli's create_parquet
and features_to_dataframe functions to avoid the flatdict dependency.
"""

import json

import geopandas as gpd
import pyarrow as pa
import shapely.geometry


def features_to_dataframe(features, columns):
    """Convert GeoJSON-like features to a GeoDataFrame.

    Args:
        features: List of GeoJSON-like feature dictionaries with geometry and properties.
        columns: List of column names to include in the GeoDataFrame.

    Returns:
        GeoDataFrame with the features converted to rows.
    """
    rows = []
    for feature in features:
        feature_id = feature.get("id")
        geometry = (
            shapely.geometry.shape(feature["geometry"])
            if "geometry" in feature
            else None
        )
        row = {
            "id": feature_id,
            "geometry": geometry,
        }
        properties = feature.get("properties", {})
        row.update(properties)
        rows.append(row)

    return gpd.GeoDataFrame(rows, columns=columns, geometry="geometry", crs="EPSG:4326")


def create_parquet(data, columns, collection, output_file, config, compression=None):
    """Write a GeoDataFrame to a Parquet file with fiboa metadata.

    Args:
        data: GeoDataFrame to write.
        columns: List of column names to include in the output.
        collection: Dictionary with fiboa collection metadata.
        output_file: Path to the output Parquet file.
        config: Configuration dictionary (not used in this simplified version).
        compression: Compression algorithm to use (default: 'zstd').
    """
    if compression is None:
        compression = "zstd"

    # Write to Parquet using geopandas built-in method
    # which handles geometry serialization properly
    data[columns].to_parquet(
        output_file,
        compression=compression,
        index=False,
    )

    # Now reopen and add the fiboa metadata
    # This is needed to add custom metadata to the parquet file
    import pyarrow.parquet as pq

    parquet_file = pq.read_table(output_file)
    existing_metadata = parquet_file.schema.metadata or {}
    existing_metadata[b"fiboa"] = json.dumps(collection).encode("utf-8")
    parquet_file = parquet_file.replace_schema_metadata(existing_metadata)
    pq.write_table(parquet_file, output_file, compression=compression)
