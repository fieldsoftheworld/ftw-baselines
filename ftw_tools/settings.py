# Collection id + bands for inference download command

AWS_SENTINEL_URL = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"
COLLECTION_ID = "sentinel-2-l2a"
BANDS_OF_INTEREST = ["red", "green", "blue", "nir"]

MSPC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
MSPC_BANDS_OF_INTEREST = ["B04", "B03", "B02", "B08"]

# Supported file formats for the inference polygon command
SUPPORTED_POLY_FORMATS_TXT = "Available file extensions: .parquet (GeoParquet, fiboa-compliant), .fgb (FlatGeoBuf), .gpkg (GeoPackage), .geojson / .json / .ndjson (GeoJSON)"

# List of all available countries
ALL_COUNTRIES = [
    "belgium",
    "cambodia",
    "croatia",
    "estonia",
    "portugal",
    "slovakia",
    "south_africa",
    "sweden",
    "austria",
    "brazil",
    "corsica",
    "denmark",
    "france",
    "india",
    "latvia",
    "luxembourg",
    "finland",
    "germany",
    "kenya",
    "lithuania",
    "netherlands",
    "rwanda",
    "slovenia",
    "spain",
    "vietnam",
]
