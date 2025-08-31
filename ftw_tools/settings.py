# Collection id + bands for inference download command

AWS_SENTINEL_URL = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"
COLLECTION_ID = "sentinel-2-c1-l2a"
BANDS_OF_INTEREST = ["red", "green", "blue", "nir"]

MSPC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
MSPC_BANDS_OF_INTEREST = ["B04", "B03", "B02", "B08"]

EARTHSEARCH_URL = "https://earth-search.aws.element84.com/v1"

# Supported file formats for the inference polygon command
SUPPORTED_POLY_FORMATS_TXT = "Available file extensions: .parquet (GeoParquet, fiboa-compliant), .fgb (FlatGeoBuf), .gpkg (GeoPackage), .geojson / .json / .ndjson (GeoJSON)"

# List of all available countries
ALL_COUNTRIES = [
    "austria",
    "belgium",
    "brazil",
    "cambodia",
    "corsica",
    "croatia",
    "denmark",
    "estonia",
    "finland",
    "france",
    "germany",
    "india",
    "kenya",
    "latvia",
    "lithuania",
    "luxembourg",
    "netherlands",
    "portugal",
    "rwanda",
    "slovakia",
    "slovenia",
    "south_africa",
    "spain",
    "sweden",
    "vietnam",
]

TEMPORAL_OPTIONS = [
    "stacked",
    "windowA",
    "windowB",
    "median",
    "rgb",
]

LULC_COLLECTIONS = [
    "io-lulc-annual-v02",
    "esa-worldcover",
]
