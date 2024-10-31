# Microsoft Planetary Computer API URL + collection id + bands for inference download command
MSPC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION_ID = "sentinel-2-l2a"
BANDS_OF_INTEREST = ["B04", "B03", "B02", "B08"]

# Supported file formats for the inference polygon command
SUPPORTED_POLY_FORMATS_TXT = "Available file extensions: .parquet (GeoParquet, fiboa-compliant), .fgb (FlatGeoBuf), .gpkg (GeoPackage), .geojson and .json (GeoJSON)"

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
    "vietnam"
]