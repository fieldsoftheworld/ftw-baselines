import hashlib
import os

import numpy as np
import scipy.spatial.distance


def compute_md5(file_path):
    """Compute the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    except FileNotFoundError:
        return None
    return hash_md5.hexdigest()


def validate_checksums(checksum_file, root_directory):
    """Validate checksums stored in a checksum file."""
    if not os.path.isfile(checksum_file):
        print(f"Checksum file not found: {checksum_file}")
        return

    with open(checksum_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        
        stored_checksum, file_path = parts
        file_path = os.path.join(root_directory, file_path)
        current_checksum = compute_md5(file_path)
        
        if current_checksum != stored_checksum:
            print("Checksum mismatch: {file_path}")
            return False
    return True


def euclidean_distance(pt1, pt2):
    """Computes the euclidean distance between two points.

    Args:
        pt1: tuple of floats
        pt2: tuple of floats

    Returns:
        distance between the two points
    """
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def featurize(polygon):
    """Computes morphological features of a shapely polygon.

    Features:
    - Area
    - Perimeter
    - Shape index -- perimeter / perimeter of square with equal area
    - Fractal dimension -- area / perimeter
    - Perimeter index -- perimeter / perimeter of circle with equal area
    - Detour -- perimeter of convex hull
    - Detour index -- detour / perimeter of equal area circle
    - Range -- longest distance between two vertices on the shape
    - Range index -- range / 2* diameter of equal area circle
    - Exchange -- shared area of the building footprint and the equal area circle with
        the same centroid
    - Exchange index -- exchange / shape's area
    - Cohesion -- average distance between 30 randomly selected interior points
    - Cohesion index -- cohesion / radius of equal area circle
    - Proximity -- average euclidean distance from all interor points to the centroid
    - Proximity index -- proximity / 2/3 * radius of equal area circle
    - Spin -- average squared euclidean distance between all interior points to the
        centroid
    - Spin index -- spin / 0.5 * squared radius of equal area circle
    - Length -- length of bounding box
    - Width -- width of bounding box
    - Length width ratio -- length / width
    - Vertices -- number of vertices

    Args:
        polygon: A shapely.geometry.Polygon to extract features from

    Returns:
        dictionary of features as described above
    """
    area = polygon.area
    perimeter = polygon.exterior.length

    coords = list(polygon.exterior.coords)

    minx, miny, maxx, maxy = polygon.bounds
    length = euclidean_distance((minx, miny), (maxx, miny))
    width = euclidean_distance((minx, miny), (minx, maxy))
    length, width = max(length, width), min(length, width)

    perimeter_of_equal_area_square = 4 * np.sqrt(area)

    radius_of_equal_area_circle = np.sqrt(area / np.pi)
    diameter_of_equal_area_circle = 2 * radius_of_equal_area_circle
    perimeter_of_equal_area_circle = 2 * np.pi * radius_of_equal_area_circle

    detour = polygon.convex_hull.exterior.length

    range_ = max(scipy.spatial.distance.pdist(coords[:-1]))

    temp_radius = np.sqrt(polygon.area / np.pi)
    exchange = (polygon & polygon.centroid.buffer(temp_radius)).area

    centroid_pt = list(polygon.centroid.coords)
    distances_to_centroids = scipy.spatial.distance.cdist(centroid_pt, coords)[0]
    proximity = np.mean(distances_to_centroids)

    spin = np.mean(distances_to_centroids**2)

    results = {
        "area": area,
        "perimeter": perimeter,
        "shape_index": perimeter / perimeter_of_equal_area_square,
        "fractal_dimension": area / perimeter,
        "perimeter_index": perimeter / perimeter_of_equal_area_circle,
        "detour": detour,
        "detour_index": detour / perimeter_of_equal_area_circle,
        "range": range_,
        "range_index": range_ / (2 * diameter_of_equal_area_circle),
        "exchange": exchange,
        "exchange_index": exchange / area,
        "proximity": proximity,
        "proximity_index": proximity / ((2 / 3) * radius_of_equal_area_circle),
        "spin": spin,
        "spin_index": spin / (0.5 * radius_of_equal_area_circle**2.0),
        "length": length,
        "width": width,
        "length_width_ratio": length / width,
        "vertices": len(coords) - 1,  # -1 because the first and last point are always the same
    }
    return results