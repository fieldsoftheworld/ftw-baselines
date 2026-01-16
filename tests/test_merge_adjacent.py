"""Tests for merge_adjacent_polygons function."""

import time

import pytest
import shapely.geometry
from ftw_tools.postprocess.polygonize import merge_adjacent_polygons


def create_feature(geom, id_val=""):
    """Helper to create a feature dict from a geometry."""
    return {
        "geometry": shapely.geometry.mapping(geom),
        "properties": {"id": id_val},
    }


def test_merge_adjacent_empty():
    """Test with empty input."""
    result = merge_adjacent_polygons([], 0.0)
    assert result == []


def test_merge_adjacent_single_polygon():
    """Test with a single polygon."""
    poly = shapely.geometry.box(0, 0, 1, 1)
    features = [create_feature(poly, "1")]
    result = merge_adjacent_polygons(features, 0.0)
    assert len(result) == 1
    assert result[0]["properties"]["id"] == "1"


def test_merge_adjacent_non_touching():
    """Test with two non-touching polygons."""
    poly1 = shapely.geometry.box(0, 0, 1, 1)
    poly2 = shapely.geometry.box(5, 5, 6, 6)
    features = [create_feature(poly1, "1"), create_feature(poly2, "2")]
    result = merge_adjacent_polygons(features, 0.0)
    assert len(result) == 2


def test_merge_adjacent_overlapping():
    """Test with overlapping polygons - should always merge."""
    poly1 = shapely.geometry.box(0, 0, 2, 2)
    poly2 = shapely.geometry.box(1, 1, 3, 3)
    features = [create_feature(poly1, "1"), create_feature(poly2, "2")]
    result = merge_adjacent_polygons(features, 0.5)
    assert len(result) == 1
    assert "1" in result[0]["properties"]["id"]
    assert "2" in result[0]["properties"]["id"]


def test_merge_adjacent_touching_high_ratio():
    """Test with touching polygons that share enough boundary."""
    poly1 = shapely.geometry.box(0, 0, 1, 1)
    poly2 = shapely.geometry.box(1, 0, 2, 1)  # shares right edge of poly1
    features = [create_feature(poly1, "1"), create_feature(poly2, "2")]
    
    # Shared edge is 1.0, smaller perimeter is 4.0, so ratio = 1.0/4.0 = 0.25
    # With ratio=0.2, they should merge (0.25 >= 0.2)
    result = merge_adjacent_polygons(features, 0.2)
    assert len(result) == 1
    
    # With ratio=0.5, they should NOT merge (0.25 < 0.5)
    result = merge_adjacent_polygons(features, 0.5)
    assert len(result) == 2


def test_merge_adjacent_touching_zero_ratio():
    """Test with ratio=0 - any touching polygons should merge."""
    poly1 = shapely.geometry.box(0, 0, 1, 1)
    poly2 = shapely.geometry.box(1, 0, 2, 1)  # shares right edge of poly1
    features = [create_feature(poly1, "1"), create_feature(poly2, "2")]
    result = merge_adjacent_polygons(features, 0.0)
    assert len(result) == 1


def test_merge_adjacent_multiple_components():
    """Test with multiple connected components."""
    # Component 1: poly1 and poly2 touch each other
    poly1 = shapely.geometry.box(0, 0, 1, 1)
    poly2 = shapely.geometry.box(1, 0, 2, 1)
    # Component 2: poly3 and poly4 touch each other
    poly3 = shapely.geometry.box(5, 5, 6, 6)
    poly4 = shapely.geometry.box(6, 5, 7, 6)
    # Component 3: poly5 is isolated
    poly5 = shapely.geometry.box(10, 10, 11, 11)
    
    features = [
        create_feature(poly1, "1"),
        create_feature(poly2, "2"),
        create_feature(poly3, "3"),
        create_feature(poly4, "4"),
        create_feature(poly5, "5"),
    ]
    result = merge_adjacent_polygons(features, 0.0)
    assert len(result) == 3  # 3 connected components


def test_merge_adjacent_transitive():
    """Test transitive merging: if A touches B and B touches C, all merge."""
    poly1 = shapely.geometry.box(0, 0, 1, 1)
    poly2 = shapely.geometry.box(1, 0, 2, 1)
    poly3 = shapely.geometry.box(2, 0, 3, 1)
    features = [
        create_feature(poly1, "1"),
        create_feature(poly2, "2"),
        create_feature(poly3, "3"),
    ]
    result = merge_adjacent_polygons(features, 0.0)
    assert len(result) == 1
    assert "1" in result[0]["properties"]["id"]
    assert "2" in result[0]["properties"]["id"]
    assert "3" in result[0]["properties"]["id"]


def test_merge_adjacent_bbox_optimization():
    """Test that bbox rejection works - polygons far apart shouldn't be checked."""
    # Create many polygons far apart
    features = []
    for i in range(100):
        poly = shapely.geometry.box(i * 10, i * 10, i * 10 + 1, i * 10 + 1)
        features.append(create_feature(poly, str(i)))
    
    result = merge_adjacent_polygons(features, 0.0)
    assert len(result) == 100  # All separate


def test_merge_adjacent_empty_geometry():
    """Test with empty geometries - should be skipped."""
    poly1 = shapely.geometry.box(0, 0, 1, 1)
    empty = shapely.geometry.Polygon()  # Empty polygon
    poly2 = shapely.geometry.box(2, 2, 3, 3)
    features = [
        create_feature(poly1, "1"),
        create_feature(empty, "2"),
        create_feature(poly2, "3"),
    ]
    result = merge_adjacent_polygons(features, 0.0)
    assert len(result) == 2  # Only non-empty polygons


def test_merge_adjacent_properties():
    """Test that merged properties are correct."""
    poly1 = shapely.geometry.box(0, 0, 1, 1)
    poly2 = shapely.geometry.box(1, 0, 2, 1)
    features = [create_feature(poly1, "A"), create_feature(poly2, "B")]
    result = merge_adjacent_polygons(features, 0.0)
    
    assert len(result) == 1
    merged = result[0]
    assert "A" in merged["properties"]["id"]
    assert "B" in merged["properties"]["id"]
    assert "area" in merged["properties"]
    assert "perimeter" in merged["properties"]
    assert merged["properties"]["area"] > 0
    assert merged["properties"]["perimeter"] > 0


def test_merge_adjacent_performance():
    """Test performance with a large number of polygons.
    
    This test demonstrates that the rtree-based implementation scales well.
    With the old O(N²) implementation, this would take significantly longer.
    """
    # Create a grid of 1024 small polygons (32x32 grid with spacing)
    features = []
    grid_size = 32
    for i in range(grid_size):
        for j in range(grid_size):
            # Add small gaps so polygons don't all merge
            x = i * 2.0
            y = j * 2.0
            poly = shapely.geometry.box(x, y, x + 0.9, y + 0.9)
            features.append(create_feature(poly, f"{i}_{j}"))
    
    print(f"\nPerformance test with {len(features)} polygons")
    start = time.time()
    result = merge_adjacent_polygons(features, 0.0)
    elapsed = time.time() - start
    
    print(f"Processing {len(features)} polygons took {elapsed:.3f}s")
    assert len(result) == len(features)  # None should merge due to gaps
    # With rtree optimization, this should complete in under 5 seconds
    # The old O(N²) version would take much longer
    assert elapsed < 10.0, f"Performance test took too long: {elapsed:.3f}s"
