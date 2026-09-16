import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from sitex.viz import city3d

LOCAL_EPSG = 32649


def _square(cx, cy, size):
    h = size / 2
    return Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)])


def test_robust_range_clips_outliers():
    # 998 bulk values spread narrowly (4-6) plus 2 extreme outliers: the 2nd/98th
    # percentiles fall inside the bulk's own spread, well short of the true min/max.
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.uniform(4.0, 6.0, 998), [0.0, 100.0]])
    lo, hi = city3d.robust_range(values, pct=(2, 98))
    assert 3.5 < lo < 6.5
    assert 3.5 < hi < 6.5
    assert hi - lo < 100.0  # narrower than the true min/max


def test_robust_range_handles_flat_data():
    values = np.full(10, 3.0)
    lo, hi = city3d.robust_range(values)
    assert lo == hi == 3.0


def test_sample_raster_at_centroids_reads_correct_value(tmp_path):
    values = np.full((10, 10), 42.0)
    values[0, 0] = 99.0
    transform = from_origin(0, 10, 1, 1)
    raster_path = tmp_path / "test.tif"
    with rasterio.open(
        raster_path, "w", driver="GTiff", height=10, width=10, count=1,
        dtype="float64", crs=f"EPSG:{LOCAL_EPSG}", transform=transform,
    ) as dst:
        dst.write(values, 1)

    buildings = gpd.GeoDataFrame({"uID": [0]}, geometry=[_square(5, 5, 1)], crs=f"EPSG:{LOCAL_EPSG}")
    vals, outside = city3d.sample_raster_at_centroids(buildings, raster_path)
    assert vals[0] == pytest.approx(42.0)
    assert not outside[0]


def test_sample_vector_at_centroids_joins_by_containment(tmp_path):
    buildings = gpd.GeoDataFrame({"uID": [0, 1]}, geometry=[_square(2, 2, 1), _square(50, 50, 1)], crs=f"EPSG:{LOCAL_EPSG}")
    grid = gpd.GeoDataFrame({"value": [7.5]}, geometry=[_square(2, 2, 10)], crs=f"EPSG:{LOCAL_EPSG}")
    grid_path = tmp_path / "grid.gpkg"
    grid.to_file(grid_path)

    vals, outside = city3d.sample_vector_at_centroids(buildings, grid_path, "value")
    assert vals[0] == pytest.approx(7.5)
    assert not outside[0]
    assert outside[1]  # building 1's centroid falls outside the grid polygon


def test_build_colored_mesh_produces_matching_face_colors():
    bldg_local = gpd.GeoDataFrame(
        {"height_est": [5.0]}, geometry=[_square(0, 0, 4)], crs=f"EPSG:{LOCAL_EPSG}"
    )
    base_z = np.array([100.0])
    values = np.array([0.5])
    outside = np.array([False])

    verts, faces, face_colors = city3d.build_colored_mesh(bldg_local, base_z, values, outside, "viridis", vmin=0, vmax=1)
    assert verts.shape[1] == 3
    assert faces.shape[1] == 3
    assert len(face_colors) == len(faces)
    assert face_colors.dtype == np.uint8


def test_build_colored_mesh_uses_no_data_color_when_outside():
    bldg_local = gpd.GeoDataFrame(
        {"height_est": [5.0]}, geometry=[_square(0, 0, 4)], crs=f"EPSG:{LOCAL_EPSG}"
    )
    base_z = np.array([100.0])
    values = np.array([np.nan])
    outside = np.array([True])

    _, _, face_colors = city3d.build_colored_mesh(
        bldg_local, base_z, values, outside, "viridis", vmin=0, vmax=1, no_data_color=(0.1, 0.2, 0.3)
    )
    expected = tuple(int(c * 255) for c in (0.1, 0.2, 0.3))
    assert tuple(face_colors[0]) == expected
