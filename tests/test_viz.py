import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, Polygon

from sitex.viz import building_overlays as bo
from sitex.viz import heatmaps as hm
from sitex.viz import interactive_webmap as iw
from sitex.viz import raster_scenes as rs

LOCAL_EPSG = 32649


def _square(cx, cy, size):
    h = size / 2
    return Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)])


def _write_raster(path, values, transform=None):
    transform = transform or from_origin(0, values.shape[0], 1, 1)
    meta = {
        "driver": "GTiff", "height": values.shape[0], "width": values.shape[1], "count": 1,
        "dtype": "float64", "crs": f"EPSG:{LOCAL_EPSG}", "transform": transform,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(values, 1)
    return path


# ── raster_scenes ─────────────────────────────────────────────────────────────

def test_find_latest_uhi_scene_picks_newest(tmp_path):
    import time

    old_dir = tmp_path / "scene_old"
    new_dir = tmp_path / "scene_new"
    for d in (old_dir, new_dir):
        d.mkdir()
        for name in ("temperature_lst.tif", "ndvi.tif", "hei_50_50.tif"):
            (d / name).write_text("x")
    import os

    old_time = time.time() - 100
    os.utime(old_dir / "temperature_lst.tif", (old_time, old_time))

    result = rs.find_latest_uhi_scene(tmp_path)
    assert result == new_dir


def test_find_latest_uhi_scene_raises_when_incomplete(tmp_path):
    incomplete = tmp_path / "scene1"
    incomplete.mkdir()
    (incomplete / "temperature_lst.tif").write_text("x")  # missing ndvi.tif / hei file

    with pytest.raises(FileNotFoundError):
        rs.find_latest_uhi_scene(tmp_path)


def test_load_change_rasters_computes_symmetric_range(tmp_path):
    early = np.full((10, 10), 0.1)
    recent = np.full((10, 10), 0.4)
    recent[0, 0] = 0.9  # an outlier the 98th-percentile clip should tame
    for name, arr in [("ndvi_2017", early), ("ndvi_2026", recent), ("ndbi_2017", early), ("ndbi_2026", recent)]:
        _write_raster(tmp_path / f"{name}.tif", arr)

    result = rs.load_change_rasters(tmp_path, 2017, 2026)
    assert result["ndvi_change"]["arr"].shape == (10, 10)
    assert result["ndvi_change"]["vmin"] == -result["ndvi_change"]["vmax"]
    assert result["ndvi_change"]["vmax"] < 0.8  # the outlier shouldn't dominate the 98th pctile


# ── building_overlays ───────────────────────────────────────────────────────

@pytest.fixture
def raster_and_buildings():
    values = np.arange(100, dtype="float64").reshape(10, 10)
    transform = from_origin(0, 10, 1, 1)
    buildings = gpd.GeoDataFrame(geometry=[_square(2, 7, 1.5), _square(7, 3, 1.5)], crs=f"EPSG:{LOCAL_EPSG}")

    class Bounds:
        left, bottom, right, top = 0, 0, 10, 10

    return values, transform, Bounds(), buildings


def test_outline_mask_marks_only_boundary_not_interior(raster_and_buildings):
    values, transform, bounds, buildings = raster_and_buildings
    outline = bo.outline_mask(buildings, values.shape, transform, bounds)
    buildings_clip = buildings.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]
    from rasterio.features import rasterize

    filled = rasterize(
        buildings_clip.geometry, out_shape=values.shape, transform=transform,
        fill=0, default_value=1, dtype="uint8", all_touched=True,
    )
    outline_count = outline.sum()
    filled_count = filled.sum()
    # The outline of a filled polygon should touch fewer (or equal, for tiny shapes) pixels
    # than the filled interior — it is a boundary trace, not a fill.
    assert outline_count <= filled_count


def test_render_outline_overlay_returns_rgba(raster_and_buildings):
    values, transform, bounds, buildings = raster_and_buildings
    rgba = bo.render_outline_overlay(values, transform, bounds, buildings, cmap="viridis")
    assert rgba.shape == (10, 10, 4)
    assert rgba.dtype == np.uint8


# ── interactive_webmap ────────────────────────────────────────────────────────

def test_colorize_to_rgba_shape():
    arr = np.random.rand(5, 5)
    rgba = iw.colorize_to_rgba(arr, "viridis")
    assert rgba.shape == (5, 5, 4)


# ── heatmaps ─────────────────────────────────────────────────────────────────

def test_compute_kde_density_peaks_near_points():
    x = np.array([5.0, 5.1, 4.9, 5.0])
    y = np.array([5.0, 5.0, 5.1, 4.9])
    density = hm.compute_kde_density(x, y, xlim=(0, 10), ylim=(0, 10), grid_size=50)
    assert density.shape == (50, 50)
    # The density's peak location should be near the cluster's own centre, not at an edge.
    peak_row, peak_col = np.unravel_index(np.argmax(density), density.shape)
    assert 15 < peak_row < 35
    assert 15 < peak_col < 35


def test_compute_kde_density_empty_for_fewer_than_two_points():
    density = hm.compute_kde_density(np.array([5.0]), np.array([5.0]), xlim=(0, 10), ylim=(0, 10), grid_size=20)
    assert np.all(density == 0)


def test_rasterize_building_mask_nonzero_where_building_is():
    buildings = gpd.GeoDataFrame(geometry=[_square(5, 5, 2)], crs=f"EPSG:{LOCAL_EPSG}")
    mask = hm.rasterize_building_mask(buildings, xlim=(0, 10), ylim=(0, 10), grid_size=100)
    assert mask.sum() > 0
    assert mask.shape == (100, 100)


def test_render_gehl_heatmap_produces_opaque_rgba():
    points = gpd.GeoDataFrame(geometry=[Point(5, 5), Point(5.1, 5.1), Point(4.9, 4.9)], crs=f"EPSG:{LOCAL_EPSG}")
    buildings = gpd.GeoDataFrame(geometry=[_square(3, 3, 1)], crs=f"EPSG:{LOCAL_EPSG}")
    rgba = hm.render_gehl_heatmap(points, buildings, xlim=(0, 10), ylim=(0, 10), cmap="Blues", grid_size=50)
    assert rgba.shape == (50, 50, 4)
    assert (rgba[..., 3] == 255).all()  # background is always opaque black
