import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Polygon

from sitex.arch.morphology import (
    classify_building_type,
    compute_intersection_density,
    compute_morphology,
    compute_tessellation,
    detect_blocks,
    estimate_floors_height,
    export_enriched_buildings,
    extrude_footprint,
    join_block_to_buildings,
    join_landuse,
    load_buildings,
    load_streets_main_component,
    plot_building_types,
    select_export_columns,
    shift_to_local_origin,
)

LOCAL_EPSG = 32649


def _square(cx, cy, size):
    h = size / 2
    return Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)])


@pytest.fixture
def buildings_gpkg(tmp_path):
    # A small house, a mid-size building, and a large warehouse-shaped footprint.
    geoms = [_square(0, 0, 6), _square(50, 0, 15), _square(100, 0, 40)]
    gdf = gpd.GeoDataFrame(
        {"id": ["a", "b", "c"], "height": [np.nan] * 3, "num_floors": [np.nan] * 3, "class": [None] * 3},
        geometry=geoms,
        crs=f"EPSG:{LOCAL_EPSG}",
    )
    path = tmp_path / "buildings.gpkg"
    gdf.to_file(path)
    return path


@pytest.fixture
def streets_gpkg(tmp_path):
    lines = [
        LineString([(-20, 0), (60, 0)]),
        LineString([(60, 0), (130, 0)]),
        LineString([(60, 0), (60, 60)]),
        LineString([(60, 60), (-20, 60)]),
        LineString([(-20, 60), (-20, 0)]),
    ]
    gdf = gpd.GeoDataFrame(geometry=lines, crs=f"EPSG:{LOCAL_EPSG}")
    path = tmp_path / "streets.gpkg"
    gdf.to_file(path)
    return path


def test_load_buildings_assigns_uid_and_projects(buildings_gpkg):
    bldg = load_buildings(buildings_gpkg, LOCAL_EPSG)
    assert list(bldg["uID"]) == [0, 1, 2]
    assert bldg.crs.to_epsg() == LOCAL_EPSG


def test_compute_morphology_adds_expected_columns(buildings_gpkg):
    bldg = load_buildings(buildings_gpkg, LOCAL_EPSG)
    bldg = compute_morphology(bldg)
    for col in ["area", "perimeter", "lal", "convexity", "elongation", "rectangularity", "eri"]:
        assert col in bldg.columns
    # A perfect square: area = size^2, convexity/rectangularity ~= 1.
    assert bldg.loc[bldg["uID"] == 0, "area"].iloc[0] == pytest.approx(36.0)
    assert bldg["convexity"].min() > 0.99


def test_compute_tessellation_returns_one_cell_per_building(buildings_gpkg):
    bldg = load_buildings(buildings_gpkg, LOCAL_EPSG)
    bldg = compute_morphology(bldg)
    tessellation, limit = compute_tessellation(bldg)
    assert len(tessellation) == len(bldg)
    assert set(tessellation["uID"]) == set(bldg["uID"])


def test_load_streets_main_component_filters_disconnected(tmp_path):
    lines = [
        LineString([(0, 0), (10, 0)]),
        LineString([(10, 0), (10, 10)]),
        LineString([(1000, 1000), (1010, 1000)]),  # disconnected fragment
    ]
    gdf = gpd.GeoDataFrame(geometry=lines, crs=f"EPSG:{LOCAL_EPSG}")
    path = tmp_path / "streets.gpkg"
    gdf.to_file(path)

    streets, graph, main_nodes = load_streets_main_component(path, LOCAL_EPSG)
    assert len(streets) == 2
    assert (1000.0, 1000.0) not in main_nodes


def test_detect_blocks_and_intersection_density(streets_gpkg, buildings_gpkg):
    streets, graph, main_nodes = load_streets_main_component(streets_gpkg, LOCAL_EPSG)
    bldg = load_buildings(buildings_gpkg, LOCAL_EPSG)
    bldg = compute_morphology(bldg)
    _, limit = compute_tessellation(bldg)

    blocks = detect_blocks(streets, limit, LOCAL_EPSG)
    assert len(blocks) > 0
    assert "block_id" in blocks.columns

    blocks = compute_intersection_density(graph, main_nodes, blocks, LOCAL_EPSG)
    assert "int_density" in blocks.columns
    assert (blocks["n_intersections"] >= 0).all()

    bldg = join_block_to_buildings(bldg, blocks)
    assert "block_id" in bldg.columns


def test_compute_tessellation_boundary_changes_block_partition(streets_gpkg, buildings_gpkg):
    # Regression test: forgetting to pass the real study-area boundary silently
    # falls back to the buildings' convex hull, which can merge distinct
    # blocks near the AOI edge into one oversized enclosure once streets are
    # clipped against a looser limit (this is exactly what happened during
    # this migration's own real-data validation, see the ARCH overview §5.4
    # discussion of representation choices mattering for correctness).
    streets, _, _ = load_streets_main_component(streets_gpkg, LOCAL_EPSG)
    bldg = load_buildings(buildings_gpkg, LOCAL_EPSG)
    bldg = compute_morphology(bldg)

    _, hull_limit = compute_tessellation(bldg)  # no boundary -> convex hull fallback
    tight_boundary = gpd.GeoDataFrame(geometry=[_square(50, 30, 200)], crs=f"EPSG:{LOCAL_EPSG}")
    _, boundary_limit = compute_tessellation(bldg, boundary=tight_boundary)

    detect_blocks(streets, hull_limit, LOCAL_EPSG)  # both must run without error
    detect_blocks(streets, boundary_limit, LOCAL_EPSG)

    assert not hull_limit.equals(boundary_limit)
    assert hull_limit.area != pytest.approx(boundary_limit.area)


def test_join_landuse_missing_file_sets_none(tmp_path, buildings_gpkg):
    bldg = load_buildings(buildings_gpkg, LOCAL_EPSG)
    bldg = join_landuse(bldg, tmp_path / "does_not_exist.gpkg", LOCAL_EPSG)
    assert bldg["lu_class"].isna().all()


def test_join_landuse_matches_polygon(tmp_path, buildings_gpkg):
    bldg = load_buildings(buildings_gpkg, LOCAL_EPSG)
    landuse = gpd.GeoDataFrame(
        {"class": ["hospital"]}, geometry=[_square(0, 0, 20)], crs=f"EPSG:{LOCAL_EPSG}"
    )
    path = tmp_path / "landuse.gpkg"
    landuse.to_file(path)

    bldg = join_landuse(bldg, path, LOCAL_EPSG)
    matched = bldg.loc[bldg["uID"] == 0, "lu_class"].iloc[0]
    assert matched == "hospital"
    assert pd.isna(bldg.loc[bldg["uID"] == 2, "lu_class"].iloc[0])


def test_classify_building_type_landuse_overrides_morphology(buildings_gpkg):
    bldg = load_buildings(buildings_gpkg, LOCAL_EPSG)
    bldg = compute_morphology(bldg)
    bldg["int_density"] = 0.0
    # Building "c" is large (1600 m^2) — morphology alone would call it industrial —
    # but a hospital landuse tag must override that.
    bldg["lu_class"] = [None, None, "hospital"]
    bldg = classify_building_type(bldg)
    assert bldg.loc[bldg["uID"] == 2, "bldg_type"].iloc[0] == "civic"


def test_classify_building_type_morphology_fallback(buildings_gpkg):
    bldg = load_buildings(buildings_gpkg, LOCAL_EPSG)
    bldg = compute_morphology(bldg)
    bldg["int_density"] = 0.0
    bldg["lu_class"] = None
    bldg = classify_building_type(bldg)
    # Large (1600 m^2), high eri, low intersection density -> industrial fallback rule.
    assert bldg.loc[bldg["uID"] == 2, "bldg_type"].iloc[0] == "industrial"


def test_estimate_floors_height_is_reproducible(buildings_gpkg):
    bldg = load_buildings(buildings_gpkg, LOCAL_EPSG)
    bldg = compute_morphology(bldg)

    result_a = estimate_floors_height(bldg, seed=42)
    result_b = estimate_floors_height(bldg, seed=42)
    pd.testing.assert_series_equal(result_a["height_est"], result_b["height_est"])
    pd.testing.assert_series_equal(result_a["num_floors_est"], result_b["num_floors_est"])
    assert "volume_est" in result_a.columns
    assert (result_a["volume_est"] == result_a["area"] * result_a["height_est"]).all()


def test_estimate_floors_height_keeps_known_values(buildings_gpkg):
    bldg = load_buildings(buildings_gpkg, LOCAL_EPSG)
    bldg = compute_morphology(bldg)
    bldg.loc[bldg["uID"] == 0, "height"] = 12.5
    bldg.loc[bldg["uID"] == 0, "num_floors"] = 4

    bldg = estimate_floors_height(bldg)
    assert bldg.loc[bldg["uID"] == 0, "height_est"].iloc[0] == 12.5
    assert bldg.loc[bldg["uID"] == 0, "num_floors_est"].iloc[0] == 4


def test_select_export_columns_orders_original_then_derived(buildings_gpkg):
    bldg = load_buildings(buildings_gpkg, LOCAL_EPSG)
    bldg = compute_morphology(bldg)
    bldg = estimate_floors_height(bldg)
    out = select_export_columns(bldg)
    assert list(out.columns[:3]) == ["id", "height", "num_floors"]
    assert "area" in out.columns
    assert "height_est" in out.columns


def test_export_enriched_buildings_writes_gpkg(tmp_path, buildings_gpkg):
    bldg = load_buildings(buildings_gpkg, LOCAL_EPSG)
    bldg = compute_morphology(bldg)
    bldg = estimate_floors_height(bldg)
    out_path = tmp_path / "enriched.gpkg"
    export_enriched_buildings(bldg, out_path)
    assert out_path.exists()
    reloaded = gpd.read_file(out_path)
    assert len(reloaded) == 3


def test_shift_to_local_origin_translates_and_clears_crs(buildings_gpkg):
    bldg = load_buildings(buildings_gpkg, LOCAL_EPSG)
    shifted = shift_to_local_origin(bldg, origin_easting=50, origin_northing=0)
    assert shifted.crs is None
    assert shifted.geometry.iloc[1].centroid.x == pytest.approx(0.0, abs=1e-6)


def test_plot_building_types_returns_folium_map(buildings_gpkg):
    import folium

    bldg = load_buildings(buildings_gpkg, LOCAL_EPSG)
    bldg = compute_morphology(bldg)
    bldg["int_density"] = 0.0
    bldg["lu_class"] = None
    bldg = classify_building_type(bldg)

    m = plot_building_types(bldg, center_lat=13.98, center_lon=108.00)
    assert isinstance(m, folium.Map)


def test_extrude_footprint_produces_closed_solid():
    poly = _square(0, 0, 10)
    result = extrude_footprint(poly, base_z=100.0, height=5.0)
    assert result is not None
    verts, faces = result
    assert verts.shape == (8, 3)  # 4 ring points x (bottom + top)
    assert verts[:4, 2].max() == pytest.approx(100.0)
    assert verts[4:, 2].max() == pytest.approx(105.0)
    assert faces.shape[1] == 3
