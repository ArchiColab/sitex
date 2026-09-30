import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from sitex.network import accessibility_common as ac
from sitex.network import green_accessibility as ga
from sitex.network import poi_accessibility as pa

LOCAL_EPSG = 32649
WALK_SPEED_MPS = 4.8 * 1000 / 3600


def _grid_walk_graph():
    """A small 3x3 grid of nodes, 100 m apart, as a MultiDiGraph shaped like an
    OSMnx walking graph (x/y node attrs in degrees-ish units, length + travel_time
    edge attrs) — no network access needed since ox.nearest_nodes only needs the
    graph's own coordinates."""
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"
    coord_step = 0.001  # ~111 m at the equator, close enough for nearest-node tests
    for i in range(3):
        for j in range(3):
            node_id = i * 3 + j
            G.add_node(node_id, x=108.0 + j * coord_step, y=13.98 + i * coord_step)

    def add_edge(u, v, length):
        G.add_edge(u, v, length=length, travel_time=length / WALK_SPEED_MPS)
        G.add_edge(v, u, length=length, travel_time=length / WALK_SPEED_MPS)

    for i in range(3):
        for j in range(2):
            add_edge(i * 3 + j, i * 3 + j + 1, 100.0)
    for i in range(2):
        for j in range(3):
            add_edge(i * 3 + j, (i + 1) * 3 + j, 100.0)
    return G


def _poi_gdf(records):
    return gpd.GeoDataFrame(
        [{"id": r[0], "name": r[0], "geometry": Point(r[1], r[2])} for r in records],
        crs="EPSG:4326",
    )


# ── accessibility_common ─────────────────────────────────────────────────────

def test_compute_gini_none_for_insufficient_data():
    assert ac.compute_gini([1.0]) is None
    assert ac.compute_gini([]) is None


def test_compute_gini_zero_for_equal_values():
    assert ac.compute_gini([5.0, 5.0, 5.0, 5.0]) == pytest.approx(0.0, abs=1e-6)


def test_compute_gini_positive_for_unequal_values():
    g = ac.compute_gini([1.0, 1.0, 1.0, 100.0])
    assert 0 < g <= 1


def test_build_h3_grid_covers_boundary():
    from shapely.geometry import box

    boundary = box(108.0, 13.98, 108.01, 13.99)
    grid = ac.build_h3_grid(boundary, resolution=9)
    assert len(grid) > 0
    assert set(grid.columns) >= {"h3_id", "geometry", "lat", "lon"}


def test_assign_origin_nodes_picks_nearest():
    G = _grid_walk_graph()
    grid = gpd.GeoDataFrame({"lon": [108.0], "lat": [13.98]}, geometry=[Point(108.0, 13.98)], crs="EPSG:4326")
    result = ac.assign_origin_nodes(grid, G)
    assert result["origin_node"].iloc[0] == 0  # node 0 is exactly at (108.0, 13.98)


# ── poi_accessibility ─────────────────────────────────────────────────────────

def test_compute_facility_isochrones_and_union(monkeypatch):
    G = _grid_walk_graph()
    pois = _poi_gdf([("clinic", 108.0, 13.98)])

    facility_polys = pa.compute_facility_isochrones(G, pois, time_bands=[2, 5])
    assert len(facility_polys) == 1
    poi_row, bands = facility_polys[0]
    assert set(bands.keys()) == {2, 5}
    # A larger time band must reach at least as much area as a smaller one.
    assert bands[5].area >= bands[2].area

    unioned = pa.union_category_isochrones(facility_polys, time_bands=[2, 5])
    assert unioned[5].area == pytest.approx(bands[5].area)


def test_extract_service_areas_uses_max_time():
    G = _grid_walk_graph()
    pois = _poi_gdf([("clinic", 108.0, 13.98)])
    facility_polys = pa.compute_facility_isochrones(G, pois, time_bands=[2, 5])
    service_areas = pa.extract_service_areas(facility_polys, max_time=5)
    assert len(service_areas) == 1
    assert service_areas.iloc[0]["poi_name"] == "clinic"


def test_capture_buildings_counts_inside_polygon():
    from shapely.geometry import box

    poly_wgs = box(107.999, 13.979, 108.002, 13.982)  # covers the 3x3 grid's SW corner
    buildings = gpd.GeoDataFrame(
        {"area": [50.0, 50.0], "num_floors_est": [2, 2]},
        geometry=[Point(500000, 1546000), Point(600000, 1546000)],
        crs=f"EPSG:{LOCAL_EPSG}",
    )
    buildings["floor_area_m2"] = buildings["area"] * buildings["num_floors_est"]
    buildings["centroid"] = buildings.geometry.centroid

    # Reproject the WGS84 polygon's equivalent footprint isn't needed here — just
    # confirm the function's own reprojection + within() pipeline runs and returns
    # a well-formed per-band table for a polygon and a None band.
    result = pa.capture_buildings(buildings, {5: poly_wgs, 10: None}, local_epsg=LOCAL_EPSG)
    assert list(result["time_min"]) == [5, 10]
    assert result.loc[result["time_min"] == 10, "n_buildings"].iloc[0] == 0
    assert (result["n_buildings_pct"] <= 100).all()


def test_compute_category_accessibility_nearest_poi_wins():
    G = _grid_walk_graph()
    grid = ac.assign_origin_nodes(
        gpd.GeoDataFrame({"lon": [108.0], "lat": [13.98]}, geometry=[Point(108.0, 13.98)], crs="EPSG:4326"),
        G,
    )
    pois = _poi_gdf([("near", 108.0, 13.98), ("far", 108.002, 13.982)])
    result = pa.compute_category_accessibility(G, grid, pois, max_time_min=15)
    assert result["access_time_min"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert result["nearest_poi_name"].iloc[0] == "near"


def test_compute_accessibility_summary_shape():
    grid = gpd.GeoDataFrame({"access_time_min": [1.0, 2.0, 3.0, float("inf")]})
    pois = _poi_gdf([("a", 108.0, 13.98)])
    capture_row = pd.Series({"n_buildings_pct": 80.0, "floor_area_pct": 75.0})
    summary = pa.compute_accessibility_summary("Healthcare", pois, grid, capture_row, max_walk_time=15)
    assert summary["category"] == "Healthcare"
    assert summary["n_facilities"] == 1
    assert summary["pct_buildings_15min"] == 80.0
    assert summary["gini_access_time"] is not None


def test_distinct_color_returns_hex_and_varies():
    colors = [pa.distinct_color(i, 5) for i in range(5)]
    assert all(c.startswith("#") and len(c) == 7 for c in colors)
    assert len(set(colors)) == 5


# ── green_accessibility ───────────────────────────────────────────────────────

def test_label_green_space_priority_order():
    row = pd.Series({"leisure": "park", "landuse": "forest"})
    assert ga._label_green_space(row) == "leisure:park"

    row2 = pd.Series({"landuse": "forest"})
    assert ga._label_green_space(row2) == "landuse:forest"

    row3 = pd.Series({})
    assert ga._label_green_space(row3) == "unknown"


def test_cluster_green_spaces_smallest_is_cluster_zero():
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(
        {"green_space": ["a", "b", "c"]},
        geometry=[box(0, 0, 10, 10), box(0, 0, 100, 100), box(0, 0, 50, 50)],
        crs=f"EPSG:{LOCAL_EPSG}",
    )
    clustered = ga.cluster_green_spaces(gdf, local_epsg=LOCAL_EPSG, num_clusters=3)
    smallest_row = clustered.loc[clustered["area_m2"].idxmin()]
    largest_row = clustered.loc[clustered["area_m2"].idxmax()]
    assert smallest_row["cluster"] < largest_row["cluster"]


def test_select_public_green_drops_farmland_and_water():
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(
        {"green_space": ["leisure:park", "landuse:farmland", "natural:water", "leisure:playground", "landuse:orchard"]},
        geometry=[box(0, 0, 1, 1)] * 5, crs="EPSG:4326",
    )
    public = ga.select_public_green(gdf)
    assert set(public["green_space"]) == {"leisure:park", "leisure:playground"}


def _write_worldcover(tmp_path, array):
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "wc.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
        dtype="uint8", crs=f"EPSG:{LOCAL_EPSG}", transform=from_origin(500000, 1550000, 10, 10),
    ) as dst:
        dst.write(array.astype("uint8"), 1)
    return path


def test_worldcover_green_patches_merges_classes_and_drops_small(tmp_path):
    wc = np.full((60, 60), 50)          # built-up background, 10 m pixels
    wc[0:20, 0:10] = 10                 # tree cover 2 ha ...
    wc[0:20, 10:20] = 30                # ... touching grassland 2 ha: one 4 ha patch
    wc[40:45, 40:45] = 10               # isolated 0.25 ha of trees: below 1 ha
    wc[40:50, 0:10] = 40                # cropland: not green environment
    patches = ga.worldcover_green_patches(_write_worldcover(tmp_path, wc), LOCAL_EPSG, min_area_ha=1.0)
    assert len(patches) == 1
    assert patches["area_m2"].iloc[0] == pytest.approx(40_000)
    assert patches["tree_share"].iloc[0] == pytest.approx(0.5)
    assert patches.crs.to_epsg() == 4326


def test_green_access_nodes_uses_edge_not_centre():
    from shapely.geometry import box

    G = _grid_walk_graph()
    # A long strip along the grid's east column (x = 108.002), reaching far north
    # of the graph: its centre is nearest to node 8, but nodes 2, 5 and 8 all lie
    # inside it.
    strip = gpd.GeoDataFrame(geometry=[box(108.0018, 13.9795, 108.0022, 13.99)], crs="EPSG:4326")
    assert set(ga.green_access_nodes(G, strip, LOCAL_EPSG, edge_buffer_m=5)) == {2, 5, 8}


def test_green_access_nodes_falls_back_to_nearest_node():
    from shapely.geometry import box

    G = _grid_walk_graph()
    far = gpd.GeoDataFrame(geometry=[box(108.0000, 13.9750, 108.0003, 13.9753)], crs="EPSG:4326")
    assert ga.green_access_nodes(G, far, LOCAL_EPSG, edge_buffer_m=5) == [0]


def test_compute_green_accessibility_measures_to_the_edge():
    from shapely.geometry import box

    G = _grid_walk_graph()
    strip = gpd.GeoDataFrame(geometry=[box(108.0018, 13.9795, 108.0022, 13.99)], crs="EPSG:4326")
    boundary = box(108.0015, 13.9795, 108.0025, 13.9805)  # hexes around node 2
    grid = ga.compute_green_accessibility(G, boundary, strip, walk_time_minutes=15,
                                          h3_resolution=12, local_epsg=LOCAL_EPSG, edge_buffer_m=5)
    at_node_2 = grid[grid["origin_node"] == 2]
    assert len(at_node_2) > 0
    # Node 2 is inside the green strip: 0 min (the old centre-based walk was 200 m, 2.5 min).
    assert at_node_2["access_time_min"].max() == pytest.approx(0.0, abs=1e-9)


def test_compute_green_accessibility_no_green_is_all_inf():
    from shapely.geometry import box

    G = _grid_walk_graph()
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    grid = ga.compute_green_accessibility(G, box(108.0, 13.98, 108.002, 13.982), empty,
                                          h3_resolution=11, local_epsg=LOCAL_EPSG)
    assert np.isinf(grid["access_time_min"]).all()


def test_summarise_green_access_uses_reachable_hexes_only():
    grid = gpd.GeoDataFrame({"access_time_min": [2.0, 4.0, 6.0, float("inf")]})
    row = ga.summarise_green_access(grid, walk_time_minutes=15, layer="public")
    assert row["pct_hexes_within"] == pytest.approx(75.0)
    assert row["median_walk_min"] == pytest.approx(4.0)
    assert row["gini_walk_time"] is not None


def test_classify_green_gap_four_classes():
    from shapely.geometry import box

    geoms = [box(i, 0, i + 1, 1) for i in range(4)]
    public = gpd.GeoDataFrame({"h3_id": list("abcd"), "access_time_min": [5, float("inf"), 5, float("inf")]},
                              geometry=geoms, crs="EPSG:4326")
    env = gpd.GeoDataFrame({"h3_id": list("abcd"), "access_time_min": [3, 3, float("inf"), float("inf")]},
                           geometry=geoms, crs="EPSG:4326")
    gap = ga.classify_green_gap(public, env, walk_time_minutes=15)
    assert list(gap["gap_class"]) == ["both", "environment_only", "public_only", "none"]
    assert set(gap["gap_class"]) <= set(ga.GAP_CLASSES)


def test_compute_green_accessibility_zero_at_green_space_node():
    G = _grid_walk_graph()
    green = gpd.GeoDataFrame(geometry=[Point(108.0, 13.98)], crs="EPSG:4326")

    from shapely.geometry import box

    boundary = box(108.0, 13.98, 108.002, 13.982)
    grid = ga.compute_green_accessibility(G, boundary, green, walk_time_minutes=15, h3_resolution=12)
    assert (grid["access_time_min"] < float("inf")).any()
    assert grid["access_time_min"].min() == pytest.approx(0.0, abs=1.0)


def test_first_available_basemap_falls_back(monkeypatch):
    import sys
    import types

    from sitex.network import poi_accessibility as pa

    fake = types.ModuleType("leafmap.basemaps")
    fake.get_xyz_dict = lambda: {"Esri.WorldStreetMap": 1, "Esri.WorldImagery": 1}
    monkeypatch.setitem(sys.modules, "leafmap.basemaps", fake)
    assert pa._first_available_basemap(["CartoDB.Positron", "Esri.WorldStreetMap"]) == "Esri.WorldStreetMap"
    fake.get_xyz_dict = lambda: {"CartoDB.Positron": 1}
    assert pa._first_available_basemap(["CartoDB.Positron", "Esri.WorldStreetMap"]) == "CartoDB.Positron"
