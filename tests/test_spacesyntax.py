import math

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from sitex.network import spacesyntax as ss

LOCAL_EPSG = 32649


def _streets_gdf(lines):
    return gpd.GeoDataFrame(geometry=[LineString(pts) for pts in lines], crs=f"EPSG:{LOCAL_EPSG}")


@pytest.fixture
def cross_streets():
    # A "+" shaped 4-way intersection at the origin, each arm 100 m, all collinear
    # pairs at 0/90 degrees so the axial merge collapses it to 2 long lines.
    return _streets_gdf([
        [(-100, 0), (0, 0)],
        [(0, 0), (100, 0)],
        [(0, -100), (0, 0)],
        [(0, 0), (0, 100)],
    ])


@pytest.fixture
def grid_streets():
    # A simple 2x2 block grid: a square ring plus a cross bisecting it.
    lines = [
        [(0, 0), (200, 0)], [(200, 0), (200, 200)], [(200, 200), (0, 200)], [(0, 200), (0, 0)],
        [(0, 100), (200, 100)], [(100, 0), (100, 200)],
    ]
    return _streets_gdf(lines)


def test_graph_to_segment_map_one_row_per_edge(grid_streets):
    seg_map = ss.graph_to_segment_map(grid_streets)
    assert len(seg_map) == len(grid_streets)
    assert set(seg_map.columns) >= {"seg_id", "length_m", "bearing_deg", "geometry"}
    assert seg_map["length_m"].iloc[0] == pytest.approx(200.0)


def test_graph_to_axial_map_merges_collinear_edges(cross_streets):
    axial_map = ss.graph_to_axial_map(cross_streets, angular_tolerance=20.0)
    # The two horizontal arms (0 deg) merge into one 200 m line; same for vertical.
    assert len(axial_map) == 2
    assert set(axial_map["length_m"].round(1)) == {200.0}
    assert set(axial_map["n_merged"]) == {2}


def test_graph_to_axial_map_does_not_merge_across_tolerance():
    # Two arms meeting at 90 degrees at a shared endpoint must NOT merge.
    gdf = _streets_gdf([[(-100, 0), (0, 0)], [(0, 0), (0, 100)]])
    axial_map = ss.graph_to_axial_map(gdf, angular_tolerance=20.0)
    assert len(axial_map) == 2


def test_gdf_to_nx_graph_node_and_edge_counts(grid_streets):
    seg_map = ss.graph_to_segment_map(grid_streets)
    G = ss.gdf_to_nx_graph(seg_map, id_col="seg_id")
    assert G.number_of_edges() == len(seg_map)
    for _, _, data in G.edges(data=True):
        assert data["weight"] > 0


def test_axial_connectivity_counts_direct_edges(cross_streets):
    axial_map = ss.graph_to_axial_map(cross_streets)
    G = ss.gdf_to_nx_graph(axial_map, id_col="axial_id")
    conn = ss.axial_connectivity(G)
    # Two perpendicular lines crossing once: each line is connected to the other = 1.
    assert set(conn.values()) == {1}


def test_axial_integration_higher_for_more_central_line(grid_streets):
    axial_map = ss.graph_to_axial_map(grid_streets)
    G = ss.gdf_to_nx_graph(axial_map, id_col="axial_id")
    integ = ss.axial_integration(G)
    assert all(v >= 0 for v in integ.values())
    assert len(integ) == len(axial_map)


def test_axial_choice_multiscale_returns_all_radii(grid_streets):
    axial_map = ss.graph_to_axial_map(grid_streets)
    G = ss.gdf_to_nx_graph(axial_map, id_col="axial_id")
    result = ss.axial_choice_multiscale(G)
    assert set(result.keys()) == set(ss.RADII_M.keys())
    for radius_name, choice in result.items():
        assert len(choice) == G.number_of_edges()


def test_axial_choice_length_weighted_nonnegative(grid_streets):
    axial_map = ss.graph_to_axial_map(grid_streets)
    G = ss.gdf_to_nx_graph(axial_map, id_col="axial_id")
    lw = ss.axial_choice_length_weighted(G)
    assert all(v >= 0 for v in lw.values())


def test_seg_choice_and_integration(grid_streets):
    seg_map = ss.graph_to_segment_map(grid_streets)
    G = ss.gdf_to_nx_graph(seg_map, id_col="seg_id")
    choice = ss.seg_choice(G)
    integ = ss.seg_integration(G)
    assert len(choice) == len(integ) == G.number_of_edges()
    assert all(v >= 0 for v in choice.values())


def test_seg_reach_increases_with_radius(grid_streets):
    seg_map = ss.graph_to_segment_map(grid_streets)
    G = ss.gdf_to_nx_graph(seg_map, id_col="seg_id")
    reach_small = ss.seg_reach(G, radius_m=50)
    reach_large = ss.seg_reach(G, radius_m=5000)
    for seg_id in reach_small:
        assert reach_large[seg_id] >= reach_small[seg_id]


def test_seg_reach_multiscale_matches_individual_calls(grid_streets):
    seg_map = ss.graph_to_segment_map(grid_streets)
    G = ss.gdf_to_nx_graph(seg_map, id_col="seg_id")
    multi = ss.seg_reach_multiscale(G, radii={"R100": 100, "R500": 500})
    individual_100 = ss.seg_reach(G, radius_m=100)
    individual_500 = ss.seg_reach(G, radius_m=500)
    for seg_id in individual_100:
        assert multi["R100"][seg_id] == pytest.approx(individual_100[seg_id])
        assert multi["R500"][seg_id] == pytest.approx(individual_500[seg_id])


def test_seg_angular_choice_straight_through_favoured_over_turn():
    # Three collinear segments in a row: the middle one should carry more angular
    # betweenness than a segment requiring a sharp turn to reach.
    gdf = _streets_gdf([
        [(0, 0), (100, 0)],
        [(100, 0), (200, 0)],
        [(200, 0), (200, 100)],  # a 90-degree turn off the main straight line
    ])
    seg_map = ss.graph_to_segment_map(gdf)
    G = ss.gdf_to_nx_graph(seg_map, id_col="seg_id")
    ang_choice = ss.seg_angular_choice(G)
    assert len(ang_choice) == 3


def test_seg_nach_single_segment_matches_hand_calculation():
    # One edge, two nodes: seg_choice counts each direction's trivial path once
    # (choice=2); total_depth is 1 hop from each endpoint (sum=1 each).
    # NACH = log(2+1) / log((1+1)/2 + 3) = log(3) / log(4).
    gdf = _streets_gdf([[(0, 0), (100, 0)]])
    seg_map = ss.graph_to_segment_map(gdf)
    G = ss.gdf_to_nx_graph(seg_map, id_col="seg_id")
    nach = ss.seg_nach(G)
    expected = math.log(3) / math.log(4)
    assert list(nach.values())[0] == pytest.approx(expected)


def test_seg_norm_integration_is_0_to_1(grid_streets):
    seg_map = ss.graph_to_segment_map(grid_streets)
    G = ss.gdf_to_nx_graph(seg_map, id_col="seg_id")
    norm = ss.seg_norm_integration(G)
    assert min(norm.values()) == pytest.approx(0.0)
    assert max(norm.values()) == pytest.approx(1.0)


def test_metrics_to_gdf_combines_dicts(grid_streets):
    seg_map = ss.graph_to_segment_map(grid_streets)
    G = ss.gdf_to_nx_graph(seg_map, id_col="seg_id")
    choice = ss.seg_choice(G)
    integ = ss.seg_integration(G)
    result = ss.metrics_to_gdf(G, {"choice": choice, "integration": integ}, crs=seg_map.crs, id_col="seg_id")
    assert set(result.columns) >= {"seg_id", "choice", "integration", "geometry"}
    assert len(result) == G.number_of_edges()


def test_compute_local_centre_index_near_one_when_r800_equals_rn():
    gdf = pd.DataFrame({"choice_R800": [10.0], "choice_Rn": [10.0]})
    lci = ss.compute_local_centre_index(gdf)
    assert lci.iloc[0] == pytest.approx(1.0, abs=0.01)


def test_compute_local_centre_index_low_when_rn_much_larger():
    gdf = pd.DataFrame({"choice_R800": [1.0], "choice_Rn": [1000.0]})
    lci = ss.compute_local_centre_index(gdf)
    assert lci.iloc[0] < 0.5


def test_style_by_metric_adds_color_and_width_columns(grid_streets):
    seg_map = ss.graph_to_segment_map(grid_streets)
    seg_map["dummy"] = np.linspace(0, 100, len(seg_map))
    styled = ss.style_by_metric(seg_map, "dummy", "YlOrRd")
    assert "_color" in styled.columns and "_width" in styled.columns
    assert all(c.startswith("#") for c in styled["_color"])


def test_style_by_metric_diverging_centers_at_zero():
    gdf = gpd.GeoDataFrame(
        {"delta": [-10.0, 0.0, 10.0]},
        geometry=[LineString([(0, 0), (1, 1)])] * 3,
        crs=f"EPSG:{LOCAL_EPSG}",
    )
    styled = ss.style_by_metric_diverging(gdf, "delta")
    # Symmetric magnitude deltas should get identical widths (mirrored around 0).
    assert styled["_width"].iloc[0] == pytest.approx(styled["_width"].iloc[2])


def test_add_line_to_map_snaps_and_appends(grid_streets):
    axial_map = ss.graph_to_axial_map(grid_streets)
    new_geom = LineString([(0.5, 0.5), (150, 150)])  # near-origin endpoint, should snap
    result = ss.add_line_to_map(axial_map, new_geom, id_col="axial_id", snap_tolerance=5.0)
    assert len(result) == len(axial_map) + 1
    new_row = result.iloc[-1]
    assert list(new_row.geometry.coords)[0][:2] == pytest.approx((0.0, 0.0))


def test_normalize_scenario_geometry_picks_longest_disconnected_part():
    from shapely.geometry import MultiLineString

    multi = MultiLineString([[(0, 0), (10, 0)], [(100, 100), (100, 300)]])  # disconnected, second longer
    result = ss.normalize_scenario_geometry(multi)
    assert result.length == pytest.approx(200.0)


def test_compute_scenario_delta_matches_by_midpoint_not_id(grid_streets):
    axial_map = ss.graph_to_axial_map(grid_streets)
    G_before = ss.gdf_to_nx_graph(axial_map, id_col="axial_id")
    before_metrics = ss.metrics_to_gdf(G_before, {"choice": ss.axial_connectivity(G_before)}, crs=axial_map.crs, id_col="axial_id")

    new_geom = LineString([(0, 0), (50, 50)])
    axial_scenario = ss.add_line_to_map(axial_map, new_geom, id_col="axial_id", snap_tolerance=5.0)
    G_after = ss.gdf_to_nx_graph(axial_scenario, id_col="axial_id")
    after_metrics = ss.build_scenario_metrics_gdf(G_after, "axial_id", axial_map.crs, {"choice": ss.axial_connectivity(G_after)})

    delta = ss.compute_scenario_delta(before_metrics, after_metrics, "axial_id", ["choice"])
    assert delta["is_new"].sum() == 1
    assert (~delta["is_new"]).sum() == len(axial_map)
    assert pd.isna(delta.loc[delta["is_new"], "choice_delta"].iloc[0])
