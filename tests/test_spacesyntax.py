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


def test_axial_connectivity_counts_touching_and_crossing_lines(grid_streets):
    # Every line of the ring-plus-cross grid touches or crosses exactly 3 others
    # (the old edge count returned 1 for every line).
    axial_map = ss.graph_to_axial_map(grid_streets)
    G = ss.gdf_to_nx_graph(axial_map, id_col="axial_id")
    conn = ss.axial_connectivity(G)
    assert len(conn) == len(axial_map) == 6
    assert set(conn.values()) == {3}


def test_axial_integration_higher_for_more_central_line(grid_streets):
    axial_map = ss.graph_to_axial_map(grid_streets)
    G = ss.gdf_to_nx_graph(axial_map, id_col="axial_id")
    integ = ss.axial_integration(G)
    assert all(v >= 0 for v in integ.values())
    assert len(integ) == len(axial_map)


@pytest.fixture
def comb_streets():
    # One 400 m main street crossed in its middle part by three 100 m side streets
    # that don't touch each other. Only the line graph links them: in the
    # end-point graph none of the four lines share a node.
    return _streets_gdf([
        [(0, 0), (400, 0)],
        [(100, -50), (100, 50)],
        [(200, -50), (200, 50)],
        [(300, -50), (300, 50)],
    ])


def _axial_G(gdf):
    return ss.gdf_to_nx_graph(ss.graph_to_axial_map(gdf, angular_tolerance=20.0), id_col="axial_id")


def _main_id(G):
    return next(d["id"] for _, _, d in G.edges(data=True) if d["geometry"].length == pytest.approx(400))


def test_axial_line_graph_links_mid_line_crossings(comb_streets):
    G = _axial_G(comb_streets)
    L = ss.axial_line_graph(G)
    main = _main_id(G)
    assert L.degree(main) == 3
    assert L.number_of_edges() == 3  # side streets only meet the main street


def test_axial_integration_matches_hillier_hanson(grid_streets):
    # Every line of the grid sees 3 lines at depth 1 and 2 at depth 2:
    # k = 6, MD = 7/5, RA = 2(MD-1)/(k-2) = 0.2,
    # D_k = 2(k(log2((k+2)/3) - 1) + 1) / ((k-1)(k-2)), integration = D_k / RA.
    k = 6
    d_k = 2 * (k * (math.log2((k + 2) / 3) - 1) + 1) / ((k - 1) * (k - 2))
    expected = d_k / 0.2
    integ = ss.axial_integration(_axial_G(grid_streets))
    assert all(v == pytest.approx(expected) for v in integ.values())


def test_axial_integration_main_street_most_integrated(comb_streets):
    G = _axial_G(comb_streets)
    integ = ss.axial_integration(G)
    main = _main_id(G)
    assert integ[main] == max(integ.values())
    assert all(v > 0 for v in integ.values())


def test_axial_choice_counts_routes_through_the_main_street(comb_streets):
    # The 3 side-street pairs can only connect via the main street: Choice = 3;
    # side streets are never between two other lines: Choice = 0.
    G = _axial_G(comb_streets)
    choice = ss.axial_choice(G)
    main = _main_id(G)
    assert choice[main] == pytest.approx(3.0)
    assert all(v == pytest.approx(0.0) for k, v in choice.items() if k != main)


def test_axial_choice_length_weighted_uses_origin_and_destination_lengths(comb_streets):
    # 3 side-street pairs, each 100 m x 100 m.
    G = _axial_G(comb_streets)
    lw = ss.axial_choice_length_weighted(G)
    assert lw[_main_id(G)] == pytest.approx(3 * 100 * 100)


def test_axial_choice_metric_radius_limits_routes(comb_streets):
    # Side streets are 100 m apart along the main street; midpoint-to-midpoint
    # distance between neighbouring side streets is 100 m, between the outer two
    # 200 m. A 150 m radius keeps only the 2 neighbouring pairs.
    G = _axial_G(comb_streets)
    multi = ss.axial_choice_multiscale(G, radii={"R150": 150, "Rn": float("inf")})
    main = _main_id(G)
    assert multi["Rn"][main] == pytest.approx(3.0)
    assert multi["R150"][main] == pytest.approx(2.0)


def test_add_line_to_map_snaps_onto_a_line_interior(comb_streets):
    # A proposed street ending 1 m short of the middle of the main street must
    # snap onto it and be linked in the line graph.
    axial_map = ss.graph_to_axial_map(comb_streets)
    new_geom = LineString([(150, 100), (150, 1)])
    result = ss.add_line_to_map(axial_map, new_geom, id_col="axial_id", snap_tolerance=2.0)
    assert list(result.iloc[-1].geometry.coords)[-1][:2] == pytest.approx((150.0, 0.0))
    L = ss.axial_line_graph(ss.gdf_to_nx_graph(result, id_col="axial_id"))
    assert L.degree(int(result.iloc[-1]["axial_id"])) == 1


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


def _straight_then_turn(rotate_deg=0.0):
    # seg 0 and seg 1 run straight on; seg 2 turns 90 degrees off the end of seg 1.
    pts = [[(0, 0), (100, 0)], [(100, 0), (200, 0)], [(200, 0), (200, 100)]]
    a = math.radians(rotate_deg)
    rot = lambda x, y: (x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a))
    gdf = _streets_gdf([[rot(*p) for p in line] for line in pts])
    return ss.gdf_to_nx_graph(ss.graph_to_segment_map(gdf), id_col="seg_id")


def test_seg_angular_integration_matches_hand_calculation():
    # Dual graph: 0-1 turn cost 0 (straight), 1-2 turn cost 1 (90 degrees).
    # Total angular depth: seg0 = 0 + 1 = 1, seg1 = 0 + 1 = 1, seg2 = 1 + 1 = 2.
    # Integration = n^2 / TD with n = 2: 4, 4, 2.
    ang_int = ss.seg_angular_integration(_straight_then_turn())
    assert ang_int[0] == pytest.approx(4.0)
    assert ang_int[1] == pytest.approx(4.0)
    assert ang_int[2] == pytest.approx(2.0)


def test_seg_angular_integration_ignores_network_orientation():
    # Depth is turn cost, not compass bearing, so rotating the whole network
    # must not change the result (the old bearing-weighted version did).
    before = ss.seg_angular_integration(_straight_then_turn(0))
    after = ss.seg_angular_integration(_straight_then_turn(37))
    for seg_id, value in before.items():
        assert after[seg_id] == pytest.approx(value)


def test_seg_nach_matches_hand_calculation():
    # Angular choice (every least-angle path, endpoints included) = 4, 6, 4;
    # angular total depth = 1, 1, 2. NACH = log(CH + 1) / log(TD + 3).
    nach = ss.seg_nach(_straight_then_turn())
    assert nach[0] == pytest.approx(math.log(5) / math.log(4))
    assert nach[1] == pytest.approx(math.log(7) / math.log(4))
    assert nach[2] == pytest.approx(math.log(5) / math.log(5))
    assert nach[1] > nach[0]


def test_seg_nach_single_segment_is_zero():
    # One segment: no route passes through it and it has no depth -> log(1) / log(3) = 0.
    gdf = _streets_gdf([[(0, 0), (100, 0)]])
    G = ss.gdf_to_nx_graph(ss.graph_to_segment_map(gdf), id_col="seg_id")
    assert list(ss.seg_nach(G).values())[0] == pytest.approx(0.0)


def test_seg_nach_metric_infinite_radius_equals_seg_nach():
    G = _straight_then_turn()
    metric = ss.seg_nach_metric(G)
    plain = ss.seg_nach(G)
    for seg_id, value in plain.items():
        assert metric[seg_id] == pytest.approx(value)


def test_seg_nach_metric_matches_hand_calculation_with_small_radius():
    # Segments are 100 m long, so the average step between neighbours is 100 m.
    # R150 reaches only direct neighbours: seg 0 sees {0, 1}, seg 1 sees {0, 1, 2},
    # seg 2 sees {1, 2}. Least-angle routes inside each catchment:
    #   origin 0: 0->1            choice: +1 on 0, 1;        depth 0
    #   origin 1: 1->0, 1->2      choice: +1 on 1,0 and 1,2; depth 0 + 1 = 1
    #   origin 2: 2->1            choice: +1 on 2, 1;        depth 1
    # Choice = seg0: 2, seg1: 4, seg2: 2. Total depth = seg0: 0, seg1: 1, seg2: 1.
    nach = ss.seg_nach_metric(_straight_then_turn(), radius_m=150)
    assert nach[0] == pytest.approx(math.log(3) / math.log(3))
    assert nach[1] == pytest.approx(math.log(5) / math.log(4))
    assert nach[2] == pytest.approx(math.log(3) / math.log(4))


def test_seg_nach_metric_choice_term_matches_angular_choice_metric(grid_streets):
    # Guard against the two metric-radius functions drifting apart.
    G = ss.gdf_to_nx_graph(ss.graph_to_segment_map(grid_streets), id_col="seg_id")
    choice = ss.seg_angular_choice_metric(G, radius_m=300)
    nach = ss.seg_nach_metric(G, radius_m=300)
    for seg_id, ch in choice.items():
        if ch == 0:
            assert nach[seg_id] == pytest.approx(0.0)
        else:
            assert nach[seg_id] > 0


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
