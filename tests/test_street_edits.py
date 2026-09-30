import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from sitex.network import spacesyntax as ss
from sitex.network import street_edits as se

LOCAL_EPSG = 32649


@pytest.fixture
def network():
    # Two parallel east-west streets, 100 m apart, joined at their west ends:
    #   3 ----------- 4
    #   |
    #   1 ----------- 2
    xy = {1: (0, 0), 2: (200, 0), 3: (0, 100), 4: (200, 100)}
    nodes = gpd.GeoDataFrame(
        {"osmid": list(xy), "x": [p[0] for p in xy.values()], "y": [p[1] for p in xy.values()],
         "street_count": [2, 1, 2, 1]},
        geometry=[Point(p) for p in xy.values()], crs=f"EPSG:{LOCAL_EPSG}",
    )
    pairs = [(1, 2), (3, 4), (1, 3)]
    edges = gpd.GeoDataFrame(
        {"u": [u for u, _ in pairs], "v": [v for _, v in pairs], "key": 0,
         "osmid": ["10", "11", "12"], "highway": "residential", "name": ["South", "North", "West"],
         "oneway": False, "length": [200.0, 200.0, 100.0]},
        geometry=[LineString([xy[u], xy[v]]) for u, v in pairs], crs=f"EPSG:{LOCAL_EPSG}",
    )
    return nodes, edges


def _sketch(*geoms, **columns):
    return gpd.GeoDataFrame(columns, geometry=list(geoms), crs=f"EPSG:{LOCAL_EPSG}")


def test_street_between_two_streets_splits_both(network):
    nodes, edges = network
    # Drawn 1 m short of each street, as a MultiLineString like QGIS saves it.
    sketch = _sketch(MultiLineString([[(100, 1), (100, 99)]]))
    nodes_out, edges_out, report = se.add_streets_to_network(nodes, edges, sketch)

    assert len(nodes_out) == len(nodes) + 2
    assert len(edges_out) == len(edges) + 3  # 2 streets split in two, 1 street added
    assert list(report["snapped_to"]) == ["along an existing street"] * 2
    assert list(report["moved_m"]) == [1.0, 1.0]

    added = edges_out[edges_out["source"] == "added"]
    assert len(added) == 1
    assert added["length"].iloc[0] == pytest.approx(100.0)
    assert set(nodes_out.loc[nodes_out["osmid"] < 0, "street_count"]) == {3}
    # The split pieces keep their street's attributes and share its length.
    assert edges_out.loc[edges_out["name"] == "South", "length"].tolist() == pytest.approx([100.0, 100.0])
    assert not edges_out.duplicated(["u", "v", "key"]).any()
    assert se.in_main_network(edges_out).all()


def test_result_is_connected_for_space_syntax(network):
    nodes, edges = network
    _, edges_out, _ = se.add_streets_to_network(nodes, edges, _sketch(LineString([(100, 1), (100, 99)])))
    segment_map = ss.graph_to_segment_map(edges_out)
    G = ss.gdf_to_nx_graph(segment_map, id_col="seg_id")
    # Shortest route from node 2 to node 4 now uses the new street: 100 + 100 + 100 m.
    import networkx as nx
    assert nx.shortest_path_length(G, (200.0, 0.0), (200.0, 100.0), weight="weight") == pytest.approx(300.0)


def test_end_near_a_junction_snaps_to_it(network):
    nodes, edges = network
    sketch = _sketch(LineString([(198, 3), (198, 97)]))
    nodes_out, edges_out, report = se.add_streets_to_network(nodes, edges, sketch)
    assert list(report["snapped_to"]) == ["existing junction"] * 2
    assert len(nodes_out) == len(nodes)
    assert len(edges_out) == len(edges) + 1
    added = edges_out[edges_out["source"] == "added"].iloc[0]
    assert {added["u"], added["v"]} == {2, 4}
    assert nodes_out.set_index("osmid").loc[[2, 4], "street_count"].tolist() == [2, 2]


def test_crossing_is_connected_only_when_asked(network):
    nodes, edges = network
    # Starts at the south street, crosses the north street, ends in open land.
    sketch = _sketch(LineString([(100, 0), (100, 150)]))

    nodes_out, edges_out, report = se.add_streets_to_network(nodes, edges, sketch, connect_crossings=True)
    assert report["snapped_to"].tolist() == ["along an existing street", "nothing (dead end)"]
    assert (edges_out["source"] == "added").sum() == 2  # split at the crossing
    assert len(nodes_out) == len(nodes) + 3

    nodes_out, edges_out, _ = se.add_streets_to_network(nodes, edges, sketch, connect_crossings=False)
    assert (edges_out["source"] == "added").sum() == 1
    assert len(nodes_out) == len(nodes) + 2
    assert len(edges_out[edges_out["name"] == "North"]) == 1  # bridge: not split


def test_new_streets_join_each_other_and_take_sketch_attributes(network):
    nodes, edges = network
    sketch = _sketch(
        LineString([(100, 0), (100, 50)]),
        LineString([(100.5, 50.5), (150, 50)]),
        name=["Spine", None], highway=["tertiary", None],
    )
    nodes_out, edges_out, report = se.add_streets_to_network(nodes, edges, sketch, name="New street")
    assert report["snapped_to"].tolist() == [
        "along an existing street", "another new street", "another new street", "nothing (dead end)",
    ]
    assert len(nodes_out) == len(nodes) + 3  # the two loose ends meet in one node
    added = edges_out[edges_out["source"] == "added"]
    assert added["name"].tolist() == ["Spine", "New street"]
    assert added["highway"].tolist() == ["tertiary", "residential"]
    assert se.in_main_network(edges_out).all()


def test_polygons_and_unprojected_networks_are_rejected(network):
    nodes, edges = network
    with pytest.raises(ValueError, match="lines"):
        se.add_streets_to_network(nodes, edges, _sketch(Polygon([(0, 0), (1, 0), (1, 1)])))
    with pytest.raises(ValueError, match="metric"):
        se.add_streets_to_network(nodes.to_crs(4326), edges.to_crs(4326), _sketch(LineString([(100, 1), (100, 99)])))


def test_save_and_load_round_trip(network, tmp_path):
    nodes, edges = network
    nodes_out, edges_out, _ = se.add_streets_to_network(nodes, edges, _sketch(LineString([(100, 1), (100, 99)])))
    path = se.save_network(nodes_out, edges_out, tmp_path / "updated.gpkg")
    nodes_back, edges_back = se.load_network(path)
    assert len(nodes_back) == len(nodes_out) and len(edges_back) == len(edges_out)
    assert edges_back["oneway"].dtype == bool
