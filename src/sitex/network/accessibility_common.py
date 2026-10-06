"""Shared building blocks for the accessibility notebooks.

``03-NA06-Green_Accessibility.ipynb`` uses a constant-pace walking graph and an H3
hex grid over the same study area, and computes a Gini index of spatial equity. :func:`load_walk_graph` reads the same saved
``osm/{slug}_walk.gpkg`` that the isochrone notebooks (``03-NA03`` to ``03-NA05``) use.
"""

from __future__ import annotations

import geopandas as gpd
import networkx as nx

from sitex.core.hexgrid import build_h3_grid  # noqa: F401  (moved there; kept importable from here)

# Walking speed used in every accessibility and isochrone calculation: 4.8 km/h
# (80 m per minute), the speed Transport for London assumes for walk times in its
# PTAL method. Transport for London (2015), *Assessing transport connectivity in
# London*, Table 2.1, p. 18. https://content.tfl.gov.uk/connectivity-assessment-guide.pdf
DEFAULT_WALK_SPEED_KMH = 4.8


def build_walk_graph(place: str, walk_speed_kmh: float = DEFAULT_WALK_SPEED_KMH):
    """Download the walking network for ``place`` and assign a constant pedestrian
    pace to every edge's ``travel_time``.

    Deliberately **not** ``ox.add_edge_speeds()``, which imputes driving speeds by
    highway type (e.g. ~40 km/h on a residential street) — that would make every
    isochrone/accessibility result several times too generous for a walking
    analysis. Returns ``(G, boundary_poly)``.
    """
    import osmnx as ox

    G = ox.graph_from_place(place, network_type="walk")
    walk_speed_mps = walk_speed_kmh * 1000 / 3600
    for _u, _v, _k, data in G.edges(keys=True, data=True):
        data["travel_time"] = data["length"] / walk_speed_mps

    boundary = ox.geocode_to_gdf(place)
    return G, boundary.geometry.iloc[0]


def load_walk_graph(path, walk_speed_kmh: float = DEFAULT_WALK_SPEED_KMH):
    """Load a walking network saved by ``00-Data-Acquisition.ipynb``
    (``osm/{slug}_walk.gpkg``, layers ``nodes`` and ``edges``) and assign the same
    constant pedestrian pace as :func:`build_walk_graph`.

    The file is stored in the local UTM zone; the graph is returned in EPSG:4326
    (x = lon, y = lat), like a freshly downloaded OSMnx graph, so the H3 grid and
    ``ox.nearest_nodes`` work on it unchanged. ``length`` stays in metres.
    """
    import osmnx as ox

    nodes = gpd.read_file(path, layer="nodes").to_crs(epsg=4326).set_index("osmid")
    nodes["x"] = nodes.geometry.x
    nodes["y"] = nodes.geometry.y
    edges = gpd.read_file(path, layer="edges").to_crs(epsg=4326).set_index(["u", "v", "key"])
    # The GeoPackage stores every street once (undirected); a pedestrian can walk
    # it both ways, so add the reverse direction of every edge.
    G = ox.graph_from_gdfs(nodes, edges).to_undirected().to_directed()

    walk_speed_mps = walk_speed_kmh * 1000 / 3600
    for _u, _v, _k, data in G.edges(keys=True, data=True):
        data["travel_time"] = data["length"] / walk_speed_mps
    return G


ESRI_WORLD_IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


def base_map(center, zoom_start: int = 13):
    """Folium map with two switchable basemaps: CartoDB Positron (default, so
    coloured layers read clearly) and Esri satellite imagery. The switch only
    appears once the caller adds a ``folium.LayerControl``."""
    import folium

    m = folium.Map(location=center, zoom_start=zoom_start, tiles="cartodbpositron", control_scale=True)
    folium.TileLayer(tiles=ESRI_WORLD_IMAGERY, attr="Esri World Imagery", name="Satellite", overlay=False).add_to(m)
    return m


def assign_origin_nodes(grid: gpd.GeoDataFrame, G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """Add ``origin_node`` (each hex's nearest graph node) and ``snap_dist_m`` (distance in
    metres from the hex centroid to that node; a large value means the hex is far from
    any street)."""
    import osmnx as ox

    grid = grid.copy()
    nodes, dists = ox.nearest_nodes(G, grid["lon"].values, grid["lat"].values, return_dist=True)
    grid["origin_node"] = nodes
    grid["snap_dist_m"] = dists
    return grid


def compute_gini(values) -> float | None:
    """Gini index of spatial inequality (0 = perfectly equal, 1 = maximally unequal).

    Returns ``None`` if fewer than 2 data points are given (an equity number isn't
    meaningful with one or zero valid cells).
    """
    from inequality.gini import Gini

    values = list(values)
    if len(values) < 2:
        return None
    return Gini(values).g
