"""Shared building blocks for the two accessibility notebooks (POI, Green).

Both ``03-NA02-POI_Accessibility.ipynb`` and ``03-NA03-Green Accessibility.ipynb``
build a constant-pace walking graph and an H3 hex grid over the same study area, and
both compute a Gini index of spatial equity — this module is the one shared
definition instead of two independently-maintained copies (the exact kind of
duplication the ARCH/ENV overviews already flag for ``utm_epsg_from_lonlat()``).
"""

from __future__ import annotations

import geopandas as gpd
import networkx as nx

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


def build_h3_grid(boundary_poly, resolution: int) -> gpd.GeoDataFrame:
    """H3 hex grid covering ``boundary_poly``. Columns: h3_id, geometry, lat, lon."""
    import h3
    from shapely.geometry import Polygon

    exterior_coords = [(lat, lng) for lng, lat in boundary_poly.exterior.coords]
    h3_poly = h3.LatLngPoly(exterior_coords)
    hex_ids = h3.h3shape_to_cells(h3_poly, resolution)

    hex_data = []
    for h_id in hex_ids:
        boundary_pts = h3.cell_to_boundary(h_id)
        poly = Polygon([(lng, lat) for lat, lng in boundary_pts])
        centroid = poly.centroid
        hex_data.append({"h3_id": h_id, "geometry": poly, "lat": centroid.y, "lon": centroid.x})

    return gpd.GeoDataFrame(hex_data, crs="EPSG:4326")


def assign_origin_nodes(grid: gpd.GeoDataFrame, G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """Add an ``origin_node`` column: each hex's nearest graph node."""
    import osmnx as ox

    grid = grid.copy()
    grid["origin_node"] = ox.nearest_nodes(G, grid["lon"].values, grid["lat"].values)
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
