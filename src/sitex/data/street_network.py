"""OSM street-network acquisition for one :class:`~sitex.data.aoi.AOI`.

Extracted from ``00-Data-OSM_Streetnetwork.ipynb`` — acquisition only (download +
export). The original notebook's intersection-density hexbin/KDE analysis isn't data
acquisition and already has a home in the Space Syntax / accessibility notebooks, so
it isn't ported here.
"""

from __future__ import annotations

from pathlib import Path

import osmnx as ox

from sitex.data.aoi import AOI

DEFAULT_NETWORK_TYPES = ("drive", "walk", "drive_service")


def download_street_networks(
    aoi: AOI,
    network_types: tuple[str, ...] = DEFAULT_NETWORK_TYPES,
) -> dict[str, "ox.MultiDiGraph"]:
    """One graph per network type, truncated to ``aoi.boundary`` (the real admin
    polygon for Method 1, or the AOI rectangle for Method 2 — either way the result is
    clipped to the actual AOI shape, not just its bounding box) and reprojected to
    ``aoi.local_epsg``."""
    graphs = {}
    for network_type in network_types:
        graph = ox.graph_from_polygon(aoi.boundary, network_type=network_type)
        graphs[network_type] = ox.project_graph(graph, to_crs=aoi.local_epsg)
    return graphs


def save_street_networks(graphs: dict[str, "ox.MultiDiGraph"], output_dir: Path, slug: str) -> dict[str, Path]:
    """Save each graph as its own GeoPackage: ``{slug}_{network_type}.gpkg``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for network_type, graph in graphs.items():
        path = output_dir / f"{slug}_{network_type}.gpkg"
        ox.save_graph_geopackage(graph, filepath=path)
        paths[network_type] = path
    return paths
