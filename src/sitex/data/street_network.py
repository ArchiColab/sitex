"""OSM street-network acquisition for one :class:`~sitex.data.aoi.AOI`.

Acquisition only (download + export); network analysis lives in ``sitex.network``.

Two routes give the same result, a dict of projected OSMnx graphs:

- ``download_street_networks`` asks a public Overpass server through OSMnx. Those servers
  are shared and can be unreachable or slow.
- ``street_networks_from_geofabrik`` crops a Geofabrik country extract on the local
  computer with the ``osmium`` command (install ``osmium-tool``: it is a program, not a
  Python package). The extract is updated daily, so it can be up to a day older than the
  live OSM data.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import osmnx as ox

from sitex.data.aoi import AOI

DEFAULT_NETWORK_TYPES = ("drive", "walk", "drive_service")

# Ways left out of each network type, as osmium tag expressions: a way is dropped when any
# one of them matches. Copied from OSMnx 2.0.7 (``_get_network_filter`` and
# ``settings.default_access``), so a graph made from a Geofabrik extract holds the same
# kinds of streets as one from Overpass. OSMnx matches its word lists anywhere in a tag
# value, osmium matches whole values.
_NOT_DRIVABLE = (
    "abandoned,bridleway,bus_guideway,construction,corridor,cycleway,elevator,escalator,"
    "footway,no,path,pedestrian,planned,platform,proposed,raceway,razed,rest_area,steps,track"
)
_EXCLUDE = {
    "drive": [
        "w/area=yes", "w/access=private",
        f"w/highway={_NOT_DRIVABLE},service,services",
        "w/motor_vehicle=no", "w/motorcar=no",
        "w/service=alley,driveway,emergency_access,parking,parking_aisle,private",
    ],
    "drive_service": [
        "w/area=yes", "w/access=private",
        f"w/highway={_NOT_DRIVABLE},services",
        "w/motor_vehicle=no", "w/motorcar=no",
        "w/service=emergency_access,parking,parking_aisle,private",
    ],
    "walk": [
        "w/area=yes", "w/access=private",
        "w/highway=abandoned,bus_guideway,construction,cycleway,motor,no,planned,platform,"
        "proposed,raceway,razed,rest_area,services",
        "w/foot=no", "w/service=private",
        "w/sidewalk=separate", "w/sidewalk:both=separate",
        "w/sidewalk:left=separate", "w/sidewalk:right=separate",
    ],
}


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


def download_geofabrik_extract(
    url: str,
    dest_dir: Path,
    contact: str,
    refresh: bool = False,
) -> Path:
    """Download a Geofabrik ``.osm.pbf`` file into ``dest_dir`` and return its path.

    An existing file is reused unless ``refresh`` is True. The download goes through a
    ``.part`` file, so an interrupted download never leaves a half file under the final
    name. ``contact`` goes into the User-Agent so the server operators can reach you."""
    import requests

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / url.rsplit("/", 1)[-1]
    if dest.exists() and not refresh:
        return dest

    part = dest.with_name(dest.name + ".part")
    headers = {"User-Agent": f"sitex-workshop (contact: {contact})"}
    with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=total, unit="B", unit_scale=True)
        except ImportError:
            bar = None
        with open(part, "wb") as f:
            for block in response.iter_content(1 << 20):
                f.write(block)
                if bar is not None:
                    bar.update(len(block))
        if bar is not None:
            bar.close()
    if total and part.stat().st_size != total:
        raise IOError(f"Incomplete download: {part.stat().st_size:,} of {total:,} bytes")
    part.replace(dest)
    return dest


def _run_osmium(*args) -> None:
    if shutil.which("osmium") is None:
        raise RuntimeError(
            "The `osmium` command is not installed. Install the program osmium-tool: "
            "`conda install -c conda-forge osmium-tool` in a conda environment, or "
            "`apt-get install osmium-tool` in Google Colab."
        )
    result = subprocess.run(["osmium", *map(str, args), "--overwrite"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"osmium {args[0]} failed:\n{result.stderr}")


def crop_geofabrik_extract(aoi: AOI, pbf_path: Path, work_dir: Path, margin_m: float = 500) -> Path:
    """Cut ``pbf_path`` to ``aoi.boundary`` plus a ``margin_m`` margin and return the
    cropped ``.osm.pbf``. The margin (500 m, as OSMnx uses for its own download) keeps the
    streets at the edge connected to their real intersections; the graphs are truncated
    to the exact boundary later. Streets are kept whole (``complete_ways``)."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    boundary_path = work_dir / f"{aoi.slug}_boundary.geojson"
    area_pbf = work_dir / f"{aoi.slug}_area.osm.pbf"

    margin = gpd.GeoSeries([aoi.boundary], crs=4326).to_crs(aoi.local_epsg).buffer(margin_m).to_crs(4326)
    margin.to_file(boundary_path, driver="GeoJSON")
    _run_osmium("extract", "-p", boundary_path, "--strategy", "complete_ways", "-o", area_pbf, pbf_path)
    return area_pbf


def street_networks_from_geofabrik(
    aoi: AOI,
    pbf_path: Path,
    work_dir: Path,
    network_types: tuple[str, ...] = DEFAULT_NETWORK_TYPES,
    margin_m: float = 500,
) -> dict[str, "ox.MultiDiGraph"]:
    """Same result as :func:`download_street_networks`, built from a Geofabrik extract:
    one graph per network type, truncated to ``aoi.boundary`` and reprojected to
    ``aoi.local_epsg``. Intermediate files are written to ``work_dir``.

    ``aoi.boundary`` is the admin polygon for Method 1 and the drawn rectangle for
    Method 2, so a drawn area is cropped to its box."""
    unknown = [t for t in network_types if t not in _EXCLUDE]
    if unknown:
        raise ValueError(f"Unknown network type {unknown}; choose from {sorted(_EXCLUDE)}")

    work_dir = Path(work_dir)
    area_pbf = crop_geofabrik_extract(aoi, pbf_path, work_dir, margin_m)

    graphs = {}
    for network_type in network_types:
        kept_pbf = work_dir / f"{aoi.slug}_{network_type}_ways.osm.pbf"
        ways_xml = work_dir / f"{aoi.slug}_{network_type}.osm"
        # First remove the ways this network type leaves out, then keep the highway ways
        # with the nodes they use (the second step also drops the nodes of removed ways).
        _run_osmium("tags-filter", area_pbf, "-i", *_EXCLUDE[network_type], "-o", kept_pbf)
        _run_osmium("tags-filter", kept_pbf, "w/highway", "-o", ways_xml)

        graph = ox.graph_from_xml(
            ways_xml, bidirectional=network_type in ox.settings.bidirectional_network_types
        )
        graph = ox.truncate.truncate_graph_polygon(graph, aoi.boundary)
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
