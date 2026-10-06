"""Walking isochrones for a chosen set of amenity types, packaged for a QGIS layout.

A lighter sibling of :mod:`sitex.network.poi_accessibility`: no H3 grid, no equity
metrics — just "which area can I reach on foot from each facility in 5/10/15
minutes", on a street network the student chooses, written to one GeoPackage in
the local metric CRS.

Two things differ from ``poi_accessibility`` on purpose:

- **Any street network.** :func:`network_from_files` builds the walking graph from
  any line layers — the OSM walk network saved by the data-acquisition notebook, a
  street layer drawn or edited in QGIS, or both combined (e.g. the existing network
  plus a proposed new street). Lines are snapped to a small grid and split at every
  crossing, so a student's hand-drawn street connects wherever it touches the
  existing network. :func:`network_from_osm` downloads the OSM walk network
  instead.
- **Isochrone shape** (``shape=``): ``"concave"`` (default) is a concave hull of the
  street junctions reached within the time, following Dolmajian (2023): a Delaunay
  triangulation of the junctions, keeping the triangles whose circumradius is at or
  below a percentile (starting at the 85th), raised one step at a time until the
  shape is a single polygon that contains every street walked. At the 100th
  percentile it equals the convex hull. Following the same article, the walk does
  not stop at the last junction: the end of every partly walked street (the
  "extended ego graph") and the facility itself are part of the shape. ``"convex"`` is the convex hull itself, the
  common OSMnx/NetworkX recipe (``ego_graph`` + convex hull) used by
  ``poi_accessibility``: smoother, but it also covers corners the network doesn't
  reach.

  Dolmajian, A. (2023). *Creating isochrones: what is the optimal way?* Medium.
  https://medium.com/@arthur.dolmajian/creating-isochrones-what-is-the-optimal-way-dfc77a2ca13a

**Why the concave hull, not a buffer around the walked streets.** Tested on the
Pleiku building footprints (21,389 buildings; 20 healthcare facilities and 25
kindergartens; 5/10/15 min): a building counts as truly reachable when the walk
along the streets plus the short walk from the street to the building fits in the
time. The concave hull matched that best (F1 0.948 healthcare, 0.956 kindergartens),
ahead of the convex hull (0.943 / 0.952) and of street buffers of 15-100 m (best at
50 m: 0.913 / 0.932). Half of Pleiku's buildings lie within 18 m of a walkable street,
90% within 44 m and 95% within 60 m. The test script is
``scripts/isochrone_shape_test.py`` (Experiments folder).

Walking speed: 4.8 km/h, as in Transport for London (2015), *Assessing transport
connectivity in London*, Table 2.1 (see :mod:`sitex.network.accessibility_common`).
Cycling speed: 15 km/h (:func:`with_speed`), the low end of the median speeds
Schuhmacher et al. (2025) measured for cyclists in Hamburg, 15.0-21.7 km/h depending on
the infrastructure: *Cycling Speeds in Urban Traffic*, Findings,
https://doi.org/10.32866/001c.141204. The bike uses the same network as walking.

Everything is computed in ``local_epsg`` (metres), so the exported layers measure
correctly in QGIS without reprojection.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, MultiPoint
from shapely.ops import substring, unary_union

from .accessibility_common import DEFAULT_WALK_SPEED_KMH, base_map

DEFAULT_TIME_BANDS = [5, 10, 15]
DEFAULT_BIKE_SPEED_KMH = 15.0  # Schuhmacher et al. (2025), low end of 15.0-21.7 km/h
ISOCHRONE_SHAPES = ("concave", "convex")
# Concave hull (Dolmajian, 2023): first percentile of the triangle circumradii tried.
CONCAVE_START_PERCENTILE = 85
# Also require every walked street inside the shape (the article's second condition).
CONCAVE_CONTAIN_STREETS = True


# ── 1. Street network ────────────────────────────────────────────────────────

def _read_lines(path: Path, local_epsg: int) -> gpd.GeoDataFrame:
    """Line features from a GeoPackage/GeoJSON/Shapefile. For an OSMnx GeoPackage
    (layers ``nodes`` + ``edges``) the ``edges`` layer is used."""
    import fiona

    layers = fiona.listlayers(path)
    layer = "edges" if "edges" in layers else layers[0]
    gdf = gpd.read_file(path, layer=layer).to_crs(epsg=local_epsg)
    gdf = gdf[gdf.geom_type.isin(["LineString", "MultiLineString"])]
    return gdf[["geometry"]]


def graph_from_lines(
    lines: gpd.GeoSeries, local_epsg: int, walk_speed_kmh: float = DEFAULT_WALK_SPEED_KMH, snap_m: float = 1.0,
) -> nx.MultiDiGraph:
    """Walkable graph from plain line geometries in ``local_epsg``.

    Coordinates are snapped to a ``snap_m`` grid (so ends drawn a few decimetres
    apart meet), all lines are split at every crossing, and each piece becomes a
    two-way edge with ``length`` (m) and ``travel_time`` (s) at a constant walking
    speed. Only the largest connected part is kept — a disconnected fragment would
    give any facility that snaps onto it a tiny, misleading isochrone.
    """
    geoms = shapely.set_precision(np.asarray(lines.explode(index_parts=False).values), snap_m)
    noded = shapely.get_parts(shapely.union_all(geoms))
    speed_mps = walk_speed_kmh * 1000 / 3600

    G = nx.MultiDiGraph(crs=f"EPSG:{local_epsg}")
    for seg in noded:
        if seg.is_empty or seg.length == 0:
            continue
        (x0, y0), (x1, y1) = seg.coords[0], seg.coords[-1]
        u, v = (round(x0, 2), round(y0, 2)), (round(x1, 2), round(y1, 2))
        G.add_node(u, x=u[0], y=u[1])
        G.add_node(v, x=v[0], y=v[1])
        attrs = dict(length=seg.length, travel_time=seg.length / speed_mps)
        G.add_edge(u, v, geometry=seg, **attrs)
        G.add_edge(v, u, geometry=LineString(seg.coords[::-1]), **attrs)

    largest = max(nx.weakly_connected_components(G), key=len)
    G = G.subgraph(largest).copy()
    return nx.convert_node_labels_to_integers(G)


def network_from_files(
    paths: list[Path], local_epsg: int, walk_speed_kmh: float = DEFAULT_WALK_SPEED_KMH, snap_m: float = 1.0,
) -> nx.MultiDiGraph:
    """Walking graph from one or more line files, combined (see :func:`graph_from_lines`)."""
    lines = pd.concat([_read_lines(Path(p), local_epsg) for p in paths], ignore_index=True)
    return graph_from_lines(lines.geometry, local_epsg, walk_speed_kmh, snap_m)


def network_from_osm(place: str, local_epsg: int, walk_speed_kmh: float = DEFAULT_WALK_SPEED_KMH) -> nx.MultiDiGraph:
    """OSM walk network for ``place``, at a constant walking pace, projected to ``local_epsg``."""
    import osmnx as ox

    from .accessibility_common import build_walk_graph

    G, _boundary = build_walk_graph(place, walk_speed_kmh)
    return ox.project_graph(G, to_crs=f"EPSG:{local_epsg}")


def with_speed(G: nx.MultiDiGraph, speed_kmh: float) -> nx.MultiDiGraph:
    """A copy of ``G`` with every edge's ``travel_time`` recomputed from its ``length``
    at ``speed_kmh`` -- e.g. the walk network ridden by bike. ``G`` is not changed."""
    H = G.copy()
    speed_mps = speed_kmh * 1000 / 3600
    for _, _, data in H.edges(data=True):
        data["travel_time"] = data["length"] / speed_mps
    return H


def network_edges(G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """One line per street segment (the two directions of a two-way edge merged)."""
    rows, seen = [], set()
    for u, v, data in G.edges(data=True):
        if (v, u) in seen:
            continue
        seen.add((u, v))
        rows.append({"length_m": round(data["length"], 1), "geometry": _edge_geometry(G, u, v, data)})
    return gpd.GeoDataFrame(rows, crs=G.graph["crs"])


def _edge_geometry(G, u, v, data) -> LineString:
    geom = data.get("geometry")
    if geom is None:
        geom = LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])
    return geom


# ── 2. Facilities on the network ─────────────────────────────────────────────

def load_points(
    path: Path, local_epsg: int, where: dict | None = None, label: str = "Facilities", layer: str | None = None,
) -> gpd.GeoDataFrame:
    """Facilities from any point file — an already-cleaned POI export, or points a
    student drew in QGIS — kept as they are, no classification.

    ``where`` keeps only rows whose column values are in the given lists, e.g.
    ``{"keep": [1], "amenity_type": ["Kindergarten"]}``. Missing ``id``/``name``
    columns are filled in, and every row gets ``amenity = label``.
    """
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    for col, values in (where or {}).items():
        if col not in gdf.columns:
            raise KeyError(f"{Path(path).name} has no column {col!r}; columns: {list(gdf.columns)}")
        gdf = gdf[gdf[col].isin(values)]
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.geom_type == "Point"].to_crs(epsg=local_epsg).reset_index(drop=True)

    if "id" not in gdf.columns:
        gdf["id"] = [f"p{i + 1}" for i in range(len(gdf))]
    if "name" not in gdf.columns:
        gdf["name"] = None
    gdf["name"] = gdf["name"].fillna(pd.Series([f"{label} {i + 1}" for i in range(len(gdf))]))
    gdf["amenity"] = label
    if "amenity_type" not in gdf.columns:
        gdf["amenity_type"] = label
    return gdf


def snap_facilities(G: nx.MultiDiGraph, pois: gpd.GeoDataFrame, max_snap_m: float = 250) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Attach each facility to its nearest junction. Facilities farther than
    ``max_snap_m`` from the network (outside the study area, or misplaced) are
    returned separately instead of being forced onto a distant street.
    Returns ``(on_network, off_network)``, both in the graph's CRS."""
    import osmnx as ox

    pois = pois.to_crs(G.graph["crs"]).copy()
    nodes, dists = ox.nearest_nodes(G, pois.geometry.x.values, pois.geometry.y.values, return_dist=True)
    pois["node"] = nodes
    pois["snap_dist_m"] = np.round(dists, 1)
    near = pois["snap_dist_m"] <= max_snap_m
    return pois[near].reset_index(drop=True), pois[~near].reset_index(drop=True)


# ── 3. Isochrones ────────────────────────────────────────────────────────────

def _walked_streets(G, nodes: set) -> list:
    """Every street whose two ends are both in ``nodes``, once per street."""
    out, seen = [], set()
    for u, v, data in G.subgraph(nodes).edges(data=True):
        if (v, u) not in seen:
            seen.add((u, v))
            out.append(_edge_geometry(G, u, v, data))
    return out


def concave_hull_dolmajian(points: MultiPoint, streets: list | None = None,
                           start_percentile: int = CONCAVE_START_PERCENTILE):
    """Concave hull of ``points`` after Dolmajian (2023).

    Delaunay-triangulate the points and keep the triangles whose circumradius is at
    or below the ``p``-th percentile of all circumradii. Starting at
    ``start_percentile``, ``p`` is raised by 1 until the shape is a single polygon
    (and, if ``streets`` is given, contains every one of those lines); at 100 every
    triangle is kept, which is the convex hull. The article allows either condition
    alone or both. Returns ``None`` for fewer than three points or points on one line.
    """
    tris = shapely.get_parts(shapely.delaunay_triangles(points))
    if len(tris) == 0:
        return None
    corners = np.array([np.asarray(t.exterior.coords)[:3] for t in tris])
    a = np.linalg.norm(corners[:, 1] - corners[:, 2], axis=1)
    b = np.linalg.norm(corners[:, 0] - corners[:, 2], axis=1)
    c = np.linalg.norm(corners[:, 0] - corners[:, 1], axis=1)
    area = shapely.area(tris)
    radius = np.where(area > 0, a * b * c / (4 * np.maximum(area, 1e-12)), np.inf)
    streets = shapely.union_all(streets) if streets else None

    shape = None
    for p in range(start_percentile, 101):
        keep = tris[radius <= np.percentile(radius, p)] if p < 100 else tris
        shape = shapely.union_all(keep)
        if shape.geom_type == "Polygon" and (streets is None or shape.buffer(0.5).covers(streets)):
            break
    return shape if shape is not None and shape.geom_type == "Polygon" else None


def _partly_walked(G, times: dict, limit: float, weight: str) -> list[LineString]:
    """The walked part of every street that is left but not finished within ``limit``
    — the "extended ego graph" of Dolmajian (2023). Without it the walk stops at the
    last junction reached, and on long blocks the rest of the time is lost."""
    parts = []
    for u, t_u in times.items():
        if t_u >= limit:
            continue
        for v, keyed in G[u].items():
            data = min(keyed.values(), key=lambda d: d[weight])
            cost = data[weight]
            if cost <= 0 or t_u + cost <= limit:
                continue  # walked to the end: v itself is reached
            geom = _edge_geometry(G, u, v, data)
            # Orient the geometry u -> v (an OSMnx edge can store it either way).
            start = shapely.Point(geom.coords[0])
            if start.distance(shapely.Point(G.nodes[u]["x"], G.nodes[u]["y"])) > \
               start.distance(shapely.Point(G.nodes[v]["x"], G.nodes[v]["y"])):
                geom = LineString(geom.coords[::-1])
            parts.append(substring(geom, 0.0, (limit - t_u) / cost, normalized=True))
    return [g for g in parts if g.length > 0]


def _hull(G, nodes: list, shape: str, extra_points: list | None = None, extra_streets: list | None = None,
          contain_streets: bool = CONCAVE_CONTAIN_STREETS):
    """The isochrone polygon around ``nodes`` (graph node ids) plus ``extra_points``
    (e.g. the facility itself and the ends of partly walked streets) — ``None`` if
    fewer than three points, since a point or a line has no area."""
    coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in nodes] + [p.coords[0] for p in (extra_points or [])]
    pts = MultiPoint(coords)
    if shape == "concave":
        streets = (_walked_streets(G, set(nodes)) + list(extra_streets or [])) if contain_streets else None
        return concave_hull_dolmajian(pts, streets)
    hull = pts.convex_hull
    return hull if hull.geom_type == "Polygon" else None


def _reach_polygon(G, times: dict, limit: float, shape: str, weight: str, origin=None):
    """Hull of everything walked within ``limit``: the junctions reached, the end of
    every partly walked street (extended ego graph), and ``origin`` (the facility)."""
    partial = _partly_walked(G, times, limit, weight)
    extra = [shapely.Point(g.coords[-1]) for g in partial] + ([origin] if origin is not None else [])
    return _hull(G, [n for n, d in times.items() if d <= limit], shape, extra, partial)


def _facility_polygons(
    G: nx.MultiDiGraph, facilities: gpd.GeoDataFrame, limits: list, weight: str,
    shape: str, out_col: str, out_values: list,
) -> gpd.GeoDataFrame:
    """One network search per facility out to ``max(limits)`` (in ``weight`` units),
    then one polygon per facility per limit — shared by isochrones (seconds) and
    distance service areas (metres)."""
    if shape not in ISOCHRONE_SHAPES:
        raise ValueError(f"shape must be one of {ISOCHRONE_SHAPES}, got {shape!r}")

    rows = []
    for _, fac in facilities.iterrows():
        reach = nx.single_source_dijkstra_path_length(G, fac["node"], cutoff=max(limits), weight=weight)
        for limit, value in zip(limits, out_values):
            poly = _reach_polygon(G, reach, limit, shape, weight, origin=fac.geometry)
            if poly is None:
                continue
            rows.append({
                "id": fac["id"], "name": fac["name"], "amenity": fac.get("amenity"),
                "amenity_type": fac.get("amenity_type"), out_col: value, "geometry": poly,
            })
    return gpd.GeoDataFrame(rows, columns=["id", "name", "amenity", "amenity_type", out_col, "geometry"],
                            geometry="geometry", crs=G.graph["crs"])


def facility_isochrones(
    G: nx.MultiDiGraph, facilities: gpd.GeoDataFrame, time_bands: list = DEFAULT_TIME_BANDS,
    shape: str = "concave",
) -> gpd.GeoDataFrame:
    """One walking search per facility (out to the largest band), one polygon per
    facility per band: the hull around every street junction reached within that time
    (``shape="concave"``, see :func:`concave_hull_dolmajian`; ``"convex"`` for the
    convex hull). ``facilities`` must carry a ``node`` column (:func:`snap_facilities`) and
    an ``amenity`` column — what the isochrones are drawn for: a group ("Healthcare")
    or a single type ("Kindergarten"). A facility that reaches fewer than three
    junctions in a band gets no polygon for it.
    Columns: ``id, name, amenity, amenity_type, time_min, geometry``."""
    return _facility_polygons(
        G, facilities, [t * 60 for t in time_bands], "travel_time", shape, "time_min", list(time_bands),
    )


def dissolve_isochrones(fac_iso: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge all facilities of an ``amenity`` per band: the area within ``time_min``
    of the *nearest* one. Each band is then merged with the smaller bands, so they
    always nest (5 inside 10 inside 15): two hulls drawn from different sets of
    junctions do not have to contain each other on their own. The same step is used
    in Raczyński's chronographic isochrones and on spatialworkflow.io."""
    out = fac_iso.dissolve(by=["amenity", "time_min"], as_index=False)[["amenity", "time_min", "geometry"]]
    out = out.sort_values(["amenity", "time_min"]).reset_index(drop=True)
    for _, idx in out.groupby("amenity").groups.items():
        prev = None
        for i in idx:
            if prev is not None:
                out.at[i, "geometry"] = shapely.union_all([out.at[i, "geometry"], prev])
            prev = out.at[i, "geometry"]
    out["area_ha"] = (out.area / 10_000).round(1)
    return out.sort_values(["amenity", "time_min"], ascending=[True, False]).reset_index(drop=True)


def isochrone_bands(isochrones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Non-overlapping rings (0-5, 5-10, 10-15 min) per amenity — the layer to style
    with a graduated colour ramp in QGIS without transparent bands stacking up."""
    rows = []
    for amenity, grp in isochrones.groupby("amenity"):
        grp = grp.sort_values("time_min")
        prev_t, prev_geom = 0, None
        for _, r in grp.iterrows():
            ring = r.geometry if prev_geom is None else shapely.make_valid(r.geometry.difference(prev_geom))
            rows.append({
                "amenity": amenity, "time_min": r.time_min,
                "band": f"{prev_t}-{r.time_min} min", "area_ha": round(ring.area / 10_000, 1), "geometry": ring,
            })
            prev_t, prev_geom = r.time_min, r.geometry
    return gpd.GeoDataFrame(rows, crs=isochrones.crs)


def service_areas(fac_iso: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Each facility's own polygon at the largest band, kept separate (they overlap)."""
    max_t = fac_iso["time_min"].max()
    return fac_iso[fac_iso["time_min"] == max_t].reset_index(drop=True)


def facility_service_areas(
    G: nx.MultiDiGraph, facilities: gpd.GeoDataFrame, distances_m: list, shape: str = "concave",
) -> gpd.GeoDataFrame:
    """Each facility's own reach within ``distances_m`` walked along the streets —
    a service radius measured on the network, not as the crow flies: the hull around
    the junctions reached within each distance. Kept per facility, so neighbouring
    areas overlap.
    Columns: ``id, name, amenity, amenity_type, distance_m, area_ha, geometry``."""
    out = _facility_polygons(G, facilities, list(distances_m), "length", shape, "distance_m", list(distances_m))
    out["area_ha"] = (out.area / 10_000).round(1)
    return out.sort_values(["distance_m", "id"], ascending=[False, True]).reset_index(drop=True)


def nearest_facility_catchments(
    G: nx.MultiDiGraph, facilities: gpd.GeoDataFrame, max_distance_m: float, shape: str = "concave",
) -> gpd.GeoDataFrame:
    """Split the walkable area between facilities: every street junction goes to
    its *nearest* facility along the network (within ``max_distance_m``), so the
    areas don't overlap — "which facility serves which neighbourhood".

    One network search from all facilities at once; each reached junction gets a
    Voronoi cell, and the cells are merged per facility. Each facility's area is then
    clipped to the hull of its own junctions (``shape``, as for the isochrones), so it
    does not spread past the last junction it reaches.
    Columns: ``id, name, amenity, n_facilities_here, area_ha, geometry``.
    """
    by_node: dict = {}
    for _, fac in facilities.iterrows():
        by_node.setdefault(fac["node"], []).append(fac)

    lengths, paths = nx.multi_source_dijkstra(G, set(by_node), cutoff=max_distance_m, weight="length")
    nodes = list(lengths)
    owner = [paths[n][0] for n in nodes]

    pts = shapely.points([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in nodes])
    cells = shapely.get_parts(shapely.voronoi_polygons(shapely.multipoints(pts)))
    tree = shapely.STRtree(cells)
    pt_idx, cell_idx = tree.query(pts, predicate="within")
    cell_owner = dict(zip(cell_idx, [owner[i] for i in pt_idx]))

    rows = []
    for node, facs in by_node.items():
        mine = [cells[c] for c, o in cell_owner.items() if o == node]
        hull = _hull(G, [n for n, o in zip(nodes, owner) if o == node], shape)
        if not mine or hull is None:
            continue
        area = unary_union(mine).intersection(hull)
        if area.is_empty:
            continue
        fac = facs[0]
        rows.append({
            "id": fac["id"], "name": fac["name"], "amenity": fac.get("amenity"),
            # two facilities snapped to the same junction share one catchment
            "n_facilities_here": len(facs), "area_ha": round(area.area / 10_000, 1), "geometry": area,
        })
    return gpd.GeoDataFrame(rows, crs=G.graph["crs"])


def count_buildings(areas: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame, col: str = "n_buildings") -> gpd.GeoDataFrame:
    """Add ``col``: the number of buildings whose centre lies in each area — a simple
    stand-in for the people a facility serves, when no population data is at hand.
    Service areas overlap, so a building can be counted for several facilities there;
    catchments don't, so each building is counted once."""
    centres = buildings.to_crs(areas.crs).geometry.centroid.values
    tree = shapely.STRtree(centres)
    out = areas.copy()
    out[col] = [len(tree.query(g, predicate="contains")) for g in out.geometry]
    return out


def count_facilities_by_hex(G: nx.MultiDiGraph, hexes: gpd.GeoDataFrame, facilities: gpd.GeoDataFrame,
                            minutes: float = 15) -> pd.DataFrame:
    """Number of facilities within ``minutes`` of walking, from the centre of every hexagon.

    Each hexagon centre is attached to its nearest junction, and walked from there
    along the network (like the facilities in :func:`snap_facilities`). ``facilities``
    must be snapped already (column ``node``); a facility listed twice counts once.

    Returns one row per hexagon of ``hexes``: ``h3_id``, ``n_amenities`` and
    ``snap_dist_m`` (metres from the hexagon centre to its junction; a large value
    means the hexagon is far from any street, so its count says little).
    """
    import osmnx as ox

    centres = hexes.to_crs(G.graph["crs"]).geometry.centroid
    origin, snap = ox.nearest_nodes(G, centres.x.values, centres.y.values, return_dist=True)

    unique = facilities.drop_duplicates("id") if "id" in facilities.columns else facilities
    per_node: dict = {}
    for node in unique["node"]:
        per_node[node] = per_node.get(node, 0) + 1

    reached: dict = {}  # junction -> facilities within reach; the network is two-way, so
    for node, n in per_node.items():  # "facility reaches junction" equals "junction reaches facility"
        times = nx.single_source_dijkstra_path_length(G, node, cutoff=minutes * 60, weight="travel_time")
        for junction in times:
            reached[junction] = reached.get(junction, 0) + n

    return pd.DataFrame({
        "h3_id": hexes["h3_id"].values,
        "n_amenities": [reached.get(node, 0) for node in origin],
        "snap_dist_m": np.round(snap, 1),
    })


# ── 4. Export & preview ──────────────────────────────────────────────────────

def export_geopackage(path: Path, layers: dict[str, gpd.GeoDataFrame]) -> Path:
    """Write every layer into one GeoPackage (replaced if it already exists), so a
    QGIS project only has to point at one file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    for name, gdf in layers.items():
        if gdf is not None and not gdf.empty:
            gdf.to_file(path, layer=name, driver="GPKG")
    return path


BAND_COLORS = ["#1a9850", "#fee08b", "#d73027", "#762a83", "#5e4fa2"]  # inner -> outer


def preview_map(bands: gpd.GeoDataFrame, facilities: gpd.GeoDataFrame, zoom_start: int = 14, mode: str = "Walk"):
    """Folium check-map: one toggleable layer per amenity (first one shown). ``mode``
    labels the tooltip, e.g. "Walk: 5-10 min" or "Bike: 5-10 min"."""
    import folium

    bands_ll = bands.to_crs(4326)
    fac_ll = facilities.to_crs(4326)
    minx, miny, maxx, maxy = bands_ll.total_bounds
    m = base_map([(miny + maxy) / 2, (minx + maxx) / 2], zoom_start)

    times = sorted(bands_ll["time_min"].unique())
    color_of = {t: BAND_COLORS[i % len(BAND_COLORS)] for i, t in enumerate(times)}

    for i, (amenity, grp) in enumerate(bands_ll.groupby("amenity")):
        fg = folium.FeatureGroup(name=amenity, show=(i == 0))
        folium.GeoJson(
            grp[["band", "time_min", "geometry"]].sort_values("time_min", ascending=False),
            style_function=lambda f: {"fillColor": color_of[f["properties"]["time_min"]], "color": "none", "fillOpacity": 0.45},
            tooltip=folium.GeoJsonTooltip(fields=["band"], aliases=[f"{mode}:"]),
        ).add_to(fg)
        for _, p in fac_ll[fac_ll["amenity"] == amenity].iterrows():
            folium.CircleMarker([p.geometry.y, p.geometry.x], radius=4, color="#222", weight=1,
                                fill=True, fill_color="#fff", fill_opacity=1, tooltip=p["name"]).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def _distinct_color(i: int) -> str:
    import colorsys

    r, g, b = colorsys.hls_to_rgb((i * 0.618034) % 1.0, 0.55, 0.65)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def preview_service_map(catchments: gpd.GeoDataFrame, areas: gpd.GeoDataFrame, facilities: gpd.GeoDataFrame, zoom_start: int = 14):
    """Folium check-map: nearest-facility catchments (one colour per facility) and,
    as switchable outlines, each distance band of the per-facility service areas."""
    import folium

    cat_ll = catchments.to_crs(4326)
    minx, miny, maxx, maxy = cat_ll.total_bounds
    m = base_map([(miny + maxy) / 2, (minx + maxx) / 2], zoom_start)

    color_of = {fid: _distinct_color(i) for i, fid in enumerate(cat_ll["id"])}
    fg = folium.FeatureGroup(name="Nearest-facility catchments", show=True)
    folium.GeoJson(
        cat_ll[["id", "name", "area_ha", "geometry"] + (["n_buildings"] if "n_buildings" in cat_ll else [])],
        style_function=lambda f: {"fillColor": color_of[f["properties"]["id"]], "color": "#333", "weight": 0.8, "fillOpacity": 0.45},
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "area_ha"] + (["n_buildings"] if "n_buildings" in cat_ll else []),
            aliases=["Nearest facility:", "Area (ha):"] + (["Buildings:"] if "n_buildings" in cat_ll else []),
        ),
    ).add_to(fg)
    fg.add_to(m)

    for dist, grp in areas.to_crs(4326).groupby("distance_m"):
        fg = folium.FeatureGroup(name=f"Service areas, {dist} m", show=False)
        folium.GeoJson(
            grp[["name", "distance_m", "geometry"]],
            style_function=lambda f: {"fillOpacity": 0, "color": "#1B7192", "weight": 1.5},
            tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=[f"{dist} m from:"]),
        ).add_to(fg)
        fg.add_to(m)

    fac_fg = folium.FeatureGroup(name="Facilities", show=True)
    for _, p in facilities.to_crs(4326).iterrows():
        folium.CircleMarker([p.geometry.y, p.geometry.x], radius=4, color="#222", weight=1,
                            fill=True, fill_color="#fff", fill_opacity=1, tooltip=p["name"]).add_to(fac_fg)
    fac_fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
