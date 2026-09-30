"""Add streets that are missing from OSM to a saved street network.

Used by ``00-Data-Add_Missing_Streets.ipynb``. OSM often lags behind new roads. A
hand-drawn sketch (GeoJSON/GeoPackage lines from QGIS) is joined onto the network
saved by ``00-Data-Acquisition.ipynb`` (``osm/{slug}_{network_type}.gpkg``, layers
``nodes`` and ``edges``) and written back in the same layout, so every notebook that
reads the OSM network can read the updated file unchanged.

Joining means more than appending the lines: a street network is only connected
where two edges share a node. So each sketched end is moved onto the network, the
existing street it lands on is split there, and a new node is added at the split.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, substring

NODE_MERGE_M = 0.01  # points closer than this are the same junction


def load_network(path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Read the ``nodes`` and ``edges`` layers of a saved street network."""
    return gpd.read_file(path, layer="nodes"), gpd.read_file(path, layer="edges")


def save_network(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame, path) -> Path:
    """Write ``nodes`` and ``edges`` as the two layers of one GeoPackage."""
    path = Path(path)
    path.unlink(missing_ok=True)
    nodes.to_file(path, layer="nodes", driver="GPKG")
    edges.to_file(path, layer="edges", driver="GPKG")
    return path


def sketch_to_lines(new_streets: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One 2D LineString per row. QGIS often saves a single drawn line as a
    MultiLineString; parts that connect end-to-end are merged, the others become
    separate streets. Attribute columns are kept."""
    rows = []
    for _, row in new_streets.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if "Polygon" in geom.geom_type:
            raise ValueError(
                "The new streets must be drawn as lines (street centrelines), not polygons."
            )
        geom = shapely.force_2d(geom)
        if geom.geom_type == "MultiLineString":
            geom = linemerge(geom)
        for part in getattr(geom, "geoms", [geom]):
            if part.geom_type == "LineString" and part.length > NODE_MERGE_M:
                rows.append({**row.drop("geometry").to_dict(), "geometry": part})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=new_streets.crs)


class _Junctions:
    """Node lookup by position: an existing node, a node added earlier, or a new one.
    New nodes get negative ids (the OSM convention for objects not yet in OSM)."""

    def __init__(self, nodes: gpd.GeoDataFrame):
        self.ids = nodes["osmid"].tolist()
        self.xy = np.column_stack([nodes.geometry.x, nodes.geometry.y])
        self.new: dict[int, tuple] = {}
        self._next = min(0, min(self.ids)) - 1

    def nearest_existing(self, xy):
        d = np.hypot(self.xy[:, 0] - xy[0], self.xy[:, 1] - xy[1])
        i = int(d.argmin())
        return self.ids[i], tuple(self.xy[i]), float(d[i])

    def resolve(self, xy):
        """(node id, exact node position) for ``xy``; adds a node if none is there."""
        node_id, node_xy, d = self.nearest_existing(xy)
        if d <= NODE_MERGE_M:
            return node_id, node_xy
        for node_id, node_xy in self.new.items():
            if np.hypot(node_xy[0] - xy[0], node_xy[1] - xy[1]) <= NODE_MERGE_M:
                return node_id, node_xy
        node_id, self._next = self._next, self._next - 1
        self.new[node_id] = (float(xy[0]), float(xy[1]))
        return node_id, self.new[node_id]


def _points_of(geom) -> list[tuple]:
    """Junction points of an intersection result (for an overlap: its two ends)."""
    if geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [(geom.x, geom.y)]
    if geom.geom_type == "LineString":
        return [geom.coords[0][:2], geom.coords[-1][:2]]
    return [p for part in geom.geoms for p in _points_of(part)]


def _cut(line: LineString, points: list[tuple], junctions: _Junctions) -> list[tuple]:
    """Split ``line`` at ``points``. Returns ``(node_a, node_b, piece)`` per piece,
    each piece starting and ending exactly on its node."""
    stops = {0.0: line.coords[0][:2], line.length: line.coords[-1][:2]}
    for xy in points:
        d = line.project(Point(xy))
        if all(abs(d - s) > NODE_MERGE_M for s in stops):
            stops[d] = xy
    dists = sorted(stops)
    pieces = []
    for d0, d1 in zip(dists[:-1], dists[1:]):
        a, a_xy = junctions.resolve(stops[d0])
        b, b_xy = junctions.resolve(stops[d1])
        inner = list(substring(line, d0, d1).coords)[1:-1]
        pieces.append((a, b, LineString([a_xy, *inner, b_xy])))
    return pieces


def _snap_ends(lines: list[LineString], edges: gpd.GeoDataFrame, junctions: _Junctions, snap_tolerance: float):
    """Move each sketched end onto the network: to an existing junction if one is
    within ``snap_tolerance``, else onto the nearest existing street, else onto
    another new street. Ends with nothing in range stay where they are (dead ends)."""
    ends, report = {}, []
    for i, line in enumerate(lines):
        for which, xy in (("start", line.coords[0][:2]), ("end", line.coords[-1][:2])):
            _, node_xy, d_node = junctions.nearest_existing(xy)
            d_edge = edges.geometry.distance(Point(xy))
            if d_node <= snap_tolerance:
                ends[i, which], target = node_xy, "existing junction"
            elif d_edge.min() <= snap_tolerance:
                geom = edges.geometry.loc[d_edge.idxmin()]
                on_line = geom.interpolate(geom.project(Point(xy)))
                ends[i, which], target = (on_line.x, on_line.y), "along an existing street"
            else:
                ends[i, which], target = None, None
            report.append({"street": i, "end": which, "snapped_to": target})

    for row in report:
        i, which = row["street"], row["end"]
        if ends[i, which] is not None:
            continue
        xy = lines[i].coords[0 if which == "start" else -1][:2]
        best, best_d = xy, snap_tolerance
        for j, other in enumerate(lines):
            if j == i:
                continue
            for other_which in ("start", "end"):  # another new street's end first
                other_xy = ends[j, other_which] or other.coords[0 if other_which == "start" else -1][:2]
                d = float(np.hypot(other_xy[0] - xy[0], other_xy[1] - xy[1]))
                if d <= best_d:
                    best, best_d, row["snapped_to"] = other_xy, d, "another new street"
        if row["snapped_to"] is None:
            for j, other in enumerate(lines):
                if j != i and other.distance(Point(xy)) <= best_d:
                    on_line = other.interpolate(other.project(Point(xy)))
                    best, best_d = (on_line.x, on_line.y), other.distance(Point(xy))
                    row["snapped_to"] = "another new street"
        ends[i, which] = best
        row["snapped_to"] = row["snapped_to"] or "nothing (dead end)"

    snapped = []
    for i, line in enumerate(lines):
        snapped.append(LineString([ends[i, "start"], *[c[:2] for c in line.coords[1:-1]], ends[i, "end"]]))
    for row in report:
        xy = lines[row["street"]].coords[0 if row["end"] == "start" else -1][:2]
        to = ends[row["street"], row["end"]]
        row["moved_m"] = round(float(np.hypot(to[0] - xy[0], to[1] - xy[1])), 2)
    return snapped, pd.DataFrame(report)


def add_streets_to_network(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    new_streets: gpd.GeoDataFrame,
    snap_tolerance: float = 5.0,
    connect_crossings: bool = True,
    highway: str = "residential",
    name: str = "",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    """Join sketched streets onto a street network (``nodes``/``edges`` as read from
    ``osm/{slug}_{network_type}.gpkg``, in a metric CRS).

    - Each end of a new street is snapped onto the network (see :func:`_snap_ends`).
    - An existing street is split where a new street joins it, with a new node there.
    - ``connect_crossings``: where a new street crosses an existing street (or another
      new one) mid-way, both are split and joined, as at a normal junction. Set it to
      ``False`` if the crossings are bridges or underpasses.
    - New edges take ``highway`` / ``name`` from same-named columns of the sketch if it
      has them, else from the arguments.

    Returns ``(nodes, edges, report)``. Both layers get a ``source`` column (``"osm"``
    or ``"added"``); ``report`` has one row per sketched street end: what it was
    snapped to and how far it moved (m).
    """
    if edges.crs is None or not edges.crs.is_projected:
        raise ValueError("The network must be in a metric (projected) CRS; reproject it with .to_crs() first.")
    nodes = nodes.reset_index(drop=True)
    edges = edges.reset_index(drop=True)
    sketch = sketch_to_lines(new_streets.to_crs(edges.crs))
    if sketch.empty:
        raise ValueError("No line geometry found in the new streets file.")

    junctions = _Junctions(nodes)
    node_xy = dict(zip(junctions.ids, map(tuple, junctions.xy)))
    lines, report = _snap_ends(list(sketch.geometry), edges, junctions, snap_tolerance)
    keep = [i for i, line in enumerate(lines) if line.length > NODE_MERGE_M]
    sketch, lines = sketch.iloc[keep].reset_index(drop=True), [lines[i] for i in keep]

    # Every point where edges must meet: the new street ends, plus the crossings.
    points = [xy for line in lines for xy in (line.coords[0], line.coords[-1])]
    if connect_crossings:
        for i, line in enumerate(lines):
            for j in edges.sindex.query(line, predicate="intersects"):
                points += _points_of(line.intersection(edges.geometry.iloc[j]))
            for other in lines[i + 1:]:
                points += _points_of(line.intersection(other))

    # Split the existing streets at those points.
    cuts: dict[int, list] = {}
    for xy in points:
        p = Point(xy)
        for j in edges.sindex.query(p.buffer(1.0)):
            if edges.geometry.iloc[j].distance(p) <= NODE_MERGE_M:
                cuts.setdefault(int(j), []).append(xy)

    split_rows, replaced = [], []
    for j, cut_points in cuts.items():
        row = edges.iloc[j]
        pieces = _cut(row.geometry, cut_points, junctions)
        if len(pieces) < 2:
            continue  # the new street joins at this street's own end node
        replaced.append(j)
        start = row.geometry.coords[0]
        dist_u, dist_v = (np.hypot(node_xy[n][0] - start[0], node_xy[n][1] - start[1]) for n in (row["u"], row["v"]))
        start_is_u = dist_u <= dist_v
        start_node = row["u"] if start_is_u else row["v"]
        for a, b, piece in pieces:
            new = row.to_dict()
            new["u"], new["v"] = (a, b) if start_is_u else (b, a)
            if "from" in new and "to" in new:
                new["from"], new["to"] = (a, b) if row["from"] == start_node else (b, a)
            if "length" in new:
                new["length"] = row["length"] * piece.length / row.geometry.length
            new["geometry"] = piece
            split_rows.append(new)

    # Split the new streets at the same points and turn the pieces into edges.
    text_cols = [c for c in edges.columns if edges[c].dtype == object and c != "geometry"]
    new_rows = []
    for (_, attrs), line in zip(sketch.iterrows(), lines):
        on_line = [xy for xy in points if line.distance(Point(xy)) <= NODE_MERGE_M]
        for a, b, piece in _cut(line, on_line, junctions):
            new = {c: "" for c in text_cols}
            new.update(u=a, v=b, geometry=piece, length=piece.length, oneway=False, source="added")
            new["highway"] = attrs["highway"] if pd.notna(attrs.get("highway")) else highway
            new["name"] = attrs["name"] if pd.notna(attrs.get("name")) else name
            if "reversed" in edges.columns:
                new["reversed"] = "False" if edges["reversed"].dtype == object else False
            if "from" in edges.columns and "to" in edges.columns:
                new["from"], new["to"] = a, b
            new_rows.append(new)
    for n, new in enumerate(new_rows, start=1):
        if "osmid" in edges.columns:
            new["osmid"] = str(-n) if edges["osmid"].dtype == object else -n

    # Parallel edges between the same two nodes need different keys.
    kept = edges.drop(index=replaced)
    used: dict[tuple, set] = {}
    for u, v, key in zip(kept["u"], kept["v"], kept["key"]):
        used.setdefault((min(u, v), max(u, v)), set()).add(key)
    for new in split_rows + new_rows:
        keys = used.setdefault((min(new["u"], new["v"]), max(new["u"], new["v"])), set())
        new["key"] = next(k for k in range(len(keys) + 1) if k not in keys)
        keys.add(new["key"])

    if "source" not in kept.columns:
        kept = kept.assign(source="osm")
    added_edges = gpd.GeoDataFrame(split_rows + new_rows, geometry="geometry", crs=edges.crs)
    added_edges["source"] = added_edges["source"].fillna("osm") if "source" in added_edges else "osm"
    edges_out = gpd.GeoDataFrame(pd.concat([kept, added_edges], ignore_index=True), geometry="geometry", crs=edges.crs)
    edges_out = edges_out[[*edges.columns.drop("geometry"), *edges_out.columns.difference(edges.columns), "geometry"]]
    edges_out = edges_out.astype({c: edges[c].dtype for c in ("u", "v", "key", "from", "to", "oneway") if c in edges.columns})

    new_nodes = gpd.GeoDataFrame(
        {
            "osmid": list(junctions.new),
            "x": [xy[0] for xy in junctions.new.values()],
            "y": [xy[1] for xy in junctions.new.values()],
            "source": "added",
        },
        geometry=[Point(xy) for xy in junctions.new.values()],
        crs=nodes.crs,
    )
    if "source" not in nodes.columns:
        nodes = nodes.assign(source="osm")
    nodes_out = gpd.GeoDataFrame(pd.concat([nodes, new_nodes], ignore_index=True), geometry="geometry", crs=nodes.crs)
    nodes_out = nodes_out[[*nodes.columns.drop("geometry"), "geometry"]]

    # street_count (streets meeting at a node) for the nodes whose streets changed.
    if "street_count" in nodes_out.columns:
        changed = set(junctions.new) | {n for new in new_rows for n in (new["u"], new["v"])}
        ends = pd.concat([edges_out["u"], edges_out["v"]]).value_counts()
        is_changed = nodes_out["osmid"].isin(changed)
        counts = nodes_out.loc[is_changed, "osmid"].map(ends).fillna(0)
        nodes_out.loc[is_changed, "street_count"] = counts.values
        nodes_out["street_count"] = nodes_out["street_count"].astype(nodes["street_count"].dtype)

    return nodes_out, edges_out, report


def in_main_network(edges: gpd.GeoDataFrame) -> pd.Series:
    """True for every edge that belongs to the largest connected part of the network."""
    G = nx.Graph()
    G.add_edges_from(zip(edges["u"], edges["v"]))
    main = max(nx.connected_components(G), key=len)
    return edges["u"].isin(main)
