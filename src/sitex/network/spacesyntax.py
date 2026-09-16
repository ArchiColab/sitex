"""Space Syntax — configurational analysis of a street network.

Extracted from ``03-NA01-SpaceSyntax.ipynb``. Function set adapted from Nabil Mohareb
(2024), The American University in Cairo (space-syntaxNM). A street edges GeoDataFrame
becomes two maps — **axial** (fewer, longer, merged lines; topological hop-count depth)
and **segment** (every individual street segment; true metric distance and angular turn
cost) — analysed almost entirely in parallel: they answer related but distinct
questions and share helper functions, never the same underlying graph.
"""

from __future__ import annotations

import math
from collections import defaultdict

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import linemerge

RADII_M = {
    "R400": 400,           # pedestrian local
    "R800": 800,           # pedestrian catchment
    "R1200": 1200,         # neighbourhood
    "R2000": 2000,         # neighbourhood-regional
    "Rn": float("inf"),    # global (whole system)
}


# ── 1. Axial & segment maps from a streets GeoDataFrame ─────────────────────

def _dedupe_bidirectional(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """OSMnx stores both u->v and v->u for the same physical edge; keep one."""
    if "u" not in gdf.columns or "v" not in gdf.columns:
        return gdf
    seen, keep = set(), []
    for i, row in gdf.iterrows():
        key = (min(row["u"], row["v"]), max(row["u"], row["v"]), row.get("key", 0))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return gdf.loc[keep].reset_index(drop=True)


def _bearing(geom) -> float:
    c = list(geom.coords)
    dx, dy = c[-1][0] - c[0][0], c[-1][1] - c[0][1]
    return math.degrees(math.atan2(dy, dx)) % 180


def graph_to_segment_map(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Each undirected edge becomes one segment. Columns: seg_id, length_m, bearing_deg, geometry."""
    gdf = _dedupe_bidirectional(gdf[gdf.geometry.geom_type == "LineString"].copy().reset_index(drop=True))
    return gpd.GeoDataFrame(
        {
            "seg_id": range(len(gdf)),
            "length_m": gdf.geometry.length.values,
            "bearing_deg": [_bearing(g) for g in gdf.geometry],
            "geometry": gdf.geometry.values,
        },
        crs=gdf.crs,
    )


def graph_to_axial_map(gdf: gpd.GeoDataFrame, angular_tolerance: float = 20.0) -> gpd.GeoDataFrame:
    """Connected edges whose bearings agree within ``angular_tolerance`` degrees are
    merged into single axial lines via union-find grouping at shared endpoints — a
    computational stand-in for the classical hand-digitised "least-line" axial map.
    Columns: axial_id, length_m, bearing_deg, n_merged, geometry.
    """
    gdf = _dedupe_bidirectional(gdf[gdf.geometry.geom_type == "LineString"].copy().reset_index(drop=True))

    def _bearing_diff(b1, b2):
        diff = abs(b1 - b2) % 180
        return min(diff, 180 - diff)

    gdf["bearing"] = [_bearing(g) for g in gdf.geometry]

    node_edges = defaultdict(list)
    for i, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        start = tuple(round(c, 2) for c in coords[0][:2])
        end = tuple(round(c, 2) for c in coords[-1][:2])
        node_edges[start].append(i)
        node_edges[end].append(i)

    parent = list(range(len(gdf)))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x, y):
        parent[find(x)] = find(y)

    for edge_idxs in node_edges.values():
        for i in range(len(edge_idxs)):
            for j in range(i + 1, len(edge_idxs)):
                ei, ej = edge_idxs[i], edge_idxs[j]
                if _bearing_diff(gdf.loc[ei, "bearing"], gdf.loc[ej, "bearing"]) <= angular_tolerance:
                    union(ei, ej)

    groups = defaultdict(list)
    for i in range(len(gdf)):
        groups[find(i)].append(i)

    records = []
    for indices in groups.values():
        geoms = [gdf.loc[i, "geometry"] for i in indices]
        merged = linemerge(geoms) if len(geoms) > 1 else geoms[0]
        parts = merged.geoms if merged.geom_type == "MultiLineString" else [merged]
        for part in parts:
            records.append({
                "axial_id": len(records),
                "length_m": part.length,
                "bearing_deg": _bearing(part),
                "n_merged": len(indices),
                "geometry": part,
            })

    return gpd.GeoDataFrame(records, crs=gdf.crs)


def export_lines_to_dxf(gdf: gpd.GeoDataFrame, filepath) -> None:
    """Export a LineString GeoDataFrame as a DXF (R2010) — 2-point chords only.

    Flattens every line to a straight start/end chord: harmless for axial lines
    (already straight) but would silently straighten a curved segment — use
    GeoJSON/GPKG, not a DXF round-trip, as the scenario-testing sketch input
    (Section 6 of the original notebook explains why in detail).
    """
    import ezdxf

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    for geom in gdf.geometry:
        coords = list(geom.coords)
        if len(coords) >= 2:
            msp.add_line(coords[0][:2], coords[-1][:2])
    doc.saveas(filepath)


# ── 2. NetworkX graph from an axial/segment GeoDataFrame ────────────────────

def gdf_to_nx_graph(gdf: gpd.GeoDataFrame, id_col: str) -> nx.Graph:
    """Nodes are endpoint coordinates rounded to 6 decimals; edges carry id, geometry,
    weight (length, m), and angle (radians)."""
    G = nx.Graph()
    for _, row in gdf.iterrows():
        geom = row.geometry
        coords = list(geom.coords)
        start = tuple(round(c, 6) for c in coords[0][:2])
        end = tuple(round(c, 6) for c in coords[-1][:2])
        angle = abs(math.atan2(end[1] - start[1], end[0] - start[0]))
        G.add_edge(start, end, id=int(row[id_col]), geometry=geom, weight=geom.length, angle=angle)
        G.nodes[start]["pos"] = start
        G.nodes[end]["pos"] = end
    return G


# ── 3. Axial metrics ─────────────────────────────────────────────────────────

def axial_connectivity(G: nx.Graph) -> dict:
    """Directly connected axial lines per line."""
    conn = {data["id"]: 0 for _, _, data in G.edges(data=True)}
    for _, _, data in G.edges(data=True):
        conn[data["id"]] += 1
    return conn


def axial_integration(G: nx.Graph, radius: float = float("inf")) -> dict:
    """Real Relative Asymmetry (RRA) closeness (inverse mean topological depth)."""
    integration = {data["id"]: 0 for _, _, data in G.edges(data=True)}
    for u, v, data in G.edges(data=True):
        edge_id = data["id"]
        lengths = nx.single_source_shortest_path_length(G, u, cutoff=radius)
        depths = {t: d for t, d in lengths.items() if t != u}
        n = len(depths)
        if n > 1:
            md = sum(depths.values()) / n
            ra = 2 * (md - 1) / (n - 1)
            dv = (n ** 2 - 1) / (8 * (n - 1))
            rra = ra / dv
            integration[edge_id] = 1 / rra if rra != 0 else 0
    return integration


def axial_choice(G: nx.Graph, radius: float = float("inf"), weight: str = "weight", normalize: bool = False) -> dict:
    """Betweenness Choice: how many shortest paths (within ``radius`` metres) pass
    through each axial line. ``weight=None`` switches to topological hops instead."""
    choice = {data["id"]: 0.0 for _, _, data in G.edges(data=True)}
    nodes = list(G.nodes())
    n = len(nodes)

    for source in nodes:
        _, paths = nx.single_source_dijkstra(G, source, cutoff=radius, weight=weight)
        for target, path in paths.items():
            if target == source:
                continue
            for u, v in nx.utils.pairwise(path):
                if G.has_edge(u, v):
                    choice[G[u][v]["id"]] += 1

    if normalize and n > 2:
        denom = (n - 1) * (n - 2)
        choice = {k: v / denom for k, v in choice.items()}
    return choice


def axial_choice_length_weighted(G: nx.Graph, radius: float = float("inf"), weight: str = "weight") -> dict:
    """Length-weighted betweenness Choice (Hillier & Iida, 2005): each path
    contributes ``l(origin) * l(destination)`` to every segment it crosses, so
    longer segments generate proportionally more through-movement potential."""
    choice = {data["id"]: 0.0 for _, _, data in G.edges(data=True)}
    for source in G.nodes():
        _, paths = nx.single_source_dijkstra(G, source, cutoff=radius, weight=weight)
        for target, path in paths.items():
            if target == source or len(path) < 2:
                continue
            edge_pairs = list(nx.utils.pairwise(path))
            fu, fv = edge_pairs[0]
            lu, lv = edge_pairs[-1]
            l_orig = G[fu][fv]["weight"] if G.has_edge(fu, fv) else 1.0
            l_dest = G[lu][lv]["weight"] if G.has_edge(lu, lv) else 1.0
            path_w = l_orig * l_dest
            for u, v in edge_pairs:
                if G.has_edge(u, v):
                    choice[G[u][v]["id"]] += path_w
    return choice


def axial_choice_multiscale(G: nx.Graph, radii: dict | None = None, weight: str = "weight", normalize: bool = False) -> dict:
    """Axial Choice at every radius in ``radii`` (default: R400/R800/R1200/R2000/Rn)
    in one call. Returns ``{radius_name: choice_dict}``."""
    radii = radii or RADII_M
    return {name: axial_choice(G, radius=r, weight=weight, normalize=normalize) for name, r in radii.items()}


# ── 4. Segment metrics ───────────────────────────────────────────────────────

def seg_choice(G: nx.Graph, radius: float = float("inf")) -> dict:
    """Topological betweenness (hop-count shortest paths) for each segment."""
    choice = {data["id"]: 0 for _, _, data in G.edges(data=True)}
    for source in G.nodes():
        for path in nx.single_source_dijkstra_path(G, source, cutoff=radius).values():
            for u, v in nx.utils.pairwise(path):
                if G.has_edge(u, v):
                    choice[G[u][v]["id"]] += 1
    return choice


def seg_integration(G: nx.Graph, radius: float = float("inf")) -> dict:
    """Closeness via n² / total_depth."""
    integ = {data["id"]: 0 for _, _, data in G.edges(data=True)}
    for u, v, data in G.edges(data=True):
        lengths = nx.single_source_shortest_path_length(G, u, cutoff=radius)
        depths = {t: d for t, d in lengths.items() if t != u}
        n, td = len(depths), sum(depths.values())
        if n > 0 and td > 0:
            integ[data["id"]] = (n ** 2) / td
    return integ


def _build_dual_graph(G: nx.Graph):
    """Dual (line) graph for Angular Segment Analysis: nodes are segment IDs, edges
    connect segments sharing an intersection node, weighted by true angular turn
    cost in [0, 2] (0 = straight through, 2 = full reversal) — the depthmapX
    convention (``angle_difference_degrees / 90``), not the raw bearing difference
    an earlier version of this analysis mistakenly used as an edge weight.
    """
    edge_info = {}
    for u, v, data in G.edges(data=True):
        eid = data["id"]
        geom = data.get("geometry")
        coords = list(geom.coords)
        dx = coords[-1][0] - coords[0][0]
        dy = coords[-1][1] - coords[0][1]
        bearing = math.degrees(math.atan2(dy, dx)) % 180
        edge_info[eid] = {"u": u, "v": v, "bearing": bearing, "length": data["weight"]}

    node_to_eids = defaultdict(list)
    for eid, info in edge_info.items():
        node_to_eids[info["u"]].append(eid)
        node_to_eids[info["v"]].append(eid)

    L = nx.Graph()
    for eid, info in edge_info.items():
        L.add_node(eid, length=info["length"])

    for eids in node_to_eids.values():
        for i in range(len(eids)):
            for j in range(i + 1, len(eids)):
                ei, ej = eids[i], eids[j]
                diff = abs(edge_info[ei]["bearing"] - edge_info[ej]["bearing"]) % 180
                ang_deg = min(diff, 180 - diff)
                ang_cost = ang_deg / 90.0
                if not L.has_edge(ei, ej) or L[ei][ej]["weight"] > ang_cost:
                    L.add_edge(ei, ej, weight=ang_cost)
    return L, edge_info


def seg_angular_choice(G: nx.Graph, radius: float = float("inf")) -> dict:
    """Angular Betweenness Choice via the dual graph — depthmapX's Angular Segment
    Analysis (ASA): shortest paths minimise total angular turn cost, which
    correlates strongly with pedestrian/vehicular movement (Hillier & Iida, 2005;
    Turner, 2007)."""
    L, _ = _build_dual_graph(G)
    choice = {node: 0.0 for node in L.nodes()}
    for source in L.nodes():
        _, paths = nx.single_source_dijkstra(L, source, cutoff=radius, weight="weight")
        for target, path in paths.items():
            if target == source or len(path) < 2:
                continue
            for seg_id in path:
                choice[seg_id] += 1
    return choice


def seg_angular_choice_metric(G: nx.Graph, radius_m: float = float("inf")) -> dict:
    """Angular betweenness restricted to a metric catchment radius — combines a
    metric radius with angular cost, the standard Space Syntax practice."""
    L, edge_info = _build_dual_graph(G)

    L_metric = nx.Graph()
    for eid in edge_info:
        L_metric.add_node(eid)
    for ei, ej in L.edges():
        avg_len = (edge_info[ei]["length"] + edge_info[ej]["length"]) / 2
        L_metric.add_edge(ei, ej, weight=avg_len)

    choice = {node: 0.0 for node in L.nodes()}
    for source in L.nodes():
        metric_dist, _ = nx.single_source_dijkstra(L_metric, source, cutoff=radius_m, weight="weight")
        reachable = set(metric_dist.keys())
        sub = L.subgraph(reachable)
        _, ang_paths = nx.single_source_dijkstra(sub, source, weight="weight")
        for target, path in ang_paths.items():
            if target == source or len(path) < 2:
                continue
            for seg_id in path:
                choice[seg_id] += 1
    return choice


def seg_choice_length_weighted(G: nx.Graph, radius: float = float("inf")) -> dict:
    """Length-weighted topological betweenness (Hillier & Iida, 2005)."""
    choice = {data["id"]: 0.0 for _, _, data in G.edges(data=True)}
    for source in G.nodes():
        _, paths = nx.single_source_dijkstra(G, source, cutoff=radius, weight="weight")
        for target, path in paths.items():
            if target == source or len(path) < 2:
                continue
            edge_pairs = list(nx.utils.pairwise(path))
            fu, fv = edge_pairs[0]
            lu, lv = edge_pairs[-1]
            l_o = G[fu][fv]["weight"] if G.has_edge(fu, fv) else 1.0
            l_d = G[lu][lv]["weight"] if G.has_edge(lu, lv) else 1.0
            w = l_o * l_d
            for u, v in edge_pairs:
                if G.has_edge(u, v):
                    choice[G[u][v]["id"]] += w
    return choice


def seg_angular_integration(G: nx.Graph, radius: float = float("inf")) -> dict:
    """Closeness based on angular depth from each segment."""
    ang_int = {data["id"]: 0 for _, _, data in G.edges(data=True)}
    for u, v, data in G.edges(data=True):
        lengths = nx.single_source_dijkstra_path_length(G, u, weight="angle", cutoff=radius)
        depths = {t: d for t, d in lengths.items() if t != u}
        n, mad = len(depths), sum(depths.values())
        if n > 0 and mad > 0:
            ang_int[data["id"]] = n / (mad / n)
    return ang_int


def seg_nach(G: nx.Graph, radius: float = float("inf")) -> dict:
    """Normalised Angular Choice: ``log(choice+1) / log(mean_total_depth+3)`` — high
    values mean short, straight, central routes."""
    ch = seg_choice(G, radius)
    total_depth = {node: 0 for node in G.nodes()}
    for node in G.nodes():
        for t, d in nx.single_source_shortest_path_length(G, node, cutoff=radius).items():
            total_depth[t] += d
    nach = {}
    for u, v, data in G.edges(data=True):
        eid = data["id"]
        c, tdu, tdv = ch.get(eid, 0), total_depth.get(u, 0), total_depth.get(v, 0)
        if c == 0 or tdu == 0 or tdv == 0:
            nach[eid] = 0
        else:
            try:
                nach[eid] = math.log(c + 1) / math.log((tdu + tdv) / 2 + 3)
            except ValueError:
                nach[eid] = 0
    return nach


def seg_norm_integration(G: nx.Graph, radius: float = float("inf")) -> dict:
    """Min-max normalised segment integration, for cross-comparison."""
    integ = seg_integration(G, radius)
    lo, hi = min(integ.values()), max(integ.values())
    if lo == hi:
        return {k: 1.0 for k in integ}
    return {k: (v - lo) / (hi - lo) for k, v in integ.items()}


def seg_reach(G: nx.Graph, radius_m: float = 800) -> dict:
    """Metric Reach: total street length (m) reachable within ``radius_m`` metres
    of either endpoint of a segment (union of both catchments)."""
    node_reach = {}
    for node in G.nodes():
        dists = nx.single_source_dijkstra_path_length(G, node, cutoff=radius_m, weight="weight")
        node_reach[node] = set(dists.keys())

    reach = {}
    for u, v, data in G.edges(data=True):
        reachable = node_reach[u] | node_reach[v]
        sub = G.subgraph(reachable)
        reach[data["id"]] = sum(d["weight"] for _, _, d in sub.edges(data=True))
    return reach


def seg_reach_multiscale(G: nx.Graph, radii: dict | None = None) -> dict:
    """Metric Reach at multiple radii (default R400/R800/R2000) in one precomputed
    pass — one Dijkstra per node up to the largest radius, then filtered per radius."""
    radii = radii or {"R400": 400, "R800": 800, "R2000": 2000}
    max_r = max(radii.values())

    node_dists = {
        node: nx.single_source_dijkstra_path_length(G, node, cutoff=max_r, weight="weight")
        for node in G.nodes()
    }

    results = {}
    for name, r in radii.items():
        reach = {}
        for u, v, data in G.edges(data=True):
            reach_u = {n for n, d in node_dists[u].items() if d <= r}
            reach_v = {n for n, d in node_dists[v].items() if d <= r}
            sub = G.subgraph(reach_u | reach_v)
            reach[data["id"]] = sum(d["weight"] for _, _, d in sub.edges(data=True))
        results[name] = reach
    return results


# ── 5. Merge metrics into a GeoDataFrame ─────────────────────────────────────

def metrics_to_gdf(G: nx.Graph, metrics_dict: dict, crs, id_col: str = "id") -> gpd.GeoDataFrame:
    """Combine multiple ``{id: value}`` metric dicts into one GeoDataFrame, one row
    per graph edge."""
    rows = []
    for _, _, d in G.edges(data=True):
        row = {id_col: d["id"], "geometry": d["geometry"]}
        for col, m in metrics_dict.items():
            row[col] = m.get(d["id"], 0)
        rows.append(row)
    return gpd.GeoDataFrame(rows, crs=crs)


# ── 6. Folium visualization (not matplotlib — see style_by_metric's docstring) ──

def style_by_metric(gdf: gpd.GeoDataFrame, col: str, cmap_name: str, clip: tuple = (5, 95), log_scale: bool = True) -> gpd.GeoDataFrame:
    """Add ``_color``/``_width`` columns for a folium line map, styled by ``col``.

    Values are optionally log1p-transformed then clipped to a percentile range
    before colour-mapping — a handful of extreme betweenness values would
    otherwise wash out the rest of the network's contrast. This colorizes via
    pure array/colormap lookup, not a matplotlib ``GeoDataFrame.plot()`` call —
    the latter has been observed to crash the interpreter outright in this
    project's environment (see the ARCH-module Building Morphology migration).
    """
    import matplotlib as mpl
    import matplotlib.colors as mcolors

    raw = gdf[col].fillna(0).astype(float).values
    vals = np.log1p(raw) if log_scale else raw.copy()
    p_lo, p_hi = np.percentile(vals, clip[0]), np.percentile(vals, clip[1])
    if p_hi <= p_lo:
        p_hi = p_lo + 1.0
    norm = mcolors.Normalize(vmin=p_lo, vmax=p_hi, clip=True)
    cmap_fn = mpl.colormaps[cmap_name]
    out = gdf.copy()
    out["_color"] = [mcolors.to_hex(cmap_fn(norm(v))) for v in vals]
    out["_width"] = [0.8 + 4.2 * float(norm(v)) for v in vals]
    return out


def style_by_metric_diverging(gdf: gpd.GeoDataFrame, col: str, cmap_name: str = "coolwarm", clip_abs_percentile: float = 98) -> gpd.GeoDataFrame:
    """Like ``style_by_metric``, but for a signed delta: the colour scale is
    centred on 0 (``vmin=-vmax, vmax=vmax``) instead of stretched to the data's
    own min/max, so a positive and a negative value of the same magnitude get
    symmetric colours."""
    import matplotlib as mpl
    import matplotlib.colors as mcolors

    raw = gdf[col].fillna(0).astype(float).values
    vmax = np.percentile(np.abs(raw), clip_abs_percentile) or 1.0
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax, clip=True)
    cmap_fn = mpl.colormaps[cmap_name]
    out = gdf.copy()
    out["_color"] = [mcolors.to_hex(cmap_fn(norm(v))) for v in raw]
    out["_width"] = [1.0 + 3.5 * abs(float(norm(v)) - 0.5) * 2 for v in raw]
    return out


def add_metric_layer(m, gdf: gpd.GeoDataFrame, style_col: str, name: str, cmap: str, tooltip_cols: list, id_col: str = "id", show: bool = False, log_scale: bool = True, diverging: bool = False) -> None:
    """Style ``gdf`` by ``style_col`` and add it as a toggleable folium GeoJson layer."""
    import folium

    styler = style_by_metric_diverging if diverging else style_by_metric
    kwargs = {} if diverging else {"log_scale": log_scale}
    s = styler(gdf, style_col, cmap, **kwargs)

    keep = [id_col, "_color", "_width", "geometry"] + [
        c for c in tooltip_cols if c in s.columns and c not in (id_col, "_color", "_width", "geometry")
    ]
    folium.GeoJson(
        s[keep].__geo_interface__,
        name=name,
        style_function=lambda f: {
            "color": f["properties"]["_color"],
            "weight": f["properties"]["_width"],
            "opacity": 0.85,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[id_col] + [c for c in tooltip_cols if c in s.columns],
            aliases=[id_col] + [c for c in tooltip_cols if c in s.columns],
        ),
        show=show,
    ).add_to(m)


def compute_local_centre_index(gdf: gpd.GeoDataFrame, choice_r800_col: str = "choice_R800", choice_rn_col: str = "choice_Rn") -> pd.Series:
    """``log(Choice_R800 + 1) / log(Choice_Rn + 1)`` — near 1.0 means a segment
    matters as much locally as globally (a genuine local centre); a low value
    means it only matters city-wide (an arterial/bypass)."""
    return (np.log1p(gdf[choice_r800_col]) / (np.log1p(gdf[choice_rn_col]) + 1e-6)).round(3)


# ── 7. Scenario testing ──────────────────────────────────────────────────────

def add_line_to_map(gdf: gpd.GeoDataFrame, new_geom, id_col: str, snap_tolerance: float = 2.0) -> gpd.GeoDataFrame:
    """Append a proposed street to an axial/segment map, snapping its endpoints to
    the nearest existing vertex within ``snap_tolerance`` metres. Returns ``gdf``
    plus one new row."""
    existing_pts = set()
    for geom in gdf.geometry:
        coords = list(geom.coords)
        existing_pts.add(coords[0][:2])
        existing_pts.add(coords[-1][:2])

    def _snap(pt):
        best, best_d = pt, snap_tolerance
        for cand in existing_pts:
            d = math.dist(pt, cand)
            if d < best_d:
                best, best_d = cand, d
        return best

    coords = list(new_geom.coords)
    start = _snap(coords[0][:2])
    end = _snap(coords[-1][:2])
    snapped_geom = LineString([start, *coords[1:-1], end])

    new_row = {c: None for c in gdf.columns}
    new_row["geometry"] = snapped_geom
    new_row[id_col] = int(gdf[id_col].max()) + 1
    if "length_m" in gdf.columns:
        new_row["length_m"] = snapped_geom.length
    if "bearing_deg" in gdf.columns:
        dx, dy = end[0] - start[0], end[1] - start[1]
        new_row["bearing_deg"] = math.degrees(math.atan2(dy, dx)) % 180
    if "n_merged" in gdf.columns:
        new_row["n_merged"] = 1

    new_gdf = gpd.GeoDataFrame([new_row], crs=gdf.crs)
    return gpd.GeoDataFrame(pd.concat([gdf, new_gdf], ignore_index=True), crs=gdf.crs)


def normalize_scenario_geometry(new_geom):
    """Merge a MultiLineString sketch into one LineString (QGIS often exports even
    a single sketched line as a MultiLineString); if the parts don't connect
    end-to-end, keep the longest one."""
    if new_geom.geom_type != "MultiLineString":
        return new_geom
    merged = linemerge(new_geom)
    if merged.geom_type == "MultiLineString":
        merged = max(merged.geoms, key=lambda g: g.length)
    return merged


def build_scenario_metrics_gdf(G: nx.Graph, id_col: str, crs, metrics_dict: dict) -> gpd.GeoDataFrame:
    """Same idea as ``metrics_to_gdf``, kept separate because scenario code reads
    more clearly naming its own "after" step explicitly."""
    return metrics_to_gdf(G, metrics_dict, crs, id_col=id_col)


def _midpoint_key(geom) -> tuple:
    """Rounded midpoint — a stable before/after match key, since raw ids are just
    ``range(len(gdf))`` and shift once a line is appended."""
    return tuple(round(c, 3) for c in geom.interpolate(0.5, normalized=True).coords[0])


def compute_scenario_delta(before_df: gpd.GeoDataFrame, after_df: gpd.GeoDataFrame, id_col: str, metric_cols: list) -> gpd.GeoDataFrame:
    """Compare before/after metric GeoDataFrames, matched by line midpoint (not by
    id, since ids shift once a line is appended)."""
    before_lookup = {_midpoint_key(r.geometry): r for _, r in before_df.iterrows()}
    rows = []
    for _, r in after_df.iterrows():
        before = before_lookup.get(_midpoint_key(r.geometry))
        row = {id_col: r[id_col], "geometry": r.geometry, "is_new": before is None}
        for col in metric_cols:
            row[f"{col}_after"] = r[col]
            row[f"{col}_delta"] = (r[col] - before[col]) if before is not None else None
        rows.append(row)
    return gpd.GeoDataFrame(rows, crs=after_df.crs)
