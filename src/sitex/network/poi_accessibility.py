"""POI accessibility — isochrone, service area, and H3-grid accessibility/equity
for essential-service points of interest.

Extracted from ``03-NA02-POI_Accessibility.ipynb``. Two deliberately different
methods answer two different questions: Sections 1-2's per-facility convex hull
answers "what can *this facility* reach" (a specific, inspectable polygon per
facility — what most GIS tools call an isochrone/service area); the H3 grid answers
"for *every point* in the area, what's the walk to the *nearest* facility," which the
coverage % and Gini index need and a per-facility polygon can't give cheaply. They
are not expected to trace the same boundary — see the original notebook's own
Methodology section for the full comparison.
"""

from __future__ import annotations

import colorsys

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point
from shapely.ops import unary_union

from .accessibility_common import compute_gini

DEFAULT_TIME_BANDS = [5, 10, 15]


# ── 1. Isochrones (per-facility convex hull) ─────────────────────────────────

def _polygon_from_points(points: list, min_buffer_deg: float = 0.0004):
    """Convex hull of a set of points; a 0/1/2-point hull degenerates to
    nothing/a point/a line, so those get a small buffer instead — every facility
    still renders as an area."""
    if not points:
        return None
    hull = MultiPoint(points).convex_hull
    if hull.geom_type in ("Point", "LineString"):
        hull = hull.buffer(min_buffer_deg)
    return hull


def compute_facility_isochrones(G: nx.MultiDiGraph, poi_subset: gpd.GeoDataFrame, time_bands: list = DEFAULT_TIME_BANDS) -> list:
    """One Dijkstra search per facility, out to the largest time band; a convex-hull
    polygon per band is then just a filter on that one distance dict — equivalent to
    ``nx.ego_graph`` once per (facility, band) pair without repeating the search per
    band. Returns a list of ``(poi_row, {t: polygon})`` tuples, one per facility."""
    import osmnx as ox

    max_time = max(time_bands)
    facility_polys = []
    for _, poi in poi_subset.iterrows():
        node = ox.nearest_nodes(G, poi.geometry.x, poi.geometry.y)
        lengths = nx.single_source_dijkstra_path_length(G, node, cutoff=max_time * 60, weight="travel_time")
        bands = {}
        for t in time_bands:
            pts = [Point(G.nodes[n]["x"], G.nodes[n]["y"]) for n, d in lengths.items() if d <= t * 60]
            bands[t] = _polygon_from_points(pts)
        facility_polys.append((poi, bands))
    return facility_polys


def union_category_isochrones(facility_polys: list, time_bands: list = DEFAULT_TIME_BANDS) -> dict:
    """A category's isochrone band = the union of every facility's own hull at that time."""
    bands = {}
    for t in time_bands:
        polys = [b[t] for _, b in facility_polys if b[t] is not None]
        bands[t] = unary_union(polys) if polys else None
    return bands


# ── 2. Service areas & building capture ──────────────────────────────────────

def extract_service_areas(facility_polys: list, max_time: int) -> gpd.GeoDataFrame:
    """Each facility's own polygon at the largest time band = its service area —
    kept separate (allowed to overlap), unlike a nearest-facility partition."""
    rows = []
    for poi, bands in facility_polys:
        poly = bands.get(max_time)
        if poly is None:
            continue
        rows.append({"poi_id": poi["id"], "poi_name": poi["name"], "geometry": poly})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def capture_buildings(buildings: gpd.GeoDataFrame, isochrone_bands: dict, local_epsg: int) -> pd.DataFrame:
    """Buildings (and floor area) captured inside each isochrone band.

    ``buildings`` must already carry a ``floor_area_m2`` column and a ``centroid``
    geometry column in ``local_epsg``.
    """
    rows = []
    for t, poly in isochrone_bands.items():
        if poly is None:
            rows.append({"time_min": t, "n_buildings": 0, "floor_area_m2": 0.0, "n_buildings_pct": 0.0, "floor_area_pct": 0.0})
            continue
        poly_local = gpd.GeoSeries([poly], crs="EPSG:4326").to_crs(local_epsg).iloc[0]
        inside = buildings["centroid"].within(poly_local)
        rows.append({
            "time_min": t,
            "n_buildings": int(inside.sum()),
            "floor_area_m2": float(buildings.loc[inside, "floor_area_m2"].sum()),
            "n_buildings_pct": 100 * inside.sum() / len(buildings),
            "floor_area_pct": 100 * buildings.loc[inside, "floor_area_m2"].sum() / buildings["floor_area_m2"].sum(),
        })
    return pd.DataFrame(rows)


# ── 3. H3-grid accessibility & equity ────────────────────────────────────────

def compute_category_accessibility(G: nx.MultiDiGraph, base_grid: gpd.GeoDataFrame, poi_subset: gpd.GeoDataFrame, max_time_min: int) -> gpd.GeoDataFrame:
    """One multi-source Dijkstra pass (on the reversed graph) from every POI in
    ``poi_subset`` simultaneously — every hex gets both its travel time to the
    nearest facility and which facility that is, in a single pass. ``base_grid``
    must already carry an ``origin_node`` column (``accessibility_common.assign_origin_nodes``).
    """
    import osmnx as ox

    G_rev = G.reverse()
    dest_nodes = ox.nearest_nodes(G, poi_subset.geometry.x.values, poi_subset.geometry.y.values)
    node_to_poi = {}
    for node, (_, poi_row) in zip(dest_nodes, poi_subset.iterrows()):
        node_to_poi.setdefault(node, poi_row)

    lengths, paths = nx.multi_source_dijkstra(G_rev, set(dest_nodes), cutoff=max_time_min * 60, weight="travel_time")
    nearest_source = {node: path[0] for node, path in paths.items()}

    grid = base_grid.copy()
    grid["access_time_min"] = [lengths.get(n, float("inf")) / 60.0 for n in grid["origin_node"]]
    grid["nearest_poi_node"] = [nearest_source.get(n) for n in grid["origin_node"]]
    grid["nearest_poi_name"] = [node_to_poi[n]["name"] if n in node_to_poi else None for n in grid["nearest_poi_node"]]
    return grid


def compute_accessibility_summary(group: str, pois: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, capture_at_max_time: pd.Series, max_walk_time: int) -> dict:
    """One row of the cross-category summary table: median access time, % hexes
    within the walk-time budget, % buildings/floor-area captured, and the Gini
    index of access-time equity."""
    reachable = grid[grid["access_time_min"] < 999]
    gini = compute_gini(reachable["access_time_min"].values) if len(reachable) else None

    return {
        "category": group,
        "n_facilities": len(pois),
        "median_access_min": reachable["access_time_min"].median() if len(reachable) else np.nan,
        "pct_hexes_15min": 100 * (grid["access_time_min"] <= max_walk_time).mean(),
        "pct_buildings_15min": capture_at_max_time["n_buildings_pct"],
        "pct_floor_area_15min": capture_at_max_time["floor_area_pct"],
        "gini_access_time": gini,
    }


# ── 4. Folium visualization ──────────────────────────────────────────────────

def distinct_color(i: int, n: int) -> str:
    """Deterministic, evenly-spaced hue per index — avoids depending on matplotlib
    colormaps (see the ARCH-module Building Morphology migration for why matplotlib
    rendering is unreliable in this project's environment)."""
    hue = (i / max(n, 1)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.85)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def plot_isochrones(results: dict, boundary_poly, category_colors: dict, default_show: str | None = None):
    """Nested 5/10/15-minute walking bands per category, one togglable FeatureGroup
    per category. ``results`` is ``{category: {"isochrones": {t: polygon}}}``."""
    import folium

    center = [boundary_poly.centroid.y, boundary_poly.centroid.x]
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron", control_scale=True)
    band_opacity = {5: 0.55, 10: 0.35, 15: 0.18}

    for group, res in results.items():
        color = category_colors[group]
        fg = folium.FeatureGroup(name=group, show=(group == default_show))
        for t in sorted(res["isochrones"], reverse=True):
            poly = res["isochrones"][t]
            if poly is None:
                continue
            folium.GeoJson(
                poly,
                style_function=lambda x, c=color, op=band_opacity.get(t, 0.3): {"fillColor": c, "color": c, "weight": 1, "fillOpacity": op},
                tooltip=f"{group} — within {t} min walk",
            ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def plot_service_area(service_area: gpd.GeoDataFrame, pois: gpd.GeoDataFrame, max_walk_time: int):
    """Each facility's own reach shown together — overlapping colours mean
    overlapping service areas (a block with a genuine choice of provider)."""
    import folium

    gdf = service_area.reset_index(drop=True)
    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=14, tiles="CartoDB positron", control_scale=True)

    for i, row in gdf.iterrows():
        color = distinct_color(i, len(gdf))
        folium.GeoJson(
            row.geometry,
            style_function=lambda x, c=color: {"fillColor": c, "color": "#333333", "weight": 0.6, "fillOpacity": 0.4},
            tooltip=f"{row['poi_name']} — {max_walk_time} min reach",
        ).add_to(m)

    for _, row in pois.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x], radius=3, color="#111111", weight=1,
            fill=True, fill_color="white", fill_opacity=1, tooltip=row["name"],
        ).add_to(m)
    return m


def plot_accessibility_hexgrid(results: dict, boundary_poly, category_colors: dict, max_display_min: int = 20, default_show: str | None = None):
    """Every hex, coloured by walking time to the nearest facility of that
    category — the literal per-hex value the Gini index measures, with no
    dissolving or hulling. ``results`` is ``{category: {"grid": gdf}}``."""
    import folium

    center = [boundary_poly.centroid.y, boundary_poly.centroid.x]
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron", control_scale=True)

    def _style_factory(color):
        def _style(feature):
            t = feature["properties"]["access_time_min"]
            if t is None or t >= 999:
                return {"fillColor": color, "color": "#999999", "weight": 0.2, "fillOpacity": 0.0}
            opacity = 0.9 * (1 - min(t, max_display_min) / max_display_min) + 0.05
            return {"fillColor": color, "color": "#999999", "weight": 0.2, "fillOpacity": opacity}
        return _style

    for group, res in results.items():
        color = category_colors[group]
        grid = res["grid"][["h3_id", "access_time_min", "nearest_poi_name", "geometry"]].copy()
        grid["access_time_min"] = grid["access_time_min"].replace([np.inf, -np.inf], 999)

        fg = folium.FeatureGroup(name=f"{group} — H3 access time", show=(group == default_show))
        folium.GeoJson(
            grid, style_function=_style_factory(color),
            tooltip=folium.GeoJsonTooltip(fields=["access_time_min", "nearest_poi_name"], aliases=["Walk time (min):", "Nearest facility:"]),
        ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def build_combined_map(results: dict, boundary_poly, category_colors: dict, max_walk_time: int, max_display_min: int = 20, default_show: str | None = None):
    """One map combining isochrones and the H3 access grid for every category —
    all togglable from a single layer control.

    Per-facility service areas are deliberately left out here: with a whole
    category's worth of facilities (dozens+), their hulls overlap so heavily the
    layer is just visual noise — that's what ``plot_service_area`` /
    ``evaluate_new_facility`` are for, where there's few enough polygons (one, for
    a candidate site) that overlap is actually informative.

    The original notebook's export step references a ``combined_map`` variable
    that is never defined anywhere in its own source (confirmed against the live
    notebook — a real bug, not a stale claim); this is what that step evidently
    intended, built from the map-building functions above rather than guessed at
    from scratch.
    """
    import folium

    center = [boundary_poly.centroid.y, boundary_poly.centroid.x]
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron", control_scale=True)
    band_opacity = {5: 0.55, 10: 0.35, 15: 0.18}

    for group, res in results.items():
        color = category_colors[group]

        iso_fg = folium.FeatureGroup(name=f"{group} — isochrone", show=(group == default_show))
        for t in sorted(res["isochrones"], reverse=True):
            poly = res["isochrones"][t]
            if poly is None:
                continue
            folium.GeoJson(
                poly,
                style_function=lambda x, c=color, op=band_opacity.get(t, 0.3): {"fillColor": c, "color": c, "weight": 1, "fillOpacity": op},
                tooltip=f"{group} — within {t} min walk",
            ).add_to(iso_fg)
        iso_fg.add_to(m)

        grid = res["grid"][["h3_id", "access_time_min", "nearest_poi_name", "geometry"]].copy()
        grid["access_time_min"] = grid["access_time_min"].replace([np.inf, -np.inf], 999)
        hex_fg = folium.FeatureGroup(name=f"{group} — H3 access time", show=False)

        def _style(feature, color=color):
            t = feature["properties"]["access_time_min"]
            if t is None or t >= 999:
                return {"fillColor": color, "color": "#999999", "weight": 0.2, "fillOpacity": 0.0}
            opacity = 0.9 * (1 - min(t, max_display_min) / max_display_min) + 0.05
            return {"fillColor": color, "color": "#999999", "weight": 0.2, "fillOpacity": opacity}

        folium.GeoJson(
            grid, style_function=_style,
            tooltip=folium.GeoJsonTooltip(fields=["access_time_min", "nearest_poi_name"], aliases=["Walk time (min):", "Nearest facility:"]),
        ).add_to(hex_fg)
        hex_fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ── 5. Interactive "what if a new facility opened here?" ─────────────────────

def build_picker_map(boundary_poly, zoom: int = 14):
    """Click-to-place map for a candidate facility site: CartoDB Positron +
    Esri Satellite basemaps, with only the marker draw tool enabled so a click
    always yields a single point rather than a shape. Read the pick back with
    ``picked_point(m)`` once the student has clicked the marker tool and then a
    spot on the map."""
    import leafmap

    center = [boundary_poly.centroid.y, boundary_poly.centroid.x]
    m = leafmap.Map(center=center, zoom=zoom)
    m.add_basemap("CartoDB.Positron")
    m.add_basemap("SATELLITE")

    for shape in ("polyline", "polygon", "rectangle", "circle"):
        setattr(m.draw_control, shape, {})

    boundary_gdf = gpd.GeoDataFrame([{"geometry": boundary_poly}], crs="EPSG:4326")
    m.add_gdf(boundary_gdf, layer_name="Study area boundary", style={"color": "#ffff00", "fillOpacity": 0, "weight": 2})
    return m


def picked_point(picker_map) -> tuple[float, float] | None:
    """``(lon, lat)`` of the point last clicked on a ``build_picker_map`` map, or
    ``None`` if nothing's been picked yet (or the last draw wasn't a point)."""
    roi = picker_map.user_roi
    if roi is None or roi["geometry"]["type"] != "Point":
        return None
    return tuple(roi["geometry"]["coordinates"])


def evaluate_new_facility(
    G: nx.MultiDiGraph, base_grid: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame,
    existing_pois: gpd.GeoDataFrame, existing_isochrones: dict, existing_service_area: gpd.GeoDataFrame,
    group: str, name: str, lon: float, lat: float,
    time_bands: list, max_time: int, local_epsg: int,
) -> dict:
    """"What if a new ``group`` facility opened at ``(lon, lat)``?" — the
    candidate's own isochrone/service area (one Dijkstra search, same pipeline as
    Section 4), plus the category-wide picture with it folded in.

    The "folded in" isochrone/service-area/building-capture are obtained by
    unioning the candidate's own hull onto the already-computed
    ``existing_isochrones``/``existing_service_area`` rather than rerunning
    Section 4's per-facility search over every existing facility — the H3 grid
    still needs its own multi-source Dijkstra pass (nearest facility can change
    for hexes now closer to the candidate), but that's one pass regardless of how
    many facilities feed it.
    """
    new_poi = gpd.GeoDataFrame([{"id": "candidate", "name": name, "geometry": Point(lon, lat)}], crs="EPSG:4326")

    candidate_polys = compute_facility_isochrones(G, new_poi, time_bands)
    candidate_isochrones = union_category_isochrones(candidate_polys, time_bands)
    candidate_service_area = extract_service_areas(candidate_polys, max_time)

    updated_isochrones = {}
    for t in time_bands:
        polys = [p for p in (existing_isochrones.get(t), candidate_isochrones.get(t)) if p is not None]
        updated_isochrones[t] = unary_union(polys) if polys else None
    updated_service_area = pd.concat([existing_service_area, candidate_service_area], ignore_index=True)
    updated_building_capture = capture_buildings(buildings, updated_isochrones, local_epsg)

    updated_pois = pd.concat([existing_pois, new_poi], ignore_index=True)
    updated_grid = compute_category_accessibility(G, base_grid, updated_pois, max_time)
    updated_summary = compute_accessibility_summary(
        group, updated_pois, updated_grid, updated_building_capture.set_index("time_min").loc[max_time], max_time,
    )

    return dict(
        new_poi=new_poi,
        candidate_isochrones=candidate_isochrones,
        candidate_service_area=candidate_service_area,
        updated_pois=updated_pois,
        updated_isochrones=updated_isochrones,
        updated_service_area=updated_service_area,
        updated_building_capture=updated_building_capture,
        updated_grid=updated_grid,
        updated_summary=updated_summary,
    )


def build_new_facility_map(existing_grid: gpd.GeoDataFrame, new_res: dict, boundary_poly, color: str, max_display_min: int = 20):
    """Before/after comparison map for a candidate facility from
    ``evaluate_new_facility``: the category's H3 access grid with and without the
    candidate (togglable), the candidate's own isochrone bands, and a marker at
    the picked point."""
    import folium

    center = [boundary_poly.centroid.y, boundary_poly.centroid.x]
    m = folium.Map(location=center, zoom_start=14, tiles="CartoDB positron", control_scale=True)
    band_opacity = {5: 0.55, 10: 0.35, 15: 0.18}

    def _style(feature):
        t = feature["properties"]["access_time_min"]
        if t is None or t >= 999:
            return {"fillColor": color, "color": "#999999", "weight": 0.2, "fillOpacity": 0.0}
        opacity = 0.9 * (1 - min(t, max_display_min) / max_display_min) + 0.05
        return {"fillColor": color, "color": "#999999", "weight": 0.2, "fillOpacity": opacity}

    def _grid_layer(grid, layer_name, show):
        g = grid[["h3_id", "access_time_min", "nearest_poi_name", "geometry"]].copy()
        g["access_time_min"] = g["access_time_min"].replace([np.inf, -np.inf], 999)
        fg = folium.FeatureGroup(name=layer_name, show=show)
        folium.GeoJson(
            g, style_function=_style,
            tooltip=folium.GeoJsonTooltip(fields=["access_time_min", "nearest_poi_name"], aliases=["Walk time (min):", "Nearest facility:"]),
        ).add_to(fg)
        fg.add_to(m)

    _grid_layer(existing_grid, "Before — H3 access time", show=False)
    _grid_layer(new_res["updated_grid"], "After — H3 access time (+ candidate)", show=True)

    iso_fg = folium.FeatureGroup(name="Candidate — isochrone", show=True)
    for t in sorted(new_res["candidate_isochrones"], reverse=True):
        poly = new_res["candidate_isochrones"][t]
        if poly is None:
            continue
        folium.GeoJson(
            poly,
            style_function=lambda x, op=band_opacity.get(t, 0.3): {"fillColor": "#ff5500", "color": "#ff5500", "weight": 1, "fillOpacity": op},
            tooltip=f"Candidate — within {t} min walk",
        ).add_to(iso_fg)
    iso_fg.add_to(m)

    poi = new_res["new_poi"].iloc[0]
    folium.Marker(
        location=[poi.geometry.y, poi.geometry.x],
        popup=poi["name"],
        icon=folium.Icon(color="orange", icon="plus", prefix="fa"),
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
