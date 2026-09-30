"""Green space accessibility — clustering, walking-time accessibility, and spatial
equity (Gini index).

Walking speed is a constant 4.8 km/h on every edge, matching
``sitex.network.poi_accessibility``'s convention — not OSMnx's ``add_edge_speeds()``,
which imputes driving speeds by highway type and would make every accessibility
result several times too generous for a walking analysis.

Two green layers answer two questions: **public green** (OSM places mapped for
recreation: parks, gardens, playgrounds...) is where people may go; the **green
environment** (ESA WorldCover tree cover, shrubland and grassland patches) is where
green physically exists, public or not. Comparing the two shows where green is nearby
but not public.
"""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from .accessibility_common import (
    DEFAULT_WALK_SPEED_KMH,
    assign_origin_nodes,
    build_h3_grid,
    compute_gini,
)

# OSM green-space tags, merged from OSM wiki/Landcover and wiki/Green_space.
# Ordered by confidence: core green-space types first, optional/contextual last.
OSM_GREEN_TAGS = {
    "landuse": [
        "forest", "meadow", "grass", "recreation_ground", "village_green",
        "orchard", "vineyard", "plant_nursery",
        "farmland",  # optional — large coverage
        "greenhouse_horticulture", "allotments", "flowerbed",
    ],
    "natural": ["wood", "scrub", "heath", "grassland", "wetland", "tree_row", "tree"],
    "landcover": ["grass", "trees", "shrubbery", "flowerbed"],
    "leisure": [
        "park", "garden", "nature_reserve", "dog_park", "playground", "pitch",
        "golf_course", "sports_centre", "common", "stadium",
    ],
    "water": ["pond", "lake", "reservoir"],
}

_LABEL_TAG_PRIORITY = ["leisure", "landuse", "natural", "landcover"]


def _label_green_space(row) -> str:
    """Human-readable label from whichever OSM tag column has a value."""
    for key in _LABEL_TAG_PRIORITY:
        if key in row.index and pd.notna(row[key]):
            return f"{key}:{row[key]}"
    return "unknown"


def fetch_green_spaces(place: str, tags: dict = OSM_GREEN_TAGS) -> gpd.GeoDataFrame:
    """Green space polygons from OSM, with a derived ``green_space`` label column."""
    import osmnx as ox

    gdf = ox.features_from_place(place, tags=tags)
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf["green_space"] = gdf.apply(_label_green_space, axis=1)
    return gdf


# OSM labels (``green_space`` column) of green space mapped for public recreation.
# Farmland, orchards, plantations, nurseries and water are left out: they are green,
# but not places people go to spend time.
PUBLIC_GREEN_LABELS = (
    "leisure:park", "leisure:garden", "leisure:nature_reserve", "leisure:dog_park",
    "leisure:playground", "leisure:pitch", "leisure:common",
    "landuse:recreation_ground", "landuse:village_green",
)


def select_public_green(green_gdf: gpd.GeoDataFrame, labels=PUBLIC_GREEN_LABELS) -> gpd.GeoDataFrame:
    """Keep only the OSM green spaces mapped for public recreation (``labels``)."""
    return green_gdf[green_gdf["green_space"].isin(labels)].copy()


# ESA WorldCover classes counted as green environment.
WC_GREEN_CLASSES = {10: "tree cover", 20: "shrubland", 30: "grassland"}


def worldcover_green_patches(
    wc_path,
    local_epsg: int,
    classes=tuple(WC_GREEN_CLASSES),
    min_area_ha: float = 1.0,
    boundary=None,
) -> gpd.GeoDataFrame:
    """Connected patches of WorldCover green (default: tree cover, shrubland,
    grassland) of at least ``min_area_ha``, as polygons in EPSG:4326.

    Neighbouring green pixels of different classes form one patch. ``boundary``
    (a WGS84 polygon) optionally clips the patches to the study area. Columns:
    ``green_space`` ("worldcover:green"), ``area_m2``, ``tree_share`` (share of the
    patch's pixels that are tree cover), geometry.
    """
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape

    with rasterio.open(wc_path) as src:
        wc = src.read(1)
        transform, crs = src.transform, src.crs

    green = np.isin(wc, list(classes)).astype("uint8")
    polys = [shape(geom) for geom, value in shapes(green, mask=green == 1, transform=transform) if value == 1]
    empty = gpd.GeoDataFrame(columns=["green_space", "area_m2", "tree_share", "geometry"], geometry="geometry", crs="EPSG:4326")
    if not polys:
        return empty

    patches = gpd.GeoDataFrame(geometry=polys, crs=crs)
    if boundary is not None:
        patches = gpd.clip(patches, gpd.GeoSeries([boundary], crs="EPSG:4326").to_crs(crs))
        patches = patches[~patches.geometry.is_empty].explode(index_parts=False).reset_index(drop=True)

    local = patches.to_crs(epsg=local_epsg)
    patches["area_m2"] = local.geometry.area.values
    patches = patches[patches["area_m2"] >= min_area_ha * 10_000].reset_index(drop=True)
    if patches.empty:
        return empty

    # share of tree-cover pixels per patch (pixel centres inside the patch)
    rows, cols = np.nonzero(green)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    px = gpd.GeoDataFrame({"tree": wc[rows, cols] == 10}, geometry=gpd.points_from_xy(xs, ys), crs=crs)
    joined = gpd.sjoin(px, patches[["geometry"]], predicate="within")
    tree_share = joined.groupby("index_right")["tree"].mean()
    patches["tree_share"] = tree_share.reindex(patches.index).fillna(0.0).values

    patches["green_space"] = "worldcover:green"
    return patches[["green_space", "area_m2", "tree_share", "geometry"]].to_crs("EPSG:4326")


def green_access_nodes(G: nx.MultiDiGraph, green_gdf: gpd.GeoDataFrame, local_epsg: int, edge_buffer_m: float = 25.0) -> list:
    """Walking-network junctions where a green space can be entered: every node
    inside a green space or within ``edge_buffer_m`` of its edge, plus, for each
    green space, the single node nearest to it (so a green space with no street
    next to it is still reachable).

    Using the edge instead of the centre matters for large patches: the centre
    of a 50 ha park can be 400 m from its nearest entrance.
    """
    import osmnx as ox

    if green_gdf.empty:
        return []
    nodes = ox.graph_to_gdfs(G, edges=False)[["geometry"]].to_crs(epsg=local_epsg)
    polys = green_gdf[["geometry"]].to_crs(epsg=local_epsg).reset_index(drop=True)

    buffered = polys.copy()
    buffered["geometry"] = polys.buffer(edge_buffer_m)
    near = gpd.sjoin(nodes, buffered, predicate="within")
    access = set(near.index)

    _, node_pos = nodes.sindex.nearest(polys.geometry, return_all=False)
    access |= set(nodes.index[node_pos])
    return list(access)


def compute_green_accessibility(
    G: nx.MultiDiGraph,
    boundary_poly,
    green_spaces: gpd.GeoDataFrame,
    walk_time_minutes: int = 15,
    h3_resolution: int = 10,
    local_epsg: int | None = None,
    edge_buffer_m: float = 25.0,
) -> gpd.GeoDataFrame:
    """Walking-time accessibility to the nearest green space, per H3 hex — one
    multi-source Dijkstra pass over the walking network, seeded from the
    junctions at the edge of every green space (``green_access_nodes``)
    simultaneously. A single cutoff (``walk_time_minutes``), not a multi-band
    isochrone; hexes beyond it get ``inf``.

    Seeding from the edge rather than from the junction nearest each green
    space's centre matters for large parks and forests, where walking to the
    centre would overstate the walk.
    """
    grid = assign_origin_nodes(build_h3_grid(boundary_poly, h3_resolution), G)
    if local_epsg is None:
        local_epsg = grid.estimate_utm_crs().to_epsg()

    dest_nodes = green_access_nodes(G, green_spaces, local_epsg, edge_buffer_m)
    grid = grid.copy()
    if not dest_nodes:
        grid["access_time_min"] = float("inf")
        return grid

    G_rev = G.reverse()
    dist_from_green = nx.multi_source_dijkstra_path_length(
        G_rev, dest_nodes, cutoff=walk_time_minutes * 60, weight="travel_time"
    )
    grid["access_time_min"] = [dist_from_green.get(node, float("inf")) / 60.0 for node in grid["origin_node"]]
    return grid


def summarise_green_access(grid: gpd.GeoDataFrame, walk_time_minutes: int, layer: str) -> dict:
    """One row of the comparison table: share of hexes within the walking time,
    median walking time and Gini index (both over the hexes within reach only)."""
    times = grid["access_time_min"].to_numpy(dtype=float)
    within = times[times <= walk_time_minutes]
    return {
        "green_layer": layer,
        "pct_hexes_within": 100 * len(within) / len(times) if len(times) else np.nan,
        "median_walk_min": float(np.median(within)) if len(within) else np.nan,
        "gini_walk_time": compute_gini(within),
    }


GAP_CLASSES = {
    "both": ("Public green and green environment nearby", "#1a9850"),
    "environment_only": ("Green nearby, but no public green", "#fdae61"),
    "public_only": ("Public green only", "#66bd63"),
    "none": ("No green nearby", "#d73027"),
}


def classify_green_gap(grid_public: gpd.GeoDataFrame, grid_environment: gpd.GeoDataFrame, walk_time_minutes: int) -> gpd.GeoDataFrame:
    """Combine the two layers per hex (same H3 grid): walking time to public green,
    walking time to green environment, and a ``gap_class`` (keys of
    ``GAP_CLASSES``). ``environment_only`` hexes are the planning opportunity: green
    exists within reach, but none of it is mapped as public."""
    env = grid_environment.set_index("h3_id")["access_time_min"]
    out = grid_public[["h3_id", "geometry"]].copy()
    out["access_public_min"] = grid_public["access_time_min"].values
    out["access_environment_min"] = env.reindex(out["h3_id"]).values
    pub_ok = out["access_public_min"] <= walk_time_minutes
    env_ok = out["access_environment_min"] <= walk_time_minutes
    out["gap_class"] = np.select(
        [pub_ok & env_ok, env_ok & ~pub_ok, pub_ok & ~env_ok],
        ["both", "environment_only", "public_only"],
        default="none",
    )
    return out


def cluster_green_spaces(gdf: gpd.GeoDataFrame, local_epsg: int, num_clusters: int = 3) -> gpd.GeoDataFrame:
    """K-means clustering of green spaces by footprint area, labels sorted so
    cluster 0 = smallest, cluster N = largest."""
    from scipy.cluster.vq import kmeans as scipy_kmeans
    from scipy.cluster.vq import vq as scipy_vq

    gdf = gdf.copy()
    gdf["area_m2"] = gdf.to_crs(epsg=local_epsg).geometry.area
    area_values = gdf["area_m2"].values.astype(float)
    centroids, _ = scipy_kmeans(area_values, num_clusters)
    labels, _ = scipy_vq(area_values, centroids)
    order = centroids.argsort()
    rank = {old: new for new, old in enumerate(order)}
    gdf["cluster"] = [rank[l] for l in labels]
    return gdf


# ── Folium visualization ──────────────────────────────────────────────────────

def _map_center(gdf: gpd.GeoDataFrame) -> list:
    """[lat, lon] of the middle of the layer's bounding box (no centroid of
    lat/lon polygons, which geopandas warns about)."""
    minx, miny, maxx, maxy = gdf.to_crs("EPSG:4326").total_bounds
    return [(miny + maxy) / 2, (minx + maxx) / 2]


def plot_green_clusters(gdf: gpd.GeoDataFrame):
    """One togglable FeatureGroup per cluster; layer names derived from the
    dominant OSM tag values in each cluster."""
    import folium

    center = _map_center(gdf)
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron", control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", overlay=False,
    ).add_to(m)

    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    for cluster_id in sorted(gdf["cluster"].unique()):
        subset = gdf[gdf["cluster"] == cluster_id]
        top_types = subset["green_space"].value_counts().head(2).index.tolist()
        layer_name = f"Cluster {cluster_id} — {' · '.join(top_types)} ({len(subset)})"
        color = colors[cluster_id % len(colors)]
        fg = folium.FeatureGroup(name=layer_name)

        for _, row in subset.iterrows():
            sim_geo = row.geometry.simplify(0.0001)
            tooltip = f"<b>{row['green_space']}</b><br>Cluster: {cluster_id}<br>Area: {row['area_m2']:,.0f} m²"
            folium.GeoJson(
                sim_geo,
                style_function=lambda x, c=color: {"fillColor": c, "color": "#444444", "weight": 1, "fillOpacity": 0.6},
                tooltip=tooltip,
            ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def _base_map(gdf: gpd.GeoDataFrame):
    import folium

    center = _map_center(gdf)
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron", control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", overlay=False,
    ).add_to(m)
    return m


def plot_green_layers(public_gdf: gpd.GeoDataFrame, environment_gdf: gpd.GeoDataFrame, boundary_poly=None):
    """Both green layers on one map: public green (OSM) on top of the green
    environment (WorldCover patches)."""
    import folium

    m = _base_map(environment_gdf if not environment_gdf.empty else public_gdf)
    if boundary_poly is not None:
        folium.GeoJson(boundary_poly, name="Study area",
                       style_function=lambda x: {"fill": False, "color": "#333333", "weight": 2}).add_to(m)
    if not environment_gdf.empty:
        env = environment_gdf.copy()
        env["area_ha"] = (env["area_m2"] / 10_000).round(1)
        env["tree_pct"] = (100 * env["tree_share"]).round(0)
        folium.GeoJson(
            env[["area_ha", "tree_pct", "geometry"]], name=f"Green environment, WorldCover ({len(env)})",
            style_function=lambda x: {"fillColor": "#a6d96a", "color": "#66a61e", "weight": 1, "fillOpacity": 0.5},
            tooltip=folium.GeoJsonTooltip(fields=["area_ha", "tree_pct"], aliases=["Area (ha)", "Tree cover (%)"]),
        ).add_to(m)
    if not public_gdf.empty:
        folium.GeoJson(
            public_gdf[["green_space", "geometry"]], name=f"Public green, OSM ({len(public_gdf)})",
            style_function=lambda x: {"fillColor": "#006837", "color": "#004529", "weight": 2, "fillOpacity": 0.8},
            tooltip=folium.GeoJsonTooltip(fields=["green_space"], aliases=["Type"]),
        ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def plot_green_gap(gap_grid: gpd.GeoDataFrame, public_gdf: gpd.GeoDataFrame | None = None):
    """H3 map of ``classify_green_gap``: one toggleable layer per class, so the
    layer control doubles as the legend."""
    import folium

    m = _base_map(gap_grid)
    for key, (label, color) in GAP_CLASSES.items():
        subset = gap_grid[gap_grid["gap_class"] == key]
        if subset.empty:
            continue
        share = 100 * len(subset) / len(gap_grid)
        folium.GeoJson(
            subset[["h3_id", "geometry"]], name=f"{label} ({share:.0f}%)",
            style_function=lambda x, c=color: {"fillColor": c, "color": c, "weight": 0.3, "fillOpacity": 0.65},
        ).add_to(m)
    if public_gdf is not None and not public_gdf.empty:
        folium.GeoJson(
            public_gdf[["green_space", "geometry"]], name="Public green (OSM)",
            style_function=lambda x: {"fill": False, "color": "#004529", "weight": 2},
        ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def plot_green_accessibility(grid_gdf: gpd.GeoDataFrame, green_gdf: gpd.GeoDataFrame,
                             legend_name: str = "Walking Time to Green Space (min)"):
    """Choropleth of walking-time accessibility to the nearest green space."""
    import folium

    plot_gdf = grid_gdf[grid_gdf["access_time_min"] < 999].copy()
    center = _map_center(plot_gdf)
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron", control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", overlay=False,
    ).add_to(m)

    folium.Choropleth(
        geo_data=plot_gdf, data=plot_gdf, columns=["h3_id", "access_time_min"],
        key_on="feature.properties.h3_id", fill_color="YlGn_r", fill_opacity=0.7,
        line_opacity=0.2, legend_name=legend_name, name="Green Accessibility Grid",
    ).add_to(m)
    folium.GeoJson(
        green_gdf[["geometry"]], name="Green Spaces",
        style_function=lambda x: {"fillColor": "green", "color": "darkgreen", "weight": 1},
    ).add_to(m)
    folium.LayerControl().add_to(m)
    return m


def plot_green_inequality(grid_gdf: gpd.GeoDataFrame, gini_score: float,
                          legend_name: str = "Walking Time to Green Space (min)"):
    """H3 choropleth of walking-time inequality — darker red = longer walk = higher
    inequality within this AOI."""
    import folium

    plot_gdf = grid_gdf[grid_gdf["access_time_min"] < 999].copy()
    center = _map_center(plot_gdf)
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron", control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", overlay=False,
    ).add_to(m)

    folium.Choropleth(
        geo_data=plot_gdf, data=plot_gdf, columns=["h3_id", "access_time_min"],
        key_on="feature.properties.h3_id", fill_color="RdYlGn_r", fill_opacity=0.75,
        line_opacity=0.15, legend_name=legend_name,
        name="Spatial Green Inequality", bins=6,
    ).add_to(m)
    folium.LayerControl().add_to(m)
    return m
