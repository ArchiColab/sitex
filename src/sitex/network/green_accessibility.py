"""Green space accessibility — clustering, walking-time accessibility, and spatial
equity (Gini index).

Extracted from ``03-NA03-Green Accessibility.ipynb``. Walking speed is a constant
4.8 km/h on every edge, matching ``sitex.network.poi_accessibility``'s convention —
not OSMnx's ``add_edge_speeds()``, which imputes driving speeds by highway type and
would make every accessibility result several times too generous for a walking
analysis (an inconsistency the two sibling notebooks actually had until this was
aligned — see the project's Accessibility overview).
"""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import pandas as pd

from .accessibility_common import DEFAULT_WALK_SPEED_KMH, assign_origin_nodes, build_h3_grid

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


def compute_green_accessibility(
    G: nx.MultiDiGraph,
    boundary_poly,
    green_spaces: gpd.GeoDataFrame,
    walk_time_minutes: int = 15,
    h3_resolution: int = 10,
) -> gpd.GeoDataFrame:
    """Walking-time accessibility to the nearest green space, per H3 hex — one
    multi-source Dijkstra pass over the walking network, seeded from every green
    space's centroid simultaneously. A single cutoff (``walk_time_minutes``), not a
    multi-band isochrone."""
    import osmnx as ox

    grid = assign_origin_nodes(build_h3_grid(boundary_poly, h3_resolution), G)

    green_centroids = green_spaces.geometry.centroid
    dest_nodes = list(set(ox.nearest_nodes(G, green_centroids.x.values, green_centroids.y.values)))

    G_rev = G.reverse()
    dist_from_green = nx.multi_source_dijkstra_path_length(
        G_rev, dest_nodes, cutoff=walk_time_minutes * 60, weight="travel_time"
    )

    grid = grid.copy()
    grid["access_time_min"] = [dist_from_green.get(node, float("inf")) / 60.0 for node in grid["origin_node"]]
    return grid


# ── Folium visualization ──────────────────────────────────────────────────────

def plot_green_clusters(gdf: gpd.GeoDataFrame):
    """One togglable FeatureGroup per cluster; layer names derived from the
    dominant OSM tag values in each cluster."""
    import folium

    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
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


def plot_green_accessibility(grid_gdf: gpd.GeoDataFrame, green_gdf: gpd.GeoDataFrame):
    """Choropleth of walking-time accessibility to the nearest green space."""
    import folium

    plot_gdf = grid_gdf[grid_gdf["access_time_min"] < 999].copy()
    center = [plot_gdf.geometry.centroid.y.mean(), plot_gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron", control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", overlay=False,
    ).add_to(m)

    folium.Choropleth(
        geo_data=plot_gdf, data=plot_gdf, columns=["h3_id", "access_time_min"],
        key_on="feature.properties.h3_id", fill_color="YlGn_r", fill_opacity=0.7,
        line_opacity=0.2, legend_name="Walking Time to Green Space (min)", name="Green Accessibility Grid",
    ).add_to(m)
    folium.GeoJson(
        green_gdf, name="Green Spaces",
        style_function=lambda x: {"fillColor": "green", "color": "darkgreen", "weight": 1},
    ).add_to(m)
    folium.LayerControl().add_to(m)
    return m


def plot_green_inequality(grid_gdf: gpd.GeoDataFrame, gini_score: float):
    """H3 choropleth of walking-time inequality — darker red = longer walk = higher
    inequality within this AOI."""
    import folium

    plot_gdf = grid_gdf[grid_gdf["access_time_min"] < 999].copy()
    center = [plot_gdf.geometry.centroid.y.mean(), plot_gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron", control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", overlay=False,
    ).add_to(m)

    folium.Choropleth(
        geo_data=plot_gdf, data=plot_gdf, columns=["h3_id", "access_time_min"],
        key_on="feature.properties.h3_id", fill_color="RdYlGn_r", fill_opacity=0.75,
        line_opacity=0.15, legend_name="Walking Time to Green Space (min)",
        name="Spatial Green Inequality", bins=6,
    ).add_to(m)
    folium.LayerControl().add_to(m)
    return m
