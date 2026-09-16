"""Interactive webmaps consolidating the UHI/change raster layers and the classified
POIs into single Folium maps.

Extracted from ``04-VIZ-InteractiveWebmap.ipynb`` and ``04-VIZ-InteractiveWebmap-Gehl.ipynb``.
Both were already matplotlib-crash-safe in their original form — ``colorize()`` only
calls a matplotlib colormap object directly (a pure array lookup), never
``Figure``/``Axes``/``imshow`` — so this module ports them close to verbatim.

An earlier version also sampled a raster value per building centroid to draw a
"crisp" per-building vector layer. At this project's 30 m/10 m source resolution that
crispness was misleading — many centroids share the same pixel, so it read as
building-level precision the data doesn't actually have — so that layer was dropped
in favour of the plain raster overlay + building-footprint outline below.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np


def reproject_array_to_wgs84(arr: np.ndarray, transform, src_crs):
    """Warp an in-memory array to EPSG:4326 (no file I/O) — the CRS folium/Leaflet expects.
    Returns ``(dst_arr, bounds)`` with ``bounds = (left, bottom, right, top)``."""
    from rasterio.transform import array_bounds
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    dst_crs = "EPSG:4326"
    src_bounds = array_bounds(arr.shape[0], arr.shape[1], transform)
    dst_transform, width, height = calculate_default_transform(src_crs, dst_crs, arr.shape[1], arr.shape[0], *src_bounds)
    dst_arr = np.full((height, width), np.nan, dtype="float64")
    reproject(
        arr, dst_arr,
        src_transform=transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        src_nodata=np.nan, dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return dst_arr, array_bounds(height, width, dst_transform)


def colorize_to_rgba(arr: np.ndarray, cmap: str, vmin=None, vmax=None) -> np.ndarray:
    """Matplotlib colormap *object* lookup only (no Figure/draw) — safe in this
    environment. Equivalent to ``sitex.core.raster_viz.colorize_array``, kept as its
    own thin wrapper here to match the original notebook's own ``colorize()`` name."""
    from ..core.raster_viz import colorize_array

    return colorize_array(arr, cmap, vmin=vmin, vmax=vmax)


def build_uhi_change_webmap(center: tuple, raster_layers: dict, buildings_web: gpd.GeoDataFrame, default_visible: str | None = None, zoom_start: int = 13):
    """One folium map: a raster ``ImageOverlay`` per ``raster_layers`` entry plus a
    plain building-footprints outline layer, all toggleable.

    ``raster_layers`` is ``{name: (rgba_array, (left, bottom, right, top))}``.
    """
    import folium

    m = folium.Map(location=list(center), zoom_start=zoom_start, tiles="CartoDB positron", control_scale=True, prefer_canvas=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", overlay=False,
    ).add_to(m)

    for name, (rgba, bounds) in raster_layers.items():
        left, bottom, right, top = bounds
        folium.raster_layers.ImageOverlay(
            image=rgba, bounds=[[bottom, left], [top, right]],
            name=f"Raster — {name}", opacity=0.8, show=(name == default_visible),
        ).add_to(m)

    folium.GeoJson(
        buildings_web[["geometry"]], name="Building footprints",
        style_function=lambda _: {"color": "#666666", "weight": 0.5, "fillOpacity": 0}, show=False,
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def build_gehl_point_webmap(center: tuple, pois_web: gpd.GeoDataFrame, buildings_web: gpd.GeoDataFrame, category_style: dict, building_color: str = "#9e9e9e", zoom_start: int = 13):
    """Plain categorical point map: every classified POI, coloured by Gehl
    activity/Third Space, over grey building footprints — the point-data counterpart
    to a KDE heatmap (``sitex.viz.heatmaps``), for when individual places should stay
    identifiable and clickable rather than blurred into a density surface.

    ``category_style`` is ``{name: {"col": bool_column, "color": hex}}``.
    """
    import folium

    m = folium.Map(location=list(center), zoom_start=zoom_start, tiles="CartoDB dark_matter", control_scale=True, prefer_canvas=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", overlay=False,
    ).add_to(m)

    folium.GeoJson(
        buildings_web[["geometry"]], name="Building footprints",
        style_function=lambda _: {"color": building_color, "weight": 0.5, "fillColor": building_color, "fillOpacity": 0.35},
        show=True,
    ).add_to(m)

    for name, spec in category_style.items():
        subset = pois_web[pois_web[spec["col"]]]
        layer = folium.FeatureGroup(name=f"{name}  (n={len(subset):,})", show=True)
        for _, row in subset.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x], radius=4, color=spec["color"],
                fill=True, fill_color=spec["color"], fill_opacity=0.85, weight=0.5,
                popup=f"{row.get('name', '')} — {name}",
            ).add_to(layer)
        layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
