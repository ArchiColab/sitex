"""Area-of-interest resolution — shared by every ``00-Data-*`` acquisition notebook.

Two methods, both producing the same :class:`AOI` shape so downstream fetchers (DEM,
Sentinel-2, Landsat, OSM street network, Overture) never need to know which one was
used:

- ``resolve_aoi_by_place()`` — geocode an OSM administrative boundary (default).
- ``build_aoi_map()`` / ``resolve_aoi_from_map()`` — draw a rectangle on a plain
  ``ipyleaflet`` map by clicking two opposite corners, or keep a default box around a
  point. Deliberately not ``leafmap`` (which wraps ipyleaflet with its own extra
  toolbar/plugin JS) — that extra surface area is exactly the kind of thing that can
  fail to load in a locked-down Jupyter frontend even when plain ipyleaflet renders
  fine. Click-to-draw pattern adapted from VoxCity's ``geoprocessor/mapping/rectangle.py``,
  reimplemented here without a dependency on the ``voxcity`` package.

Replaces the ``utm_epsg_from_lonlat()`` / bbox-building logic that was independently
copy-pasted into the remote-sensing, OSM street-network, and Overture notebooks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import osmnx as ox
from shapely.geometry import box

from sitex.core.geo import utm_epsg_from_lonlat


@dataclass
class AOI:
    """One resolved area of interest, in EPSG:4326 unless noted."""

    bbox: tuple[float, float, float, float]  # (west, south, east, north)
    geojson: dict[str, Any]  # Feature, Polygon
    boundary: Any  # shapely geometry — the real admin polygon (Method 1) or the AOI
    # rectangle itself (Method 2); pass this to graph_from_polygon / clip_to_mask so
    # results are trimmed to the actual AOI shape, not just its bounding box.
    center_lat: float
    center_lon: float
    local_epsg: int
    slug: str


def _slugify(text: str) -> str:
    return re.sub(r"[^\w]+", "_", text.split(",")[0]).strip("_").lower()


def _bbox_to_geojson(bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    west, south, east, north = bbox
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [west, south], [east, south],
                [east, north], [west, north],
                [west, south],
            ]],
        },
        "properties": {},
    }


def _bbox_polygon(bbox: tuple[float, float, float, float], color: str, fill_opacity: float):
    """An ``ipyleaflet.Polygon`` tracing ``bbox`` — shared by the default-box preview
    and the box the student draws."""
    from ipyleaflet import Polygon as LeafletPolygon

    west, south, east, north = bbox
    return LeafletPolygon(
        locations=[(south, west), (north, west), (north, east), (south, east)],
        color=color, weight=2, fill_color=color, fill_opacity=fill_opacity,
    )


def resolve_aoi_by_place(place_name: str) -> AOI:
    """Method 1 (default): geocode ``place_name`` to its OSM administrative boundary.

    Validate the name on https://www.openstreetmap.org/ first — a near-miss silently
    returns the wrong place rather than raising.
    """
    place_gdf = ox.geocoder.geocode_to_gdf(place_name)
    boundary = place_gdf.geometry.iloc[0]
    bbox = tuple(place_gdf.total_bounds)  # (west, south, east, north)

    centroid = boundary.centroid
    center_lat, center_lon = centroid.y, centroid.x
    local_epsg = utm_epsg_from_lonlat(center_lon, center_lat)

    return AOI(
        bbox=bbox,
        geojson=_bbox_to_geojson(bbox),
        boundary=boundary,
        center_lat=center_lat,
        center_lon=center_lon,
        local_epsg=local_epsg,
        slug=_slugify(place_name),
    )


def _bbox_from_point(lon: float, lat: float, dist_m: float) -> tuple[float, float, float, float]:
    """Bounding box (WGS84) centred on a point, ``dist_m`` in each direction."""
    from shapely.geometry import Point

    pt_proj, crs_proj = ox.projection.project_geometry(Point(lon, lat))
    box_proj = pt_proj.buffer(dist_m)
    box_ll = ox.projection.project_geometry(box_proj, crs=crs_proj, to_latlong=True)[0]
    return tuple(box_ll.bounds)  # (west, south, east, north)


def build_aoi_map(center_lat: float, center_lon: float, dist_m: float = 1000, zoom: int = 14):
    """Method 2: an interactive map for drawing a rectangle AOI by clicking two opposite
    corners — plain ``ipyleaflet``, no ``leafmap``, no ``voxcity`` (see module docstring
    for why).

    Returns ``(widget, default_bbox)`` — display the widget, click two opposite corners
    to draw a custom AOI (leaving it undrawn keeps the yellow default box), then pass
    both to :func:`resolve_aoi_from_map`.
    """
    import time as _time

    import ipywidgets as widgets
    from ipyleaflet import Circle, Map, Polygon as LeafletPolygon, TileLayer

    from sitex.core.raster_viz import SATELLITE_ATTR, SATELLITE_TILES

    default_bbox = _bbox_from_point(center_lon, center_lat, dist_m)

    m = Map(center=[center_lat, center_lon], zoom=zoom, scroll_wheel_zoom=True)
    m.add(TileLayer(url=SATELLITE_TILES, attribution=SATELLITE_ATTR, name="Satellite"))
    m.add(_bbox_polygon(default_bbox, color="#FFFF00", fill_opacity=0.08))
    m.default_style = {"cursor": "crosshair"}

    drawn_bbox: dict[str, Any] = {"value": None}
    state: dict[str, Any] = {"clicks": [], "markers": [], "preview": None, "drawn_layer": None}
    _last_mousemove = [0.0]

    status = widgets.HTML(
        value="<i>Click two opposite corners to draw a custom AOI — yellow shows the "
              "default box, kept if you don't draw anything.</i>"
    )

    def _clear(key: str) -> None:
        layer = state.get(key)
        if layer is not None:
            try:
                m.remove(layer)
            except Exception:
                pass
            state[key] = None

    def _clear_markers() -> None:
        while state["markers"]:
            try:
                m.remove(state["markers"].pop())
            except Exception:
                pass

    def _handle_interaction(**kwargs):
        if kwargs.get("type") == "click":
            coords = kwargs.get("coordinates")
            if not coords:
                return
            lat, lon = coords
            state["clicks"].append((lon, lat))
            _clear_markers()
            for clon, clat in state["clicks"]:
                pt = Circle(location=(clat, clon), radius=2, color="red", fill_color="red", fill_opacity=1.0)
                m.add(pt)
                state["markers"].append(pt)

            if len(state["clicks"]) == 1:
                _clear("preview")
                _clear("drawn_layer")
                drawn_bbox["value"] = None
                status.value = "<i>Click 1/2 — click the opposite corner</i>"
                return

            (lon1, lat1), (lon2, lat2) = state["clicks"]
            bbox = (min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))
            _clear("preview")
            _clear("drawn_layer")
            state["drawn_layer"] = _bbox_polygon(bbox, color="red", fill_opacity=0.15)
            m.add(state["drawn_layer"])
            drawn_bbox["value"] = bbox
            state["clicks"] = []
            status.value = "<i>Custom AOI drawn — click again to redraw, or run the next cell to confirm.</i>"

        elif kwargs.get("type") == "mousemove":
            now = _time.time()
            if now - _last_mousemove[0] < 0.05 or len(state["clicks"]) != 1:
                return
            _last_mousemove[0] = now
            coords = kwargs.get("coordinates")
            if not coords:
                return
            lat_c, lon_c = coords
            lon1, lat1 = state["clicks"][0]
            locs = [(lat1, lon1), (lat1, lon_c), (lat_c, lon_c), (lat_c, lon1)]
            if state["preview"] is not None:
                state["preview"].locations = locs
            else:
                state["preview"] = LeafletPolygon(
                    locations=locs, color="#6bc2e5", weight=2, fill_color="#6bc2e5", fill_opacity=0.1,
                )
                m.add(state["preview"])

    m.on_interaction(_handle_interaction)

    ui = widgets.VBox([status, m])
    ui._sitex_drawn_bbox = drawn_bbox  # read back by resolve_aoi_from_map
    return ui, default_bbox


def resolve_aoi_from_map(
    m,
    default_bbox: tuple[float, float, float, float],
    slug: str = "custom_aoi",
) -> AOI:
    """Read the rectangle drawn on ``m`` (from :func:`build_aoi_map`), or fall back to
    ``default_bbox`` if nothing was drawn."""
    drawn = getattr(m, "_sitex_drawn_bbox", {"value": None})["value"]
    bbox = tuple(drawn) if drawn is not None else tuple(default_bbox)
    west, south, east, north = bbox

    center_lat, center_lon = (south + north) / 2, (west + east) / 2
    local_epsg = utm_epsg_from_lonlat(center_lon, center_lat)

    return AOI(
        bbox=bbox,
        geojson=_bbox_to_geojson(bbox),
        boundary=box(west, south, east, north),
        center_lat=center_lat,
        center_lon=center_lon,
        local_epsg=local_epsg,
        slug=slug,
    )
