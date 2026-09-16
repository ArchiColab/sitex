"""Area-of-interest resolution — shared by every ``00-Data-*`` acquisition notebook.

Two methods, both producing the same :class:`AOI` shape so downstream fetchers (DEM,
Sentinel-2, Landsat, OSM street network, Overture) never need to know which one was
used:

- ``resolve_aoi_by_place()`` — geocode an OSM administrative boundary (default).
- ``build_aoi_map()`` / ``resolve_aoi_from_map()`` — draw a rectangle on a Leafmap
  widget, or keep a default box around a point.

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
    """Method 2: a Leafmap widget pre-loaded with a default AOI rectangle to redraw.

    Returns ``(map_widget, default_bbox)`` — display the widget, let the student draw a
    rectangle, then pass it to :func:`resolve_aoi_from_map`.
    """
    import leafmap

    default_bbox = _bbox_from_point(center_lon, center_lat, dist_m)

    m = leafmap.Map(center=[center_lat, center_lon], zoom=zoom)
    m.add_basemap("SATELLITE")
    m.add_geojson(
        _bbox_to_geojson(default_bbox),
        layer_name="Default AOI",
        style={"color": "#FFFF00", "weight": 2, "fillOpacity": 0.08},
    )
    return m, default_bbox


def resolve_aoi_from_map(
    m,
    default_bbox: tuple[float, float, float, float],
    slug: str = "custom_aoi",
) -> AOI:
    """Read the rectangle drawn on ``m`` (from :func:`build_aoi_map`), or fall back to
    ``default_bbox`` if nothing was drawn."""
    bbox = tuple(m.user_roi_bounds()) if m.user_roi is not None else tuple(default_bbox)
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
