"""Terrain — hillshade and exploratory flood inundation from a DTM.

Reads the same ``dtm_gedtm30.tif`` as ``sitex.arch.terrain`` (DEM Contour) but asks a
different question of it: not the ground's shape, but where water would collect
(flood fill).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
from scipy import ndimage

from ..core.config import CityConfig
from ..core.geo import local_epsg_from_lonlat


def load_dtm(dtm_path: Path) -> xr.DataArray:
    """Load a DTM GeoTIFF in its native CRS (not yet reprojected)."""
    return rioxarray.open_rasterio(dtm_path, masked=True).squeeze("band", drop=True)


def load_dtm_for_city(city: CityConfig, dtm_filename: str = "dtm_gedtm30.tif") -> xr.DataArray:
    return load_dtm(city.data_dir / "dem" / dtm_filename)


def compute_hillshade_gradient(dtm: xr.DataArray) -> np.ndarray:
    """Normalized ``[0, 1]`` hillshade approximation from the DTM's own gradient.

    Not a true sun-angle hillshade (no azimuth/altitude) — a cheap gradient-
    magnitude proxy that needs no extra library, good enough for a quick
    terrain-roughness read.
    """
    arr = dtm.values.astype(float)
    dy, dx = np.gradient(arr)
    shade = -dx + dy
    return (shade - np.nanmin(shade)) / (np.nanmax(shade) - np.nanmin(shade))


# ── Exploratory flood inundation (flood fill) ───────────────────────────────

def find_lowest_point(dtm: xr.DataArray):
    """Return ``(seed_rc, seed_lat, seed_lon, elev_min)`` — the lowest DTM cell,
    used as the flood-fill seed (the most flood-prone starting point)."""
    arr = np.where(np.isnan(dtm.values), np.inf, dtm.values.astype(float))
    seed_rc = tuple(int(i) for i in np.unravel_index(np.argmin(arr), arr.shape))  # first lowest cell
    elev_min = int(np.floor(arr[seed_rc]))  # whole metres, rounded down (the DEM may have decimals)
    seed_lat = float(dtm.y[seed_rc[0]])
    seed_lon = float(dtm.x[seed_rc[1]])
    return seed_rc, seed_lat, seed_lon, elev_min


def pixel_area_m2(dtm: xr.DataArray, local_epsg: int | None = None) -> float:
    """True pixel area in m², read by reprojecting to a metric UTM CRS.

    A degree-based DTM's pixels aren't square in metres (1° longitude is
    compressed by cos(latitude) relative to 1° latitude), so a fixed
    "30 x 30 m" assumption is wrong away from the equator, and silently
    wrong again for a different-resolution DEM. ``local_epsg`` defaults to
    a zone auto-derived from the DTM's own extent centroid.
    """
    if local_epsg is None:
        minx, miny, maxx, maxy = dtm.rio.bounds()
        local_epsg = local_epsg_from_lonlat((minx + maxx) / 2, (miny + maxy) / 2)
    res = dtm.rio.reproject(f"EPSG:{local_epsg}").rio.resolution()
    return abs(res[0] * res[1])


def flood_fill(elevation: np.ndarray, seed_rc: tuple, water_level: float) -> np.ndarray:
    """Boolean mask of cells connected to ``seed_rc`` at or below ``water_level``.

    Two-step flood fill: threshold by elevation, then keep only the
    connected component containing the seed — this is what excludes
    isolated low pixels (e.g. a depression on a hilltop) that share the
    elevation but water could never physically reach.
    """
    below = elevation <= water_level
    if not below[seed_rc[0], seed_rc[1]]:
        return np.zeros_like(below)
    labeled, _ = ndimage.label(below)
    seed_label = labeled[seed_rc[0], seed_rc[1]]
    return labeled == seed_label


def flood_extent_km2(mask: np.ndarray, pixel_area_m2_value: float) -> float:
    """Flooded area in km², from a flood-fill mask and its true per-pixel area."""
    return float(mask.sum()) * pixel_area_m2_value / 1e6


def flood_depth(elevation: np.ndarray, mask: np.ndarray, water_level: float) -> np.ndarray:
    """Per-pixel flood depth (``water_level - elevation``) inside ``mask``, ``NaN`` outside."""
    return np.where(mask, water_level - elevation, np.nan)
