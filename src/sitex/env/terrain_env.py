"""Terrain — canopy height, hillshade, and exploratory flood inundation from a DTM/DSM pair.

Extracted from ``02-ENV-DEM Terrain.ipynb``. Reads the same ``dtm_nasadem.tif`` /
``dsm_aw3d30.tif`` pair as ``sitex.arch.terrain`` (DEM Contour) but asks a
different question of it: not the ground's shape, but what's above it (CHM)
and where water would collect (flood fill).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
from scipy import ndimage

from ..core.config import CityConfig
from ..core.geo import utm_epsg_from_lonlat


def load_dtm_dsm(dtm_path: Path, dsm_path: Path) -> tuple[xr.DataArray, xr.DataArray]:
    """Load a DTM/DSM GeoTIFF pair, each in its own native CRS (not yet reprojected)."""
    dtm = rioxarray.open_rasterio(dtm_path, masked=True).squeeze("band", drop=True)
    dsm = rioxarray.open_rasterio(dsm_path, masked=True).squeeze("band", drop=True)
    return dtm, dsm


def load_dtm_dsm_for_city(
    city: CityConfig, dtm_filename: str = "dtm_nasadem.tif", dsm_filename: str = "dsm_aw3d30.tif"
) -> tuple[xr.DataArray, xr.DataArray]:
    return load_dtm_dsm(city.data_dir / "dem" / dtm_filename, city.data_dir / "dem" / dsm_filename)


# ── Canopy Height Model ──────────────────────────────────────────────────────

def compute_chm(dtm: xr.DataArray, dsm: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Canopy/surface Height Model = DSM - DTM, negative values clipped to 0.

    DTM (NASADEM, radar) captures bare earth; DSM (AW3D30, optical stereo)
    captures the top surface — buildings, trees, structures. The DSM is
    reprojected onto the DTM's own pixel grid first since the two rasters
    may not share an identical grid. Negative results after that alignment
    are sensor-noise/registration artefacts, not real canopy height, and are
    clipped to 0. Returns ``(chm, dsm_matched)`` — the latter for a
    DTM/DSM/CHM comparison plot.
    """
    dsm_matched = dsm.rio.reproject_match(dtm)
    chm = (dsm_matched - dtm).clip(min=0)
    chm.name = "CHM"
    chm.attrs["units"] = "meters"
    chm.attrs["description"] = "Canopy/Surface Height Model (DSM - DTM)"
    return chm, dsm_matched


def compute_hillshade_gradient(dtm: xr.DataArray) -> np.ndarray:
    """Normalized ``[0, 1]`` hillshade approximation from the DTM's own gradient.

    Not a true sun-angle hillshade (no azimuth/altitude) — a cheap gradient-
    magnitude proxy that needs no extra library, good enough for a quick
    terrain-roughness read.
    """
    arr = dtm.values.astype(float)
    dy, dx = np.gradient(arr)
    shade = -dx + dy
    return (shade - shade.min()) / (shade.max() - shade.min())


# ── Exploratory flood inundation (flood fill) ───────────────────────────────

def find_lowest_point(dtm: xr.DataArray):
    """Return ``(seed_rc, seed_lat, seed_lon, elev_min)`` — the lowest DTM cell,
    used as the flood-fill seed (the most flood-prone starting point)."""
    arr = np.where(np.isnan(dtm.values), np.inf, dtm.values.astype(float))
    elev_min = int(np.nanmin(arr[arr != np.inf]))
    seed_rc = tuple(zip(*np.where(arr == elev_min)))[0]
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
        local_epsg = utm_epsg_from_lonlat((minx + maxx) / 2, (miny + maxy) / 2)
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
