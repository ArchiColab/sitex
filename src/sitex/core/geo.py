"""Coordinate-system helpers shared across every SiteX module.

Consolidates the ``utm_epsg_from_lonlat()`` / CAD-origin pair that was
independently copy-pasted into 5+ notebooks across the ARCH, ENV, Space
Syntax, and Accessibility phases of the original Pleiku case study — one
definition here instead of five.
"""

from __future__ import annotations

import math

import numpy as np
from pyproj import Transformer


def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    """Return the EPSG code of the UTM zone containing ``(lon, lat)``.

    Vietnam alone spans UTM zone 48N (west of 108°E) and zone 49N (east of
    108°E) — the split is why this must be derived per-site rather than
    hardcoded once for a case study and reused elsewhere.
    """
    zone = math.floor((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def compute_cad_origin(
    lon: float,
    lat: float,
    epsg: int,
    round_m: int = 1000,
) -> tuple[float, float]:
    """Reproject ``(lon, lat)`` into ``epsg`` and floor to the nearest ``round_m``.

    Gives the ``(easting, northing)`` a CAD user re-enters as the project
    base point / survey point (UCS) in Revit or Rhino, so every export that
    uses this same origin lands on one shared drawing origin.
    """
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    origin_easting = math.floor(easting / round_m) * round_m
    origin_northing = math.floor(northing / round_m) * round_m
    return float(origin_easting), float(origin_northing)


def xy_to_rowcol(transform, xs: np.ndarray, ys: np.ndarray) -> tuple:
    """Inverse-affine ``(x, y)`` -> ``(row, col)``, computed from a rasterio/Affine
    transform's own coefficients via elementwise scalar arithmetic.

    Deliberately not ``rasterio.transform.rowcol()``: that function's internal
    ``_transform`` has been observed to crash the interpreter outright with array
    input in this project's environment — inconsistently (it can succeed on one call
    and crash on a very similar one), which rules out a ``try``/``except`` guard. This
    is the same root cause as the ``numpy.cov()`` / ``scikit-learn`` ``KMeans``
    crashes found elsewhere in this project: a broken BLAS backend in this numpy
    build. ``xs``/``ys`` arrays here only ever go through elementwise
    multiply/add/subtract — confirmed safe — never a matrix product.
    """
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    det = a * e - b * d
    dx = np.asarray(xs) - c
    dy = np.asarray(ys) - f
    cols = (e * dx - b * dy) / det
    rows = (a * dy - d * dx) / det
    return np.floor(rows).astype(int), np.floor(cols).astype(int)
