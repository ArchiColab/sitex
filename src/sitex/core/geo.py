"""Coordinate-system helpers shared across every SiteX module: the local metric CRS
(``local_epsg_from_lonlat()``: the UTM zone from ``utm_epsg_from_lonlat()``, or
VN-2000 inside Vietnam on request) and the CAD origin.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

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


@lru_cache(maxsize=1)
def _vn2000_provinces() -> list:
    """``[(prepared polygon, polygon, {scheme: epsg}), ...]`` from the bundled province layer."""
    from shapely.geometry import shape
    from shapely.prepared import prep

    path = Path(__file__).parent / "data" / "vn2000_provinces.geojson"
    features = json.loads(path.read_text(encoding="utf-8"))["features"]
    return [
        (prep(geom), geom, {"2025": int(f["properties"]["epsg_2025"]), "epsg": int(f["properties"]["epsg"])})
        for f in features
        for geom in [shape(f["geometry"])]
    ]


def vn2000_epsg_from_lonlat(
    lon: float, lat: float, scheme: str = "2025", max_gap_deg: float = 0.05
) -> int | None:
    """EPSG code of the VN-2000 CRS used by the province containing ``(lon, lat)``.

    Vietnam assigns a VN-2000 TM-3 central meridian per province, so the zone cannot
    be read from longitude alone. The bundled layer holds the pre-2025 provinces, each
    with two zone codes:

    - ``scheme="2025"`` (default): the meridian of the new province the old one was
      merged into (Thong tu 24/2025/TT-BNNMT, in force since 1 July 2025; table cross-checked between
      tracdiahoangphat.com/kinh-tuyen-truc-cac-tinh and dodacrtk.com).
    - ``scheme="epsg"``: the pre-2025 assignment, from the area-of-use lists of the
      EPSG registry (provinces not named there use the 5897 / 5898 longitude-band zones).

    Province outlines are from https://github.com/nguyenduy1133/Free-GIS-Data,
    simplified to about 100 m. A point just outside every polygon (a coastline miss)
    takes the nearest province within ``max_gap_deg`` degrees; further out, ``None``.
    """
    from shapely.geometry import Point

    if scheme not in ("2025", "epsg"):
        raise ValueError(f"scheme must be '2025' or 'epsg', got {scheme!r}")
    pt = Point(lon, lat)
    zones = _vn2000_provinces()
    for prepared, _, codes in zones:
        if prepared.contains(pt):
            return codes[scheme]
    gap, codes = min(((geom.distance(pt), codes) for _, geom, codes in zones), key=lambda t: t[0])
    return codes[scheme] if gap <= max_gap_deg else None


def local_epsg_from_lonlat(lon: float, lat: float, scheme: str | None = None) -> int:
    """Local metric CRS for a site: the UTM zone (default). With ``scheme="2025"`` or
    ``"epsg"`` it is the site's VN-2000 zone inside Vietnam (UTM outside it)."""
    if scheme is None:
        return utm_epsg_from_lonlat(lon, lat)
    return vn2000_epsg_from_lonlat(lon, lat, scheme) or utm_epsg_from_lonlat(lon, lat)


def cad_frame(lon: float, lat: float, local_epsg: int, vn2000: bool = False, round_m: int = 1000):
    """CRS and origin for DXF/OBJ files: the site's ``local_epsg`` (UTM), or with
    ``vn2000=True`` the VN-2000 zone of the province, for documents submitted to
    the authorities.

    Returns ``(cad_epsg, (origin_easting, origin_northing), to_cad)``. The origin is
    the point ``(lon, lat)`` in ``cad_epsg``, floored to ``round_m``. ``to_cad(x, y)``
    takes coordinates in ``local_epsg`` (numbers or arrays) and returns CAD
    coordinates relative to that origin. With ``vn2000=False`` this is just a shift;
    with ``vn2000=True`` each point is converted first, because the VN-2000 grid is
    slightly rotated against UTM. Raises ``ValueError`` outside Vietnam.
    """
    cad_epsg = local_epsg
    if vn2000:
        cad_epsg = vn2000_epsg_from_lonlat(lon, lat)
        if cad_epsg is None:
            raise ValueError("The site lies outside Vietnam: no VN-2000 zone")
    origin = compute_cad_origin(lon, lat, cad_epsg, round_m)
    transformer = Transformer.from_crs(f"EPSG:{local_epsg}", f"EPSG:{cad_epsg}", always_xy=True)

    def to_cad(x, y):
        e, n = transformer.transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return e - origin[0], n - origin[1]

    return cad_epsg, origin, to_cad


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
