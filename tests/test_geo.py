import numpy as np
from rasterio.transform import from_origin

from sitex.core.geo import compute_cad_origin, utm_epsg_from_lonlat, xy_to_rowcol


def test_zone_just_west_of_108e_is_48n():
    assert utm_epsg_from_lonlat(lon=107.9, lat=13.98) == 32648


def test_zone_at_108e_is_49n():
    # Pleiku sits right at the 108 degrees E split the case study relies on.
    assert utm_epsg_from_lonlat(lon=108.00, lat=13.98) == 32649


def test_southern_hemisphere_uses_327xx_prefix():
    assert utm_epsg_from_lonlat(lon=108.00, lat=-1.0) == 32749


def test_compute_cad_origin_floors_to_round_m():
    lon, lat, epsg = 108.00, 13.98, 32649
    easting, northing = compute_cad_origin(lon, lat, epsg, round_m=1000)
    assert easting % 1000 == 0
    assert northing % 1000 == 0

    # Un-floored coordinates should land within one round_m step of the origin.
    from pyproj import Transformer

    raw_easting, raw_northing = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    ).transform(lon, lat)
    assert 0 <= raw_easting - easting < 1000
    assert 0 <= raw_northing - northing < 1000


def test_xy_to_rowcol_matches_rasterio_for_north_up_transform():
    # A plain north-up transform (no rotation) — compare against rasterio's own
    # rowcol() on a handful of points to confirm the manual inverse is correct,
    # even though the library itself avoids calling rowcol() with array input
    # (see xy_to_rowcol's docstring for why).
    from rasterio.transform import rowcol as rasterio_rowcol

    transform = from_origin(500000, 1548000, 10, 10)
    xs = np.array([500005.0, 500055.0, 500500.0])
    ys = np.array([1547995.0, 1547500.0, 1540000.0])

    rows, cols = xy_to_rowcol(transform, xs, ys)
    for x, y, expected_row, expected_col in zip(xs, ys, *[list(v) for v in [rows, cols]]):
        ref_row, ref_col = rasterio_rowcol(transform, x, y)  # scalar call — safe in isolation
        assert expected_row == ref_row
        assert expected_col == ref_col


def test_xy_to_rowcol_origin_pixel():
    transform = from_origin(0, 10, 1, 1)
    rows, cols = xy_to_rowcol(transform, np.array([0.5]), np.array([9.5]))
    assert rows[0] == 0
    assert cols[0] == 0
