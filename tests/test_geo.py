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


def test_vn2000_zone_is_by_province_not_longitude():
    from sitex.core.geo import vn2000_epsg_from_lonlat

    assert vn2000_epsg_from_lonlat(108.00, 13.98) == 9217   # Pleiku, Gia Lai (2025: 108-15)
    assert vn2000_epsg_from_lonlat(108.00, 13.98, scheme="epsg") == 9218   # pre-2025: 108-30
    assert vn2000_epsg_from_lonlat(107.07, 10.35) == 9210   # Vung Tau, now in Ho Chi Minh City
    assert vn2000_epsg_from_lonlat(106.70, 10.78) == 9210   # Ho Chi Minh City
    assert vn2000_epsg_from_lonlat(105.85, 21.03) == 5897   # Ha Noi (longitude-band zone)
    assert vn2000_epsg_from_lonlat(108.80, 15.12) == 5898   # Quang Ngai


def test_vn2000_coastline_miss_takes_nearest_province_and_far_points_return_none():
    from sitex.core.geo import vn2000_epsg_from_lonlat

    assert vn2000_epsg_from_lonlat(106.95, 10.40) is not None   # just off the Can Gio coast
    assert vn2000_epsg_from_lonlat(100.50, 13.75) is None       # Bangkok


def test_local_epsg_is_utm_by_default_and_vn2000_on_request():
    from sitex.core.geo import local_epsg_from_lonlat

    assert local_epsg_from_lonlat(108.00, 13.98) == 32649
    assert local_epsg_from_lonlat(100.50, 13.75) == 32647
    assert local_epsg_from_lonlat(108.00, 13.98, scheme="2025") == 9217
    assert local_epsg_from_lonlat(100.50, 13.75, scheme="2025") == 32647  # outside Vietnam: UTM


def test_cad_frame_default_is_utm_shift_and_matches_compute_cad_origin():
    from sitex.core.geo import cad_frame, compute_cad_origin

    epsg, origin, to_cad = cad_frame(108.00, 13.98, 32649)
    assert epsg == 32649
    assert origin == compute_cad_origin(108.00, 13.98, 32649, 1000)
    x, y = to_cad([origin[0] + 10.0, origin[0] + 20.0], [origin[1] + 5.0, origin[1] + 6.0])
    assert list(x) == [10.0, 20.0] and list(y) == [5.0, 6.0]


def test_cad_frame_vn2000_uses_province_zone_and_converts_each_point():
    import numpy as np
    from pyproj import Transformer
    from sitex.core.geo import cad_frame

    epsg, origin, to_cad = cad_frame(108.00, 13.98, 32649, vn2000=True)
    assert epsg == 9217
    ux, uy = Transformer.from_crs(4326, 32649, always_xy=True).transform(108.00, 13.98)
    vx, vy = Transformer.from_crs(4326, 9217, always_xy=True).transform(108.00, 13.98)
    x, y = to_cad(np.array([ux, ux + 1000.0]), np.array([uy, uy]))
    assert abs(x[0] - (vx - origin[0])) < 1e-3 and abs(y[0] - (vy - origin[1])) < 1e-3
    # the two grids are rotated against each other: 1 km east in UTM gains northing in VN-2000
    assert abs(y[1] - y[0]) > 1.0
    assert abs(x[1] - x[0] - 1000.0) < 5.0


def test_cad_frame_vn2000_outside_vietnam_raises():
    import pytest
    from sitex.core.geo import cad_frame

    with pytest.raises(ValueError, match="outside Vietnam"):
        cad_frame(100.50, 13.75, 32647, vn2000=True)
