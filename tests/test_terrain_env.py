import numpy as np
import pytest
import xarray as xr

from sitex.env.terrain_env import (
    compute_chm,
    compute_hillshade_gradient,
    find_lowest_point,
    flood_depth,
    flood_extent_km2,
    flood_fill,
    pixel_area_m2,
)


def _make_dataarray(values, x, y, crs="EPSG:4326"):
    da = xr.DataArray(values[np.newaxis, :, :], dims=("band", "y", "x"), coords={"band": [1], "y": y, "x": x})
    da.rio.write_crs(crs, inplace=True)
    return da.squeeze("band", drop=True)


@pytest.fixture
def dtm_dsm():
    size = 10
    x = np.linspace(108.000, 108.001, size)
    y = np.linspace(13.990, 13.989, size)  # descending -> north-up
    dtm_vals = np.full((size, size), 700.0)
    dsm_vals = dtm_vals.copy()
    dsm_vals[2:4, 2:4] = 715.0  # a 15 m "canopy" patch
    dsm_vals[0, 0] = 699.0  # a registration artefact -> negative CHM before clipping
    dtm = _make_dataarray(dtm_vals, x, y)
    dsm = _make_dataarray(dsm_vals, x, y)
    return dtm, dsm


def test_compute_chm_matches_dsm_minus_dtm_and_clips_negatives(dtm_dsm):
    dtm, dsm = dtm_dsm
    chm, dsm_matched = compute_chm(dtm, dsm)

    assert float(chm.max()) == pytest.approx(15.0)
    assert float(chm.min()) >= 0.0  # the -1 m artefact must be clipped to 0
    assert float(chm.sel(y=dtm.y[0], x=dtm.x[0])) == 0.0
    assert chm.name == "CHM"
    assert chm.attrs["units"] == "meters"


def test_compute_hillshade_gradient_is_normalized():
    # A flat or uniformly-sloped DTM has a constant gradient everywhere, which
    # would divide by zero when normalizing (min == max) — use a bump, whose
    # gradient genuinely varies across the surface, like any real terrain.
    size = 10
    x = np.linspace(108.000, 108.001, size)
    y = np.linspace(13.990, 13.989, size)
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    dtm = _make_dataarray(700.0 + 20.0 * np.exp(-((xx - 5) ** 2 + (yy - 5) ** 2) / 8), x, y)

    shade = compute_hillshade_gradient(dtm)
    assert shade.shape == dtm.shape
    assert shade.min() == pytest.approx(0.0)
    assert shade.max() == pytest.approx(1.0)


def test_find_lowest_point_locates_minimum():
    x = np.linspace(108.0, 108.001, 5)
    y = np.linspace(14.0, 13.999, 5)
    vals = np.full((5, 5), 700.0)
    vals[3, 1] = 650.0  # the lowest cell
    dtm = _make_dataarray(vals, x, y)

    seed_rc, seed_lat, seed_lon, elev_min = find_lowest_point(dtm)
    assert seed_rc == (3, 1)
    assert elev_min == 650
    assert seed_lat == pytest.approx(float(dtm.y[3]))
    assert seed_lon == pytest.approx(float(dtm.x[1]))


def test_pixel_area_m2_close_to_expected_for_30m_dem():
    # NASADEM-like resolution: ~0.0002778 deg/pixel (~30 m at the equator).
    size = 20
    step = 0.0002778
    x = 108.0 + np.arange(size) * step
    y = 13.99 - np.arange(size) * step
    vals = np.full((size, size), 700.0)
    dtm = _make_dataarray(vals, x, y)

    area = pixel_area_m2(dtm, local_epsg=32649)
    assert 850 < area < 950  # ~30 x 30 m, not the naive assumed 900 exactly


def test_flood_fill_excludes_disconnected_low_pixel():
    elevation = np.array([
        [700, 700, 700, 700, 700],
        [700, 690, 690, 700, 700],
        [700, 690, 690, 700, 700],
        [700, 700, 700, 700, 700],
        [700, 700, 700, 700, 690],  # isolated low pixel: same elevation, NOT reachable by water
    ], dtype=float)
    seed_rc = (1, 1)  # inside the connected depression

    mask = flood_fill(elevation, seed_rc, water_level=692)
    assert mask[1, 1] and mask[1, 2] and mask[2, 1] and mask[2, 2]
    # Elevation alone (690 <= 692) would flag [4, 4] too — connectivity must rule it out.
    assert not mask[4, 4]
    assert mask.sum() == 4


def test_flood_fill_seed_above_water_returns_empty():
    elevation = np.full((3, 3), 700.0)
    mask = flood_fill(elevation, seed_rc=(1, 1), water_level=650)
    assert not mask.any()


def test_flood_extent_km2_scales_with_pixel_count_and_area():
    mask = np.array([[True, True], [False, True]])
    assert flood_extent_km2(mask, pixel_area_m2_value=1_000_000) == pytest.approx(3.0)


def test_flood_depth_only_inside_mask():
    elevation = np.array([[690.0, 695.0], [700.0, 685.0]])
    mask = np.array([[True, False], [False, True]])
    depth = flood_depth(elevation, mask, water_level=692)

    assert depth[0, 0] == pytest.approx(2.0)
    assert depth[1, 1] == pytest.approx(7.0)
    assert np.isnan(depth[0, 1])
    assert np.isnan(depth[1, 0])
