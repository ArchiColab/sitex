import numpy as np
import pytest
import xarray as xr

from sitex.arch.terrain import (
    export_contours_dxf,
    extract_contours,
    generate_contours,
    smooth_nan_aware,
)


def test_smooth_nan_aware_sigma_zero_is_noop():
    arr = np.array([[1.0, 2.0], [3.0, np.nan]])
    out = smooth_nan_aware(arr, sigma=0)
    assert out is arr


def test_smooth_nan_aware_does_not_pull_edges_toward_zero():
    # A flat 100 m plateau with a NaN "no-data" strip along one edge.
    arr = np.full((20, 20), 100.0)
    arr[:, 0] = np.nan
    out = smooth_nan_aware(arr, sigma=2)
    interior = out[:, 5:15]
    # Naive gaussian_filter (NaN treated as 0) would pull values near the
    # NaN edge well below 100; the NaN-aware version should not.
    assert np.all(interior > 95.0)


def test_extract_contours_on_synthetic_cone():
    size = 60
    x = np.linspace(0, 590, size)
    y = np.linspace(0, 590, size)
    xx, yy = np.meshgrid(x, y)
    cx, cy = 300, 300
    arr = 700 - 0.05 * np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)  # cone peak 700 m

    gdf = extract_contours(x, y, arr, interval=5.0, local_epsg=32649)

    assert len(gdf) > 0
    assert set(gdf.columns) == {"ELEV", "geometry"}
    assert gdf.crs.to_epsg() == 32649
    # Every extracted level must be a multiple of the interval.
    assert all(elev % 5.0 == 0 for elev in gdf["ELEV"].unique())


def _write_synthetic_dtm(path):
    size = 40
    lon = np.linspace(108.000, 108.006, size)
    lat = np.linspace(13.990, 13.984, size)  # descending -> north-up raster
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    elev = 700 + 50 * np.exp(
        -(((lon_grid - 108.003) ** 2 + (lat_grid - 13.987) ** 2) / 2e-5)
    )
    da = xr.DataArray(
        elev[np.newaxis, :, :],
        dims=("band", "y", "x"),
        coords={"band": [1], "y": lat, "x": lon},
    )
    da.rio.write_crs("EPSG:4326", inplace=True)
    da.rio.to_raster(path)


def test_generate_contours_end_to_end(tmp_path):
    dtm_path = tmp_path / "dtm_synthetic.tif"
    _write_synthetic_dtm(dtm_path)

    gdf = generate_contours(dtm_path, local_epsg=32649, interval=5.0, smooth_sigma=1)

    assert len(gdf) > 0
    assert gdf.crs.to_epsg() == 32649
    assert gdf["ELEV"].min() >= 695
    assert gdf["ELEV"].max() <= 755


def test_export_contours_dxf_shifts_to_local_origin(tmp_path):
    from shapely.geometry import LineString
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {"ELEV": [700.0], "geometry": [LineString([(175100, 1547200), (175150, 1547250)])]},
        crs="EPSG:32649",
    )
    out_path = tmp_path / "contours.dxf"
    export_contours_dxf(gdf, out_path, origin_easting=175000, origin_northing=1547000)

    assert out_path.exists()

    import ezdxf

    doc = ezdxf.readfile(out_path)
    msp = doc.modelspace()
    polylines = list(msp.query("POLYLINE"))
    assert len(polylines) == 1
    points = [(p.x, p.y, p.z) for p in polylines[0].points()]
    # Shifted near the origin, not raw UTM coordinates in the hundreds of km.
    assert points[0] == pytest.approx((100, 200, 700.0))
    assert points[1] == pytest.approx((150, 250, 700.0))
