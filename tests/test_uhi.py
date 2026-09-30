import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from sitex.env import uhi

LOCAL_EPSG = 32649


def _write_band(path, values, dtype="uint16"):
    transform = from_origin(500000, 1548000, 30, 30)
    meta = {
        "driver": "GTiff",
        "height": values.shape[0],
        "width": values.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": f"EPSG:{LOCAL_EPSG}",
        "transform": transform,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(values.astype(dtype), 1)
    return path


def test_resolve_band_paths_uses_scene_info_directory_not_stored_string(tmp_path):
    scene_dir = tmp_path / "data" / "landsat" / "output" / "SCENE123"
    scene_dir.mkdir(parents=True)
    scene_info_path = scene_dir / "scene_info.json"

    # Stored paths deliberately point somewhere that doesn't exist relative to the
    # current working directory, or to a completely different machine's layout —
    # only the filename should matter.
    scene_info = {"band_paths": {"B4": "..\\data\\landsat\\output\\SCENE123\\SCENE123_B4_clipped.tif"}}

    resolved = uhi.resolve_band_paths(scene_info, scene_info_path)
    assert resolved["B4"] == scene_dir / "SCENE123_B4_clipped.tif"


def test_default_bbox_is_centered_and_reasonable_size():
    bbox = uhi.default_bbox(13.98, 108.0, dist_m=1000)
    west, south, east, north = bbox
    assert west < 108.0 < east
    assert south < 13.98 < north
    # ~1000 m at this latitude should be a small fraction of a degree.
    assert 0.005 < (east - west) < 0.05


def test_bbox_to_geojson_is_closed_polygon():
    feature = uhi.bbox_to_geojson(0, 0, 1, 1)
    coords = feature["geometry"]["coordinates"][0]
    assert coords[0] == coords[-1]
    assert len(coords) == 5


def test_dn_to_reflectance_and_compute_ndvi():
    dn_4 = np.array([[10000.0, 0.0]])
    dn_5 = np.array([[20000.0, 15000.0]])
    refl_4 = uhi.dn_to_reflectance(dn_4)
    refl_5 = uhi.dn_to_reflectance(dn_5)
    ndvi = uhi.compute_ndvi(refl_4, refl_5, dn_4=dn_4)

    expected_valid = (refl_5[0, 0] - refl_4[0, 0]) / (refl_5[0, 0] + refl_4[0, 0])
    assert ndvi[0, 0] == pytest.approx(expected_valid)
    assert np.isnan(ndvi[0, 1])  # dn_4 == 0 -> masked to NaN regardless of reflectance math


def test_compute_lst_masks_zero_dn_and_converts_to_celsius():
    dn_10 = np.array([[0.0, 44000.0]])
    lst = uhi.compute_lst(dn_10)
    assert np.isnan(lst[0, 0])
    expected = uhi.ST_MULT * 44000.0 + uhi.ST_ADD - 273.15
    assert lst[0, 1] == pytest.approx(expected)


def test_fractional_vegetation_cover_bounds():
    # At NDVI == BARESOIL_NDVI_MAX, FVC should be 0; at NDVI == VEGETATION_NDVI_MIN, FVC should be 1.
    ndvi = np.array([uhi.BARESOIL_NDVI_MAX, uhi.VEGETATION_NDVI_MIN])
    fvc = uhi.fractional_vegetation_cover(ndvi)
    assert fvc[0] == pytest.approx(0.0)
    assert fvc[1] == pytest.approx(1.0)


def test_compute_emissivity_nbem_classifies_by_ndvi_regime():
    ndvi = np.array([-0.5, 0.35, 0.8])  # baresoil, mixed, vegetation
    red_reflectance = np.array([0.1, 0.1, 0.1])
    emissivity = uhi.compute_emissivity_nbem(ndvi, red_reflectance)

    expected_baresoil = uhi.RED_BAND_COEFF_A_10 - uhi.RED_BAND_COEFF_B_10 * 0.1
    assert emissivity[0] == pytest.approx(expected_baresoil)
    assert emissivity[2] > emissivity[0]  # vegetation emissivity is higher than bare soil
    assert not np.isnan(emissivity[1])  # mixed pixel gets a blended value, not NaN


def test_compute_emissivity_nbem_nan_propagates():
    ndvi = np.array([np.nan, 0.8])
    red_reflectance = np.array([0.1, 0.1])
    emissivity = uhi.compute_emissivity_nbem(ndvi, red_reflectance)
    assert np.isnan(emissivity[0])


def test_compute_uhi_intensity_matches_manual_calculation():
    ndvi = np.array([0.5, 0.5, 0.1, 0.1])  # 2 vegetated, 2 built-up
    lst = np.array([30.0, 32.0, 40.0, 42.0])

    result = uhi.compute_uhi_intensity(ndvi, lst, ndvi_threshold=0.3)
    assert result["mean_lst_vegetation"] == pytest.approx(31.0)
    assert result["mean_lst_built_up"] == pytest.approx(41.0)
    assert result["uhi_intensity"] == pytest.approx(10.0)
    assert result["vegetation_pct"] == pytest.approx(50.0)


def test_compute_uhi_intensity_excludes_nan_pixels_from_built_up():
    ndvi = np.array([0.5, np.nan, 0.1])
    lst = np.array([30.0, 99.0, 40.0])
    result = uhi.compute_uhi_intensity(ndvi, lst, ndvi_threshold=0.3)
    # The NaN-NDVI pixel must not be counted as built-up (its LST of 99 would skew the mean).
    assert result["mean_lst_built_up"] == pytest.approx(40.0)


def test_normalize01_maps_min_max_to_0_1():
    arr = np.array([10.0, 20.0, 30.0])
    normed = uhi.normalize01(arr)
    assert normed[0] == pytest.approx(0.0)
    assert normed[-1] == pytest.approx(1.0)
    assert normed[1] == pytest.approx(0.5)


def test_compute_hei_weights_applied_correctly():
    lst = np.array([0.0, 10.0])   # normalizes to [0, 1]
    ndvi = np.array([0.0, 1.0])   # normalizes to [0, 1] -> veg_deficit = [1, 0]

    hei = uhi.compute_hei(lst, ndvi, weights={"50_50": (0.5, 0.5)})
    # pixel 0: lst_norm=0, veg_deficit=1 -> 0.5*0 + 0.5*1 = 0.5
    # pixel 1: lst_norm=1, veg_deficit=0 -> 0.5*1 + 0.5*0 = 0.5
    assert hei["50_50"][0] == pytest.approx(0.5)
    assert hei["50_50"][1] == pytest.approx(0.5)


def test_hei_sensitivity_range_zero_when_weights_agree():
    hei_arrays = {"a": np.array([0.5, 0.5]), "b": np.array([0.5, 0.5])}
    assert np.allclose(uhi.hei_sensitivity_range(hei_arrays), 0.0)


def test_hei_sensitivity_range_positive_when_weights_disagree():
    hei_arrays = {"a": np.array([0.2, 0.8]), "b": np.array([0.6, 0.6])}
    result = uhi.hei_sensitivity_range(hei_arrays)
    assert result[0] == pytest.approx(0.4)
    assert result[1] == pytest.approx(0.2)


def test_compute_ndvi_lst_density_shape():
    rng = np.random.default_rng(0)
    ndvi = rng.uniform(-1, 1, size=1000)
    lst = rng.uniform(20, 50, size=1000)
    density, xedges, yedges = uhi.compute_ndvi_lst_density(ndvi, lst, bins=10)
    assert density.shape == (10, 10)
    assert density.sum() == pytest.approx(1000)


def test_clip_to_geojson_crops_to_aoi(tmp_path):
    values = np.arange(400, dtype="uint16").reshape(20, 20)
    band_path = tmp_path / "band.tif"
    _write_band(band_path, values)

    # AOI covering a sub-window in WGS84 -> reprojected internally to the band's own CRS.
    import pyproj

    transformer = pyproj.Transformer.from_crs(f"EPSG:{LOCAL_EPSG}", "EPSG:4326", always_xy=True)
    lon1, lat1 = transformer.transform(500050, 1547850)
    lon2, lat2 = transformer.transform(500250, 1547950)
    aoi = uhi.bbox_to_geojson(min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))

    out_path = uhi.clip_to_geojson(band_path, aoi, target_dir=tmp_path)
    with rasterio.open(out_path) as src:
        assert src.width < 20
        assert src.height < 20


def test_calc_ndvi_reads_writes_and_returns_array(tmp_path):
    dn_4 = np.full((5, 5), 10000, dtype="uint16")
    dn_5 = np.full((5, 5), 20000, dtype="uint16")
    b4_path = _write_band(tmp_path / "b4.tif", dn_4)
    b5_path = _write_band(tmp_path / "b5.tif", dn_5)

    out_path = tmp_path / "ndvi.tif"
    ndvi, meta = uhi.calc_ndvi(b4_path, b5_path, target_path=out_path)

    assert out_path.exists()
    with rasterio.open(out_path) as src:
        reloaded = src.read(1)
    assert np.allclose(reloaded, ndvi, equal_nan=True)


def test_reproject_geotiff_fills_gaps_with_nan_not_zero(tmp_path):
    values = np.full((10, 10), 5.0)
    src_path = tmp_path / "src.tif"
    transform = from_origin(500000, 1548000, 30, 30)
    meta = {
        "driver": "GTiff", "height": 10, "width": 10, "count": 1, "dtype": "float64",
        "crs": f"EPSG:{LOCAL_EPSG}", "transform": transform, "nodata": None,
    }
    with rasterio.open(src_path, "w", **meta) as dst:
        dst.write(values, 1)

    out_path = tmp_path / "reprojected.tif"
    uhi.reproject_geotiff(src_path, out_path, target_crs=4326)

    with rasterio.open(out_path) as src:
        arr = src.read(1)
    # Corners outside the rotated footprint must be NaN, not 0 (0 would crush a
    # narrow-range colour stretch — see reproject_geotiff's docstring).
    assert np.isnan(arr[0, 0]) or arr[0, 0] == pytest.approx(5.0)
    assert not np.any(arr == 0.0)


def test_create_rgba_color_image_writes_uint8_rgba(tmp_path):
    values = np.linspace(0, 100, 100).reshape(10, 10)
    src_path = tmp_path / "band.tif"
    transform = from_origin(0, 10, 1, 1)
    meta = {
        "driver": "GTiff", "height": 10, "width": 10, "count": 1, "dtype": "float64",
        "crs": "EPSG:4326", "transform": transform,
    }
    with rasterio.open(src_path, "w", **meta) as dst:
        dst.write(values, 1)

    out_path = tmp_path / "colored.tif"
    result_path, image = uhi.create_rgba_color_image(src_path, out_path, colormap="viridis")

    assert result_path.exists()
    assert image.shape == (10, 10, 4)
    assert image.dtype == np.uint8

    array_rgba, bounds = uhi.load_raster_rgba(out_path)
    assert array_rgba.shape == (10, 10, 4)
