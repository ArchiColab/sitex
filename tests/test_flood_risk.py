import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin

from sitex.env import flood_risk as fr

LOCAL_EPSG = 32649


def test_compute_ndwi_known_values():
    green = np.array([0.4])
    nir = np.array([0.1])
    ndwi = fr.compute_ndwi(green, nir)
    assert ndwi[0] == pytest.approx((0.4 - 0.1) / (0.4 + 0.1))


def test_compute_ndwi_handles_zero_denominator():
    green = np.array([0.0])
    nir = np.array([0.0])
    ndwi = fr.compute_ndwi(green, nir)
    assert np.isnan(ndwi[0])


def test_classify_water_thresholds_correctly():
    ndwi = np.array([0.3, 0.0, -0.2, np.nan])
    water = fr.classify_water(ndwi)
    assert list(water) == [1, 0, 0, 0]


def test_classify_water_custom_threshold():
    ndwi = np.array([0.1, 0.2, 0.3])
    water = fr.classify_water(ndwi, threshold=0.15)
    assert list(water) == [0, 1, 1]


def test_classify_flood_risk_all_four_classes():
    # dry, permanent, seasonal flood, recession
    water_dry = np.array([0, 1, 0, 1], dtype=np.uint8)
    water_flood = np.array([0, 1, 1, 0], dtype=np.uint8)
    classified = fr.classify_flood_risk(water_dry, water_flood)
    assert list(classified) == [0, 1, 2, 3]


def test_classify_flood_risk_shape_mismatch_raises():
    with pytest.raises(ValueError):
        fr.classify_flood_risk(np.zeros((2, 2)), np.zeros((3, 3)))


def test_export_flood_risk_to_geojson_only_includes_water_classes(tmp_path):
    classified = np.array([
        [0, 1, 2],
        [3, 0, 0],
    ], dtype=np.int8)
    transform = from_origin(500000, 1548000, 10, 10)
    profile = {"transform": transform, "crs": f"EPSG:{LOCAL_EPSG}"}

    out_path = tmp_path / "flood_risk.geojson"
    gdf = fr.export_flood_risk_to_geojson(classified, profile, out_path)

    assert out_path.exists()
    assert set(gdf["flood_class"]) == {1, 2, 3}
    assert set(gdf["label"]) == {
        fr.FLOOD_RISK_CLASS_LABELS[1], fr.FLOOD_RISK_CLASS_LABELS[2], fr.FLOOD_RISK_CLASS_LABELS[3],
    }
    assert gdf.crs.to_epsg() == 4326


def test_export_flood_risk_to_geojson_uses_local_epsg_fallback(tmp_path):
    classified = np.array([[1]], dtype=np.int8)
    transform = from_origin(500000, 1548000, 10, 10)
    profile = {"transform": transform}  # no crs key at all

    out_path = tmp_path / "flood_risk.geojson"
    gdf = fr.export_flood_risk_to_geojson(classified, profile, out_path, local_epsg=LOCAL_EPSG)
    assert len(gdf) == 1


def test_refine_with_landcover_water_upgrades_recession_and_dry_to_permanent():
    # classes: dry(0), permanent(1), flood(2), recession(3)
    classified = np.array([0, 1, 2, 3], dtype=np.int8)
    wc_permanent = np.array([True, False, True, True])
    refined = fr.refine_with_landcover_water(classified, wc_permanent)
    # class 0 -> 1 (landcover confirms permanent), class 1 stays 1,
    # class 2 (flood signal) untouched even though landcover disagrees,
    # class 3 -> 1 (landcover confirms permanent, recession was noise)
    assert list(refined) == [1, 1, 2, 1]


def test_refine_with_landcover_water_leaves_unconfirmed_pixels_alone():
    classified = np.array([0, 3], dtype=np.int8)
    wc_permanent = np.array([False, False])
    refined = fr.refine_with_landcover_water(classified, wc_permanent)
    assert list(refined) == [0, 3]


def test_refine_with_landcover_water_shape_mismatch_raises():
    with pytest.raises(ValueError):
        fr.refine_with_landcover_water(np.zeros((2, 2)), np.zeros((3, 3), dtype=bool))


def test_landcover_agreement_stats_known_values():
    # 2 class-1 pixels, 1 confirmed by landcover -> 50%
    # 2 class-3 pixels, both confirmed -> 100%
    classified = np.array([1, 1, 3, 3], dtype=np.int8)
    wc_permanent = np.array([True, False, True, True])
    stats = fr.landcover_agreement_stats(classified, wc_permanent)
    assert stats["class1_permanent_confirmed_by_landcover_pct"] == pytest.approx(50.0)
    assert stats["class3_recession_confirmed_permanent_by_landcover_pct"] == pytest.approx(100.0)


def test_landcover_agreement_stats_handles_empty_class():
    classified = np.array([1, 1], dtype=np.int8)  # no class-2 pixels at all
    wc_permanent = np.array([True, True])
    stats = fr.landcover_agreement_stats(classified, wc_permanent)
    assert np.isnan(stats["class2_flood_but_landcover_says_permanent_pct"])


def test_build_flood_risk_webmap_returns_folium_map():
    import folium

    gdf = gpd.GeoDataFrame(
        {"flood_class": [2], "label": [fr.FLOOD_RISK_CLASS_LABELS[2]]},
        geometry=gpd.points_from_xy([105.4], [10.5]).buffer(0.001),
        crs="EPSG:4326",
    )
    m = fr.build_flood_risk_webmap(gdf, center=(10.5, 105.4), dry_date="2024-03", flood_date="2024-10")
    assert isinstance(m, folium.Map)
