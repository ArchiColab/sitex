import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from sitex.env import change_detection as cd

LOCAL_EPSG = 32649


def _write_composite(path, b04, b08, b11):
    transform = from_origin(500000, 1548000, 10, 10)
    meta = {
        "driver": "GTiff", "height": b04.shape[0], "width": b04.shape[1], "count": 3,
        "dtype": "uint16", "crs": f"EPSG:{LOCAL_EPSG}", "transform": transform,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(b04.astype("uint16"), 1)
        dst.write(b08.astype("uint16"), 2)
        dst.write(b11.astype("uint16"), 3)
    return path


def test_normalize_band_scales_and_masks_zero():
    arr = np.array([0.0, 5000.0, 10000.0, 15000.0])
    normed = cd.normalize_band(arr)
    assert np.isnan(normed[0])
    assert normed[1] == pytest.approx(0.5)
    assert normed[2] == pytest.approx(1.0)
    assert normed[3] == pytest.approx(1.0)  # clipped


def test_compute_ndvi_known_values():
    nir = np.array([0.5])
    red = np.array([0.1])
    ndvi = cd.compute_ndvi(nir, red)
    assert ndvi[0] == pytest.approx((0.5 - 0.1) / (0.5 + 0.1))


def test_compute_ndvi_handles_zero_denominator():
    nir = np.array([0.0])
    red = np.array([0.0])
    ndvi = cd.compute_ndvi(nir, red)
    assert np.isnan(ndvi[0])


def test_compute_ndbi_known_values():
    swir = np.array([0.4])
    nir = np.array([0.2])
    ndbi = cd.compute_ndbi(swir, nir)
    assert ndbi[0] == pytest.approx((0.4 - 0.2) / (0.4 + 0.2))


def test_compute_change_is_t2_minus_t1():
    t1 = np.array([0.2, 0.5])
    t2 = np.array([0.3, 0.4])
    change = cd.compute_change(t1, t2)
    assert np.allclose(change, [0.1, -0.1])


def test_compute_change_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cd.compute_change(np.zeros((2, 2)), np.zeros((3, 3)))


def test_threshold_change_classifies_correctly():
    change = np.array([-0.2, 0.0, 0.2, np.nan])
    classified = cd.threshold_change(change, low=-0.05, high=0.05)
    assert list(classified) == [-1, 0, 1, 0]


def test_binary_change_mask():
    classified = np.array([-1, 0, 1, 0], dtype=np.int8)
    mask = cd.binary_change_mask(classified)
    assert list(mask) == [1, 0, 1, 0]


class _FakeKMeans:
    """Deterministic stand-in for sklearn's KMeans.

    This environment's real `sklearn.cluster.KMeans` crashes the interpreter
    outright on `.fit()` (`threadpoolctl` -> a stray/incompatible MKL DLL raises
    `OSError: [WinError -1066598273]`), reproduced even with a bare `KMeans().fit()`
    and no other imports at all — a pre-existing environment package conflict,
    unrelated to `kmeans_change()`'s own logic. Mocking it here tests the actual
    code this module owns (nodata handling, re-ordering clusters by magnitude so
    label 0 is always the smallest change) independently of whether sklearn's
    threading runtime works in this environment.
    """

    def __init__(self, n_clusters, random_state=None, n_init="auto"):
        self.n_clusters = n_clusters

    def fit(self, X):
        # Assign each point to whichever of `n_clusters` evenly-spaced quantile
        # bins its value falls into — deterministic, no threadpoolctl involved.
        values = X.flatten()
        edges = np.quantile(values, np.linspace(0, 1, self.n_clusters + 1)[1:-1])
        self.labels_ = np.searchsorted(edges, values)
        self.cluster_centers_ = np.array(
            [[values[self.labels_ == k].mean()] for k in range(self.n_clusters)]
        )
        return self


def test_kmeans_change_labels_ordered_by_magnitude(monkeypatch):
    # Three well-separated clusters of change magnitude: near-zero, moderate, large.
    change = np.concatenate([
        np.random.default_rng(0).normal(0, 0.005, 30),
        np.random.default_rng(1).normal(0.3, 0.005, 30),
        np.random.default_rng(2).normal(-0.6, 0.005, 30),
    ])
    monkeypatch.setattr(cd, "KMeans", _FakeKMeans)
    labels = cd.kmeans_change(change, n_clusters=3, random_state=42)
    assert set(np.unique(labels)) == {0, 1, 2}

    # Label 0 should correspond to the near-zero cluster (smallest |change|).
    near_zero_labels = labels[:30]
    assert len(set(near_zero_labels)) == 1
    assert near_zero_labels[0] == 0

    # Label 2 should correspond to the largest-magnitude cluster (-0.6).
    large_labels = labels[60:]
    assert len(set(large_labels)) == 1
    assert large_labels[0] == 2


def test_kmeans_change_flags_nodata(monkeypatch):
    change = np.array([0.0, 0.1, np.nan, 0.2, np.nan] * 10)
    monkeypatch.setattr(cd, "KMeans", _FakeKMeans)
    labels = cd.kmeans_change(change, n_clusters=2)
    assert (labels[np.isnan(change)] == -9999).all()
    assert (labels[~np.isnan(change)] != -9999).all()


def test_save_raster_writes_geotiff(tmp_path):
    array = np.ones((5, 5))
    transform = from_origin(0, 5, 1, 1)
    profile = {
        "driver": "GTiff", "height": 5, "width": 5, "count": 1, "dtype": "float32",
        "crs": "EPSG:4326", "transform": transform,
    }
    out_path = tmp_path / "sub" / "test.tif"
    result = cd.save_raster(array, profile, out_path)
    assert result.exists()
    with rasterio.open(result) as src:
        assert np.allclose(src.read(1), 1.0)


def test_export_change_to_geojson_only_includes_changed_pixels(tmp_path):
    classified = np.array([
        [0, 0, -1],
        [1, 0, 0],
    ], dtype=np.int8)
    transform = from_origin(500000, 1548000, 10, 10)
    profile = {"transform": transform, "crs": f"EPSG:{LOCAL_EPSG}"}

    out_path = tmp_path / "change.geojson"
    gdf = cd.export_change_to_geojson(classified, profile, out_path)

    assert out_path.exists()
    assert set(gdf["change_class"]) == {-1, 1}
    assert set(gdf["label"]) == {"Urban expansion", "Revegetation"}
    assert gdf.crs.to_epsg() == 4326


def test_export_change_to_geojson_uses_local_epsg_fallback(tmp_path):
    classified = np.array([[1]], dtype=np.int8)
    transform = from_origin(500000, 1548000, 10, 10)
    profile = {"transform": transform}  # no crs key at all

    out_path = tmp_path / "change.geojson"
    gdf = cd.export_change_to_geojson(classified, profile, out_path, local_epsg=LOCAL_EPSG)
    assert len(gdf) == 1


def test_build_change_webmap_returns_folium_map(tmp_path):
    import folium

    gdf = gpd.GeoDataFrame(
        {"change_class": [-1], "label": ["Urban expansion"]},
        geometry=gpd.points_from_xy([108.0], [13.98]).buffer(0.001),
        crs="EPSG:4326",
    )
    m = cd.build_change_webmap(gdf, center=(13.98, 108.0), t1_year=2017, t2_year=2026)
    assert isinstance(m, folium.Map)
