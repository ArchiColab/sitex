"""Urban change detection — NDVI/NDBI change between two Sentinel-2 dates.

Extracted from ``02-ENV-Urban_change_detection.ipynb``. Reads a 3-band Sentinel-2
L2A composite per date (B04=band 1, B08=band 2, B11=band 3, from
``00-Data-Acquisition_RemoteSensing.ipynb``'s OpenEO step, which already resamples
B11 from its native 20 m to 10 m), computes NDVI and NDBI, differences them across
two dates, and classifies the result two ways: a fixed threshold and unsupervised
K-Means.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import shape
from sklearn.cluster import KMeans  # import at top level — avoids a DLL-load-order crash on Windows

DEFAULT_DECREASE_THRESHOLD = -0.05  # below -> vegetation loss / urban expansion
DEFAULT_INCREASE_THRESHOLD = 0.05  # above -> vegetation gain / revegetation

CHANGE_CLASS_COLORS = {
    -1: (215, 48, 39),   # urban expansion / loss — red
    0: (255, 255, 191),  # no significant change — yellow
    1: (26, 152, 80),    # revegetation / gain — green
}


# ── Band loading & preprocessing ─────────────────────────────────────────────

def load_band(tif_path: Path, band_index: int = 1):
    """Load one band from a GeoTIFF. Returns ``(array: float32, profile: dict)``."""
    with rasterio.open(tif_path) as src:
        array = src.read(band_index).astype(np.float32)
        profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    return array, profile


def normalize_band(array: np.ndarray) -> np.ndarray:
    """Sentinel-2 L2A DN (scaled by 10000) -> [0, 1] reflectance; DN 0 -> NaN (no-data)."""
    arr = array.astype(np.float32) / 10000.0
    arr = np.clip(arr, 0.0, 1.0)
    arr[arr == 0] = np.nan
    return arr


# ── Spectral indices ─────────────────────────────────────────────────────────

def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red). >0.3 healthy vegetation, ~0 bare/urban, <0 water."""
    denom = nir + red
    denom = np.where(denom == 0, np.nan, denom)
    return np.clip((nir - red) / denom, -1.0, 1.0)


def compute_ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDBI = (SWIR - NIR) / (SWIR + NIR). >0 built-up, <0 vegetation/water."""
    denom = swir + nir
    denom = np.where(denom == 0, np.nan, denom)
    return np.clip((swir - nir) / denom, -1.0, 1.0)


# ── Change detection ─────────────────────────────────────────────────────────

def compute_change(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """change = T2 - T1. Positive -> gain, negative -> loss."""
    if t1.shape != t2.shape:
        raise ValueError(f"Shape mismatch: {t1.shape} vs {t2.shape}")
    return t2 - t1


def threshold_change(
    change: np.ndarray,
    low: float = DEFAULT_DECREASE_THRESHOLD,
    high: float = DEFAULT_INCREASE_THRESHOLD,
) -> np.ndarray:
    """Classify change: -1 significant decrease, 0 no significant change, +1 significant increase.

    Nodata pixels (non-finite) are classed as 0 (no change) rather than excluded.
    """
    classified = np.zeros_like(change, dtype=np.int8)
    classified[change < low] = -1
    classified[change > high] = 1
    classified[~np.isfinite(change)] = 0
    return classified


def binary_change_mask(classified: np.ndarray) -> np.ndarray:
    """1 = changed, 0 = unchanged."""
    return (classified != 0).astype(np.uint8)


def kmeans_change(change: np.ndarray, n_clusters: int = 3, random_state: int = 42) -> np.ndarray:
    """Unsupervised K-Means clustering of change magnitude.

    Labels are re-ordered so 0 = smallest change magnitude, n-1 = largest — the raw
    cluster IDs KMeans assigns are arbitrary, so this makes the label ordering
    meaningful and reproducible across runs. Nodata pixels are flagged ``-9999``.
    """
    valid_mask = np.isfinite(change)
    valid_pixels = change[valid_mask].reshape(-1, 1)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(valid_pixels)

    labels = np.full(change.shape, -9999, dtype=np.int32)
    labels[valid_mask] = km.labels_

    order = np.argsort(np.abs(km.cluster_centers_).flatten())
    remap = {old: new for new, old in enumerate(order)}
    result = np.copy(labels)
    for old, new in remap.items():
        result[labels == old] = new
    result[~valid_mask] = -9999
    return result


def save_raster(array: np.ndarray, profile: dict, path: Path) -> Path:
    """Save a 2D array as a single-band, LZW-compressed GeoTIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    p.update(dtype="float32", count=1, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(array.astype(np.float32), 1)
    return path


# ── Vector export & webmap ───────────────────────────────────────────────────

def export_change_to_geojson(classified: np.ndarray, profile: dict, output_path: Path, local_epsg: int | None = None) -> gpd.GeoDataFrame:
    """Vectorize the classified raster to GeoJSON polygons — only changed pixels
    (class != 0) are exported; unchanged area is not vectorised."""
    from rasterio.features import shapes

    transform = profile["transform"]
    crs = profile.get("crs") or (f"EPSG:{local_epsg}" if local_epsg else None)
    if crs is None:
        raise ValueError("No CRS in `profile` and no `local_epsg` fallback given.")

    mask = classified != 0
    results = list(shapes(classified.astype(np.int32), mask=mask, transform=transform))

    geometries, values = [], []
    for geom, val in results:
        geometries.append(shape(geom))
        values.append(int(val))

    gdf = gpd.GeoDataFrame(
        {
            "change_class": values,
            "label": ["Urban expansion" if v == -1 else "Revegetation" for v in values],
        },
        geometry=geometries,
        crs=crs,
    ).to_crs("EPSG:4326")

    gdf.to_file(output_path, driver="GeoJSON")
    return gdf


_CHANGE_LABEL_COLORS = {"Urban expansion": "#e74c3c", "Revegetation": "#27ae60"}


def build_change_webmap(
    gdf: gpd.GeoDataFrame,
    center: tuple,
    t1_year,
    t2_year,
    zoom_start: int = 14,
    index_used: str | None = None,
    low_threshold: float = DEFAULT_DECREASE_THRESHOLD,
    high_threshold: float = DEFAULT_INCREASE_THRESHOLD,
    pct_expansion: float | None = None,
    pct_revegetation: float | None = None,
    pct_unchanged: float | None = None,
):
    """Folium map of change polygons — shareable as a standalone HTML file.

    The legend box always explains what the two colors mean and the threshold
    that separates "changed" from "unchanged" pixels; passing ``index_used``
    and the ``pct_*`` summary stats (from the same run's ``threshold_change``
    output) adds a one-line caption with the period, index, and area shares —
    the same numbers ``run_change_pipeline`` prints to the console — so the
    map is readable on its own once shared as a standalone HTML file.
    """
    import folium

    m = folium.Map(location=list(center), zoom_start=zoom_start, tiles="CartoDB positron")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", overlay=False,
    ).add_to(m)

    folium.GeoJson(
        gdf,
        name=f"Urban Change ({t1_year}→{t2_year})",
        style_function=lambda feat: {
            "fillColor": _CHANGE_LABEL_COLORS.get(feat["properties"]["label"], "#95a5a6"),
            "color": "black", "weight": 0.3, "fillOpacity": 0.65,
        },
        tooltip=folium.GeoJsonTooltip(fields=["label", "change_class"], aliases=["Change type:", "Class value:"]),
    ).add_to(m)

    caption = f"{t1_year} → {t2_year}"
    if index_used:
        caption += f" · {index_used.upper()}-driven"
    stats_line = ""
    if pct_expansion is not None and pct_revegetation is not None and pct_unchanged is not None:
        stats_line = (
            f"<div style='margin-top:4px;color:#555;'>"
            f"Expansion {pct_expansion:.1f}% &middot; Revegetation {pct_revegetation:.1f}% "
            f"&middot; Unchanged {pct_unchanged:.1f}%</div>"
        )

    legend_html = f"""
    <div style='position:fixed;bottom:30px;right:10px;z-index:9999;
         background:white;padding:12px;border-radius:6px;
         box-shadow:0 2px 6px rgba(0,0,0,0.2);font-family:Arial;font-size:13px;
         max-width:260px;'>
        <b>Urban Change Detection</b><br>
        <span style='color:#555;'>{caption}</span><br>
        <i style='background:#e74c3c;width:14px;height:14px;
           display:inline-block;border-radius:2px;'></i>&nbsp;Urban expansion
           (change &lt; {low_threshold:+.2f})<br>
        <i style='background:#27ae60;width:14px;height:14px;
           display:inline-block;border-radius:2px;'></i>&nbsp;Revegetation
           (change &gt; {high_threshold:+.2f})<br>
        <span style='color:#777;font-size:11px;'>Unshaded area = no significant change
           (only changed pixels are vectorised)</span>
        {stats_line}
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    return m
