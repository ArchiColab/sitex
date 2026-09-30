"""NDWI-based flood-risk mapping — water-state comparison between two Sentinel-2 dates.

Same shape as ``sitex.env.change_detection`` (NDVI/NDBI urban change), applied to flood
mapping instead: compute NDWI per date, threshold each date into a binary water mask,
then combine the two masks into flood-risk classes (permanent water, seasonal flood
water, dry land, recession). This differs from urban change detection's continuous
index-difference approach because flood risk is about *where water newly appears*
between a dry-season and a flood-season date, not how far a continuous index moved.

Reuses ``load_band``/``normalize_band``/``save_raster`` from ``change_detection`` rather
than duplicating them — both modules read the same kind of Sentinel-2 L2A composite.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import shape

from sitex.env.change_detection import load_band, normalize_band, save_raster  # noqa: F401 (re-exported)

DEFAULT_NDWI_THRESHOLD = 0.0  # McFeeters (1996): NDWI > 0 -> water


# ── Spectral index ───────────────────────────────────────────────────────────

def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDWI = (Green - NIR) / (Green + NIR) (McFeeters 1996). >0 water, <=0 non-water."""
    denom = green + nir
    denom = np.where(denom == 0, np.nan, denom)
    return np.clip((green - nir) / denom, -1.0, 1.0)


def classify_water(ndwi: np.ndarray, threshold: float = DEFAULT_NDWI_THRESHOLD) -> np.ndarray:
    """Binary water mask from NDWI. 1 = water, 0 = non-water; non-finite input -> 0."""
    return (np.isfinite(ndwi) & (ndwi > threshold)).astype(np.uint8)


# ── Flood-risk classification ────────────────────────────────────────────────

FLOOD_RISK_CLASS_LABELS = {
    0: "Dry land (both dates)",
    1: "Permanent water (both dates)",
    2: "Seasonal flood water (dry season land, flood season water)",
    3: "Recession (flood-season-only water at the dry-season date)",
}

FLOOD_RISK_CLASS_COLORS = {
    0: (247, 233, 195),  # dry land — tan
    1: (33, 102, 172),   # permanent water — dark blue
    2: (103, 169, 207),  # seasonal flood water — light blue
    3: (178, 24, 43),    # recession — red (flag for review, see docstring below)
}


def classify_flood_risk(water_dry_season: np.ndarray, water_flood_season: np.ndarray) -> np.ndarray:
    """Combine two binary water masks (:func:`classify_water`, one per date) into the
    4 classes in ``FLOOD_RISK_CLASS_LABELS``.

    Class 2 (dry -> water) is the actual flood signal. Class 3 (water only at the
    dry-season date) is the reverse and should be treated as a data-quality flag rather
    than a real recession — the two input dates span a full season, not a single event,
    so a pixel water only at the *dry-season* date is usually cloud/shadow contamination
    in one of the two composites, not genuine hydrology.
    """
    if water_dry_season.shape != water_flood_season.shape:
        raise ValueError(f"Shape mismatch: {water_dry_season.shape} vs {water_flood_season.shape}")
    classified = np.zeros(water_dry_season.shape, dtype=np.int8)
    classified[(water_dry_season == 1) & (water_flood_season == 1)] = 1
    classified[(water_dry_season == 0) & (water_flood_season == 1)] = 2
    classified[(water_dry_season == 1) & (water_flood_season == 0)] = 3
    return classified


# ── Land-cover cross-check ────────────────────────────────────────────────────

def refine_with_landcover_water(classified: np.ndarray, wc_permanent_water: np.ndarray) -> np.ndarray:
    """Correct ``classified`` (from :func:`classify_flood_risk`) using an independent
    permanent-water mask from land cover (e.g. ESA WorldCover class 80, aligned to the
    same grid via ``sitex.data.landcover.align_categorical_to_grid``).

    Reclassifies class 3 ("recession" — already flagged as likely cloud noise) and
    class 0 ("dry land") pixels to class 1 ("permanent water") wherever land cover
    independently confirms permanent water there — cloud contamination in one date's
    NDWI composite is a far more likely explanation than a permanent water body
    genuinely appearing or disappearing between the two dates. Class 2 ("seasonal flood
    water" — the actual flood signal) is left untouched even where land cover disagrees,
    since overriding it would erase the one class this notebook exists to find; use
    :func:`landcover_agreement_stats` to see how often that disagreement happens instead
    of silently resolving it.
    """
    if classified.shape != wc_permanent_water.shape:
        raise ValueError(f"Shape mismatch: {classified.shape} vs {wc_permanent_water.shape}")
    refined = classified.copy()
    confirmed_permanent = wc_permanent_water.astype(bool) & np.isin(classified, (0, 3))
    refined[confirmed_permanent] = 1
    return refined


def _pct_overlap(mask: np.ndarray, wc_permanent_water: np.ndarray) -> float:
    denom = int(mask.sum())
    if denom == 0:
        return float("nan")
    return 100 * float((mask & wc_permanent_water).sum()) / denom


def landcover_agreement_stats(classified: np.ndarray, wc_permanent_water: np.ndarray) -> dict:
    """Diagnostic counts comparing ``classified`` against an independent permanent-water
    mask from land cover, *before* any refinement — how much each flood-risk class
    already agrees with land cover's permanent-water call, including how many class-2
    ("seasonal flood") pixels land cover disagrees with (flagged here, not resolved —
    see :func:`refine_with_landcover_water`)."""
    if classified.shape != wc_permanent_water.shape:
        raise ValueError(f"Shape mismatch: {classified.shape} vs {wc_permanent_water.shape}")
    wc = wc_permanent_water.astype(bool)
    return {
        "class1_permanent_confirmed_by_landcover_pct": _pct_overlap(classified == 1, wc),
        "class3_recession_confirmed_permanent_by_landcover_pct": _pct_overlap(classified == 3, wc),
        "class0_dry_but_landcover_says_permanent_pct": _pct_overlap(classified == 0, wc),
        "class2_flood_but_landcover_says_permanent_pct": _pct_overlap(classified == 2, wc),
    }


# ── Vector export & webmap ───────────────────────────────────────────────────

def export_flood_risk_to_geojson(
    classified: np.ndarray, profile: dict, output_path: Path, local_epsg: int | None = None
) -> gpd.GeoDataFrame:
    """Vectorize the classified raster to GeoJSON polygons — only water-related
    classes (1, 2, 3) are exported; dry land (class 0) is not vectorised."""
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
            "flood_class": values,
            "label": [FLOOD_RISK_CLASS_LABELS[v] for v in values],
        },
        geometry=geometries,
        crs=crs,
    ).to_crs("EPSG:4326")

    gdf.to_file(output_path, driver="GeoJSON")
    return gdf


_FLOOD_LABEL_COLORS_HEX = {
    label: "#%02x%02x%02x" % FLOOD_RISK_CLASS_COLORS[cls] for cls, label in FLOOD_RISK_CLASS_LABELS.items()
}


def build_flood_risk_webmap(
    gdf: gpd.GeoDataFrame,
    center: tuple,
    dry_date: str,
    flood_date: str,
    zoom_start: int = 11,
    pct_permanent: float | None = None,
    pct_flood: float | None = None,
    pct_dry: float | None = None,
):
    """Folium map of flood-risk polygons — shareable as a standalone HTML file.

    The legend always explains the 3 shaded classes and the two source dates; passing
    the ``pct_*`` summary stats (from the same run's ``classify_flood_risk`` output)
    adds a one-line caption with area shares, so the map is readable on its own once
    shared as a standalone HTML file.
    """
    import folium

    m = folium.Map(location=list(center), zoom_start=zoom_start, tiles="CartoDB positron")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", overlay=False,
    ).add_to(m)

    folium.GeoJson(
        gdf,
        name=f"Flood risk ({dry_date}→{flood_date})",
        style_function=lambda feat: {
            "fillColor": _FLOOD_LABEL_COLORS_HEX.get(feat["properties"]["label"], "#95a5a6"),
            "color": "black", "weight": 0.3, "fillOpacity": 0.65,
        },
        tooltip=folium.GeoJsonTooltip(fields=["label", "flood_class"], aliases=["Class:", "Value:"]),
    ).add_to(m)

    stats_line = ""
    if pct_permanent is not None and pct_flood is not None and pct_dry is not None:
        stats_line = (
            f"<div style='margin-top:4px;color:#555;'>"
            f"Permanent water {pct_permanent:.1f}% &middot; Seasonal flood {pct_flood:.1f}% "
            f"&middot; Dry land {pct_dry:.1f}%</div>"
        )

    c1, c2, c3 = FLOOD_RISK_CLASS_COLORS[1], FLOOD_RISK_CLASS_COLORS[2], FLOOD_RISK_CLASS_COLORS[3]
    legend_html = f"""
    <div style='position:fixed;bottom:30px;right:10px;z-index:9999;
         background:white;padding:12px;border-radius:6px;
         box-shadow:0 2px 6px rgba(0,0,0,0.2);font-family:Arial;font-size:13px;
         max-width:280px;'>
        <b>Flood Risk &mdash; NDWI</b><br>
        <span style='color:#555;'>Dry season {dry_date} &rarr; Flood season {flood_date}</span><br>
        <i style='background:rgb{c1};width:14px;height:14px;
           display:inline-block;border-radius:2px;'></i>&nbsp;Permanent water<br>
        <i style='background:rgb{c2};width:14px;height:14px;
           display:inline-block;border-radius:2px;'></i>&nbsp;Seasonal flood water<br>
        <i style='background:rgb{c3};width:14px;height:14px;
           display:inline-block;border-radius:2px;'></i>&nbsp;Recession (likely cloud noise &mdash; see notes)<br>
        <span style='color:#777;font-size:11px;'>Unshaded area = dry land at both dates</span>
        {stats_line}
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    return m
