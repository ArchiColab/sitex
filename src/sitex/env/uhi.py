"""Urban Heat Island — NDVI, NBEM emissivity (narrative only), LST, UHI intensity, and
Heat Exposure Index from Landsat 8/9 Collection 2 Level-2 bands.

Extracted from ``02-ENV-UHI-analysis.ipynb``. **Level-2 only**: Landsat 8/9 Level-1
scenes aren't available for this AOI (a Global-South-specific USGS ground-station
gap), so LST is read directly from USGS's own ``ST_B10`` Surface Temperature band
rather than derived via a manual mono-window retrieval — which also sidesteps that
algorithm's known sensitivity to Vietnam's humid tropical atmosphere. Emissivity is
still computed (NBEM, reproducing ``pylandtemp``'s formula by hand — that package
depends on ``pkg_resources``, broken on current Colab runtimes) for land-cover
narrative only; it does not feed into LST, since ``ST_B10`` already bakes in its own
atmospheric correction and per-pixel emissivity.
"""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
import rasterio.warp
from shapely.geometry import shape

# Landsat Collection 2 Level-2 fixed radiometric rescaling constants (USGS Landsat
# 8-9 Data Users Handbook) — identical for every scene.
SR_REFLECTANCE_MULT = 2.75e-05
SR_REFLECTANCE_ADD = -0.2
ST_MULT = 0.00341802
ST_ADD = 149.0

# NBEM emissivity constants (Band 10 only) — Li & Meng (2018), Yu, Guo & Wu (2014).
NDVI_MIN, NDVI_MAX = -1.0, 1.0
BARESOIL_NDVI_MAX = 0.2
VEGETATION_NDVI_MIN = 0.5
EMISSIVITY_SOIL_10 = 0.9668
EMISSIVITY_VEG_10 = 0.9863
RED_BAND_COEFF_A_10 = 0.973
RED_BAND_COEFF_B_10 = 0.047
GEOMETRICAL_FACTOR = 0.55

DEFAULT_NDVI_THRESHOLD = 0.3

HEI_WEIGHTS = {
    "50_50": (0.5, 0.5),
    "70_30": (0.7, 0.3),
    "30_70": (0.3, 0.7),
}


# ── Scene discovery (Phase-1 handoff) ────────────────────────────────────────

def find_latest_scene_info(landsat_output_dir: Path) -> Path:
    """Most recently written ``scene_info.json`` under a Phase-1 Landsat output dir."""
    candidates = sorted(Path(landsat_output_dir).glob("*/scene_info.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No scene_info.json found under {landsat_output_dir} — run the remote-sensing "
            "acquisition notebook first."
        )
    return candidates[-1]


def load_scene_info(scene_info_path: Path) -> dict:
    """Load and validate a Phase-1 ``scene_info.json`` — Level-2 only.

    Raises if the referenced scene is Level-1: this module reads LST directly from
    ``ST_B10`` (Level-2 only).
    """
    with open(scene_info_path, encoding="utf-8") as f:
        scene_info = json.load(f)
    level = scene_info.get("product_level", "L2")
    if level != "L2":
        raise ValueError(f"Scene {scene_info.get('landsat_id')} is {level}, but this module only supports Level-2.")
    return scene_info


def resolve_band_paths(scene_info: dict, scene_info_path: Path) -> dict:
    """Resolve ``scene_info["band_paths"]`` relative to ``scene_info_path``'s own
    directory, instead of trusting the stored path strings.

    The acquisition notebook writes those strings relative to wherever *it* happens
    to run from (``notebooks/``) — fragile the moment ``scene_info.json`` is read
    from anywhere else. Every band file the acquisition notebook writes lives
    alongside its own ``scene_info.json``, so reconstructing the path from that
    directory plus the file's own name is both correct and portable.
    """
    scene_dir = Path(scene_info_path).parent
    return {band: scene_dir / PureWindowsPath(path).name for band, path in scene_info["band_paths"].items()}


def bbox_to_geojson(west: float, south: float, east: float, north: float) -> dict:
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
        },
        "properties": {},
    }


def default_bbox(lat: float, lon: float, dist_m: float = 1000) -> list:
    """Rough equirectangular bounding box (WGS84) centred on a point."""
    dlat = dist_m / 111_320
    dlon = dist_m / (111_320 * np.cos(np.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def clip_to_geojson(band_path: Path, aoi_feature: dict, target_dir: Path | None = None) -> Path:
    """Clip a GeoTIFF to an AOI polygon (GeoJSON Feature dict)."""
    band_path = Path(band_path)
    target_dir = Path(target_dir) if target_dir is not None else band_path.parent

    with rasterio.open(band_path) as src:
        aoi = gpd.GeoDataFrame(geometry=[shape(aoi_feature["geometry"])], crs="EPSG:4326").to_crs(src.crs)
        out_image, out_transform = rasterio.mask.mask(src, aoi.geometry, crop=True)
        out_meta = src.meta.copy()

    out_meta.update({"height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{band_path.stem}_clipped{band_path.suffix}"
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(out_image)
    return out_path


# ── Band I/O ─────────────────────────────────────────────────────────────────

def read_band(path: Path):
    with rasterio.open(path) as src:
        return src.read(1).astype("float64"), src.meta.copy()


def _write_single_band(array: np.ndarray, meta: dict, target_path: Path) -> Path:
    out_meta = meta.copy()
    out_meta.update(dtype="float64", count=1)
    with rasterio.open(target_path, "w", **out_meta) as dst:
        dst.write(array, 1)
    return Path(target_path)


# ── NDVI, Emissivity (NBEM), LST ─────────────────────────────────────────────

def dn_to_reflectance(dn: np.ndarray) -> np.ndarray:
    return SR_REFLECTANCE_MULT * dn + SR_REFLECTANCE_ADD


def compute_ndvi(refl_4: np.ndarray, refl_5: np.ndarray, dn_4: np.ndarray | None = None) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red) from Surface Reflectance.

    ``dn_4`` (raw DN, if given) masks true no-data pixels (DN == 0) to NaN, distinct
    from a legitimate reflectance-derived NDVI of exactly 0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (refl_5 - refl_4) / (refl_5 + refl_4)
    if dn_4 is not None:
        ndvi = np.where(dn_4 == 0, np.nan, ndvi)
    return ndvi


def calc_ndvi(band_4_path: Path, band_5_path: Path, target_path: Path | None = None):
    """Read Band 4/5 DN, convert to reflectance, compute NDVI. Returns ``(ndvi, meta)``."""
    dn_4, meta = read_band(band_4_path)
    dn_5, _ = read_band(band_5_path)
    ndvi = compute_ndvi(dn_to_reflectance(dn_4), dn_to_reflectance(dn_5), dn_4=dn_4)
    if target_path is not None:
        _write_single_band(ndvi, meta, target_path)
    return ndvi, meta


def fractional_vegetation_cover(ndvi_array: np.ndarray) -> np.ndarray:
    return ((ndvi_array - BARESOIL_NDVI_MAX) / (VEGETATION_NDVI_MIN - BARESOIL_NDVI_MAX)) ** 2


def cavity_effect(
    emissivity_veg: float, emissivity_soil: float, fvc: np.ndarray, geometrical_factor: float = GEOMETRICAL_FACTOR
) -> np.ndarray:
    return (1 - emissivity_soil) * emissivity_veg * geometrical_factor * (1 - fvc)


def compute_emissivity_nbem(ndvi_array: np.ndarray, red_reflectance: np.ndarray) -> np.ndarray:
    """Surface emissivity via the NBEM Band-10 branch (Li & Meng, 2018; Yu, Guo & Wu, 2014).

    NDVI-based bare-soil / vegetation / mixed-pixel classification, fractional
    vegetation cover, and a cavity-effect correction — reproduces ``pylandtemp``'s
    formula by hand. Narrative-only: does not feed into LST (see the module docstring).
    """
    mask_baresoil = (ndvi_array >= NDVI_MIN) & (ndvi_array < BARESOIL_NDVI_MAX)
    mask_vegetation = (ndvi_array > VEGETATION_NDVI_MIN) & (ndvi_array <= NDVI_MAX)
    mask_mixed = (ndvi_array >= BARESOIL_NDVI_MAX) & (ndvi_array <= VEGETATION_NDVI_MIN)

    fvc = np.clip(fractional_vegetation_cover(ndvi_array), 0, 1)
    cavity = cavity_effect(EMISSIVITY_VEG_10, EMISSIVITY_SOIL_10, fvc)

    emissivity = np.full_like(ndvi_array, np.nan)
    emissivity[mask_baresoil] = RED_BAND_COEFF_A_10 - RED_BAND_COEFF_B_10 * red_reflectance[mask_baresoil]
    emissivity[mask_mixed] = (
        EMISSIVITY_VEG_10 * fvc[mask_mixed] + EMISSIVITY_SOIL_10 * (1 - fvc[mask_mixed]) + cavity[mask_mixed]
    )
    emissivity[mask_vegetation] = EMISSIVITY_VEG_10 + cavity[mask_vegetation]
    return np.where(np.isnan(ndvi_array), np.nan, emissivity)


def calc_emissivity(ndvi_array: np.ndarray, band_4_path: Path, meta: dict, target_path: Path | None = None):
    dn_4, _ = read_band(band_4_path)
    emissivity = compute_emissivity_nbem(ndvi_array, dn_to_reflectance(dn_4))
    if target_path is not None:
        _write_single_band(emissivity, meta, target_path)
    return emissivity


def compute_lst(dn_10: np.ndarray) -> np.ndarray:
    """LST (°C) from USGS's own Level-2 ``ST_B10`` Surface Temperature band."""
    lst_k = ST_MULT * dn_10 + ST_ADD
    lst_celsius = lst_k - 273.15
    return np.where(dn_10 == 0, np.nan, lst_celsius)


def calc_lst(band_10_path: Path, target_path: Path | None = None):
    dn_10, meta = read_band(band_10_path)
    lst = compute_lst(dn_10)
    if target_path is not None:
        _write_single_band(lst, meta, target_path)
    return lst, meta


# ── UHI intensity ────────────────────────────────────────────────────────────

def compute_uhi_intensity(ndvi: np.ndarray, lst: np.ndarray, ndvi_threshold: float = DEFAULT_NDVI_THRESHOLD) -> dict:
    """Surface UHI intensity: mean LST gap between built-up/bare and vegetated pixels,
    using NDVI as a cheap land-cover proxy (no segmentation model needed)."""
    vegetation_mask = ndvi > ndvi_threshold
    built_up_mask = ~vegetation_mask & ~np.isnan(ndvi)

    mean_lst_vegetation = float(np.nanmean(lst[vegetation_mask]))
    mean_lst_built_up = float(np.nanmean(lst[built_up_mask]))
    veg_pct = 100.0 * float(np.nansum(vegetation_mask)) / vegetation_mask.size

    return {
        "ndvi_threshold": ndvi_threshold,
        "vegetation_pct": veg_pct,
        "mean_lst_vegetation": mean_lst_vegetation,
        "mean_lst_built_up": mean_lst_built_up,
        "uhi_intensity": mean_lst_built_up - mean_lst_vegetation,
        "vegetation_mask": vegetation_mask,
        "built_up_mask": built_up_mask,
    }


def compute_ndvi_lst_density(ndvi: np.ndarray, lst: np.ndarray, bins: int = 80):
    """2D histogram of (NDVI, LST) pairs — a density heatmap standing in for a
    many-thousand-point scatter plot, which this environment's broken matplotlib
    install can't render (see ``sitex.core.raster_viz``'s module docstring).

    Returns ``(density, xedges, yedges)``, ``density`` shaped ``(n_lst_bins, n_ndvi_bins)``
    with row 0 = lowest LST — flip vertically before treating it as a top-down image.
    """
    valid = np.isfinite(ndvi) & np.isfinite(lst)
    hist, xedges, yedges = np.histogram2d(ndvi[valid], lst[valid], bins=bins)
    return hist.T, xedges, yedges


# ── Heat Exposure Index ──────────────────────────────────────────────────────

def normalize01(array: np.ndarray) -> np.ndarray:
    return (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))


def compute_hei(lst: np.ndarray, ndvi: np.ndarray, weights: dict = HEI_WEIGHTS) -> dict:
    """Heat Exposure Index at each weighting scenario: ``w_lst * LST_norm + w_veg * VegDeficit``.

    Tested at three scenarios rather than committing to one — pixels that stay
    high-priority across all three are the more robust hot spots.
    """
    lst_norm = normalize01(lst)
    veg_deficit = 1 - normalize01(ndvi)
    return {label: w_lst * lst_norm + w_veg * veg_deficit for label, (w_lst, w_veg) in weights.items()}


def hei_sensitivity_range(hei_arrays: dict) -> np.ndarray:
    """Per-pixel spread across HEI weighting scenarios — a small range means a pixel is
    robustly high/low priority regardless of weighting; a large range means the
    weighting choice actually matters there."""
    stack = np.stack(list(hei_arrays.values()))
    return np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)


def export_hei_geotiffs(hei_arrays: dict, meta: dict, output_dir: Path, prefix: str = "hei") -> dict:
    output_dir = Path(output_dir)
    return {label: _write_single_band(array, meta, output_dir / f"{prefix}_{label}.tif") for label, array in hei_arrays.items()}


# ── Reprojection & colour export for the webmap ─────────────────────────────

def reproject_geotiff(src_path: Path, target_path: Path, target_crs: int) -> Path:
    """Reproject a single/multi-band GeoTIFF to ``target_crs``, filling gaps with NaN.

    Areas outside the source's transformed footprint (e.g. rotated corners left over
    when warping a UTM-aligned raster into EPSG:4326) get NaN rather than 0 — 0 would
    drag a colour stretch's min down and crush a naturally narrow-range layer like
    Emissivity (~0.95-0.99) into a thin sliver at the top of the colormap.
    """
    with rasterio.open(src_path) as src:
        img = src.read().astype(np.float32)
        if src.nodata is not None:
            img[img == src.nodata] = np.nan

        crs = rasterio.crs.CRS.from_epsg(target_crs)
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, crs, src.width, src.height, *src.bounds
        )
        reprojected = np.full((img.shape[0], height, width), np.nan, dtype=img.dtype)
        rasterio.warp.reproject(
            img, reprojected,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs,
            src_nodata=np.nan, dst_nodata=np.nan,
        )
        meta = src.meta.copy()
        meta.update(crs=crs, transform=transform, width=width, height=height, dtype=img.dtype, nodata=np.nan)
        with rasterio.open(target_path, "w", **meta) as dst:
            dst.write(reprojected)
    return Path(target_path)


def create_rgba_color_image(src_path: Path, target_path: Path, colormap: str = "RdBu_r"):
    """Colorize a single-band raster file and save as an RGBA GeoTIFF (map overlay /
    external GIS use). Colorization itself is ``sitex.core.raster_viz.colorize_array``
    — pure NumPy colormap lookup, never matplotlib's Figure/draw pipeline.
    """
    from ..core.raster_viz import colorize_array

    with rasterio.open(str(src_path)) as src:
        band = src.read(1)
        meta = src.meta.copy()

    image = colorize_array(band, colormap)

    meta.update(count=4, dtype=rasterio.uint8, nodata=None)
    with rasterio.open(target_path, "w", **meta) as dst:
        dst.write(np.moveaxis(image, 2, 0).astype(rasterio.uint8))
    return Path(target_path), image


def load_raster_rgba(path: Path):
    """Read an RGBA GeoTIFF, return ``(array_HWC, rasterio bounds)``."""
    with rasterio.open(path) as src:
        return np.moveaxis(src.read(), 0, 2), src.bounds


def build_uhi_webmap(center: tuple, layers: dict, aoi_geojson: dict | None = None, primary: str | None = None, zoom_start: int = 13):
    """Interactive folium webmap: one toggleable ``ImageOverlay`` per ``layers`` entry
    (``{name: (rgba_array, rasterio_bounds)}``), plus the AOI boundary if given.

    Passing a raw RGBA array directly to ``ImageOverlay`` (rather than a PNG file
    path) has been confirmed not to hit this environment's matplotlib-render crash —
    branca's array encoding path doesn't touch matplotlib's Figure/draw pipeline.
    """
    import folium

    m = folium.Map(location=list(center), zoom_start=zoom_start, tiles="CartoDB positron", control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", overlay=False,
    ).add_to(m)

    for name, (arr, bounds) in layers.items():
        folium.raster_layers.ImageOverlay(
            image=arr, name=name, opacity=0.75,
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            show=(name == primary),
        ).add_to(m)

    if aoi_geojson is not None:
        folium.GeoJson(
            aoi_geojson, name="AOI boundary",
            style_function=lambda _: {"fillOpacity": 0, "color": "black", "weight": 1.5, "dashArray": "4 4"},
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
