"""Shared raster-scene loading for the VIZ module.

Every VIZ notebook that visualizes the UHI/change layers (``04-VIZ-BuildingsOnUHISurfaces``,
``04-VIZ-BuildingMaskedUHI``, ``04-VIZ-BuildingClippedUHI``, ``04-VIZ-InteractiveWebmap``)
re-derives the same two things independently: the most recently written Landsat scene with
LST/NDVI/HEI outputs, and the Sentinel-2 NDVI/NDBI 2017→2026 change pair. One definition here
instead of four.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def find_latest_uhi_scene(landsat_output_dir: Path, hei_label: str = "50_50") -> Path:
    """Most recently written Landsat scene folder with ``temperature_lst.tif``, ``ndvi.tif``,
    and ``hei_{hei_label}.tif`` all present — i.e. one where ``02-ENV-UHI-analysis`` has
    already been run end to end."""
    landsat_output_dir = Path(landsat_output_dir)
    candidates = [
        p for p in landsat_output_dir.glob("*")
        if (p / "temperature_lst.tif").exists() and (p / "ndvi.tif").exists() and (p / f"hei_{hei_label}.tif").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No scene under {landsat_output_dir} has temperature_lst.tif / ndvi.tif / "
            f"hei_{hei_label}.tif — run the UHI analysis notebook first."
        )
    return max(candidates, key=lambda p: (p / "temperature_lst.tif").stat().st_mtime)


def _read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        return arr, src.bounds, src.crs, src.transform


def load_uhi_layer_rasters(scene_dir: Path, hei_label: str = "50_50") -> dict:
    """LST/NDVI/HEI arrays + bounds/crs/transform from one UHI-analysis scene folder."""
    scene_dir = Path(scene_dir)
    lst, lst_bounds, crs, lst_transform = _read_raster(scene_dir / "temperature_lst.tif")
    ndvi, ndvi_bounds, _, ndvi_transform = _read_raster(scene_dir / "ndvi.tif")
    hei, hei_bounds, _, hei_transform = _read_raster(scene_dir / f"hei_{hei_label}.tif")
    return {
        "lst": {"arr": lst, "bounds": lst_bounds, "transform": lst_transform, "crs": crs, "cmap": "RdBu_r", "vmin": None, "vmax": None, "label": "LST (°C)"},
        "ndvi": {"arr": ndvi, "bounds": ndvi_bounds, "transform": ndvi_transform, "crs": crs, "cmap": "RdYlGn", "vmin": -1, "vmax": 1, "label": "NDVI"},
        "hei": {"arr": hei, "bounds": hei_bounds, "transform": hei_transform, "crs": crs, "cmap": "inferno", "vmin": 0, "vmax": 1, "label": "HEI (0-1)"},
    }


def load_change_rasters(output_dir: Path, year_early: int, year_recent: int, change_threshold: float = 0.05) -> dict:
    """Sentinel-2 NDVI/NDBI per-year rasters + the ``year_recent - year_early`` delta for each,
    with a symmetric colour range from the 98th percentile of ``|change|``."""
    output_dir = Path(output_dir)

    def _pair(name):
        early_path = output_dir / f"{name}_{year_early}.tif"
        recent_path = output_dir / f"{name}_{year_recent}.tif"
        for p in (early_path, recent_path):
            if not p.exists():
                raise FileNotFoundError(f"{p} not found — run the Urban Change Detection notebook first.")
        early, bounds, crs, transform = _read_raster(early_path)
        recent, _, _, _ = _read_raster(recent_path)
        return early, recent, bounds, crs, transform

    ndvi_t1, ndvi_t2, bounds, crs, transform = _pair("ndvi")
    ndbi_t1, ndbi_t2, _, _, _ = _pair("ndbi")
    ndvi_change = ndvi_t2 - ndvi_t1
    ndbi_change = ndbi_t2 - ndbi_t1
    vmax_ndvi = float(np.nanpercentile(np.abs(ndvi_change[np.isfinite(ndvi_change)]), 98))
    vmax_ndbi = float(np.nanpercentile(np.abs(ndbi_change[np.isfinite(ndbi_change)]), 98))

    return {
        "ndvi_change": {"arr": ndvi_change, "bounds": bounds, "transform": transform, "crs": crs, "cmap": "RdYlGn", "vmin": -vmax_ndvi, "vmax": vmax_ndvi, "label": f"Δ NDVI ({year_early}→{year_recent})"},
        "ndbi_change": {"arr": ndbi_change, "bounds": bounds, "transform": transform, "crs": crs, "cmap": "RdYlGn_r", "vmin": -vmax_ndbi, "vmax": vmax_ndbi, "label": f"Δ NDBI ({year_early}→{year_recent})"},
    }
