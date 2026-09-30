"""Map panels for the building-morphology notebook -- one map per data layer, for
classroom discussion rather than statistical exploration.

Rendering goes through ``sitex.core.raster_viz`` (colormap-array + PIL). Building
heights are rasterized onto a grid first and rendered the same way as everything
else here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

from ..core.raster_viz import add_caption, add_legend, alpha_composite, colorize_array, colorize_categorical, mask_to_rgba, render_map_panel

DEFAULT_PLOT_SIZE = (700, 500)
DEFAULT_SQUARE_PLOT_SIZE = (700, 700)


def find_latest_ndvi(output_dir: Path) -> Path | None:
    """Most recent ``outputs/ndvi_{year}.tif`` from 02-ENV-Urban_Change_Detection.ipynb
    (picks the higher year, i.e. the more recent Sentinel-2 composite). Returns
    ``None`` rather than raising if none exist -- unlike
    ``sitex.viz.raster_scenes.load_change_rasters``, this doesn't need both years,
    just whichever single composite is newest, for a one-map-per-layer summary."""
    candidates = list(Path(output_dir).glob("ndvi_*.tif"))
    if not candidates:
        return None

    def _year(p):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        return int(digits) if digits else -1

    return max(candidates, key=_year)


def find_latest_uhi_lst(landsat_output_dir: Path) -> Path | None:
    """``temperature_lst.tif`` from the newest Landsat scene folder that has a
    ``scene_info.json``, matching 02-ENV-UHI_Analysis.ipynb's own discovery. Needs
    only that one file, unlike ``sitex.viz.raster_scenes.find_latest_uhi_scene``
    (which also requires ``ndvi.tif``/``hei_*.tif`` -- too strict here, where only
    the LST layer matters). Returns ``None`` rather than raising if nothing's been
    run yet."""
    landsat_output_dir = Path(landsat_output_dir)
    if not landsat_output_dir.exists():
        return None
    scene_infos = list(landsat_output_dir.glob("*/scene_info.json"))
    if not scene_infos:
        return None
    newest = max(scene_infos, key=lambda p: p.stat().st_mtime)
    lst_path = newest.parent / "temperature_lst.tif"
    return lst_path if lst_path.exists() else None


def _pixel_coords(transform, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """1D longitude/latitude coordinate arrays for a raster's transform, for
    ``render_map_panel``'s tick labels."""
    xs, _ = rasterio.transform.xy(transform, [0] * width, list(range(width)))
    _, ys = rasterio.transform.xy(transform, list(range(height)), [0] * height)
    return np.array(xs), np.array(ys)


def _read_band(path: Path, band: int = 1) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(band).astype("float64")
        nodata = src.nodata
        transform = src.transform
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, transform


def plot_building_heights_map(
    buildings_gdf,
    out_path: Path,
    value_col: str = "height_final",
    cmap_name: str = "viridis",
    plot_size: tuple = DEFAULT_SQUARE_PLOT_SIZE,
    title: str | None = None,
) -> Path:
    """Rasterize each building's ``value_col`` onto a grid and render it through the
    PIL pipeline. A building smaller than one output pixel can be lost the same
    way GBA overlays lose sub-pixel footprints elsewhere in this project (a known,
    accepted resolution limit, not a bug) -- raise ``plot_size`` if that matters for
    your AOI."""
    valid = buildings_gdf[buildings_gdf[value_col].notna()]
    minx, miny, maxx, maxy = valid.total_bounds
    width, height = plot_size
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    shapes = list(zip(valid.geometry, valid[value_col]))
    grid = rasterize(shapes, out_shape=(height, width), transform=transform, fill=np.nan, dtype="float64")

    xs, ys = _pixel_coords(transform, width, height)
    return render_map_panel(
        grid, xs, ys, cmap_name=cmap_name, title=title or "Building heights",
        cbar_label="Height (m)", plot_size=plot_size, out_path=out_path,
    )


def plot_landcover_map(
    wc_path: Path,
    out_path: Path,
    wc_classes: dict,
    plot_size: tuple = DEFAULT_SQUARE_PLOT_SIZE,
    title: str | None = None,
) -> Path:
    """Categorical land-cover map -- discrete class colors + a swatch legend, not a
    continuous colorbar (``add_legend`` instead of ``render_map_panel``)."""
    from PIL import Image

    with rasterio.open(wc_path) as src:
        data = src.read(1)

    category_colors, entries = {}, []
    for code_val, (name, hex_color) in wc_classes.items():
        if not (data == code_val).any():
            continue
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
        category_colors[code_val] = rgb
        entries.append((rgb, name))

    rgba = colorize_categorical(data, category_colors)
    out_path = Path(out_path)
    Image.fromarray(rgba, mode="RGBA").resize(plot_size, Image.NEAREST).save(out_path)
    add_caption(out_path, title or "Land cover (ESA WorldCover)")
    add_legend(out_path, entries)
    return out_path


def plot_ndvi_map(ndvi_path: Path, out_path: Path, plot_size: tuple = DEFAULT_PLOT_SIZE, title: str | None = None) -> Path:
    data, transform = _read_band(ndvi_path)
    xs, ys = _pixel_coords(transform, data.shape[1], data.shape[0])
    return render_map_panel(
        data, xs, ys, cmap_name="RdYlGn", vmin=-1, vmax=1, title=title or "NDVI",
        cbar_label="NDVI", plot_size=plot_size, out_path=out_path,
    )


def plot_uhi_map(lst_path: Path, out_path: Path, plot_size: tuple = DEFAULT_PLOT_SIZE, title: str | None = None) -> Path:
    data, transform = _read_band(lst_path)
    xs, ys = _pixel_coords(transform, data.shape[1], data.shape[0])
    return render_map_panel(
        data, xs, ys, cmap_name="inferno", title=title or "Land Surface Temperature",
        cbar_label="Temperature (°C)", plot_size=plot_size, out_path=out_path,
    )


def plot_canopy_map(canopy_path: Path, out_path: Path, plot_size: tuple = DEFAULT_PLOT_SIZE, title: str | None = None) -> Path:
    data, transform = _read_band(canopy_path)
    xs, ys = _pixel_coords(transform, data.shape[1], data.shape[0])
    return render_map_panel(
        data, xs, ys, cmap_name="Greens", vmin=0, title=title or "Tree canopy height",
        cbar_label="Canopy height (m)", plot_size=plot_size, out_path=out_path,
    )


def plot_terrain_flood_map(
    dtm_path: Path,
    out_path: Path,
    flood_margin_m: float = 5.0,
    plot_size: tuple = DEFAULT_PLOT_SIZE,
    title: str | None = None,
) -> tuple[Path, float]:
    """Terrain colored by elevation, with a red overlay for pixels within
    ``flood_margin_m`` of the AOI's own minimum elevation. No colorbar here (unlike
    the other panels) -- compositing the flood overlay on top of the colorized
    elevation array means the elevation values no longer map to a single lookup
    table `render_map_panel` could draw a colorbar for; the flood percentage goes in
    the title instead."""
    from PIL import Image

    data, _ = _read_band(dtm_path)
    min_elev = np.nanmin(data)
    flood_mask = data <= (min_elev + flood_margin_m)
    flood_pct = 100 * np.nansum(flood_mask) / np.sum(~np.isnan(data))

    base = colorize_array(data, "terrain")
    overlay = mask_to_rgba(flood_mask, color=(220, 20, 60), alpha=160)
    composited = alpha_composite(base, overlay)

    out_path = Path(out_path)
    Image.fromarray(composited, mode="RGBA").resize(plot_size, Image.NEAREST).save(out_path)
    add_caption(
        out_path,
        title or f"Terrain — red = within {flood_margin_m:.0f}m of min elevation ({flood_pct:.1f}% of AOI)",
    )
    return out_path, flood_pct
