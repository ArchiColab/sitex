"""Building footprints traced on top of UHI/change surfaces, extracted from
``04-VIZ-BuildingsOnUHISurfaces.ipynb``.

Earlier versions of this module also offered a "mask" and "clip" mode — blanking out
or GDAL-clipping every non-building pixel — but at the 30 m/10 m source resolution
here, most buildings are smaller than a single pixel, so masking/clipping throws away
most of the raster while adding no real per-building precision. Outline (tracing
footprints on top of the full raster, values intact) is the one mode that stays
useful at this resolution.

Rendering goes through ``sitex.core.raster_viz`` (colormap-array + PIL), not
matplotlib ``imshow``/``GeoDataFrame.plot()`` — both have been confirmed to crash this
project's kernel outright, independent of what's plotted.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize

from ..core.raster_viz import alpha_composite, colorize_array, mask_to_rgba


def outline_mask(buildings_gdf: gpd.GeoDataFrame, out_shape: tuple, transform, bounds) -> np.ndarray:
    """Rasterize building *boundaries* (not fills) onto the raster's own grid — a thin
    outline mask, for tracing footprints on top of a full raster without hiding the
    surface value underneath (the "outline" mode)."""
    buildings_clip = buildings_gdf.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]
    boundaries = buildings_clip.geometry.boundary
    return rasterize(boundaries, out_shape=out_shape, transform=transform, fill=0, default_value=1, dtype="uint8", all_touched=True)


def render_outline_overlay(arr: np.ndarray, transform, bounds, buildings_gdf: gpd.GeoDataFrame, cmap: str, vmin=None, vmax=None, outline_color: tuple = (217, 217, 217)) -> np.ndarray:
    """Colorize ``arr`` and trace building outlines on top, in a light grey."""
    base = colorize_array(arr, cmap, vmin=vmin, vmax=vmax)
    outline = outline_mask(buildings_gdf, arr.shape, transform, bounds)
    overlay = mask_to_rgba(outline.astype(bool), color=outline_color, alpha=230)
    return alpha_composite(base, overlay)
