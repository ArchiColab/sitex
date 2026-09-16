"""Gehl-activity KDE density heatmaps over building-footprint context.

Extracted from ``04-VIZ-POIsHeatmaps.ipynb``. The original builds each heatmap with
``seaborn.kdeplot`` on a matplotlib ``Axes`` — this project's kernel crashes outright
on any actual matplotlib figure render, independent of what's plotted. Rendering here
goes through ``sitex.core.raster_viz`` instead.

``compute_kde_density`` deliberately does **not** use ``scipy.stats.gaussian_kde``: that
function computes its bandwidth via ``numpy.cov()``, and this environment's numpy/BLAS
build crashes the interpreter outright on *any* matrix-multiplication-backed operation
— reproduced with a bare ``np.cov()`` on random data, and even a bare ``a @ a.T`` with
no scipy/sklearn involved at all. This is the same root cause behind the
``scikit-learn`` ``KMeans`` crash found while migrating ``02-ENV-Urban_change_detection``
(sklearn's centroid update is also BLAS-backed) — one underlying environment defect,
not two unrelated library issues. Elementwise operations (``np.std``, ``np.sum``,
``np.exp``, broadcasting) are confirmed safe, so the KDE here is hand-rolled from those
instead: a diagonal-bandwidth Gaussian kernel (Scott's rule) summed one data point at a
time via broadcasting, never a matrix product.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np

from ..core.raster_viz import alpha_composite, colorize_array


def compute_kde_density(x: np.ndarray, y: np.ndarray, xlim: tuple, ylim: tuple, grid_size: int = 300, bandwidth: float | None = None) -> np.ndarray:
    """Gaussian KDE of ``(x, y)`` points, evaluated on a ``grid_size × grid_size`` grid
    over ``xlim``/``ylim``. Returns a 2D density array, row 0 = ``ylim[1]`` (top), to
    match image orientation directly. Fewer than 2 points returns an all-zero grid
    (nothing to estimate a density from).

    Diagonal bandwidth via Scott's rule (``n ** (-1/6)`` for 2D), matching
    ``scipy.stats.gaussian_kde``'s default in spirit — but computed from ``np.std()``
    (an elementwise reduction) rather than ``np.cov()`` (matrix-backed; see the module
    docstring for why that matters in this environment).
    """
    grid_x, grid_y = np.meshgrid(
        np.linspace(xlim[0], xlim[1], grid_size),
        np.linspace(ylim[1], ylim[0], grid_size),  # descending -> row 0 is the top
    )
    n = len(x)
    if n < 2:
        return np.zeros((grid_size, grid_size))

    if bandwidth is None:
        factor = n ** (-1.0 / 6.0)  # Scott's rule, d=2: n ** (-1 / (d + 4))
        hx = max(float(np.std(x)) * factor, 1e-6)
        hy = max(float(np.std(y)) * factor, 1e-6)
    else:
        hx = hy = bandwidth

    density = np.zeros((grid_size, grid_size))
    for xi, yi in zip(x, y):
        density += np.exp(-0.5 * (((grid_x - xi) / hx) ** 2 + ((grid_y - yi) / hy) ** 2))
    density /= n * 2 * np.pi * hx * hy
    return density


def rasterize_building_mask(buildings_gdf: gpd.GeoDataFrame, xlim: tuple, ylim: tuple, grid_size: int = 300) -> np.ndarray:
    """Building footprints, filled, rasterized onto the same grid ``compute_kde_density``
    uses — the basemap layer under the KDE glow."""
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    transform = from_bounds(xlim[0], ylim[0], xlim[1], ylim[1], grid_size, grid_size)
    if len(buildings_gdf) == 0:
        return np.zeros((grid_size, grid_size), dtype="uint8")
    return rasterize(buildings_gdf.geometry, out_shape=(grid_size, grid_size), transform=transform, fill=0, default_value=1, dtype="uint8")


def render_gehl_heatmap(points_gdf: gpd.GeoDataFrame, buildings_gdf: gpd.GeoDataFrame, xlim: tuple, ylim: tuple, cmap: str, grid_size: int = 300, bandwidth=None, building_color: tuple = (77, 77, 77), point_color: tuple = (255, 255, 255)) -> np.ndarray:
    """Black background, grey building footprints, a KDE glow for ``points_gdf`` on top,
    and the points themselves as small dots — the full panel as one RGBA array, ready
    for ``sitex.core.raster_viz.save_array_png``.
    """
    from ..core.raster_viz import mask_to_rgba

    base = np.zeros((grid_size, grid_size, 4), dtype=np.uint8)
    base[..., 3] = 255  # opaque black background

    building_mask = rasterize_building_mask(buildings_gdf, xlim, ylim, grid_size)
    base = alpha_composite(base, mask_to_rgba(building_mask.astype(bool), color=building_color, alpha=255))

    if len(points_gdf) > 1:
        density = compute_kde_density(points_gdf.geometry.x.values, points_gdf.geometry.y.values, xlim, ylim, grid_size, bandwidth)
        if density.max() > 0:
            kde_rgba = colorize_array(density, cmap, vmin=0, vmax=density.max())
            # Fade the glow out where density is negligible instead of a hard threshold.
            alpha_scale = np.clip(density / (density.max() * 0.05), 0, 1)
            kde_rgba = kde_rgba.copy()
            kde_rgba[..., 3] = (kde_rgba[..., 3].astype(float) * alpha_scale).astype(np.uint8)
            base = alpha_composite(base, kde_rgba)

    # Scatter the points themselves as small dots.
    from rasterio.transform import from_bounds

    from ..core.geo import xy_to_rowcol

    transform = from_bounds(xlim[0], ylim[0], xlim[1], ylim[1], grid_size, grid_size)
    if len(points_gdf) > 0:
        rows, cols = xy_to_rowcol(transform, points_gdf.geometry.x.values, points_gdf.geometry.y.values)
        in_bounds = (rows >= 0) & (rows < grid_size) & (cols >= 0) & (cols < grid_size)
        base[rows[in_bounds], cols[in_bounds]] = (*point_color, 255)

    return base
