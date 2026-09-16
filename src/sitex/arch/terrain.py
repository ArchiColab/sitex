"""Ground representation — DEM raster to contour lines.

Extracted from ``01-ARCH-DEM Contour.ipynb``. Turns a DTM GeoTIFF into a
contour-line ``GeoDataFrame`` (for GIS) and a Z-elevation DXF (for CAD),
sharing the same ``CityConfig``-derived local UTM CRS and CAD/UCS origin
that ``sitex.arch.morphology`` uses for buildings — see the ARCH overview's
Section 5.4 for why that origin has to be shared, not recomputed per file.
"""

from __future__ import annotations

from pathlib import Path

import ezdxf
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
import xarray as xr
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString

from ..core.config import CityConfig


def load_dtm_utm(dtm_path: Path, local_epsg: int) -> xr.DataArray:
    """Load a DTM GeoTIFF and reproject it to a metric UTM CRS."""
    dtm = rioxarray.open_rasterio(dtm_path, masked=True).squeeze("band", drop=True)
    return dtm.rio.reproject(f"EPSG:{local_epsg}")


def smooth_nan_aware(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth ``arr``, treating NaN as missing rather than zero.

    Plain ``scipy.ndimage.gaussian_filter`` treats NaN as 0, which pulls
    values near a data edge toward 0 and erodes the terrain boundary.
    Smoothing the numerator and a 0/1 weight mask separately and dividing
    keeps edge pixels averaging only over their valid neighbours.
    """
    if sigma <= 0:
        return arr
    nan_mask = np.isnan(arr)
    filled = np.where(nan_mask, 0.0, arr)
    weights = np.where(nan_mask, 0.0, 1.0)
    smooth_num = gaussian_filter(filled, sigma=sigma)
    smooth_den = gaussian_filter(weights, sigma=sigma)
    with np.errstate(invalid="ignore"):
        return np.where(smooth_den > 0, smooth_num / smooth_den, np.nan)


def extract_contours(
    x: np.ndarray,
    y: np.ndarray,
    arr: np.ndarray,
    interval: float,
    local_epsg: int,
    elev_min: float | None = None,
    elev_max: float | None = None,
) -> gpd.GeoDataFrame:
    """Extract contour lines from a gridded elevation array at a fixed interval.

    One row per contour segment (``ELEV``, ``geometry``) — matplotlib's
    contouring emits disjoint segments per level, not one merged line, and
    that shape is kept rather than dissolved.
    """
    elev_min = int(np.nanmin(arr)) if elev_min is None else elev_min
    elev_max = int(np.nanmax(arr)) if elev_max is None else elev_max
    levels = list(np.arange(
        (elev_min // interval) * interval,
        (elev_max // interval + 2) * interval,
        interval,
    ))

    fig, ax = plt.subplots()
    cs = ax.contour(x, y, arr, levels=levels)
    plt.close(fig)

    lines, elevs = [], []
    for level, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            if len(seg) >= 2:
                lines.append(LineString(seg))
                elevs.append(float(level))

    return gpd.GeoDataFrame({"ELEV": elevs, "geometry": lines}, crs=f"EPSG:{local_epsg}")


def generate_contours(
    dtm_path: Path,
    local_epsg: int,
    interval: float = 2.0,
    smooth_sigma: float = 2.0,
    elev_min: float | None = None,
    elev_max: float | None = None,
) -> gpd.GeoDataFrame:
    """End-to-end: load a DTM GeoTIFF, reproject, NaN-aware smooth, extract contours."""
    dtm_utm = load_dtm_utm(dtm_path, local_epsg)
    arr = smooth_nan_aware(dtm_utm.values.astype(float), smooth_sigma)
    return extract_contours(dtm_utm.x.values, dtm_utm.y.values, arr, interval, local_epsg, elev_min, elev_max)


def generate_contours_for_city(
    city: CityConfig,
    dtm_filename: str = "dtm_nasadem.tif",
    **kwargs,
) -> gpd.GeoDataFrame:
    """``generate_contours()`` reading ``{city.data_dir}/dem/{dtm_filename}``."""
    dtm_path = city.data_dir / "dem" / dtm_filename
    return generate_contours(dtm_path, city.local_epsg, **kwargs)


def export_contours_dxf(
    gdf: gpd.GeoDataFrame,
    out_path: Path,
    origin_easting: float,
    origin_northing: float,
) -> None:
    """Export contour lines as 3D DXF polylines, shifted to a local CAD/UCS origin.

    DXF carries no CRS metadata; subtracting a shared local origin keeps
    geometry close to (0, 0) so CAD tools (Revit/Rhino) don't hit precision
    issues far from true UTM coordinates. Re-apply the same origin as the
    project base point / survey point in the CAD tool to georeference the
    model.
    """
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    for _, row in gdf.iterrows():
        pts_3d = [
            (px - origin_easting, py - origin_northing, float(row.ELEV))
            for px, py in row.geometry.coords
        ]
        msp.add_polyline3d(pts_3d, dxfattribs={"layer": f"ELEV_{int(row.ELEV):04d}"})
    doc.saveas(out_path)


def export_contours_dxf_for_city(gdf: gpd.GeoDataFrame, city: CityConfig, filename: str = "contours.dxf") -> Path:
    """``export_contours_dxf()`` writing to ``{city.data_dir}/dem/{filename}`` at ``city.origin``."""
    out_path = city.data_dir / "dem" / filename
    export_contours_dxf(gdf, out_path, *city.origin)
    return out_path


def plot_contours_folium(
    gdf: gpd.GeoDataFrame,
    center_lat: float,
    center_lon: float,
    interval: float,
    local_epsg: int,
    zoom_start: int = 14,
):
    """Interactive elevation-coloured contour map, reprojected to WGS84 for display."""
    import folium

    gdf_wgs = gdf.to_crs("EPSG:4326")
    cmap = cm.get_cmap("RdYlGn_r")
    norm = mcolors.Normalize(vmin=gdf_wgs["ELEV"].min(), vmax=gdf_wgs["ELEV"].max())

    def _style(feature):
        elev = feature["properties"]["ELEV"]
        rgba = cmap(norm(elev))
        return {"color": mcolors.to_hex(rgba[:3]), "weight": 1.2, "opacity": 0.85}

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles=None)
    folium.TileLayer(tiles="CartoDB positron", name="CartoDB (light)", control=True, overlay=False).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        control=True,
        overlay=False,
    ).add_to(m)
    folium.GeoJson(
        gdf_wgs,
        name=f"Contours {interval} m (EPSG:{local_epsg})",
        style_function=_style,
        tooltip=folium.GeoJsonTooltip(fields=["ELEV"], aliases=["Elevation (m)"]),
    ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m
