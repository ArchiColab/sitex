"""Ground representation — DEM raster to contour lines.

Turns a DTM GeoTIFF into a contour-line ``GeoDataFrame`` (for GIS) and a
Z-elevation DXF (for CAD), sharing the same ``CityConfig``-derived local UTM
CRS and CAD/UCS origin that ``sitex.arch.morphology`` uses for buildings, so
contours and buildings line up in CAD.
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


def write_contour_qml(
    gdf: gpd.GeoDataFrame,
    out_path: Path,
    n_classes: int = 8,
    cmap_name: str = "RdYlGn_r",
    line_width_mm: float = 0.35,
) -> Path:
    """Write a QGIS ``.qml`` style (graduated by ``ELEV``, same colour ramp as
    :func:`plot_contours_folium`) next to a contour Shapefile.

    QGIS auto-applies a sidecar style file with the exact same basename as the
    layer the first time it's added to a project — so a student who opens
    ``contours_{epsg}.shp`` in QGIS sees it already coloured by elevation, no
    manual Symbology setup needed. Verified against a real QGIS 3.40 install
    (headless PyQGIS: loads, produces a ``graduatedSymbol`` renderer with the
    expected class count and colours) — this is XML written by hand, not
    exported through QGIS itself, so that check is what makes it trustworthy
    rather than assumed.
    """
    elev_min, elev_max = float(gdf["ELEV"].min()), float(gdf["ELEV"].max())
    edges = np.linspace(elev_min, elev_max, n_classes + 1)
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=elev_min, vmax=elev_max)

    ranges_xml, symbols_xml = [], []
    for i in range(n_classes):
        lo, hi = edges[i], edges[i + 1]
        r, g, b, _ = cmap(norm((lo + hi) / 2))
        rgba = f"{int(r * 255)},{int(g * 255)},{int(b * 255)},255"
        ranges_xml.append(
            f'<range label="{lo:.0f} - {hi:.0f} m" lower="{lo:.6f}" upper="{hi:.6f}" '
            f'render="true" uuid="{{sym{i}}}" symbol="{i}"/>'
        )
        symbols_xml.append(f'''<symbol type="line" name="{i}" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleLine" locked="0" pass="0" enabled="1">
          <Option type="Map">
            <Option type="QString" name="line_color" value="{rgba}"/>
            <Option type="QString" name="line_width" value="{line_width_mm}"/>
            <Option type="QString" name="line_width_unit" value="MM"/>
            <Option type="QString" name="capstyle" value="square"/>
            <Option type="QString" name="joinstyle" value="bevel"/>
            <Option type="QString" name="line_style" value="solid"/>
          </Option>
        </layer>
      </symbol>''')

    qml = f'''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.40.7" styleCategories="AllStyleCategories">
  <renderer-v2 type="graduatedSymbol" symbollevels="0" forceraster="0" enableorderby="0" attr="ELEV" graduatedMethod="GraduatedColor">
    <ranges>
      {''.join(ranges_xml)}
    </ranges>
    <symbols>
      {''.join(symbols_xml)}
    </symbols>
  </renderer-v2>
</qgis>
'''
    Path(out_path).write_text(qml, encoding="utf-8")
    return Path(out_path)


def write_contour_qml_for_city(gdf: gpd.GeoDataFrame, shp_path: Path, **kwargs) -> Path:
    """``write_contour_qml()`` writing to ``{shp_path}`` with its extension swapped for ``.qml``."""
    return write_contour_qml(gdf, Path(shp_path).with_suffix(".qml"), **kwargs)


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
