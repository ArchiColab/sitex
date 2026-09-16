"""ESA WorldCover 10m (2021) acquisition — global, classified, no auth needed.

Promoted from ``documentations/00-Data-Acquisition_LandCover.ipynb`` Section 3, per that
notebook's own Section 6 conclusion: WorldCover-via-S3 is the cheaper module to ship —
one static snapshot, streamed directly off its public AWS bucket via GDAL's
``/vsicurl/`` (the same range-request approach ``remote_sensing.py`` already uses for
Landsat COGs), no Google Earth Engine account, no per-scene tuning.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
import rasterio.mask
from rasterio.merge import merge as rio_merge
from shapely.geometry import shape

from sitex.data.aoi import AOI

WC_BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

WC_CLASSES = {
    10: ("Tree cover", "#006400"),
    20: ("Shrubland", "#ffbb22"),
    30: ("Grassland", "#ffff4c"),
    40: ("Cropland", "#f096ff"),
    50: ("Built-up", "#fa0000"),
    60: ("Bare / sparse vegetation", "#b4b4b4"),
    70: ("Snow and ice", "#f0f0f0"),
    80: ("Permanent water bodies", "#0064c8"),
    90: ("Herbaceous wetland", "#0096a0"),
    95: ("Mangroves", "#00cf75"),
    100: ("Moss and lichen", "#fae6a0"),
}

WC_PERMANENT_WATER_CLASS = 80


def worldcover_tiles_for_bbox(bbox: tuple[float, float, float, float], tile_size: int = 3) -> list[str]:
    """3-degree WorldCover tile codes (e.g. ``'N12E108'``) covering a WGS84 bbox."""
    west, south, east, north = bbox
    lat0 = int(math.floor(south / tile_size) * tile_size)
    lat1 = int(math.floor(north / tile_size) * tile_size)
    lon0 = int(math.floor(west / tile_size) * tile_size)
    lon1 = int(math.floor(east / tile_size) * tile_size)

    tiles = []
    for lat in range(lat0, lat1 + 1, tile_size):
        for lon in range(lon0, lon1 + 1, tile_size):
            lat_code = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
            lon_code = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            tiles.append(f"{lat_code}{lon_code}")
    return tiles


def fetch_worldcover_clip(aoi: AOI, output_path: Path) -> Path:
    """Stream the WorldCover tile(s) covering ``aoi.bbox`` via ``/vsicurl/``, mosaic if
    more than one tile is needed, clip to ``aoi.geojson``, and write a single-band
    GeoTIFF (EPSG:4326)."""
    tiles = worldcover_tiles_for_bbox(aoi.bbox)

    opened = []
    for tile in tiles:
        url = f"/vsicurl/{WC_BASE_URL}/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
        try:
            opened.append(rasterio.open(url))
        except Exception:
            continue

    if not opened:
        raise RuntimeError(f"No WorldCover tiles reachable for bbox={aoi.bbox} (tried {tiles})")

    # `bounds=aoi.bbox` restricts the windowed read on each 36000x36000-pixel tile to
    # the AOI's small intersection — without it, `merge()` reads each *entire* tile
    # (~1.3GB) over the network before the later `rasterio.mask.mask()` crop ever runs.
    mosaic, mosaic_transform = rio_merge(opened, bounds=aoi.bbox)
    meta = opened[0].meta.copy()
    meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=mosaic_transform, count=1)
    for src in opened:
        src.close()

    aoi_geom = shape(aoi.geojson["geometry"])
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**meta) as tmp:
            tmp.write(mosaic)
        with memfile.open() as tmp:
            out_image, out_transform = rasterio.mask.mask(tmp, [aoi_geom], crop=True)
            out_meta = tmp.meta.copy()

    out_meta.update(height=out_image.shape[1], width=out_image.shape[2], transform=out_transform)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(out_image)
    return output_path


def align_categorical_to_grid(
    src_path: Path,
    dst_transform,
    dst_crs,
    dst_shape: tuple[int, int],
) -> np.ndarray:
    """Reproject a categorical raster (e.g. WorldCover classes) onto another raster's
    exact grid — nearest-neighbour, since the values are class codes, not continuous
    quantities. Returns the aligned array; ``0`` marks pixels outside the source
    raster's extent."""
    from rasterio.warp import Resampling, reproject

    aligned = np.zeros(dst_shape, dtype=np.uint8)
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    return aligned
