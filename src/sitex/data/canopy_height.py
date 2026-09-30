"""Meta/WRI CHMv2 tree canopy height (2026) acquisition — global, ~1m, no auth needed.

Public, unauthenticated AWS S3 bucket (``dataforgood-fb-data``) -- same "skip Google
Earth Engine" approach as :mod:`sitex.data.landcover`, streamed via GDAL's
``/vsicurl/``. Tile lookup uses the dataset's own ``tiles.geojson`` manifest (Bing
quadkey tile IDs + bbox per tile) -- the same per-tile-manifest pattern as Open
Buildings' S2 tokens, just a different tiling scheme. Verified directly against the
live bucket (not from documentation) before writing this module: tiles are
EPSG:3857, ~1.19m/pixel (~1m true ground resolution), ``uint8`` with the pixel value
*being* canopy height in metres (0 = non-vegetated) -- no scale factor.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pyproj
import rasterio
import rasterio.mask
from rasterio.merge import merge as rio_merge
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import shape

from sitex.data.aoi import AOI

CHM_BASE_URL = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3"
CHM_TILES_MANIFEST_URL = f"{CHM_BASE_URL}/tiles.geojson"


def _download_manifest(url: str, dest: Path, timeout: float = 60.0, chunk_size: int = 1 << 20) -> None:
    """Chunked download with a per-read timeout and progress prints -- a plain
    ``urlretrieve`` gives no feedback and no timeout, so a slow or stalled connection
    on this ~59MB file is indistinguishable from a genuine hang."""
    print(f"  Downloading CHMv2 tile manifest (~59MB, one-time; cached after) from {url} ...")
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp_dest, "wb") as out:
            total = resp.length or 0
            written = 0
            next_report_mb = 10
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
                if written >= next_report_mb * (1 << 20):
                    pct = f" ({100 * written / total:.0f}%)" if total else ""
                    print(f"    ...{written / (1 << 20):.0f}MB{pct}")
                    next_report_mb += 10
    except (URLError, TimeoutError) as exc:
        tmp_dest.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download CHMv2 manifest from {url}: {exc}") from exc
    tmp_dest.replace(dest)
    print(f"  Manifest saved -> {dest} ({dest.stat().st_size / (1 << 20):.1f}MB)")


def _load_chm_tile_index(manifest_cache: Path) -> list[dict]:
    """Download the CHMv2 tile manifest once (~59MB) and cache it -- every AOI after
    the first reuses the cached copy instead of re-downloading the whole global index."""
    manifest_cache = Path(manifest_cache)
    if not manifest_cache.exists():
        manifest_cache.parent.mkdir(parents=True, exist_ok=True)
        _download_manifest(CHM_TILES_MANIFEST_URL, manifest_cache)
    else:
        print(f"  Reusing cached manifest: {manifest_cache}")
    print("  Parsing manifest (~213k tiles)...")
    with open(manifest_cache, encoding="utf-8") as f:
        return json.load(f)["features"]


def chm_tiles_for_bbox(bbox: tuple[float, float, float, float], manifest_cache: Path) -> list[str]:
    """Quadkey tile IDs from the CHMv2 manifest whose bbox intersects ``bbox`` (WGS84)."""
    west, south, east, north = bbox
    tiles = []
    for feat in _load_chm_tile_index(manifest_cache):
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        txmin, txmax = min(xs), max(xs)
        tymin, tymax = min(ys), max(ys)
        if txmin < east and txmax > west and tymin < north and tymax > south:
            tiles.append(feat["properties"]["tile"])
    return tiles


def fetch_canopy_height_clip(aoi: AOI, output_path: Path, manifest_cache: Path | None = None) -> Path:
    """Stream the CHMv2 tile(s) covering ``aoi.bbox`` via ``/vsicurl/``, mosaic if more
    than one tile is needed, reproject from the source EPSG:3857 to EPSG:4326 (matches
    :func:`sitex.data.landcover.fetch_worldcover_clip`'s output CRS), clip to
    ``aoi.geojson``, and write a single-band ``uint8`` GeoTIFF -- pixel value = canopy
    height in metres."""
    output_path = Path(output_path)
    manifest_cache = Path(manifest_cache) if manifest_cache else output_path.parent / "_chmv2_tiles_manifest.geojson"

    print("Looking up CHMv2 tile(s) for this AOI...")
    tiles = chm_tiles_for_bbox(aoi.bbox, manifest_cache)
    if not tiles:
        raise RuntimeError(f"No CHMv2 tiles intersect bbox={aoi.bbox}")
    print(f"  {len(tiles)} tile(s) intersect the AOI: {tiles}")

    # GDAL has no read timeout by default over /vsicurl/ -- a stalled connection would
    # otherwise hang indefinitely with no error and no way to tell it apart from a slow
    # but working fetch.
    with rasterio.Env(GDAL_HTTP_CONNECTTIMEOUT=30, GDAL_HTTP_TIMEOUT=120):
        print("Fetching tile(s)...")
        opened = [rasterio.open(f"/vsicurl/{CHM_BASE_URL}/chm/{t}.tif") for t in tiles]
        tile_crs = opened[0].crs

        # AOI bbox reprojected into the tiles' own CRS so merge()'s bounds= restricts
        # each tile's windowed read to the AOI's small intersection, not the whole tile
        # -- same reasoning as fetch_worldcover_clip.
        to_tile_crs = pyproj.Transformer.from_crs("EPSG:4326", tile_crs, always_xy=True)
        west, south, east, north = aoi.bbox
        xs, ys = to_tile_crs.transform([west, east], [south, north])
        bounds_native = (min(xs), min(ys), max(xs), max(ys))

        mosaic, mosaic_transform = rio_merge(opened, bounds=bounds_native)
        for src in opened:
            src.close()
    print(f"  Mosaic shape: {mosaic.shape}")

    # Reproject the native-CRS mosaic to EPSG:4326.
    print("Reprojecting to EPSG:4326 and clipping to AOI...")
    dst_crs = "EPSG:4326"
    src_left, src_bottom, src_right, src_top = rasterio.transform.array_bounds(
        mosaic.shape[1], mosaic.shape[2], mosaic_transform
    )
    dst_transform, dst_width, dst_height = calculate_default_transform(
        tile_crs, dst_crs, mosaic.shape[2], mosaic.shape[1], src_left, src_bottom, src_right, src_top
    )
    reprojected = np.zeros((dst_height, dst_width), dtype=mosaic.dtype)
    reproject(
        source=mosaic[0],
        destination=reprojected,
        src_transform=mosaic_transform,
        src_crs=tile_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )

    aoi_geom = shape(aoi.geojson["geometry"])
    meta = {
        "driver": "GTiff", "dtype": reprojected.dtype, "count": 1, "crs": dst_crs,
        "transform": dst_transform, "height": dst_height, "width": dst_width,
    }
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**meta) as tmp:
            tmp.write(reprojected, 1)
        with memfile.open() as tmp:
            out_image, out_transform = rasterio.mask.mask(tmp, [aoi_geom], crop=True)
            out_meta = tmp.meta.copy()

    out_meta.update(height=out_image.shape[1], width=out_image.shape[2], transform=out_transform)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(out_image)
    print(f"Saved -> {output_path}")
    return output_path
