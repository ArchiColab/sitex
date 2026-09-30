"""Open Buildings 2.5D Temporal (Google) acquisition -- anonymous GCS bucket, no auth.

An AOI can be covered by several tiles, some in a neighbouring UTM zone. Tiles in the
AOI's own zone fill first; other-zone tiles are gap-fillers that may only fill pixels
still empty, never overwrite. Taking only the first matching tile would drop most of
the AOI, a plain cross-CRS composite would let nodata overwrite valid pixels, and
using the local zone alone would miss coverage its tile grid doesn't reach.

Pure-Python ``s2sphere`` is used for the S2 token (not ``s2geometry``, which needs a
C-extension build) -- matches Google's own manifest scheme (``_MANIFEST_S2_LEVEL = 2``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyproj
import rasterio
import rasterio.windows
import requests
import s2sphere
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import from_bounds
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform

from sitex.data.aoi import AOI

OB_BUCKET = "open-buildings-temporal-data"
OB_DEFAULT_YEAR = 2023  # most recent year in the 2016-2023 dataset
OB_NODATA = -99.0
OB_DOWNSAMPLE = 8  # ~0.5m native pixels x 8 = ~4m/px, matching the dataset's stated effective resolution


def s2_level2_tokens_for_bbox(bbox: tuple[float, float, float, float], level: int = 2) -> list[str]:
    """S2 cell tokens at ``level`` covering a WGS84 bbox -- matches the manifest scheme
    Google's own colab uses (``_MANIFEST_S2_LEVEL = 2``)."""
    west, south, east, north = bbox
    coverer = s2sphere.RegionCoverer()
    coverer.min_level = level
    coverer.max_level = level
    coverer.max_cells = 1_000_000
    region = s2sphere.LatLngRect(
        s2sphere.LatLng.from_degrees(south, west),
        s2sphere.LatLng.from_degrees(north, east),
    )
    return [cell.to_token() for cell in coverer.get_covering(region)]


def list_manifests(token: str, year: int) -> list[str]:
    """Anonymous GCS JSON API listing -- candidate manifest object names for one S2
    token and year (one manifest per UTM zone the token spans)."""
    url = f"https://storage.googleapis.com/storage/v1/b/{OB_BUCKET}/o"
    resp = requests.get(url, params={"prefix": f"v1/manifests/{token}_"}, timeout=30)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return [it["name"] for it in items if f"_{year}_" in it["name"]]


def find_intersecting_tiles(manifest_name: str, aoi_geom) -> tuple[str, list[tuple[str, object]]]:
    """Parse one manifest's tile grid and return ``(crs, [(object_path, tile_polygon), ...])``
    for tiles that actually intersect ``aoi_geom`` (WGS84)."""
    url = f"https://storage.googleapis.com/{OB_BUCKET}/{manifest_name}"
    manifest = requests.get(url, timeout=30).json()

    crs = None
    tile_polys = []
    for tileset in manifest["tilesets"]:
        for src in tileset["sources"]:
            if crs is None:
                crs = tileset["crs"]
            at = src["affineTransform"]
            transform = rasterio.Affine(at["scaleX"], 0, at["translateX"], 0, at["scaleY"], at["translateY"])
            w, h = src["dimensions"]["width"], src["dimensions"]["height"]
            corners = [transform * c for c in [(0, 0), (w, 0), (w, h), (0, h)]]
            object_path = manifest["uriPrefix"] + src["uris"][0]
            tile_polys.append((object_path, shape({"type": "Polygon", "coordinates": [corners]})))

    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    aoi_proj = shp_transform(transformer.transform, aoi_geom)
    return crs, [(u, poly) for u, poly in tile_polys if poly.intersects(aoi_proj)]


def _read_and_reproject_tile(tile_url, aoi_geom, dst_transform, dst_crs, canvas_shape, nodata_value, downsample):
    """Windowed+overview read of one tile, reprojected onto the destination grid.
    Returns ``(reprojected_array, band_descriptions)`` or ``(None, None)`` if the tile
    has no real overlap with the AOI."""
    vsicurl_url = f"/vsicurl/{tile_url.replace('gs://', 'https://storage.googleapis.com/')}"
    with rasterio.open(vsicurl_url) as src:
        transformer = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        aoi_proj = shp_transform(transformer.transform, aoi_geom)
        window = from_bounds(*aoi_proj.bounds, transform=src.transform)
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        if window.width < 1 or window.height < 1:
            return None, None

        out_h = max(1, round(window.height / downsample))
        out_w = max(1, round(window.width / downsample))
        data = src.read(window=window, out_shape=(src.count, out_h, out_w), resampling=Resampling.average)
        tile_transform = rasterio.windows.transform(window, src.transform) * rasterio.Affine.scale(
            window.width / out_w, window.height / out_h
        )
        src_crs, src_nodata, descriptions = src.crs, src.nodata, src.descriptions

    reprojected = np.full(canvas_shape, nodata_value, dtype="float32")
    for b in range(canvas_shape[0]):
        reproject(
            source=data[b], destination=reprojected[b],
            src_transform=tile_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            src_nodata=src_nodata, dst_nodata=nodata_value,
            resampling=Resampling.nearest,
        )
    return reprojected, descriptions


def fetch_open_buildings_clip(
    aoi: AOI,
    output_path: Path,
    year: int = OB_DEFAULT_YEAR,
    downsample: int = OB_DOWNSAMPLE,
) -> Path:
    """End-to-end fetch: S2 token -> candidate manifests -> intersecting tiles ->
    canvas anchored to ``aoi``'s true bounds in ``aoi.local_epsg``, same-zone tiles
    filled first, other-zone tiles used strictly as fill-only-where-still-empty
    gap-fillers. Writes a multi-band ``float32`` GeoTIFF (``building_height``,
    ``building_presence``, ... -- see band descriptions) and returns its path."""
    aoi_geom = box(*aoi.bbox)
    local_crs = rasterio.CRS.from_epsg(aoi.local_epsg)

    tokens = s2_level2_tokens_for_bbox(aoi.bbox)
    manifest_names = []
    for token in tokens:
        manifest_names.extend(list_manifests(token, year))

    matching_tiles = []
    for manifest_name in manifest_names:
        tile_crs, matches = find_intersecting_tiles(manifest_name, aoi_geom)
        matching_tiles.extend((tile_crs, u) for u, _ in matches)
    if not matching_tiles:
        raise RuntimeError(f"No Open Buildings tiles intersect bbox={aoi.bbox} for year={year}")

    transformer = pyproj.Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    aoi_local = shp_transform(transformer.transform, aoi_geom)
    txmin, tymin, txmax, tymax = aoi_local.bounds
    px = 0.5 * downsample  # native 0.5m pixels x downsample factor
    out_w = max(1, round((txmax - txmin) / px))
    out_h = max(1, round((tymax - tymin) / px))
    dst_transform = rasterio.transform.from_bounds(txmin, tymin, txmax, tymax, out_w, out_h)

    local_tiles = [u for crs, u in matching_tiles if crs == f"EPSG:{aoi.local_epsg}"]
    other_tiles = [u for crs, u in matching_tiles if crs != f"EPSG:{aoi.local_epsg}"]
    print(f"{len(local_tiles)} same-zone tile(s), {len(other_tiles)} other-zone tile(s) available as gap-fillers")

    canvas = None
    tile_descriptions = None

    def apply_tile(tile_url, role):
        nonlocal canvas, tile_descriptions
        canvas_shape = canvas.shape if canvas is not None else (3, out_h, out_w)
        reprojected, descriptions = _read_and_reproject_tile(
            tile_url, aoi_geom, dst_transform, local_crs, canvas_shape, OB_NODATA, downsample
        )
        if reprojected is None:
            return
        if canvas is None:
            canvas = np.full(canvas_shape, OB_NODATA, dtype="float32")
        if tile_descriptions is None:
            tile_descriptions = descriptions
        empty = canvas[0] == OB_NODATA
        n_filled = int((empty & (reprojected[0] != OB_NODATA)).sum())
        for b in range(canvas.shape[0]):
            band_empty = canvas[b] == OB_NODATA
            canvas[b][band_empty] = reprojected[b][band_empty]
        print(f"  [{role}] {tile_url.rsplit('/', 1)[-1]}: filled {n_filled:,} px")

    for tile_url in local_tiles:
        apply_tile(tile_url, "same-zone")

    if canvas is not None:
        nodata_before = int((canvas[0] == OB_NODATA).sum())
        if nodata_before:
            pct = 100 * nodata_before / canvas[0].size
            print(f"  {nodata_before:,} / {canvas[0].size:,} px ({pct:.1f}%) still empty after same-zone "
                  f"tiles -- filling from other-zone tiles as gap-fillers only")
            for tile_url in other_tiles:
                apply_tile(tile_url, "gap-fill")

    if canvas is None:
        raise RuntimeError(f"No Open Buildings tile actually overlapped bbox={aoi.bbox} after windowed reads")

    remaining = int((canvas[0] == OB_NODATA).sum())
    if remaining:
        pct = 100 * remaining / canvas[0].size
        print(f"  {remaining:,} px ({pct:.2f}%) still empty -- no available tile covers this "
              f"(a real dataset gap, not a bug)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": out_h, "width": out_w, "count": canvas.shape[0],
        "dtype": "float32", "crs": local_crs, "transform": dst_transform, "nodata": OB_NODATA,
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(canvas)
        for i, desc in enumerate(tile_descriptions, start=1):
            dst.set_band_description(i, desc)

    print(f"-> {canvas.shape} -> {output_path}")
    return output_path
